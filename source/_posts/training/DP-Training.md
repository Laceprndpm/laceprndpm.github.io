---
title: DP Training 从 DDP 梯度累积到 ZeRO 分片
date: 2026-10-27
tags:
  - AI-Infra
  - Training
  - DDP
  - Megatron
  - ZeRO
  - FSDP
categories:
  - 分布式训练
mathjax: true
---

以 8 个 rank、每卡累积 4 个 microbatch 为例，从普通 DDP 的完整状态副本出发，沿梯度同步、分片更新和参数聚集理解 ZeRO。重点展开 ZeRO-1 与 Megatron distributed optimizer，再用 ZeRO-2 过渡到 ZeRO-3/FSDP 的参数生命周期。

<!-- more -->

# DP Training 从 DDP 梯度累积到 ZeRO 分片

[basic-train](basic-train.md) 已经串起 Forward、Backward 和 Optimizer step，并用 $\Psi$ 表示参数量。进入数据并行后，先保持模型计算不变，让不同 rank 处理不同数据，再讨论各卡为什么重复保存训练状态，以及这些状态怎样逐步分片。

本文先设 TP、PP、CP 均为 1，DP 组大小为 $N=8$，一张 GPU 对应一个 rank。数值例沿用 2 → 3 → 2 的 ReLU MLP；显存账本采用 BF16 模型权重、FP32 梯度、显式 FP32 主权重和 FP32 Adam 状态。Megatron 源码对照固定在 `core_v0.12.0` 的提交 `d580efc68a9f0dbf1945f834f6f6200cd01d3343`。

## 1 同一个模型处理不同的数据

### 1.1 global batch 如何分到各个 rank

设每个 microbatch 有 $b_\mu=2$ 个样本，每卡累积 $K=4$ 次，则一次参数更新对应：

$$
B_{\mathrm{local}}=K b_\mu=8,\qquad
B_{\mathrm{global}}=N K b_\mu=64.
$$

把全局样本编号为 0～63，可安排为：

| rank | microbatch 0 | microbatch 1 | microbatch 2 | microbatch 3 |
| --- | --- | --- | --- | --- |
| 0 | 0、1 | 2、3 | 4、5 | 6、7 |
| 1 | 8、9 | 10、11 | 12、13 | 14、15 |
| 2 | 16、17 | 18、19 | 20、21 | 22、23 |
| … | … | … | … | … |
| 7 | 56、57 | 58、59 | 60、61 | 62、63 |

各 rank 都运行完整的同一个模型，但输入数据不同。每个 rank 的四次 backward 共同服务于一次 optimizer step，四次计算期间参数保持不变。DP 的核心是这种数据分工；PyTorch DDP 主要负责梯度同步，输入划分由数据管线负责，常见入口是 `DistributedSampler`。[DDP 接口][ddp-api]

本例每次完成一个 microbatch 的 `F → B`，便可释放本次反向不再需要的激活，再处理下一个。保留下来的是累积梯度，因此激活峰值可以小于一次处理本卡全部 8 个样本时的峰值。

### 1.2 梯度必须对应同一个全局目标

令 $L_{r,k}$ 为 rank $r$ 上第 $k$ 个 microbatch 的平均 loss，$g_{r,k}=\nabla_\theta L_{r,k}$。所有 microbatch 等大时：

$$
L=\frac{1}{NK}\sum_{r=0}^{N-1}\sum_{k=0}^{K-1}L_{r,k},
\qquad
g=\frac{1}{N}\sum_r\left(\frac{1}{K}\sum_k g_{r,k}\right).
$$

本地先得到：

$$
g_r=\frac14\left(g_{r,0}+g_{r,1}+g_{r,2}+g_{r,3}\right),
$$

再跨 8 个 rank 平均。代码中通常将每个 microbatch 的平均 loss 除以 4，由普通 DDP 完成跨 rank 的平均。若各 microbatch 的有效样本数或 token 数不同，应按实际计数加权，不能直接平均各自的均值。

归约得到的是 $g$，优化器随后根据 $g$ 和已有状态计算参数更新量。Adam 还依赖 $m/v/t$，因此完整的等价性要求参数、优化器状态、归一化和更新次数一致。

## 2 DDP 如何完成一次更新

### 2.1 从重复状态到延迟梯度同步

在所选混合精度配置下，每个 rank 都有一份完整的模型权重、主权重和 Adam 状态；梯度值则因输入不同，先在本地产生，再通过通信达成一致。

```text
rank 0：完整 W / Wmain / m / v ── 本地样本  0～ 7 ──▶ g₀
rank 1：完整 W / Wmain / m / v ── 本地样本  8～15 ──▶ g₁
  ...
rank 7：完整 W / Wmain / m / v ── 本地样本 56～63 ──▶ g₇
                                                     │
                                     跨 rank 求和并除以 8
                                                     ▼
                                      各 rank 都持有完整平均梯度 g
                                                     │
                                      各自执行相同的 optimizer step
```

这里的主权重由混合精度训练实现管理，DDP 本身不会自动为每个参数创建额外的 FP32 主副本。

若每个 microbatch 的 backward 都同步完整梯度，一个累积窗口就同步四轮。即使 microbatch 很小，梯度张量仍与参数同形状，并不会随着样本数等比例缩小。

可在前三次使用 `no_sync()`，第四次恢复同步：

```text
清空梯度
  F₀ → B₀：本地累积，不同步梯度
  F₁ → B₁：本地累积，不同步梯度
  F₂ → B₂：本地累积，不同步梯度
  F₃ → B₃：累积最后一份梯度，按 bucket 同步
等待本次同步完成
按需处理梯度并执行一次 optimizer step
```

`no_sync()` 要同时包含 forward 和 backward，退出后首次正常的前后向会同步累积梯度。[DDP no_sync][ddp-no-sync]

下面是模型已包装为 DDP 后的一次更新。`microbatches` 是本 rank 的四份数据，`loss_fn` 返回每个 microbatch 的平均 loss：

```python
from contextlib import nullcontext

optimizer.zero_grad(set_to_none=True)
for k, (x, y) in enumerate(microbatches):
    context = ddp.no_sync() if k < 3 else nullcontext()
    with context:
        prediction = ddp(x)
        loss = loss_fn(prediction, y) / 4
        loss.backward()

# 若开启 AMP，退出 no_sync 且最后一次同步 backward 完成后，先 unscale，再执行 clip_grad_norm_。
torch.nn.utils.clip_grad_norm_(ddp.parameters(), max_norm=0.25)
optimizer.step()
```

梯度同步从四轮变成一轮，同样 bucket 和 dtype 下，这部分通信量约减少 75%。

### 2.2 Bucket 和 最后一次 backward 与通信重叠

DDP 将参数梯度组织成 bucket，并通过 autograd hook 跟踪就绪状态。最后一次 backward 中，一个 bucket 内的参数梯度全部就绪后，就可以发起对应的异步 AllReduce；其余层继续反向计算。各 rank 按一致的 bucket 顺序发起 collective。[DDP 设计说明][ddp-design]

下面用多层、多 bucket 模型示意这一过程：

```text
计算：  后面几层的 backward ──▶ 前面几层的 backward ──▶ 等待通信尾部
                  │                       │
通信：            └─ bucket A AllReduce   └─ bucket B AllReduce
```

一个 bucket 可以打包多个参数，边界不一定与层边界重合。

bucket 太大，往往要等更多梯度就绪才能启动；bucket 太小，则会增加调用次数和小消息开销。判断是否有效，需看通信尾部是否缩短，以及并发通信是否拖慢了 backward。

### 2.3 更新权重

最后一次同步 backward 返回时，各 rank 的对应参数持有相同的平均梯度。初始参数和优化器状态相同，各卡随后独立执行相同的更新，就能保持参数一致。普通 DDP 因而不需要在每次 step 后再广播一遍全部新权重。[DDP 设计说明][ddp-design]

### 2.4 用同一个参数观察同步结果

配套实验沿用 basic-train 的初始 MLP，采用第 1.1 节的样本划分。第一次更新时，第一层 $W_1[0,0]$ 在各 rank 完成四次本地累积后的梯度为：

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 本地平均梯度 | 1.012375 | 1.383125 | 2.241375 | 1.698625 | 1.037000 | 2.009500 | 1.906750 | 1.296750 |

跨 rank 平均后，该参数的梯度为：

$$
g_{W_1[0,0]}=1.5731875.
$$

这是裁剪之前的梯度。单进程一次处理相同的 64 个样本，也得到同一个结果。通信改变了梯度的位置和汇总方式，没有改变求导所对应的目标函数。

## 3 ZeRO-1 把更新状态分给各个 rank

### 3.1 重复保存的状态能否只保留一份

DDP 中，各 rank 最终用相同梯度更新相同参数，因此也重复保存相同的主权重和 $m/v$。对标准逐元素 Adam，可以把参数索引集合拆成不相交的 $I_0,\ldots,I_{N-1}$：rank $r$ 只维护和更新 $I_r$ 对应的主权重及优化器状态。

各 rank 仍需要完整的低精度模型权重来执行本地 forward/backward。改变的是谁负责更新，而不是把 Linear 的矩阵乘拆成八份。[ZeRO 阶段定义][zero]

```text
                     DDP                          ZeRO-1
rank 0：      W 全量，Wmain/m/v 全量       W 全量，Wmain/m/v 的 I₀ 分片
rank 1：      W 全量，Wmain/m/v 全量       W 全量，Wmain/m/v 的 I₁ 分片
  ...
rank 7：      W 全量，Wmain/m/v 全量       W 全量，Wmain/m/v 的 I₇ 分片
```

承接前篇的每卡 $18\Psi$ 字节账本，ZeRO-1 将其中 $12\Psi$ 的主权重和 $m/v$ 均分到 $N$ 个 rank：

$$
M_{\mathrm{DDP}}=18\Psi,\qquad
M_{\mathrm{ZeRO1}}=2\Psi+4\Psi+\frac{12\Psi}{N}.
$$

取 $N=8$，每卡从 $18\Psi$ 降至 $7.5\Psi$ 字节。这里统计模型状态；激活与运行时临时空间另计。

### 3.2 ReduceScatter 将平均梯度送给负责更新的 rank

如果 rank $r$ 只更新 $I_r$，它就只需要这部分参数的全局平均梯度。无需先让所有 rank 都拿到完整平均梯度，再丢弃大部分结果。

用 ReduceScatter 将“归约”和“分发结果分片”结合：

$$
\bar g^{(r)}
=\left[\frac1N\sum_{q=0}^{N-1}g_q\right]_{I_r}.
$$

这里的求和发生在各 rank 的相同参数位置上，随后每个 rank 只得到自己负责的结果区间。调用采用 `SUM` 时要显式包含 $1/N$；也可以由框架在通信前缩放或使用平均归约实现。

```text
rank 0 本地梯度：[g₀(I₀) | g₀(I₁) | ... | g₀(I₇)]
rank 1 本地梯度：[g₁(I₀) | g₁(I₁) | ... | g₁(I₇)]
  ...
rank 7 本地梯度：[g₇(I₀) | g₇(I₁) | ... | g₇(I₇)]
                           │ ReduceScatter + 平均
                           ▼
rank 0 得到 ḡ(I₀)，rank 1 得到 ḡ(I₁)，...，rank 7 得到 ḡ(I₇)
```

梯度累积仍可保持四次本地 backward、最后一轮同步。ReduceScatter 可以按 bucket 随最后一次 backward 推进，而不必把所有通信集中在末尾。[Megatron 梯度同步代码][meg-sync]

### 3.3 分片更新后为什么还需要 AllGather

DDP 做一次 AllReduce，ZeRO-1 做一次 ReduceScatter 再加一次 AllGather，后者是否多了一份通信？先看 Ring AllReduce：它本身就由 ReduceScatter 和 AllGather 两个阶段组成。前半段得到各 rank 的归约结果分片，后半段将这些分片汇集到每个 rank。[NCCL collective 语义][nccl-collectives]

两种训练安排的区别在于优化器插在哪里，以及后半段汇集什么：

```text
DDP：    ReduceScatter 梯度 → AllGather 梯度 → 每卡完整更新
ZeRO-1： ReduceScatter 梯度 → 各卡分片更新   → AllGather 新参数
```

ZeRO-1 将完整梯度的汇集换成新参数的汇集，从而让各 rank 只维护自己负责的主权重和优化器状态。若梯度与模型参数的通信字节数相同，两种安排的主要通信量相同。[ZeRO 通信分析][zero-paper]

用 Ring 统一计量：记完整梯度大小为 $S_g$ 字节，完整模型参数大小为 $S_w$ 字节。一次 ReduceScatter 或 AllGather 都经过 $N-1$ 轮，每轮发送完整张量的 $1/N$。按每 rank 发送的有效载荷计算：

$$
V_{\mathrm{DDP}}=2\frac{N-1}{N}S_g,\qquad
V_{\mathrm{ZeRO1}}=\frac{N-1}{N}S_g+\frac{N-1}{N}S_w.
$$

本文使用 FP32 梯度和 BF16 模型参数，即 $S_g=4\Psi$、$S_w=2\Psi$。代入 $N=8$：

| 方案 | 每 rank 的主要通信发送量 |
| --- | ---: |
| DDP 梯度 AllReduce | $2\times\frac78\times4\Psi=7\Psi$ 字节 |
| ZeRO-1 梯度 ReduceScatter 与参数 AllGather | $\frac78\times(4\Psi+2\Psi)=5.25\Psi$ 字节 |

接收量与发送量相同，若统计收发总量，两种方案的结果都再乘 2。这里 ZeRO-1 的参数 AllGather 使用更低精度，因此字节数还小于 DDP；关键收益是把更新放到两个通信阶段之间，在减少状态副本的同时维持低通信量。

各 rank 用自己的平均梯度更新自己的 FP32 主权重和 $m/v$，再把新参数转换成模型计算所需的精度。此时每个 rank 只有一部分模型权重是新值，下一轮计算前需要将这些新分片汇集起来：

```text
本地四次 F/B
      │
梯度 ReduceScatter
      │
各 rank 拿到自己的全局平均梯度分片
      │
处理梯度，更新本地 Wmain / m / v 分片
      │
将更新后的主权重分片转换为模型权重分片
      │
参数 AllGather
      │
各 rank 恢复完整的新模型权重，进入下一轮
```

AllGather 汇集的是更新后的参数值，不对不同参数分片求和。原来 DDP 是各卡做完同样的完整更新；这里各卡分工更新，再交换结果。[Megatron distributed optimizer][meg-dist-doc]

对具体层而言，约束是“下一次使用它之前，新参数必须可用”。实现可以把参数 AllGather 与下一次 forward 分段重叠，不必把一次全模型参数聚集视为不可拆分的全局关卡。

### 3.4 分片梯度的 AMP 与 L2 范数裁剪

普通 DDP 同步后，每个 rank 都持有完整平均梯度，可以直接本地计算全模型的 L2 范数。ZeRO-1 的 ReduceScatter 结果则按 rank 分片，需要汇总各分片的范数统计。

若使用 loss scaling，同一个累积窗口及参与归约的 rank 使用相同的 scale；最后一次梯度同步完成后，先 unscale，再计算范数和裁剪。[AMP 累积与裁剪][amp]

每个 rank 对自己负责的梯度元素求平方和，再对这些不重复的贡献做标量归约：

$$
s_r=\sum_{i\in I_r}\bar g_i^2,\qquad
\|\bar g\|_2=\sqrt{\sum_r s_r}.
$$

所有 rank 用同一个全局范数计算裁剪系数，再作用于各自的梯度分片。普通 `clip_grad_norm_` 只统计传入的张量，各 rank 独立对分片调用它，会把全模型阈值变成多个局部阈值。DDP 的完整梯度已在各卡重复持有，其平方和只需本地计算，无需再跨 DP rank 累加。

分片也改变了溢出判断：某个 rank 检测到 Inf/NaN 时，其他 rank 的梯度分片可能仍然有限。应将各 rank 的非有限值标志按 `MAX` 归约，统一决定跳过本次更新；全部有效时，再使用全局范数裁剪并更新参数。[Megatron 溢出检查][meg-overflow]

本节只有一个 DP 维度；组合 TP、共享参数等布局时，还要确认范数统计中每份贡献的归属。

### 3.5 在 Megatron 中对应哪些对象

以下沿常规 distributed optimizer、单个优化器实例和显式 FP32 主权重路径阅读：

| 代码入口 | 对应动作 |
| --- | --- |
| `DistributedDataParallel.no_sync()` | 标记当前不是最后一个 microbatch，暂缓梯度同步 |
| `_ParamAndGradBucketGroup.start_grad_sync()` | 普通 DP 使用 AllReduce，启用 distributed optimizer 时使用 ReduceScatter |
| `DistributedOptimizer._build_model_and_main_param_groups()` | 建立模型参数分片和对应主权重分片，交给底层优化器 |
| `_copy_model_grads_to_main_grads()` | 将本 rank 对应的归约结果提供给主参数分片 |
| `step_with_ready_grads()` | 更新分片，并按配置组织后续参数同步 |
| `_ParamAndGradBucketGroup.start_param_sync()` | AllGather 更新后的模型参数分片 |

[no_sync][meg-no-sync]、[主权重分片构造][meg-shards]、[梯度分片传递][meg-grads]、[step 与参数同步][meg-step]、[参数 AllGather][meg-gather] 可以按表中的数据流串起来阅读。

**出现 ReduceScatter，并不自动意味着梯度物理存储也已缩小到 $1/N$。** Megatron 的这条路径仍有完整的连续梯度 buffer：归约后只有本 rank 对应区间是它需要的完整结果，其余区间不会因此自动释放。官方 BF16 参数、FP32 梯度的账本仍保留 $4\Psi$ 的梯度项，对应前面的 $6\Psi+12\Psi/N$。[Megatron distributed optimizer][meg-dist-doc]

参数分片按 buffer 区间划分，也不要求“一整个层只交给一张卡”。一个参数张量可以跨越多个 rank 的更新区间。实际布局中的 padding、bucket 边界和对齐，决定分片如何映射回模型张量。

## 4 ZeRO-2 进一步减少梯度驻留

上一节中，Megatron 已经使用 ReduceScatter，却仍为每个 rank 保留 $4\Psi$ 的完整梯度 buffer。这块空间让各参数的梯度在本地跨 $K$ 个 microbatch 持续累积，直到最后一次 backward 才统一归约，因此每个 optimizer step 只需一轮梯度 ReduceScatter。

要进一步将梯度驻留降到 $4\Psi/N$，典型 ZeRO-2 路径会在每个 microbatch 产生梯度后，尽早将它归约并交给负责更新的 rank，随后释放非本 rank 的部分。这样本地只需累积自己的梯度分片，但同一训练步的梯度 ReduceScatter 就从一轮增加到 $K$ 轮。[DeepSpeed 梯度归约策略][ds-coalesce]

**这里的取舍是：保留完整梯度，用显存换取跨 microbatch 的本地累积；尽早归约并分片，用重复通信换取更小的梯度驻留。** 本例 $K=4$，后者执行四轮梯度归约，而更新后的参数 AllGather 仍是一轮。

组合 PP 时，流水线还会利用多个 microbatch 摊薄启动与排空阶段的气泡。在较深流水线、较多梯度累积的配置中，逐 microbatch 归约的代价因而更突出。[Megatron 流水线分析][megatron-pp]

将梯度项从 $4\Psi$ 降为 $4\Psi/N$ 后，理想状态账本变为：

$$
M_{\mathrm{ZeRO2}}
=2\Psi+\frac{4\Psi+12\Psi}{N}
=2\Psi+\frac{16\Psi}{N}.
$$

逐 microbatch 归约并累积分片的时间线为：

```text
microbatch 0：F/B → ReduceScatter → 累积本 rank 梯度分片
microbatch 1：F/B → ReduceScatter → 累积本 rank 梯度分片
microbatch 2：F/B → ReduceScatter → 累积本 rank 梯度分片
microbatch 3：F/B → ReduceScatter → 累积本 rank 梯度分片
                                           │
                                分片更新 → 参数 AllGather
```

若改为四次都先在本地累积，再统一归约，累积窗口内就要保留完整本地梯度，不能继续只按 $4\Psi/N$ 估算该窗口的梯度驻留。

这属于执行策略的选择。DeepSpeed 0.19.9 文档中的 `coalesce_grad_reduction()` 支持把 ZeRO-2/3 的多次 backward 梯度归约合并到窗口末尾，同时保留完整本地梯度；该接口与普通 `no_sync()` 的支持范围不同，也有优化器 wrapper 和流水线限制。[DeepSpeed 梯度归约合并][ds-coalesce]

从这个开销模型看，当 $K$ 较大、完整梯度 buffer 能够容纳且通信较紧张时，ZeRO-1 的本地累积更有吸引力；当梯度驻留限制了模型或 microbatch 大小时，ZeRO-2 则提供进一步压缩显存的选择。

## 5 ZeRO-3 与 FSDP 把参数也纳入生命周期管理

### 5.1 参数平时分片，计算时聚集

ZeRO-3 再将低精度模型参数分片。训练状态不再要求每张卡长期保留一套完整模型，但一个普通 Linear 执行本地计算时仍需要对应的完整权重。因此需要在算子使用参数之前聚集它，用完后按策略重新分片。[ZeRO 阶段定义][zero]

这与 TP 的分工不同：TP 可以让各 rank 用权重分片执行局部矩阵乘；这里讨论的数据并行分片，则是先恢复当前计算单元需要的参数，再处理本 rank 的数据。

PyTorch FSDP 是这一类分片思想的实现。下面用经典 FSDP1 `FULL_SHARD` 说明时间线。参数通常按 FSDP unit 分组管理，一个 unit 可以包含一个或多个层；不需要为每次计算把全模型一次性聚集起来。[FSDP 接口][fsdp]

```text
某个 unit 的一次 microbatch

持有参数分片
      │ 参数 AllGather
      ▼
完整参数 → Forward
      │ 重新分片，释放完整参数视图对应的临时存储
      ▼
等待该 unit 的 Backward
      │ 参数 AllGather
      ▼
完整参数 → Backward → 梯度 ReduceScatter
      │
重新分片参数，保留本 rank 的梯度分片
```

前向保存的激活仍要服务于反向，参数重新分片不等于同时释放所有激活。参数、激活和梯度是三条不同的生命周期。

### 5.2 哪一种通信可以延迟

FSDP 的两种主要通信承担不同依赖：

| 通信 | 生产什么 | 谁需要它 |
| --- | --- | --- |
| 参数 AllGather | 当前 unit 的完整参数 | 即将执行的 forward/backward |
| 梯度 ReduceScatter | 各参数 owner 的归约梯度分片 | 梯度累积与后续 optimizer step |

参数没准备好，对应 unit 的计算就无法开始；而梯度通常到本次更新时才必须完整。这使两种通信具有不同的延迟空间。

经典 FSDP1 的 `no_sync()` 可以暂缓梯度同步，但会让完整梯度在窗口内累积。对按 `FULL_SHARD` 释放参数的路径，仅关闭梯度同步不会自动保留所有计算所需参数，因此不能把四个 microbatch 的全部通信都合并成最后一轮。[FSDP no_sync][fsdp-no-sync]

如果选择让完整参数留得更久，可以减少重新聚集。例如 `SHARD_GRAD_OP` 不在 forward 后立即重新分片，因而省掉对应的 backward 前参数聚集；代价是这段时间保留更多参数数据。[FSDP 分片策略][fsdp-strategy]

### 5.3 为什么小 microbatch 会使通信更突出

固定 global batch 和 DP 组大小，减小 $b_\mu$ 会增加 $K$。如果每个 microbatch 都按上面的流程重新聚集参数，同一份模型参数就会在一次 optimizer step 内反复传输，而每次聚集之后可用于覆盖通信的计算量变小。

对同一个 unit，如果预取未能在计算到达前完成，计算流就要等待参数；通信可与之前 unit 的计算重叠，但预取更多 unit 也会增加同时驻留的参数。实际调优要同时看 unit 大小、预取距离、网络带宽和 microbatch 下的 GEMM 效率。

所以这里比较的是：激活显存减少后，增加的参数传输能否被计算覆盖；节省的模型状态显存能否支持更大的 microbatch，反过来改善计算效率。分片程度更深，并不单独决定整步速度。

### 5.4 大规模训练中的实际选择

Llama 3 405B 的最多 16,384 卡训练采用包含 TP、CP、PP 和 FSDP 的四维并行。报告明确说明，为避免 backward 的额外 AllGather，他们不在 forward 后立即重新分片模型参数。这里的总卡数也不是单个 FSDP 组大小。[Llama 3 技术报告第 3.3.2 节][llama3]

高累积场景中，ZeRO-1 可将梯度归约和参数汇集集中在累积窗口边界；PyTorch FSDP 的 `SHARD_GRAD_OP` 通过保留前向参数，省去每个 microbatch 反向前的 AllGather。[FSDP 分片策略][fsdp-strategy]

## 6 将显存账本与执行时间线放在一起

### 6.1 同一 dtype 假设下的状态账本

以下沿用 BF16 模型参数、FP32 梯度、显式 FP32 主权重与 FP32 $m/v$，忽略 padding 和小型标量状态。表中是均匀分片后的逻辑存储；FSDP 的具体主副本及 buffer 复用需按实现另行核算。

| 方案 | 模型权重 | 梯度 | 主权重与 $m/v$ | 合计 | $N=8$ 时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DDP | $2\Psi$ | $4\Psi$ | $12\Psi$ | $18\Psi$ | $18\Psi$ |
| ZeRO-1 | $2\Psi$ | $4\Psi$ | $12\Psi/N$ | $6\Psi+12\Psi/N$ | $7.5\Psi$ |
| ZeRO-2 | $2\Psi$ | $4\Psi/N$ | $12\Psi/N$ | $2\Psi+16\Psi/N$ | $4\Psi$ |
| ZeRO-3 | $2\Psi/N$ | $4\Psi/N$ | $12\Psi/N$ | $18\Psi/N$ | $2.25\Psi$ |

单位均为字节。训练峰值还要考虑活跃 unit 的完整参数、激活、通信与算子临时空间；延迟梯度同步时，也要按实际驻留的完整梯度修正账本。

### 6.2 四个 microbatch 会触发哪些通信

下面按经典执行路径计算“覆盖模型相应部分的一轮通信”。一轮可包含多个 bucket/unit 的 collective；不计梯度范数、溢出标志等小型归约。

| 执行路径 | 梯度通信 | 模型参数通信 |
| --- | --- | --- |
| DDP，每次 backward 都同步 | 4 轮 AllReduce | step 后无需汇集参数 |
| DDP，前三次 `no_sync()` | 1 轮 AllReduce | step 后无需汇集参数 |
| ZeRO-1，末次同步 | 1 轮 ReduceScatter | 更新后 1 轮 AllGather |
| ZeRO-2，逐次归约并分片 | 4 轮 ReduceScatter | 更新后 1 轮 AllGather |
| 完整前后向都重新分片的 ZeRO-3/FSDP 路径 | 4 轮 ReduceScatter | 4 轮 forward 参数聚集，加 4 轮 backward 参数聚集 |

AllReduce、ReduceScatter 和 AllGather 的单轮数据量不同，梯度与参数的 dtype 也可能不同，因此不能仅凭表中的轮数比较通信字节数。它首先回答：哪些通信随累积次数 $K$ 增长，哪些集中在 optimizer step 边界。

阅读和调优时，可以按固定顺序检查：各 rank 保存什么；每个 microbatch 结束后释放了什么；哪些 collective 是下一段计算的前置条件；最终等待暴露在哪里。这样才能把“省了多少显存”与“增加了哪些依赖”对应起来。

## 附录 运行与验证

配套 [verify_dp_training.py](verify_dp_training.py) 使用 Linux 上的 8 个本地 CPU/Gloo 进程，连续执行 3 次 Adam 更新。它对照单进程 64 样本 batch，验证普通 DDP 的两种同步方式，以及教学版 `ReduceScatter → 分片 Adam → AllGather` 更新。

在笔记库根目录、已安装 PyTorch 的环境中运行：

```bash
python assets/dp-training/verify_dp_training.py --json /tmp/dp-training-verification.json
```

模型沿用前篇的 17 个参数，仅形成一个梯度 bucket；该实验检查同步次数与数值，多 bucket 的计算通信重叠由正文的机制图说明。数据按全局编号 $i=0,\ldots,63$ 构造，第 $t=0,1,2$ 步为：

$$
x_i=\left[\frac{(i\bmod11)+1}{10},\frac{((3i+t)\bmod13)+1}{10}\right],
\qquad
y_i=[0.2x_{i,0}-0.3x_{i,1},\ 0.1+0.4x_{i,0}].
$$

所有路径采用相同的平均平方误差、Adam 超参数和 0.25 的全模型梯度范数阈值。教学版为等长 collective 将 17 个参数补齐为 24 个槽位，每 rank 3 个；padding 不参与真实模型更新。这只是小模型的通信布局，不用于估计大模型的分片效率。

本次环境为 Python 3.11.15、PyTorch 2.5.1，全部数值使用 FP64。通信计数通过执行平均 AllReduce 的 DDP hook 获取，仅统计梯度 bucket。结果见 [verification.json](verification.json)：

| 检查 | 结果 |
| --- | --- |
| 各 rank 数据划分 | 64 个样本无重复、无遗漏 |
| DDP 每次同步与大 batch 参考 | 最大梯度绝对误差约 $6.66\times10^{-16}$ |
| DDP 末次同步与大 batch 参考 | 最大梯度绝对误差约 $4.44\times10^{-16}$ |
| 单 bucket 下每步梯度 AllReduce 次数 | 每次同步为 4 次，末次同步为 1 次 |
| 分片 Adam 与大 batch 参考 | 最大参数绝对误差约 $3.47\times10^{-18}$ |
| 三种路径的 Adam 状态对照 | 最大绝对误差低于 $7\times10^{-18}$ |
| 分片梯度范数 | 各 rank 的不重复平方和归约与完整梯度范数一致 |

这些是实际 CPU DDP 和教学分片实现的正确性检查。显存账本为所列 dtype 下的推导；Megatron、ZeRO-2 和 FSDP 部分依据源码及文档，没有执行其 GPU 训练或性能测量。

## 参考资料

- [PyTorch DDP 接口][ddp-api]、[no_sync][ddp-no-sync]、[内部设计][ddp-design]。
- [PyTorch AMP 梯度累积与裁剪][amp]。
- [DeepSpeed ZeRO 阶段定义][zero]、[梯度归约合并][ds-coalesce]。
- [NCCL collective 语义][nccl-collectives]、[ZeRO 原论文第 7 节][zero-paper]：AllReduce 分解与通信量。
- [Megatron distributed optimizer][meg-dist-doc]，及下列固定提交源码。
- [Megatron 流水线分析][megatron-pp]：microbatch 数量与流水线气泡。
- [PyTorch FSDP][fsdp]、[梯度同步控制][fsdp-no-sync]、[分片策略][fsdp-strategy]。
- [The Llama 3 Herd of Models][llama3]，第 3.3.2 节。

[ddp-api]: https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html
[ddp-no-sync]: https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync
[ddp-design]: https://docs.pytorch.org/docs/2.14/notes/ddp.html#internal-design
[amp]: https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-accumulation
[zero]: https://deepspeed.readthedocs.io/en/stable/zero3.html
[ds-coalesce]: https://deepspeed.readthedocs.io/en/latest/training.html#coalesced-gradient-reduction
[meg-dist-doc]: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html
[meg-no-sync]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/distributed/distributed_data_parallel.py#L424-L434
[meg-sync]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/distributed/param_and_grad_buffer.py#L279-L359
[meg-shards]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/optimizer/distrib_optimizer.py#L281-L375
[meg-grads]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/optimizer/distrib_optimizer.py#L1955-L1996
[meg-step]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/optimizer/distrib_optimizer.py#L2088-L2115
[meg-gather]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/distributed/param_and_grad_buffer.py#L193-L239
[meg-overflow]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/optimizer/optimizer.py#L382-L440
[fsdp]: https://docs.pytorch.org/docs/2.14/fsdp.html
[fsdp-no-sync]: https://docs.pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.no_sync
[fsdp-strategy]: https://docs.pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.ShardingStrategy
[llama3]: https://arxiv.org/html/2407.21783v3#S3.SS3.SSS2

[nccl-collectives]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather
[zero-paper]: https://arxiv.org/abs/1910.02054

[megatron-pp]: https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron
