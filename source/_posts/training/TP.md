---
title: TP 推理与训练
date: 2026-10-28
tags:
  - AI-Infra
  - Inference
  - Tensor-Parallel
  - Megatron
  - vLLM
  - Training
  - Sequence-Parallel
  - ZeRO
categories:
  - 分布式推理
  - 分布式训练
mathjax: true
---
从矩阵乘法的列切与行切出发，本文完整拆解了张量并行（TP）在 Transformer 推理与训练中的前反向计算图、通信原语、显存账本，以及与 GQA、序列并行（SP）和 DP/ZeRO 的协同机制。

<!-- more -->

# TP 推理与训练

## 1 推理

以下采用 dense、decoder-only、pre-norm Transformer，一张 GPU 对应一个 rank，TP 组大小为 $p$，不启用其他并行方式。数学约定为 $Y=XW$，权重按 $[d_{\mathrm{in}},d_{\mathrm{out}}]$ 表示，相关维度均可整除。

### 1.1 前向基础：输出分片与部分和

#### 1.1.1 一次矩阵乘法的两种切法

设 $X\in\mathbb{R}^{M\times d_{\mathrm{in}}}$、$W\in\mathbb{R}^{d_{\mathrm{in}}\times d_{\mathrm{out}}}$，$M$ 为输入行数。一个输出元素需要沿输入特征求和：

$$
Y_{m,j}=\sum_k X_{m,k}W_{k,j}.
$$

把输出索引 $j$ 分给不同 rank，每卡能独立完成一些输出元素；把求和索引 $k$ 分给不同 rank，每卡只能贡献这些元素的部分和。前者对应 Column Parallel，后者对应 Row Parallel。

列并行将 $W=[W_0\;W_1]$，输入 $X$ 在两卡复制：

$$
Y=XW=[XW_0\;XW_1]=[Y_0\;Y_1].
$$

![Column Parallel：完整输入乘权重列分片，得到输出分片](column-parallel.png)

图 1：Column Parallel。[Hugging Face《The Ultra-Scale Playbook》][hf-column]

行并行沿行堆叠权重分片 $W_0,W_1$，同时沿对应特征切分 $X=[X_0\;X_1]$：

$$
Y=X_0W_0+X_1W_1=P_0+P_1.
$$

![Row Parallel：输入特征与权重行对应切分，得到同一输出的部分和](row-parallel.png)

图 2：Row Parallel，图中的 $Y_0/Y_1$ 为部分和，下文记为 $P_0/P_1$。[来源][hf-row]

以图中的输入行 $[2,3]$ 为例，列并行两卡分别算完输出 $80$ 和 $180$，拼接成 $[80,180]$；行并行两卡算出 $[20,60]$ 和 $[60,120]$，相加后才得到相同结果。

| 两种切分 | 每卡输入 | 每卡权重 | 本地结果 | 恢复每卡完整输出 |
| --- | --- | --- | --- | --- |
| Column | $[M,d_{\mathrm{in}}]$，复制 | $[d_{\mathrm{in}},d_{\mathrm{out}}/p]$ | $[M,d_{\mathrm{out}}/p]$，输出分片 | AllGather，并按输出特征维组织 |
| Row | $[M,d_{\mathrm{in}}/p]$，分片 | $[d_{\mathrm{in}}/p,d_{\mathrm{out}}]$ | $[M,d_{\mathrm{out}}]$，部分和 | AllReduce(SUM) |

#### 1.1.2 通信取决于下一层需要什么

相邻层若分片方式匹配，如 Column 接 Row，中间既无需 AllGather 也无需 Scatter，直接传递本地分片即可；输入已复制时也无需再次 Broadcast。Megatron 通过 `gather_output=False` 和 `input_is_parallel=True` 表达这种衔接。[列并行][meg-column]、[行并行][meg-row]

行并行输出求和不除以 $p$。若下游需要的是归约结果的分片，可用 ReduceScatter。[collective 语义][nccl]

列并行偏置随输出切分，各卡添加 $b_r$；行并行通常先归约再添加完整 $b$，否则各卡先加 $b$ 会累计成 $p b$。逐元素激活可以作用于输出分片，却不能任意提前作用于部分和：$\phi(P_0+P_1)\ne\phi(P_0)+\phi(P_1)$。

PyTorch `Linear.weight` 按 $[d_{\mathrm{out}},d_{\mathrm{in}}]$ 存储，计算相当于 `X @ weight.T`。因此数学上的列切对应该张量第 0 维，行切对应第 1 维。[Linear 定义][linear]

### 1.2 Transformer 块：把切分串成完整前向

记隐藏维为 $h$、MLP 中间维为 $f$，query head 数为 $H_q$、KV head 数为 $H_{kv}$、每头维度为 $d$，主线取 $h=H_qd$。线性层中的 $M$ 是本次参与投影的 token 数。

#### 1.2.1 MLP：Column 与 Row 直接配对

普通两层 MLP 为：

$$
H=\phi(XW_1+b_1),\qquad Y=HW_2+b_2,
$$

其中 $W_1:[h,f]$、$W_2:[f,h]$。第一层按输出维切分，第二层沿同一中间维按输入维切分，则 rank $r$ 执行：

$$
H_r=\phi(XW_{1,r}+b_{1,r}),\qquad
P_r=H_rW_{2,r},\qquad
Y=\sum_rP_r+b_2.
$$

中间激活 $H_r:[M,f/p]$ 留在本卡，第二层输出的 $[M,h]$ 部分和需要归约。

SwiGLU 将第一层改成两个投影及逐元素门控：

$$
G=XW_g+b_g,\qquad U=XW_u+b_u,\qquad
H=\operatorname{SiLU}(G)\odot U,\qquad
Y=HW_d+b_d.
$$

$W_g,W_u:[h,f]$ 都按输出特征切分，$W_d:[f,h]$ 按输入特征切分。**同一中间特征的 gate/up 对应列与 down 对应行必须落在同一 rank**，才能在本地完成 $H_r=\operatorname{SiLU}(G_r)\odot U_r$。

若将 gate/up 打包成 $[h,2f]$ 权重，不能把 `[全部 gate | 全部 up]` 简单对半切成一卡全 gate、另一卡全 up。Megatron 在每卡保留配对的两部分，再将局部输出拆成两半相乘。[MLP 与 SwiGLU 分片][meg-mlp]

```text
每卡完整 X [M,h]
        │ Column：Wg,r、Wu,r
        ▼
Gr、Ur [M,f/p] → SiLU(Gr) ⊙ Ur → Hr [M,f/p]
        │ Row：Wd,r
        ▼
Pr [M,h] → TP AllReduce(SUM) → 每卡完整 Y [M,h]
```

#### 1.2.2 Attention：每卡拥有完整的若干个 head

先考虑 MHA，即 $H_{kv}=H_q$，且 $p$ 能整除 head 数。按完整 head 划分 Q/K/V 投影的输出列：

$$
Q_r=XW_{Q,r},\qquad K_r=XW_{K,r},\qquad V_r=XW_{V,r}.
$$

每卡拥有 $H_q/p$ 个完整 head，而非每个 head 的一段 $d$。对一个本地 head，完整计算链为：

$$
S_r=\frac{Q_rK_r^T}{\sqrt d}+\mathrm{mask},\qquad
A_r=\operatorname{softmax}(S_r),\qquad
C_r=A_rV_r.
$$

这里 $A$ 是 attention probabilities，softmax 沿 key 的序列维计算。每卡持有本地 head 所需的全部 K/V 序列位置，因此 QK、softmax、AV 都在卡内完成。[Megatron Attention 切分][megatron] RoPE 同样作用于本地完整 head。[vLLM Llama 前向][vllm-forward]

设 batch 大小为 $B$、各序列长度为 $S$，此时 $M=BS$，prefill 时每卡的逻辑形状为：

| 阶段                | 本地形状                 | 是否需要 TP 通信     |
| ----------------- | -------------------- | -------------- |
| Q/K/V 投影并重排       | $[B,H_q/p,S,d]$，MHA  | 输入已复制时无需       |
| $QK^T$ 与 softmax  | $[B,H_q/p,S,S]$，逻辑形状 | 无需，完整 key 轴在本卡 |
| $AV$              | $[B,H_q/p,S,d]$      | 无需             |
| 拼合本地 head         | $C_r:[BS,h/p]$       | 本地布局变换         |
| 输出投影 $C_rW_{O,r}$ | $[BS,h]$，部分和         | 求和得到每卡完整输出     |

最后一行来自 $W_O:[h,h]$ 的行切分：

$$
Y=[C_0\;\cdots\;C_{p-1}]
\begin{bmatrix}W_{O,0}\\\vdots\\W_{O,p-1}\end{bmatrix}
=\sum_r C_rW_{O,r}.
$$

输出投影合并各 head 对隐藏向量的贡献，归约后恢复每卡完整的 $[M,h]$ 输出。

GQA 让一组 query heads 共享一个 KV head，分配方式见第 1.4 节。decode 时，新 K/V 写入本卡对应 head 的 cache，Q 直接访问这些 head 的历史 K/V。

#### 1.2.3 残差、Norm 与两个模块之间的布局

对 pre-norm Transformer：

$$
X_{\mathrm{attn}}=X+\operatorname{Attention}(\operatorname{Norm}_1(X)),\qquad
X'=X_{\mathrm{attn}}+\operatorname{MLP}(\operatorname{Norm}_2(X_{\mathrm{attn}})).
$$

块边界的 $[M,h]$ 激活和 Norm 参数在各卡复制，LayerNorm/RMSNorm 因而能在本地沿完整隐藏维计算统计量。Attention 输出归约后与 $X$ 相加，MLP 输出归约后与 $X_{\mathrm{attn}}$ 相加。

```text
复制 X → 本地 Norm → QKV 列切 → 本地 Attention → O 行切
                                                       │ AllReduce
复制 X_attn ← 加残差 X ← 完整 Attention 输出 ←──────────────┘
   │
   └→ 本地 Norm → gate/up 列切 → 本地 SwiGLU → down 行切
                                                       │ AllReduce
复制 X' ← 加残差 X_attn ← 完整 MLP 输出 ←────────────────────┘
```

一个 Transformer block 的前向有两次 TP AllReduce：O projection 后一次，MLP down projection 后一次。[Megatron 第 3 节][megatron]

### 1.3 性能与显存占用

#### 1.3.1 分开计算权重、激活、KV cache 与缓冲区

单卡推理显存可按以下几项记账：

$$
M_{\mathrm{GPU}}=M_W+M_{\mathrm{act}}+M_{\mathrm{KV}}
+M_{\mathrm{comm/workspace}}+M_{\mathrm{runtime}}.
$$

权重常驻，KV 随请求保存，激活与临时缓冲区在消费后复用；显存峰值取决于同时存活的分配。记 $q_W$ 为权重每元素字节数，$\Psi_s$ 为可均分参数量。

| 对象 | 典型本地形状或大小 | TP 增大后的变化 |
| --- | --- | --- |
| 大矩阵权重 | 理想均分部分为 $q_W\Psi_s/p$ 字节 | 通常按 $1/p$ 减少；复制参数另算 |
| 残差、Norm 输入输出 | $[M,h]$ | 本文经典布局下仍复制 |
| SwiGLU 的 gate/up/中间结果 | 每份 $[M,f/p]$ | 按中间特征分片 |
| 本地 Attention Q 与 head 输出 | 每份 $[M,H_qd/p]$ | 按 query head 分片 |
| KV cache | 见下式 | 由本地 KV head 数决定 |
| 通信缓冲区 / workspace | 如 AllReduce 的 $[M,h]$ 输入缓冲区 | 按通信与算子配置分配 |

若不发生 KV 复制，标准 GQA+SwiGLU block 的主要参数量约为：

$$
\Psi_{\mathrm{block}}=2h^2+2hH_{kv}d+3hf.
$$

分别对应 Q/O、K/V、gate/up/down，忽略 bias 和 Norm。存在 KV 复制时，一卡主要权重更准确地写成：

$$
M_{W,\mathrm{block},r}\approx q_W
\left(\frac{2h^2+3hf}{p}+2hH_{kv,r}d\right),
$$

$H_{kv,r}$ 是本卡实际保存的 KV head 数。只有它也等于 $H_{kv}/p$ 时，整个主要权重项才都按 $1/p$ 缩放。

对 $L$ 层、每层 head 配置相同的模型，设本 TP 副本中驻留请求的历史 token 总数为 $T_{\mathrm{sum}}$，KV 每元素 $q_{KV}$ 字节，则每卡有效 KV 数据为：

$$
M_{\mathrm{KV},r}=2L\,T_{\mathrm{sum}}\,H_{kv,r}\,d\,q_{KV}.
$$

系数 2 来自 K 和 V。

例如 $L=32$、$H_q=32$、$H_{kv}=8$、$d=128$、$T_{\mathrm{sum}}=8192$，KV 使用 BF16：

| TP | 本地 Q heads | 本地 KV heads | 每卡有效 KV | 全组 KV 数据合计（含副本） |
| --- | ---: | ---: | ---: | ---: |
| 1 | 32 | 8 | 1 GiB | 1 GiB |
| 4 | 8 | 2 | 256 MiB | 1 GiB |
| 8 | 4 | 1 | 128 MiB | 1 GiB |
| 16 | 2 | 1，KV 每头跨两卡复制 | 128 MiB | 2 GiB |

按 vLLM 的 head 分配，TP 从 8 增至 16 后，Q/O 和 MLP 继续切分，每卡 KV 仍为 128 MiB，全组合计翻倍。[QKV 分配实现][vllm-qkv]

#### 1.3.2 Prefill 与 decode 的瓶颈不同

prefill 的 $M$ 是本轮处理的 prompt token 数；普通 decode 每请求追加一个 token，$M$ 约等于本轮活跃请求数，历史长度不增加投影输入行数。

设某线性层本地有 $\Psi_r$ 个权重元素，每轮从 HBM 读取一次。只计权重流量时：

$$
F_{\mathrm{linear},r}\approx2M\Psi_r,\qquad
B_{W,r}\approx q_W\Psi_r,\qquad
I_W\approx\frac{2M}{q_W}\quad\text{FLOP/byte}.
$$

TP 同时减少权重和 FLOPs，不改变这一比值；增大 $M$ 才能提高权重复用。因此，小 batch decode 即使增加 TP，仍容易受权重带宽限制。

| 阶段 | 主要工作 | TP 带来的收益 | 需要警惕的代价 |
| --- | --- | --- | --- |
| Prefill | 多 token GEMM，以及较长 query/key 轴上的 Attention | 分摊大矩阵计算、Attention heads 和临时数据 | GEMM 切得过窄后效率下降；大激活归约占带宽 |
| Decode，小活跃 batch | 小 $M$ 投影、访问历史 KV | 减少每卡权重读取及可分片 KV 读取 | 每层通信启动延迟、kernel 启动和小矩阵低利用率 |
| Decode，长历史或较大 batch | KV 读取与 Attention 工作增加 | head 分摊更有价值 | KV 复制限制收益，瓶颈可能转向 KV 带宽或计算 |

decode 的历史长度影响 KV 读取与 Attention 计算：每请求一个新 query 时，QK 与 AV 每层约需 $4T_{\mathrm{sum}}H_qd/p$ FLOPs，忽略 softmax 和 mask。GQA/MQA 减少 KV head 数；同组 query heads 在 kernel 内复用 K/V，可进一步减少重复读取。[MQA 原论文][mqa]

FlashAttention 通过分块与在线 softmax 避免在 HBM 中物化完整的 $[B,H_q/p,S,S]$ score/probability 矩阵，降低 prefill 的临时显存需求。[FlashAttention][flashattention]

#### 1.3.3 归约消息与 TP 度数

每卡输出部分和为 $[M,h]$，归约消息量 $n=q_A M h$ 不随 $p$ 缩小，其中 $q_A$ 为每元素字节数。

Ring AllReduce 的两阶段各执行 $p-1$ 步，每步传递 $n/p$ 字节，每卡发送量为 $2(p-1)n/p$，接收量相同。延迟近似为：

$$
t_{\mathrm{AR}}\approx2(p-1)\alpha+
\frac{2(p-1)}p\frac n\beta.
$$

$\alpha$ 为每步延迟，$\beta$ 为有效链路带宽。小消息更受启动延迟影响，大消息更受传输带宽影响。[AllReduce 的两阶段分解][zero-paper]

完整输出的归约未完成时，依赖它的后续计算必须等待。通信与其他就绪工作重叠，还会竞争 SM、HBM 或链路资源。

以吞吐为目标、GPU 总数固定时，可先选能容纳目标负载且满足延迟 SLA 的较小 TP 度数，将剩余 GPU 用于独立副本；容量或单请求延迟不达标时再增加 TP。

### 1.4 常见切法与适用条件

#### 1.4.1 词表维切分

embedding 和 lm_head 常沿词表维切分，各卡负责部分词条的查找或 logits。输出端可使用分布式归一化或候选汇总，不必聚集全部 logits。

#### 1.4.2 MHA、GQA、MQA 的 KV 分配

三种 Attention 的区别在于 query heads 如何共享 K/V：

| 形式 | KV head 数 | 按 head 做 TP 时的含义 |
| --- | --- | --- |
| MHA | $H_{kv}=H_q$ | 对应的 Q/K/V heads 一起分片 |
| GQA | $1<H_{kv}<H_q$ | 一组 Q heads 共享一组 K/V，需要保持分组对应 |
| MQA | $H_{kv}=1$ | 所有 Q heads 共享一组 K/V，常见纯 head TP 在每卡复制这组 K/V |

[MQA][mqa] 与 [GQA][gqa] 定义 query heads 的 KV 共享关系，TP 再将这些 heads 分配到设备上。

以 $H_q=32,H_{kv}=8$ 为例，每个 KV head 对应 4 个 query heads。TP=4 时每卡持有 2 组，TP=8 时持有 1 组；TP=16 时，一组的 4 个 Q heads 分给两卡，每卡 2 个 Q heads，共享的 KV head 在两卡各复制一份。

vLLM 的 `QKVParallelLinear` 按以下规则均分：

$$
H_{q,r}=H_q/p,\qquad
H_{kv,r}=\begin{cases}
H_{kv}/p,&p\le H_{kv},\\
1,&p>H_{kv},
\end{cases}
$$

第二种情况下，每个 KV head 的复制数为 $p/H_{kv}$。[QKVParallelLinear][vllm-qkv]

#### 1.4.3 整除与布局约束

上述 vLLM 路径要求 $H_q\bmod p=0$；当 $p\le H_{kv}$ 时要求 $H_{kv}\bmod p=0$，否则要求 $p\bmod H_{kv}=0$。[Llama Attention 的整除检查][vllm-llama]

MLP 中间维 $f$ 也需满足均分要求；打包 QKV/gate-up 的加载布局须保留 head 或门控配对。量化权重还需满足量化分组和 kernel tile 的对齐要求。

### 推理参考资料

- [The Ultra-Scale Playbook][playbook]：列、行并行原图。
- [Megatron-LM 第 3 节][megatron]；Megatron `core_v0.12.0` 的 [MLP][meg-mlp]、[ColumnParallelLinear][meg-column]、[RowParallelLinear][meg-row]。
- [PyTorch Linear][linear]、[NCCL collective 语义][nccl]。
- [MQA][mqa]、[GQA][gqa]、[FlashAttention][flashattention] 原论文。
- vLLM [QKV 分片实现][vllm-qkv]与 [Llama head 约束][vllm-llama]。

## 2 训练

记第 2.4 节引入的 DP 度数为 $D$，其余符号沿用第 1 节。

### 2.1 反向基础：梯度沿什么维度分片

#### 2.1.1 同一个线性层的两类梯度

统一记 $dT=\partial\mathcal L/\partial T$ 为损失对张量 $T$ 的梯度。对线性层 $Y=XW$：

$$
dX=dY\,W^\top,\qquad dW=X^\top dY.
$$

$dX$（dgrad）继续送往前一层；$dW$（wgrad）保留在参数拥有者处，供梯度累积与优化器使用。前向矩阵乘法和这两次反向矩阵乘法有相同的主要 FLOPs 数量级，但张量布局与就绪时间不同。

对于列并行，$W=[W_0,\ldots,W_{p-1}]$，输入 $X$ 复制，输出与上游梯度按输出维分片。rank $r$ 计算：

$$
dW_r=X^\top dY_r,\qquad
\widetilde{dX}_r=dY_r\,W_r^\top,
\qquad dX=\sum_r\widetilde{dX}_r.
$$

每张卡已经算完自己那片权重的梯度；缺少的是其他输出分片对同一个 $X$ 的梯度贡献，因此应归约 $dX$。

对于行并行，$X=[X_0,\ldots,X_{p-1}]$，$W$ 沿输入维分片，前向 $Y=\sum_r X_rW_r$。各 rank 持有相同的逻辑上游梯度 $dY$，于是：

$$
dW_r=X_r^\top dY,\qquad
dX_r=dY\,W_r^\top.
$$

这时 $dX_r$ 就是输入梯度的一片。如果前一层的输出原本就在该卡，无须把它收集成完整 $dX$。

| 配对使用时的线性层 | 前向入口 → 出口 | 本地参数梯度 | 反向激活通信 |
| --- | --- | --- | --- |
| Column parallel | 复制输入 → 输出维分片 | $dW_r$，与 $W_r$ 同布局 | 对各卡的 $\widetilde{dX}_r$ 做 AllReduce，得到复制的 $dX$ |
| Row parallel | 输入维分片 → 复制输出 | $dW_r$，与 $W_r$ 同布局 | 返回对应的 $dX_r$，通常不通信 |

**TP 组内不能把不同 $W_r$ 的梯度直接相加：它们对应不同参数。** 同一参数分片在不同数据副本间的同步属于 DP，见第 2.4 节。

#### 2.1.2 复制的梯度不代表多份独立 loss

行并行前向的 AllReduce 让各 rank 拿到同一个逻辑 $Y$。在经典 TP 图中，这些副本只是同一激活的多份存储，后续传回的是同一个逻辑 $dY$，并不是 $p$ 个独立 loss 的梯度贡献。

因此不能仅凭“前向用了 AllReduce”便再机械地在反向求和一次。Megatron 的 `reduce_from_tensor_model_parallel_region` 是前向求和、反向原样传递；对应的 `copy_to_tensor_model_parallel_region` 是前向原样传递、反向求和。二者共同表达上述分布式计算图。[TP 自动微分映射][meg-mappings]

### 2.2 Transformer 块：沿前向分工把梯度送回来

#### 2.2.1 MLP 与 SwiGLU

沿用前向的中间激活 $H_r$，普通 MLP 为：

$$
Z_r=XW_{1,r}+b_{1,r},\qquad H_r=\phi(Z_r),\qquad
Y=\sum_r H_rW_{2,r}+b_2.
$$

从复制的 $dY$ 出发，反向顺序为：

$$
dW_{2,r}=H_r^\top dY,\qquad
dH_r=dY\,W_{2,r}^\top,
$$

$$
dZ_r=dH_r\odot\phi'(Z_r),\qquad
dW_{1,r}=X^\top dZ_r,\qquad
dX=\sum_r dZ_rW_{1,r}^\top.
$$

两个 wgrad 和激活函数反向都在卡内完成，最后对 $dX$ 做一次 TP AllReduce。前向输出归约与反向输入梯度归约分别位于 MLP 的两端，中间的扩展维激活和梯度始终留在各自的 rank。

```text
前向：复制 X → 列并行 W₁,r → Zr → φ → Hr → 行并行 W₂,r → Σr → 复制 Y
反向：复制 dX ← Σr ← 局部 dX,r ← dZr ← dHr ←─────────────── 复制 dY
                           └─ dW₁,r              └─ dW₂,r
```

SwiGLU 沿用 gate/up 的 $G_r/U_r$ 与下投影 $W_{d,r}$：

$$
G_r=XW_{g,r}+b_{g,r},\qquad U_r=XW_{u,r}+b_{u,r},\qquad
H_r=\operatorname{SiLU}(G_r)\odot U_r,\qquad
Y=\sum_r H_rW_{d,r}+b_d.
$$

先通过下投影，再通过两条门控分支：

$$
dW_{d,r}=H_r^\top dY,\qquad dH_r=dY\,W_{d,r}^\top,
$$

$$
dG_r=dH_r\odot U_r\odot\operatorname{SiLU}'(G_r),\qquad
dU_r=dH_r\odot\operatorname{SiLU}(G_r),
$$

$$
dW_{g,r}=X^\top dG_r,\qquad dW_{u,r}=X^\top dU_r.
$$

偏置梯度沿 token 维求和。两条分支对输入的贡献先在卡内相加，再跨 TP 组归约：

$$
dX=\sum_r\left(dG_rW_{g,r}^\top+dU_rW_{u,r}^\top\right).
$$

融合的 gate/up 投影能一起生成局部输入梯度，只需一次 AllReduce。Megatron 将第一层输出宽度加倍，并在每个 rank 的局部输出上 `chunk` 成两支；下投影继续接收分片输入。[MLP 与门控布局][train-meg-mlp]

#### 2.2.2 Attention：从输出投影回到 Q/K/V

沿用前向的分数 $S$、概率 $A$ 和 head 输出 $C$，省略 batch/head 下标：

$$
S=QK^\top/\sqrt d+\mathrm{mask},\qquad
A=\operatorname{softmax}(S),\qquad C=AV.
$$

输出投影为 $Y=\sum_r C_rW_{O,r}$，其中 $C_r$ 是拼合本地 heads 后的 $[M,h/p]$ 张量。给定 $dY$，先得到：

$$
dW_{O,r}=C_r^\top dY,\qquad dC_r=dY\,W_{O,r}^\top.
$$

将 $dC_r$ 恢复为 head 布局，再逐 head 反向。未开启 attention dropout 时：

$$
dV=A^\top dC,\qquad dA=dC\,V^\top,
$$

$$
dS=A\odot\left(dA-\operatorname{rowsum}(dA\odot A)\right),
$$

$$
dQ=dS\,K/\sqrt d,\qquad
dK=dS^\top Q/\sqrt d.
$$

`rowsum` 沿 key 维求和并保留该维用于广播，被屏蔽位置的梯度为零。使用 attention dropout 时，先用前向一致的随机掩码和缩放传回 softmax 输出的梯度；使用 RoPE 时，再通过相应旋转的反向变换。

**每卡拥有本地 head 的完整序列和 head 维，因此这些计算都在卡内完成。** 恢复投影布局后，参数梯度留在本地：

$$
dW_{Q,r}=X^\top dQ_r,\qquad
dW_{K,r}=X^\top dK_r,\qquad
dW_{V,r}=X^\top dV_r.
$$

Q/K/V 对输入的贡献先在卡内合并，再做一次 AllReduce：

$$
\widetilde{dX}_r
=dQ_rW_{Q,r}^\top+dK_rW_{K,r}^\top+dV_rW_{V,r}^\top,
\qquad dX=\sum_r\widetilde{dX}_r.
$$

这与前向在输出投影处的归约构成一对。[经典 Megatron Attention 切分][megatron-paper]

GQA 中，同一 KV head 的 $dK$、$dV$ 要累加所有 query heads 的贡献。消费者都在本卡时局部累加；KV 投影参数跨卡复制时，在对应复制组内同步参数梯度。head 分配见第 1.4.2 节。

#### 2.2.3 残差、Norm 与需要留下的激活

对于 Pre-Norm 子块 $Y=X+F(\operatorname{Norm}(X))$，反向在本地把残差支路的 $dY$ 与 Norm 支路的梯度相加。没有 SP 时，块边界的 $X$、$Y$ 与逻辑梯度都复制；Norm 持有每个 token 的完整 $h$ 维，因此 Norm 自身的统计量和输入梯度可以在卡内计算。其复制参数的梯度在各 rank 上相同，不应再把相同值求和 $p$ 次。

训练还要把前向中间结果保留到相应反向，或者在需要时重算：

| 反向步骤 | 所需前向信息 | 标准 TP 下的主要布局 |
| --- | --- | --- |
| 线性层 wgrad | 该层输入激活 | Column 输入复制；Row 输入按中间特征分片 |
| GeLU / SwiGLU | 激活输入或等价可恢复信息；SwiGLU 两支输入 | 中间特征分片 |
| Attention | Q/K/V，以及 softmax 的结果或可重建它的统计量 | 按 head 分片，保留本 head 的序列范围 |
| Norm | 输入，以及均值/逆标准差或 RMS 统计量 | 没有 SP 时复制 |
| Dropout | 与前向一致的掩码或可重放的随机数状态 | 随算子布局；复制路径须保持所需的一致性 |

训练的 K/V 是本次前后向图中的激活；常规整序列训练不会像自回归推理那样，为持续 decode 的请求维护跨步增长的 KV cache。

### 2.3 性能与显存：哪些随 TP 减少，哪些留在关键路径

#### 2.3.1 模型状态与激活要分别记账

沿用 [basic-train](basic-train.md) 和 [DP Training](DP-Training.md) 的混合精度配置：BF16 模型权重、FP32 梯度、显式 FP32 主权重、FP32 Adam 一阶和二阶状态。一个参数对应：

| 状态 | 字节/参数 | 生命周期 |
| --- | --- | --- |
| BF16 模型权重 | 2 | 跨 iteration 保存，在 optimizer step 后更新 |
| FP32 累积梯度 | 4 | 跨 microbatch 累积，在更新边界清零或重置 |
| FP32 主权重 | 4 | 跨 iteration 保存 |
| FP32 Adam $m/v$ | 8 | 跨 iteration 保存 |
| 合计 | **18** | 尚未计激活、临时张量、padding 和框架开销 |

每卡模型状态按实际持有的参数量计算，为 $18\Psi_{\mathrm{local},r}$ 字节。若参数由均匀 TP 分片的 $\Psi_{\mathrm{shard}}$ 和全 TP 组复制的 $\Psi_{\mathrm{rep}}$ 组成：

$$
\mathcal M_{\mathrm{state}}
\approx18\left(\frac{\Psi_{\mathrm{shard}}}{p}+\Psi_{\mathrm{rep}}\right)\ \mathrm{bytes}.
$$

无 SP 时，各类激活的每卡规模为：

| 激活类别 | 每卡主要元素规模 | TP 增大后的变化 |
| --- | --- | --- |
| 块边界、Norm 输入等复制激活 | $O(Mh)$ | 不随 $p$ 等比例下降 |
| MLP 中间激活 | $O(Mf/p)$ | 均匀切分时下降；SwiGLU 还涉及两支保存 |
| MHA 的 Q/K/V、head 输出 | $O(Mh/p)$ | head 可均分时下降 |
| 若显式保存完整 attention 概率矩阵 | $O(BH_qS^2/p)$ | head 分片减少份额，序列平方项仍在 |

对于 MHA，FlashAttention 通过分块与反向重算，将 attention 核心的保存激活降至 $O(Mh/p)$；每卡前后向计算量仍为 $O(BH_qS^2d/p)$，随序列长度呈二次增长。[FlashAttention 第 3.1 节与附录 B][flash-attention]

#### 2.3.2 前后向通信与计算重叠

对一个 $M\times K$ 乘 $K\times N$ 的线性层，前向约 $2MKN$ FLOPs；dgrad 与 wgrad 各约 $2MKN$，所以不含重计算时，该层训练的三次主矩阵乘法合计约为前向的三倍。

沿用前向的经典 dense TP 布局，每个 Transformer block 有：

| 位置 | TP collective | 依赖它的下一步 |
| --- | --- | --- |
| Attention 输出投影前向 | 1 次 AllReduce | 输出残差、Norm 与后续子块 |
| MLP 第二层前向 | 1 次 AllReduce | 输出残差与下一块 |
| MLP 第一层反向 | 1 次输入梯度 AllReduce | 前面的 Norm 与 Attention 反向 |
| QKV 投影反向 | 1 次输入梯度 AllReduce | 前面的 Norm 与上一块反向 |

即 **每块前向 2 次、反向 2 次 AllReduce**。

与前向相同，反向输入梯度的每次 AllReduce 消息量仍为 $q_A M h$，不随 $p$ 缩小。但反向多了一个隐藏通信的窗口：列并行层先计算局部 dgrad，发起异步 AllReduce，再计算本层 wgrad。

```text
计算：局部 dgrad GEMM ──▶ 本层 wgrad GEMM ──▶ 等待未完成的通信 ──▶ 前一层反向
通信：                   └──────── dgrad AllReduce ───────────┘
```

因为 wgrad 依赖 $X$ 和 $dY$，不依赖已归约的 $dX$，所以可以与通信并行；前一层却要等完整 $dX$。Megatron 的固定版本正是在 wgrad 之前发起异步 dgrad AllReduce，并在返回输入梯度前等待；SP 路径则对 dgrad 发起 ReduceScatter。该实现还注明了影响排队顺序的 `CUDA_DEVICE_MAX_CONNECTIONS=1` 配置。[线性层反向与通信重叠][meg-linear-backward]

实际能隐藏多少，取决于 wgrad 时长、链路、通信占用的 GPU 资源和 kernel 排队。DP 的 bucket 通信又是另一条路径：它归约参数梯度，按 bucket 就绪触发，与这里归约激活梯度的 TP 通信不能混为一谈。[Megatron DP bucket 同步][meg-dp-bucket]

#### 2.3.3 重计算与梯度累积的取舍

激活重计算保留选定边界，反向到达该区域时先重跑前向，恢复局部所需激活，再执行反向。选择性重算只覆盖某些算子，整块 checkpoint 则重跑更大范围；目标都是用额外工作换取更短的激活保存时间。[选择性重计算][sp-paper]

重跑范围若跨越 TP collective，就可能再次付出相应通信；随机算子还必须重放一致的随机状态。

梯度累积可用较小 microbatch 完成较大的 global batch：顺序执行 `F → B` 时，一次 microbatch 的激活在反向后释放，梯度继续保留。代价是同样训练 token 数对应更多次前后向与 TP collective，较小矩阵也可能降低效率。DP 可以用 `no_sync()` 延迟跨数据副本同步；**TP 的激活归约是当前 microbatch 继续计算所必需的，不能靠 `no_sync()` 一起推迟到 optimizer step。**

### 2.4 常见组合：TP、Sequence Parallel 与 DP/ZeRO

#### 2.4.1 Megatron Sequence Parallel 改变块边界布局

无 SP 时，Norm、dropout 与残差附近的 $[S,B,h]$ 激活在 TP 组内复制。Megatron SP 沿序列维把这些区域切成 $[S/p,B,h]$：每个 token 仍拥有完整隐藏维，Norm 可以本地计算，但每卡只处理部分 token。进入 TP 线性层前再聚集序列，退出行并行时对部分和求和并留下本卡的序列片。[Sequence Parallel][sp-paper]

以一个“Column → 局部计算 → Row”子块为例：

```text
前向：序列分片 X → AllGather → 完整序列 X → Column → 局部计算 → Row → ReduceScatter → 序列分片 Y
反向：序列分片 dX ← ReduceScatter ← 局部 dX ← Column ← 局部计算 ← Row ← AllGather ← 序列分片 dY
```

| 边界 | 前向 | 反向 |
| --- | --- | --- |
| SP 输入进入 Column 区域 | AllGather 序列片，让每个参数分片看到全部 token | ReduceScatter：合并各参数分片对输入的贡献，并保留本 rank 的 token |
| Row 输出离开 TP 区域 | ReduceScatter：合并部分和，并保留本 rank 的 token | AllGather 上游梯度，让本地 Row 分片处理全部 token |

抽象布局图中，一次 AllReduce 的“归约并分片”和“再聚集”可以被安排在不同边界，从而让中间的 Norm 等区域保持序列分片。两类映射的反向分别是 ReduceScatter 与 AllGather。[SP 自动微分映射][meg-sp-mappings]

Megatron 列并行层只保存 $[S/p,B,h]$ 的分片输入。反向 wgrad 需要完整输入，因此会**额外 AllGather 一次输入激活**，让聚集与 dgrad GEMM 重叠；随后对 dgrad 发起 ReduceScatter，与 wgrad GEMM 重叠。这次输入聚集用于恢复 wgrad 所需数据，独立于图中的梯度布局通信。[SP 输入保存与反向聚集][meg-linear-backward]

SP 还改变了复制参数的梯度语义。Norm 的参数仍在 TP 组内复制，但各 rank 只处理部分 token，所以其本地参数梯度只包含部分贡献，更新前需要在 TP 组内求和。Megatron 在梯度收尾阶段归约带有 `sequence_parallel` 标记的参数梯度；Q/K Norm 等共享参数也有对应处理。这是参数量较小的额外同步，不在前面的四次主要激活通信计数之内。[Norm 参数梯度同步][meg-finalize]

SP 在进入 Attention 前聚集序列，每卡为本地 heads 处理完整上下文；Context Parallel（CP）则把 Attention 的序列范围分布在多卡上，通过 K/V 或等价的数据交换完成计算。[Megatron CP 说明][meg-cp]

#### 2.4.2 TP × DP：同一组 token 与不同组数据

用 8 张 GPU，令 TP=2、DP=4，PP=CP=1。按以下方式编排 rank：

| DP 副本 | TP shard 0 | TP shard 1 | 本次 microbatch 数据 |
| --- | --- | --- | --- |
| 0 | rank 0 | rank 1 | 数据 A |
| 1 | rank 2 | rank 3 | 数据 B |
| 2 | rank 4 | rank 5 | 数据 C |
| 3 | rank 6 | rank 7 | 数据 D |

TP 组是 `{0,1}`、`{2,3}`、`{4,5}`、`{6,7}`：组内协作计算同一模型副本对同一批数据的结果。DP 组是 `{0,2,4,6}` 和 `{1,3,5,7}`：各组成员拥有同一个 TP 参数分片，处理不同数据，随后同步这个分片的参数梯度。

因此 rank 0 的 $dW_0$ 应和 rank 2/4/6 的 $dW_0$ 归约；rank 1 的 $dW_1$ 归入另一个 DP 组。TP 通信通常补齐同一计算的激活或输入梯度，DP 通信则聚合不同样本对同一参数的梯度。

若每个 DP 副本的 microbatch 为 $b_\mu$，累积 $K$ 次，等大样本条件下：

$$
B_{\mathrm{global}}=4Kb_\mu.
$$

TP=2 不再额外乘进 global batch，因为一个 TP 组内两张卡并没有各自处理独立样本。全局梯度范数按不重复参数累加，TP 组内复制的参数只计一份。

#### 2.4.3 ZeRO 在每个 DP 组内进一步分状态

在上述编排中，ZeRO 对 `{0,2,4,6}` 内的 TP shard 0 状态做分片，对 `{1,3,5,7}` 内的 shard 1 状态做分片。它不会把两组不同参数的梯度当作副本合并。TP 决定本次算子由哪些参数片共同执行；ZeRO 决定同一片参数的训练状态由哪些数据副本保存。

| 组合 | TP 已切分的参数片内部，DP 组继续分什么 | 计算时需要补齐什么 |
| --- | --- | --- |
| TP + DDP | 参数、梯度、优化器状态在 DP 组内仍复制 | 同步该片参数梯度，各副本分别更新 |
| TP + ZeRO-1 | 优化器状态，包括该配置下的 FP32 主权重与 Adam $m/v$ | 更新后让各副本获得完整的本 TP 参数片 |
| TP + ZeRO-2 | 再分片梯度 | 梯度归约并分片；更新后恢复本 TP 参数片 |
| TP + ZeRO-3 / 完全分片路径 | 再分片模型参数 | 执行某层前，在对应 DP 组内聚集该层的本 TP 参数片 |

令 $P=\Psi_{\mathrm{shard}}/p$，DP 度数为 $D$，先只计算可均匀 TP 切分的主体参数。在前述 dtype 下，其每卡模型状态稳态近似为：

$$
\begin{aligned}
\text{TP + DDP}:\quad&18P,\\
\text{TP + ZeRO-1}:\quad&6P+12P/D,\\
\text{TP + ZeRO-2}:\quad&2P+16P/D,\\
\text{TP + ZeRO-3}:\quad&18P/D.
\end{aligned}
$$

ZeRO-3 执行某层前还会聚集该层的本 TP 参数片，预取和释放策略决定同时驻留多少完整层分片。ZeRO 各阶段的梯度累积、ReduceScatter、参数 AllGather 与更新时间线见 [DP Training](DP-Training.md)。

TP 按模型维度切权重与中间激活，SP 切块边界激活，ZeRO 在各 DP 组内进一步分片训练状态。

### 训练参考资料

- [basic-train](basic-train.md)、[DP Training](DP-Training.md)：混合精度账本与 DP 更新流程。
- [Megatron-LM 原论文][megatron-paper]：MLP 与 Attention 的配对切分。
- [Reducing Activation Recomputation in Large Transformer Models][sp-paper]：sequence parallel 与选择性激活重计算。
- [FlashAttention][flash-attention]：分块 attention、反向概率重建与保存激活。
- [Megatron Context Parallel][meg-cp]：CP 与上述 SP 的分工边界。
- Megatron `core_v0.12.0` 源码：[TP 映射][meg-mappings]、[SP 映射][meg-sp-mappings]、[线性层反向][meg-linear-backward]、[Row 输出布局][train-meg-row]、[MLP][train-meg-mlp]、[Norm 梯度收尾][meg-finalize]、[DP bucket 同步][meg-dp-bucket]。

[playbook]: https://nanotron-ultrascale-playbook.static.hf.space/index.html
[hf-column]: https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/c93890dd4b22356ee18ed9b863e6f2d0909729bc/assets/images/tp_diagram2.png
[hf-row]: https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/c93890dd4b22356ee18ed9b863e6f2d0909729bc/assets/images/tp_diagram3.png
[megatron]: https://arxiv.org/pdf/1909.08053v4#page=4
[meg-column]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/tensor_parallel/layers.py#L960-L972
[meg-row]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/tensor_parallel/layers.py#L1165-L1202
[meg-mlp]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/transformer/mlp.py#L64-L180
[linear]: https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html
[nccl]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
[zero-paper]: https://arxiv.org/pdf/1910.02054v3#page=13
[mqa]: https://arxiv.org/abs/1911.02150
[gqa]: https://arxiv.org/abs/2305.13245
[flashattention]: https://arxiv.org/abs/2205.14135
[vllm-qkv]: https://github.com/vllm-project/vllm/blob/edad8b03c5285a1c12e7e4f83c1af709b319a01b/vllm/model_executor/layers/linear.py#L1064-L1071
[vllm-llama]: https://github.com/vllm-project/vllm/blob/edad8b03c5285a1c12e7e4f83c1af709b319a01b/vllm/model_executor/models/llama.py#L143-L156
[vllm-forward]: https://github.com/vllm-project/vllm/blob/edad8b03c5285a1c12e7e4f83c1af709b319a01b/vllm/model_executor/models/llama.py#L224-L234
[megatron-paper]: https://arxiv.org/abs/1909.08053
[sp-paper]: https://arxiv.org/abs/2205.05198
[flash-attention]: https://arxiv.org/html/2205.14135v2#S3.SS1
[meg-cp]: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html
[meg-mappings]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/tensor_parallel/mappings.py#L201-L275
[meg-sp-mappings]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/tensor_parallel/mappings.py#L296-L378
[meg-linear-backward]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/tensor_parallel/layers.py#L403-L571
[train-meg-row]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/tensor_parallel/layers.py#L1147-L1202
[train-meg-mlp]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/transformer/mlp.py#L59-L136
[meg-finalize]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/distributed/finalize_model_grads.py#L190-L225
[meg-dp-bucket]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/distributed/param_and_grad_buffer.py#L279-L359
