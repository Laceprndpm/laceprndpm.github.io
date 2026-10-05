---
title: SGLang Expert Compute Backend：从接收布局到 Triton 专家计算
date: 2026-09-30
tags:
  - SGLang
  - MoE
  - EP
  - Triton
  - CUDA
  - AI-Infra
categories:
  - 框架
mathjax: true
---

# SGLang Expert Compute Backend：从接收布局到 Triton 专家计算

本文接续SGLang Runtime，介绍 dispatch 通信接收完成后、combine 通信开始前的数据适配与专家计算；只展开 Triton 后端，以 expert-major 为主线、rank-major 为布局对照。

学习终点是解释：同一个 CUDA Graph bucket 内，实际专家工作如何产生；TBO 拆成两个微批之后，哪些成本随 token 数减少，哪些成本仍然保留或重复。

**版本与范围。** 主线固定为 Triton PR #38886 的 `491a30820f6c4a65094eb97ae12604766b2f7847`：NCCL EP LL、expert-major、BF16 通信、普通 block-wise FP8 权重、gated SiLU，使用通用 Triton fused-experts 的兼容适配。采用无 bias、无额外 clamp、非交错 gate/up 的数学示例；源码还接收其他模型参数，本文不逐项展开。主线不包含后来实验中的 compact/tight-capacity，也不声称该 PR 本身支持 TBO。TBO 分析引用独立的历史实验。[PR 与适配代码][s-adapter]

rank-major 对照取基础 PR #41353 的 `e1f7de17a521dd607e1bec62081f26121cf5c6cc`

| 层次 | 要回答的问题 | 阅读结果 |
|---|---|---|
| Level 1 · 数学目标 | 每个专家最终算什么？ | 写出 gated MLP，指出路由权重的作用 |
| Level 2 · 任务组织 | 各专家的工作怎样分配给 GPU？ | 连接有效范围、tile、CTA 调度、动态工作与 compact 策略 |
| Level 3 · 源码机制 | 这套模型在实际代码中怎样运行？ | 核对一批数据的索引、grid、寻址、退出与结果恢复 |
| Level 4 · 优化技巧 | 怎样减少具体成本，需要付出什么？ | 比较容量、分块、格式转换、复用与流水线的取舍 |
| Level 5 · TBO 验证 | 哪个机制解释回归，怎样验证改进？ | 用匹配对照连接成本假设与服务表现 |

## Level 1｜数学目标：专家最终算什么？

### 1.1 一个专家的 gated MLP

输入一个 token 的行向量 $x\in\mathbb{R}^{H}$，专家中间维度为 $I$。忽略 bias 和模型特有修正，一个专家执行：

$$
g=xW_g,\qquad u=xW_u,
$$
$$
z=\operatorname{SiLU}(g)\odot u,\qquad F_e(x)=zW_d.
$$

逻辑权重形状为 $W_g,W_u\in\mathbb{R}^{H\times I}$，$W_d\in\mathbb{R}^{I\times H}$。其中 $\operatorname{SiLU}(x)=x\cdot \sigma(x)$，$\odot$ 表示逐元素乘法。

gate 和 up 实现上可以拼在同一次 GEMM 中：

```text
X_e [m_e, H]
    → gate/up GEMM → [m_e, 2I]
    → SiLU(gate) × up → [m_e, I]
    → down GEMM → [m_e, H]
```

$m_e$ 是本轮专家 e 的有效路由行数。两次 GEMM 的主要 FLOPs 为：

$$
2m_eH(2I)+2m_eIH=6m_eHI.
$$

这只统计投影的有效数学运算，不包含量化、重排、激活、padding、通信和调度成本。实际权重存储常采用 `[expert, output, input]`，因此代码执行的转置乘法与上述逻辑形状不能混淆。[两次 GEMM 的调用与激活][s-fused]

### 1.2 专家输出与 MoE 输出

Router 为原 token t 选中 $k_{route}$ 个专家，routed 分支的结果为：

$$
y_t^{routed}=\sum_{j=0}^{k_{route}-1}p_{t,j}F_{e_{t,j}}(x_t).
$$

本文主线在专家计算阶段产生各路由的 $F_e(x)$，真实路由权重 $p_{t,j}$ 留给 NCCL EP combine。一般不能将它提前乘到输入后声称等价，因为非线性下 $F_e(px)\ne pF_e(x)$。shared expert 是另一条分支，不在本篇主线中展开。

### 1.3 为什么后端不只包含两次 GEMM？

上述公式假设某 expert 的输入行已经可用。实际接收 buffer 可能含有无效容量、来自多个源 rank 的 token，以及不同的量化格式。实现必须先描述“哪些行交给哪个专家”，再执行计算，最后把结果放回 combine 能识别的位置。

这形成三个责任：**准备计算输入、执行专家网络、交付可回传的结果**。重排和量化是否需要、发生在哪个软件模块、能否融合，要看具体路径；不能把图中的每一项都当成必然独立的 kernel。

## Level 2｜任务组织：从专家工作范围到 GPU 调度

### 2.1 专家计算有哪些后端？

Level 1 给出了共同的数学目标；不同后端在分组计算、硬件优化和低精度实现上各有侧重。下面概览 SGLang 的主要 NVIDIA 专家计算选项，名称对应 `--moe-runner-backend`。

| 后端 | 实现侧重 | 典型适用场景 |
|---|---|---|
| **Triton**（`triton`） | 易于修改、扩展和调优的 grouped GEMM | 自定义 kernel，以及本文的小 M、BM 和拆批成本研究 |
| **DeepGEMM**（`deep_gemm`） | 针对支持架构优化 MoE GEMM，提供 contiguous 和 masked 布局 | block-wise FP8 专家计算；分别组织 prefill 和 decode 的不同形状 |
| **CUTLASS**（`cutlass`） | 基于 NVIDIA 模板库实现专用 GEMM | 使用对应架构、数据类型已有的 CUTLASS 实现 |
| **FlashInfer TRT-LLM**（`flashinfer_trtllm` / `flashinfer_trtllm_routed`） | 推理专用 MoE；routed 变体消费框架算好的 top-k IDs 和权重 | 使用 TRT-LLM MoE 实现的推理部署 |
| **FlashInfer CUTLASS**（`flashinfer_cutlass`） | 经 FlashInfer 接入 CUTLASS grouped GEMM 与 MoE 实现 | 支持的 FP8/FP4 模型 |
| **FlashInfer MXFP4**（`flashinfer_mxfp4`） | 针对 MXFP4 格式的低精度计算 | MXFP4 量化模型 |
| **FlashInfer CuTe DSL**（`flashinfer_cutedsl`） | 基于 CuTe DSL 的 MoE kernel，接入 ModelOpt FP4 | 支持的 NVFP4 量化模型 |

这张表用于定位，后续只展开 Triton；具体支持组合以版本为准。

### 2.2 从一个专家的矩阵乘，到一组专家的矩阵乘

如果每个 expert 都单独调用一次 GEMM，专家数量增多时，launch 和调度次数也随之增加。Grouped GEMM 将多个专家的矩阵乘组织到一次 kernel 启动中，由 kernel 内部调度各专家的计算任务；各专家仍使用自己的权重。

以下例子贯穿第二、三章：本地有两个专家，每专家接收容量 R=128，有效行数为 `[70,12]`，hidden 维度 H=256，专家中间维度 I=128。

本轮 gate/up 投影实际上两组彼此独立的矩阵乘法，分别使用 expert 0 和 expert 1 的输入与权重。

```text
expert 0：X0[70,256] × W13_0ᵀ[256,256] → G0[70,256]
expert 1：X1[12,256] × W13_1ᵀ[256,256] → G1[12,256]
```

一次调用要处理多个专家，后端需要知道每个专家有多少有效输入行，以及这些行和对应权重存在哪里。

### 2.3 一次 Grouped GEMM 的计算范围

沿用 expert-major 输入：L=2 个本地专家，每个专家预留 R=128 行，H=256，接收缓冲区形状为 `[2,128,256]`。本轮两个专家的有效输入分别为：

```text
expert 0：X[0,0:70,:]，有效行数 M0=70
expert 1：X[1,0:12,:]，有效行数 M1=12
```

缓冲区提供存储位置，counts=`[70,12]` 给出每个专家的有效边界。专家归属已经由第一维表达。其余容量可留给后续更大的批次，本次数学计算只涉及这 82 行。

对本次 gate/up grouped GEMM，每个专家都有自己的权重 `W13[e]`，按 `[输出特征,输入特征]` 存储为 `[256,256]`。需要产生的结果区域为：

| 专家 | 有效输入 | 权重 | 有效输出区域 |
|---|---|---|---|
| expert 0 | `[70,256]` | `[256,256]` | 70 行 × 256 列 |
| expert 1 | `[12,256]` | `[256,256]` | 12 行 × 256 列 |

因此，一次 grouped GEMM 的工作集合由各专家的输出区域共同组成：对 expert e，计算所有 `0≤m<M_e、0≤n<N` 的输出，每个输出沿 `0≤k<K` 累加。本例 N=2I=256、K=H=256；不同专家的 M 来自各自的 counts。

来自不同源 rank 的 token 已连续放入对应专家的有效前缀；native NCCL EP 保留接收槽位到源 token 路由身份的映射，供 combine 回传。[接收布局][s-dispatch]

若接收采用 rank-major，数据先按来源 rank/slot 组织，需要根据路由建立各专家的输入集合；计算端最终仍需获得每个专家的输入位置、有效行数和对应权重。[布局对照][s-rank]

至此，计算范围已经确定。接下来的资源划分，要把这两个输出区域拆成 GPU 可以并行处理的任务。

### 2.4 这些工作怎样划分并分配给 GPU？

#### 从 GEMM 分块到跨专家调度

这里沿用通用 GEMM 的分块方式：BM、BN 决定每个输出 tile 的行数和列数；BK 决定计算该 tile 时，每轮沿 K 维处理多少个输入特征。每轮计算 `[BM,BK] × [BK,BN]`，结果累加到同一个 `[BM,BN]` 输出块。

本例取 BM=64、BN=128、BK=128。K=256，因此每个输出 tile 经过两轮 K 维累加。两个专家分别有 70 行和 12 行输入，输出列数均为 256，按逻辑行顺序划分如下：

| 输出 tile | 专家 | 有效输出行 | 输出列 |
|---|---|---|---|
| T0 | expert 0 | `[0,64)` | `[0,128)` |
| T1 | expert 0 | `[0,64)` | `[128,256)` |
| T2 | expert 0 | `[64,70)` | `[0,128)` |
| T3 | expert 0 | `[64,70)` | `[128,256)` |
| T4 | expert 1 | `[0,12)` | `[0,128)` |
| T5 | expert 1 | `[0,12)` | `[128,256)` |

expert 0 产生 `ceil(70/64) × ceil(256/128) = 4` 个 tile，expert 1 产生 `ceil(12/64) × ceil(256/128) = 2` 个，共 6 个。

Grouped GEMM 将各专家的这些 tile 统一组织为待执行的计算任务。接下来讨论的是：这 6 个 tile 怎样分配给 CTA，以及怎样共同使用 GPU 资源。

#### 将任务映射到 CTA

有了 6 个有效任务，后端还要决定由哪些 CTA 承担，以及每个 CTA 何时结束。两种组织方式分别是：

| 方式 | 对本例的组织 | 调度职责 |
|---|---|---|
| 一 CTA 一 tile | 为 6 个有效任务分别安排 CTA；实现还可能为容量上界发射额外 CTA | CTA 根据自己的编号确定一个专家和输出区域，完成后退出 |
| Persistent CTA | 例如发射 2 个 CTA，各自连续处理多个 tile | CTA 完成当前任务后，确定下一项，直到所分配的工作结束 |

一种简单的 persistent 分配可以是 `CTA0: T0→T2→T4`、`CTA1: T1→T3→T5`。也可以采用其他顺序或设备端调度方式。Persistent 描述 CTA 连续处理任务的执行方式，其任务分配可以是确定的步进关系，也可以采用其他调度机制。[Grouped 调度][cutlass-grouped]、[固定步进示例][triton-grouped]

专家的工作按 tile 参与分配：expert 0 的 4 个 tile 可以分给不同 CTA；同一个 persistent CTA 也可以先处理 expert 0，再处理 expert 1。这样，调度可以利用整个专家集合中的并行工作。

#### Tile 大小怎样影响同时能运行多少工作？

一个 CTA 处理 tile 时，需要保存累加结果，并为输入加载、矩阵运算和流水线准备资源。BM×BN 决定逻辑累加器规模；BM×BK 和 BK×BN 决定每轮输入子块规模。相应的寄存器、共享内存和线程使用量，决定 SM 能同时容纳多少 CTA。

例如，BM64、BN128 对应 8192 个输出累加值；BM32、BN128 对应 4096 个。较小的 tile 有机会降低单个 CTA 的资源需求，但会产生更多任务，也可能增加重复加载。实际寄存器与共享内存用量还取决于数据类型、指令组织和流水线实现。[资源与流水线][cutlass-efficient]

可将执行时间先拆成两个因素理解：**有多少项工作需要完成，以及资源允许同时推进多少项工作。** 假设一个教学模型中同时只能推进两个 tile，且每个 tile 耗时近似相同，那么六项任务至少需要三轮处理；若资源允许同时推进四项，则分为四项和两项两轮，最后一轮只有一半位置有任务。

真实执行没有统一的轮次屏障，CTA 可以随资源释放继续推进；这个估算用于看清收尾现象。末尾任务少、任务耗时不同，或某些 persistent CTA 分得的工作更重，都可能使部分 SM 先空闲，而剩余工作决定整个 GEMM 的完成时间。调度需要平衡的既有任务数量，也有任务实际成本。

#### 不同 tile 之间怎样复用数据？

回到 T0–T5，同一专家内有两类复用关系：

| Tile 关系 | 可复用的数据 | 本例 |
|---|---|---|
| M 区域不同、N 区域相同 | 同一专家对应 N 区域的权重 | T0 与 T2、T1 与 T3 |
| M 区域相同、N 区域不同 | 同一批输入行的 activation | T0 与 T1、T2 与 T3 |

一个 BM64 的 M 区域如果有 64 条有效行，同一权重子块就为 64 行输出贡献计算；尾块只有 6 条有效行时，同样一块权重只服务 6 行有效结果。这是 BM 大小、专家行数与权重复用之间的直接联系。

跨 CTA 的复用可以通过缓存与任务访问顺序获得：在相近时间安排相关 tile，增加相同权重或 activation 仍在缓存中的机会。不同 CTA 的共享内存通常独立，是否能使用其他协作机制取决于实现。任务顺序、缓存容量和并行竞争共同决定复用能否兑现。[访问顺序与复用][triton-matmul]

#### 哪些关系需要等待？

普通 M/N 分块中，T0–T5 写入互不重叠的输出区域，各自累加完整 K；它们之间没有交换部分和的数学依赖。上面的数据复用关系主要影响访问成本。

专家网络不同阶段则有明确的数据依赖。对本例一行输入，gate/up 的 256 列结果包含 128 列 gate 和 128 列 up；激活乘法需要相应的两部分，down 需要由它们生成的中间特征。对于 `[0,64)` 这组行，T0 和 T1 分别产生两半投影，后续消费必须等待所需结果就绪。

普通多 kernel 路径常以完整 kernel 阶段建立顺序：gate/up 完成后运行激活，再运行 down。更细的融合或流水线实现可以对已就绪的数据建立更细粒度依赖。数学上必须满足的是每个消费者读到完整输入，具体在哪个边界同步由后端决定。

### 2.5 固定 launch 怎样承载变化的专家工作量？

前面的任务集合由 counts 决定，而 CUDA Graph 的普通 replay 复用固定启动结构、参数地址和依赖关系。后端需要把每轮变化的有效范围交给设备端，使同一组执行资源能够处理不同的任务集合。[CUDA Graph][cuda-graph]

沿用 BM64、BN128，本例 counts=`[70,12]` 时有 6 个有效输出 tile；下一轮变成 `[64,0]` 时，只剩 expert 0 的两个 N tile。输入存储和启动配置可以保持不变，设备端读取本轮 counts 或由其生成的任务描述，再选择需要完成的工作。

两种调度组织对应不同的实现方式：

- **按容量发射 CTA**：预留足够的任务位置，执行时检查当前位置是否属于有效专家范围；无工作的位置退出，有效尾块按边界加载和写回。
- **固定数量的 persistent CTA**：CTA 按本轮工作描述继续寻找任务，完成一项后再处理下一项；没有后续工作时结束。

因此，描述一次执行至少需要两个数：启动了多少 CTA，以及本轮有多少有效 tile。一个 CTA 可以处理多项工作，已发射的 CTA 也可能没有有效工作；二者的对应关系由调度方案决定。

批次变小后，耗时怎样变化，可以区分三种情况。以下假设每个有效 tile 的耗时近似为 t，并暂不计访存竞争和其他固定开销：

| 情况 | 示例 | 耗时变化的原因 |
|---|---|---|
| **有效行减少，tile 数不变** | BM64 下，某专家从 24 行减到 12 行，仍占一个 M tile | 主要块级计算未减少；改为 persistent 也不会自动减半 |
| **tile 数减少，原来已能同时执行** | 6 个 tile 减到 3 个，两者都能同时运行 | 完成时间都可能约为 t，减少的是工作量而非执行轮数 |
| **tile 数减少，原来需要多轮处理** | 并行推进能力为 3 个 tile，任务从 12 个减到 6 个 | 理想时间由约 4t 降到 2t，可能接近减半 |

第三种情况既可能发生在 persistent CTA 连续处理任务时，也可能发生在普通 CTA 受驻留资源限制、分批运行时。“轮数”只是估算，实际执行可以随资源释放继续推进。按容量发射时，无效 CTA 的判断或置零也会留下成本。

因此，分析拆批性能应先检查**有效 tile 数是否减少**，再看原有并行空间与固定开销，不能仅凭调度方式预测收益。

### 2.6 Compact 等策略改变了哪些工作？

本例为 82 条有效行预留了 256 行容量，还要按专家独立划分 tile。围绕这些规模，可以在四个位置减少工作：

| 策略 | 改变的对象 | 本例中的含义 | 新增代价或约束 |
|---|---|---|---|
| **紧凑存储** | 数据占用及后续遍历的范围 | 将 70 行和 12 行排进连续有效区域，去掉专家之间的空槽 | 搬运与位置映射；输出需能恢复原身份 |
| **紧凑任务组织** | 调度器要遍历的任务集合 | 根据有效行数只组织那 6 个有效 tile，输入数据仍可留在原位置 | 任务描述的生成、寻址及更新成本 |
| **收紧计算容量** | 固定缓冲区和辅助处理的上界 | 为较小微批选择足够而更小的计算容量 | 上界必须覆盖实际路由，且满足执行资源复用要求 |
| **减少尾块浪费** | 每个计算块覆盖的无效部分 | 选择适合专家行数的 BM/BN 等分块 | 任务数、访存复用和资源占用也随之改变 |

紧凑存储后，expert 1 的第一行可以接在 expert 0 的第 70 行之后，计算端通过各专家的起点和长度找到输入。若实际执行需要对齐或固定 shape，物理分配仍可能大于有效行总数；紧凑布局的收益要看哪些后续操作能够使用更紧的范围。

紧凑任务组织则可以直接从各专家的 counts 推导任务数，通过 offsets 或调度规则定位任务。是否同时移动 activation，是另一项实现选择。

这两类 compact 都不会自动改变各专家的矩阵大小：BM64 下，70 行仍需两个 M 块，12 行仍需一个。减少容量空槽与减少有效专家的尾块覆盖，需要分别评价。第四章再结合具体实现讨论这些策略的成本与收益。

## Level 3｜源码机制：把一个批次完整展开

从本章开始进入固定版本的 SGLang/Triton 路径。第二章的有效范围、任务划分和设备端调度，在这里对应到具体张量、索引与 kernel；兼容接入所需的格式转换单独交代。

### 3.1 固定算例与分支，先列出可核算的状态

继续使用两个本地专家、R=128、counts=`[70,12]`、H=256、I=128。本文核对固定提交的普通 AOT alignment 小批量分支、非 TMA GEMM、无 bias/LoRA/clamp，并假设没有命中调优文件或显式覆盖，采用 FP8 block-wise 默认配置。

| 项目 | 本例值 |
|---|---|
| 接收 FP8 数据 / scales | `[2,128,256]` / `[2,128,2]` |
| Adapter 输出 | `[256,256]` BF16，IDs/weights 为 `[256,1]` |
| gate/up 权重 / scales | `[2,256,256]` / `[2,2,2]` |
| down 权重 / scales | `[2,256,128]` / `[2,2,1]` |
| 两次 GEMM 的配置 | BM64、BN128、BK128、GROUP_SIZE_M32、4 warps、CUDA stages=3 |

这是源码与算术推演，没有运行 native 通信或 GPU kernel。实际安装的 kernel 包、backend override 或不同配置可能改变排序细节；下面的精确数组以同一提交的实现为依据。[配置][s-config]、[alignment 入口][s-align-entry]

### 3.2 输入准备：当前兼容接口交付什么？

`dispatch_b()` 先提交 handle 的 receive completion，再调用 `_quantize_fp8(recv_tokens, expert_counters)`。同一执行顺序上的后续 kernel 才消费接收数据；调用 `complete()` 不应被理解为 Python 已同步等待全部 GPU 运算结束。

量化后返回 `DeepEPLLDispatchOutput`。当前接入通过 adapter（接口适配代码）复用接收二维输入和逐行路由 IDs 的通用 `fused_experts`：先将 FP8 槽反量化为 BF16，再展开并构造 IDs。这些是该兼容接口的额外准备工作；expert-major 本身已经隐含专家归属。

`run_nccl_ep_triton()` 取出 received、scales、原路由表和 counts，交给 `prepare_expert_slots()`。本例将 L=2、R=128 展开为 S=L·R=256 行，启动 S 个 program，每个负责一个槽。q=e·R+s 是展开后的原输入行号，后文 p 是分组索引中的位置。[dispatcher][s-dispatch]、[adapter][s-adapter]

```python
# 解释性伪代码，省略向量化和实际 stride。
q = program_id
e, s = q // 128, q % 128
valid = s < counts[e]
hidden[q, :] = dequant(received[e, s, :], scales[e, s, :]) if valid else 0
ids[q, 0] = e if valid else -1
weights[q, 0] = 1
```

例如 q=65 有效，q=100 无效，q=130 属于 expert 1 且有效。每个有效槽的两个 scales 分别恢复特征 `[0,128)` 和 `[128,256)`；无效 received/scales 不被读取，BF16 输出行仍写零。

此后通用 fused-experts 接收的本地 IDs 完全确定：

```text
ids[0:70]    = 0
ids[70:128]  = -1
ids[128:140] = 1
ids[140:256] = -1
```

这里 top-k=1 表示每个槽已经是一条专家路由。真实模型路由权重没有进入本次数值乘法，仍由 combine 消费。Adapter 也没有使用 `expected_m` 来缩小这 256 行。

### 3.3 配置与 alignment：原行号怎样进入分组索引？

`_prepare_fused_moe_run()` 将 `hidden_states.shape[0]=256` 传给配置选择，而不是有效行总数 82。选择顺序为显式覆盖、调优文件、默认回退；本例采用回退 BM64。两次 GEMM 共享 alignment 结果，因此此提交会让 down BM 与 gate/up BM 保持一致。[配置选择][s-config]、[调用点][s-fused]

本文本地 top-k=1，`moe_align_block_size()` 根据输入容量 S 和本地专家数 L 分配索引上界；当 S≥L+1 时：

$$
Q=S+(L+1)(BM-1)=256+3\times63=445.
$$

额外的一组用于无效 expert ID；较小输入 S<L+1 时使用 Q=S·BM 的另一上界。这里 Q 是索引存储容量，当前有效分组长度由 GPU 另行写入。

函数分配 445 个索引元素、ceil(445/64)=7 个 expert block ID，以及一个设备端 padded-length 标量。接着 kernel 将输入 ID 加一作为组号，所以顺序为 `-1 → expert 0 → expert 1`。[索引分配][s-align]

对于本例，源码选择小批量排序分支。32 个排序线程各自扫描 `tid, tid+32, tid+64, ...`，再按线程前缀写入各组。这保证按专家分组，却**不保证组内按 q 升序**。[小批量选择与写入][s-align-native]

精确布局如下；所有区间都是左闭右开，padding 哨兵为 256：

| 索引位置 p | `sorted_token_ids` 内容 | 含义 |
|---|---|---|
| `[0,174)` | 174 个 ID=-1 的原行号，按线程步长顺序排列 | 有存储位置的无效容量行 |
| `[174,192)` | 18 个 256 | 无效组的补齐位置 |
| `[192,262)` | expert 0 的 70 个行号 | 有效专家行 |
| `[262,320)` | 58 个 256 | expert 0 尾块补齐 |
| `[320,332)` | 128,129,…,139 | expert 1 的 12 行 |
| `[332,445)` | 256 | expert 1 补齐和剩余索引容量 |

expert 0 的顺序开头是：

```text
[0,32,64, 1,33,65, 2,34,66, 3,35,67, 4,36,68, 5,37,69,
 6,38,7,39,...,28,60, 29,61,30,62,31,63]
```

因此原行 q=65 位于 p=197，落在完整块 `[192,256)` 的局部行索引 5（从 0 开始，即第 6 行）；尾块 `[256,320)` 中的六条有效行实际是 `[29,61,30,62,31,63]`。

对应 metadata 为：

```text
expert_ids[0:6] = [-1,-1,-1,0,0,1]
expert_ids[6]   = 未指定；正常执行不会读取它
num_tokens_post_padded = [384]
```

**无效行号与 padding 哨兵不同。** q=100 小于 256，但属于 expert=-1；它对应输出中一个需要置零的槽。哨兵 q=256 已超出输入范围，不能加载或写回。二者分别由 expert 判断和 row mask 处理。

其他规模的 alignment 可能使用原子分配组内位置，不能把此处的精确排列推广为接口保证。GEMM 依赖的是 metadata 一致性，而不是固定的组内行顺序。

### 3.4 第一次 launch：14 个 program 分别做什么？

`invoke_fused_moe_kernel()` 先将 `[256,256]` BF16 activation 量化为 FP8，得到 `[256,2]` scales；此调用没有传入 expert counts，不能因为后面的 GEMM 会早退，就说前面的量化也只处理 82 行。

GEMM 的 N=`w13.shape[1]=256`、K=`w13.shape[2]=256`。启动包装层使用索引容量计算 grid：

$$
grid_x=\left\lceil\frac{Q}{BM}\right\rceil\left\lceil\frac{N}{BN}\right\rceil
=\left\lceil\frac{445}{64}\right\rceil\times2=14.
$$

这 14 个 program 来自 7 个容量 M 块和 2 个 N 块；本轮 padded 长度和 expert IDs 再决定各自的实际工作。

实际映射使用 `GROUP_SIZE_M=32`。本例只有 7 个 M 块，所以所有 program 都在同一访问分组内，公式简化为：

```text
pid_m = pid % 7
pid_n = pid // 7
```

这是工作编号映射，不是 GPU 严格按 pid 顺序执行的保证。[映射与退出条件][s-kernel]

| 第一 N 块 pid | 第二 N 块 pid | pid_m | 索引位置 | 执行动作 |
|---:|---:|---:|---|---|
| 0 | 7 | 0 | `[0,64)` | 无效专家组，置零对应输出后退出 |
| 1 | 8 | 1 | `[64,128)` | 无效专家组，置零对应输出后退出 |
| 2 | 9 | 2 | `[128,192)` | 46 个无效行号＋18 个哨兵；仅对前者置零，退出 |
| 3 | 10 | 3 | `[192,256)` | expert 0 的完整块，执行 dot |
| 4 | 11 | 4 | `[256,320)` | expert 0 的 6 行尾块，执行 dot |
| 5 | 12 | 5 | `[320,384)` | expert 1 的 12 行尾块，执行 dot |
| 6 | 13 | 6 | 从 384 开始 | 已达本轮 padded 长度，立即退出 |

末尾逻辑块会覆盖 `[384,448)`，而索引只分配到 445。这不导致这里的越界读，因为 `pid_m*64 >= 384` 的判断在加载索引和 expert ID 之前就返回了。调整这些检查的顺序会影响正确性，不能仅把它看作性能技巧。

有效专家的 M 尾块仍使用 BM64 的运算结构；本例 N 和 K 都能整除对应块大小，没有额外 N/K 尾部。

### 3.5 跟踪 program 10：地址、scale 与 K 循环

#### FP8 GEMM 内部怎样使用 scale

3.2 的反量化和 3.4 的输入量化负责准备数据；这里讨论数据已进入 GEMM 后，scale 如何参与累加。

权重按 `[expert,N,K]` 存储，activation 每行沿 K 维的 128 个元素共享一个 scale，权重每个 128×128 的 N/K 块共享一个 scale。记量化值为 $Q^A,Q^B$，分组 $g=\lfloor k/128\rfloor$，则：

$$
A_{q,k}\approx Q^A_{q,k}s^A_{q,g},\qquad
B_{e,n,k}\approx Q^B_{e,n,k}s^B_{e,\lfloor n/128\rfloor,g}.
$$

当前 BK128 与 K 方向量化组对应，kernel 对每组执行 FP8 子块点积，再用该组 scales 恢复尺度并累加：

$$
Y_{q,n}\approx\sum_g
\left(\sum_{k\in g}Q^A_{q,k}Q^B_{e,n,k}\right)
s^A_{q,g}s^B_{e,\lfloor n/128\rfloor,g}.
$$

这是当前 block-wise FP8 GEMM 的数值计算。Scale 格式和粒度属于量化方案，BM/BN/BK 属于计算配置；本例二者在部分维度上取相同数值。下面跟踪一项输出怎样取得对应的数据和 scales。[分组累加][s-kernel]

#### Program 10 的实际寻址

program 10 对应 `pid_m=3,pid_n=1`，使用 expert 0，处理输出列 128..255。其局部行索引 5（从 0 开始，即第 6 行）从 `sorted_token_ids[197]` 取得 q=65。

设本段所用 FP8 输入、权重和 GEMM scales 都是连续存储，以下偏移以**元素**为单位；真实代码使用各张量 stride。本例跟踪输出 n=129 与归约特征 k=130：

```text
A[65,130]        → A_base + 65×256 +130             = A_base +16770
W13[0,129,130]   → W_base + 0×256×256 +129×256 +130 = W_base +33154
A_scale[65,1]   → scale offset 131
W_scale[0,1,1]  → scale offset 3
```

权重存为 `[expert,N,K]`，所以输出特征 n=129 位于权重第二维，k=130 位于第三维。这里没有先构造一整份反量化 BF16 权重矩阵；kernel 对 FP8 子块做 dot，再按 K 分组应用对应 scale。

gate/up 的两轮为：

```text
k_start=0：  加载特征 0..127，  计算局部 dot，乘该组 scales，累加
k_start=128：加载特征 128..255，计算局部 dot，乘该组 scales，累加
```

其中 k=130 只是第二轮局部点积中的一个乘积项。FP32 累加器完成两轮后转成 BF16，结果写回 `gateup[65,129]`，元素偏移 16769。写回使用原行 q，而不是分组位置 p=197。[寻址、缩放、写回][s-kernel]

作为对照，program 11 使用同一个专家和第二个 N 块，但只有六条有效行。58 个哨兵位置的 activation 加载被 mask 并填零，写回也被 mask；块仍完成两轮 K 循环。program 12 改用 expert 1 的权重，其专家维偏移不再是零。

### 3.6 激活、再量化与 down：第二次为何不同？

gate/up 工作区为 `[256,256]` BF16。以非交错 gate/up 为例，前 128 列是 gate，后 128 列是 up：

$$
Z[q,i]=\operatorname{SiLU}(G[q,i])\,G[q,i+128].
$$

激活与乘法写入 `[256,128]`。此路径将本地 IDs 传给激活实现，用于过滤无效专家行；不要把容量 256 当作所有行都执行完整激活的证据。无效行也不属于后续必须产生有意义数值的有效输出契约。[激活调用][s-fused]

激活结果 Z 是新的数值张量，down 输入使用针对 Z 生成的量化值与 scales。down 包装调用再量化 Z，scales 的逻辑形状为 `[256,1]`；本例 K 正好等于量化组大小 128，量化 helper 可以走整行快捷分支，仍满足上述 scale 契约。[量化 helper][s-quant]

第二次 GEMM 使用 `w2[2,256,128]`，共享同一套 sorted IDs/expert IDs 和 BM。N 仍为 256，所以 grid 和 program 分类仍是 14/6/6/2；K=128，只循环一轮。

对同一个 program 10、原行 q=65、输出 n=129，改看归约特征 k=2：

```text
A2[65,2]       → offset 65×128+2 = 8322
W2[0,129,2]    → offset 129×128+2 = 16514
A2_scale[65,0] → offset 65
W2_scale[0,1,0]→ offset 1
output[65,0,129] → offset 65×256+129 = 16769
```

不能把第一次的 k=130 原样拿来解释 down，因为它已经超出 down 的输入特征范围。这个例子也说明：**相同 grid 不等于相同运算量**，每个 program 的 K 长度不同。

### 3.7 为什么最后只需恢复 view？

本路径 `no_combine=True`，down 不在这里乘原路由权重或归并模型 top-k。两次非 TMA GEMM 都按原 q 寻址，所以输出 `[256,1,256]` 可以恢复为 `[2,128,256]`：

```text
q=65  → output[65,0,:]  → expert_output[0,65,:]
q=130 → output[130,0,:] → expert_output[1,2,:]
```

`view(received.shape)` 没有搬运全量数据；它能成立的前提是原行身份已在写回时保住。若未来 compact 改变了 q，必须恢复映射，不能只改 view。

返回 `DeepEPLLCombineInput` 时仍携带原路由表；NCCL EP 按保存的接收身份将专家输出送回源 token，并应用真实权重。[输出恢复][s-adapter]

### 3.8 把三个工作规模重新对上

```text
82 条有效专家行
  在 256 个接收/适配容量行中
  经 alignment 生成 384 个本轮 padded 索引位置
  索引 buffer 预留 445 个位置
  每次 GEMM 发出 14 个 program
  其中只有 6 个属于有效专家并执行 dot
```

Native 接收量化、256 槽适配、分组、两次输入量化、激活和两次 GEMM 各有自己的工作范围。最后一句“只有 6 个 dot program”不能替代整条路径的成本分析。

**本层自检：** q=65 为什么不在尾块？program 13 为什么能在读取未指定的 `expert_ids[6]` 前退出？如果只观察到两次 GEMM grid 相同，还缺哪些信息才能比较耗时？

## Level 4｜优化技巧：具体减少了哪一种成本？

第二章区分了各类成本，第三章给出了当前实现。本章据此讨论具体优化：每项指出基线成本、改法、成立条件和代价。下列“候选”表示需要另外实现或验证，不代表已存在于本文固定 PR。

### 4.1 Compact：缩小适配范围，同时承担映射成本

**基线成本。** 本例有 82 条有效路由，却展开 256 行 BF16 输入。适配会为无效行写零，GEMM 前的普通 activation 量化还按传入张量处理，索引中也保留无效组。完整 GEMM 虽会跳过该组，周边成本仍然存在。

**改法。** 以 counts 的前缀和确定各 expert 在紧凑输入中的起点，将有效槽搬入更紧的布局，并保存新行到旧槽的映射：

```text
expert 0：旧 q=0..69    → 紧凑行 0..69
expert 1：旧 q=128..139 → 紧凑行 70..81
```

旧 q=130 变成紧凑行 72。输出必须通过映射恢复到 `expert_output[1,2,:]`，不能把紧凑结果直接 view 成原接收布局。

这是理想的有效布局示意；Graph 中不能为了得到正好 82 行而每轮把 GPU counts 读回 CPU、动态分配新形状。实际实现可以使用固定安全容量、设备端 offsets 和 mask，只把有效数据打包到前缀。**逻辑 compact 与“张量 shape 每轮等于有效行总数”是两件事。**

**条件与代价。** 映射必须覆盖重复专家路由、空专家和动态 counts；代价是前缀和、pack、restore 及相应 scratch。有效率本来很高时，减少的无效处理可能抵不过新增搬运。它也没有减少每个专家按 BM 对齐的有效 tile 数。

**实现状态。** 原始兼容 PR 没有这项 compact；历史 v5 候选有。它把全容量 slot 展开缩小为 `min(L·R, R·top-k)` 的固定行上界，仍不是只处理有效 counts。历史实际形状中的 1536 行要在该版本下解释，不能套到本文 256 行教学算例上。[历史容量分析][local-diagnosis]

### 4.2 Tight-capacity：让每个微批使用更紧的安全上界

**基线成本。** 两个 TBO lane 如果各自保留整批的计算容量，即使各自只收到一半有效行，pack、量化和 metadata 仍可能处理相近的固定范围。Compact 后也可能继续发生这个问题。

**改法。** 将通信容量与计算容量分开。通信缓冲区仍满足协议要求，计算工作区则按可证明的全组路由上界配置。设 P 为 EP 参与 rank 数，每个源 rank 在本轮微批中至多处理 b 个有效 token，top-k 为 k，则一个目标 rank 接收的有效专家路由总数不会超过 P·b·k，还可与原有容量上界取更紧者。

这里 b 必须是对所有参与源 rank 都成立的上界；只有在已建立统一 bucket 等前提下，才可安全使用共同 b。不能拿本 rank 的发送行数代替，因为它可能零发送，却承接远端大量专家工作。

**条件与代价。** 所有层的 shape、路由上界和固定地址复用必须一致；不能只依据均衡路由的平均值分配。更细的容量档位会增加资源与 Graph 管理复杂度，调度不满足前提时仍需保守容量。

**实现状态。** 历史 v6 在限定的统一 decode Graph 配置中尝试过 `P × padded lane tokens × top-k`，eager 仍采用原容量。这是有条件的候选，不是通用的“所有 buffer 除以二”。[v6 范围][local-v6]

### 4.3 调 BM：少空算与更多任务之间的交换

**基线成本。** BM64 下，82 条有效行占据 192 行有效专家 tile 容量，M 方向填充率为 82/192≈42.7%。尾块仍执行对应 dot 结构。

**改法。** 缩小 BM，减少各专家尾块覆盖的无效行。保持本例 BN128、BK128 不变，可以得到以下纯算术对照：

| BM | 有效专家 M 块 | 有效专家 tile 覆盖行数 | M 行填充率 | 每次 GEMM 的有效 dot program | 索引容量 Q | launch program 数 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 2＋1=3 | 192 | 42.7% | 6 | 445 | 14 |
| 32 | 3＋1=4 | 128 | 64.1% | 8 | 349 | 22 |
| 16 | 5＋1=6 | 96 | 85.4% | 12 | 301 | 38 |
| 8 | 9＋2=11 | 88 | 93.2% | 22 | 277 | 70 |

这张表不是 GPU 测量，也不证明 BM8 已能在指定 dtype/架构上正确编译。它说明一个具体取舍：**覆盖的空行减少了，但 program 数增加了。** 较小 BM 还可能让同一权重块被更多 M 块使用，增加逻辑加载；L2 是否能消化重复请求需要实测。

**条件与代价。** 两次 GEMM 必须共享一致的 BM/alignment；检查数值正确性及最终 BN/BK/warps/stages。此源码在 SM90 的特定小 BM 配置上还会启用 `swap_ab`，所以实验变量只改 BM，并不保证编译策略完全相同。更小 BM 也不保证 Tensor Core 利用、访存或总延迟更好。[配置约束][s-config]、[kernel 分支][s-kernel]

**实现状态。** 默认回退 BM64 已有；历史 v6 试过 BM16/32，且同时包含容量优化，不能把整组改善归给 BM。新的 sweep 应固定 adapter 版本，再改变 BM。[历史候选][local-v6]

### 4.4 少做一次格式往返：移除重复量化，或融合必要转换

**基线成本。** 主线在通信后生成 FP8，adapter 又反量化到 BF16，通用 GEMM 包装层再量化回 FP8。除了算术，这意味着 BF16 工作区、写入和后续读取。第二次 GEMM 前还需要对激活结果量化。

有两类不同改法：

1. **接口改造，复用已有 FP8 数据与 scales。** 让专家入口消费正确的 FP8 槽位及索引，减少 adapter 反量化与再次量化。需要确认 scale 布局、有效性、索引寻址和量化语义；不能只是跳过一行转换调用。
2. **融合仍然需要的操作。** 例如在 pack 时顺便量化，或者在生成 `SiLU(gate)×up` 时求 group scale 并输出 FP8，从而减少中间张量的一次全量写读及一个 kernel 边界。

**条件与代价。** 求 scale 需要看到对应量化组的值，融合会引入归约、临时存储与同步需求，可能增加寄存器/共享内存压力。改变量化舍入顺序也可能改变数值结果，必须分别验证误差和模型影响。

**实现状态。** 本文基线已有 SiLU 与乘法的融合调用，但没有消除上述 FP8→BF16→FP8 往返，也未在这条执行序列中将普通激活与下一次量化合成一项。Rank-major 对照的 preprocess 展示了在数据整理时量化的另一种实现，但不能当作本文路径已获得该优化。[主线调用][s-fused]、[rank-major 对照][s-rank]

### 4.5 调整访问顺序与 BN：让数据被更多有效计算复用

**基线成本。** 同一个专家的不同 M 块需要相同权重，不同 N 块会读取相同 activation 的部分数据。多个 program 发起相同地址的加载，不保证每次都从 DRAM 取数，也不保证一定命中缓存。

**改法。** 本文已有 `GROUP_SIZE_M` 的 program 编号映射，使一组 M 块先处理同一段 N，再切换 N，以改善局部性。这个“访问分组”不是 MoE 专家分组；如果它跨过专家边界，相邻任务的权重也可能不同。Program ID 顺序是局部性设计，不是跨 SM 的严格时间顺序。[实际映射][s-kernel]

BN 也是复用手段：更大的 BN 让一次加载的 A tile 为更多输出列贡献结果，但增加 B tile 和累加器的规模。BM 更大则让权重 tile 为更多输入行服务；只有这些行真正有效，复用才转化为有效计算收益。

**条件与代价。** 访问顺序必须保持每个输出块唯一归属；BN 还要满足量化和编译约束。更大的 tile 或更激进的访问分组可能加重资源占用、破坏其他数据的缓存局部性。判断依据应是 kernel 时长和实际内存指标，不能仅按逻辑加载次数推断 DRAM 字节。[Triton 分块与局部性说明][triton-matmul]

**实现状态。** Grouped program ordering 已存在；改变分组、BN 或权重读取组织是后续候选，不是当前 BM sweep 默认同时变化的项目。

### 4.6 流水线与资源配置：隐藏加载等待，也可能放大竞争

**基线成本。** GEMM 的 K 循环交替需要加载数据和执行矩阵运算。如果下一块数据还没到，计算可能等待。另一方面，单个 program 占用太多资源，也可能减少同驻留 block 数量。

**改法。** 调整 Triton 的 `num_stages`，让编译器有机会安排多阶段循环流水线；配合 `num_warps` 和 tile 大小，平衡加载、计算和驻留资源。本文默认 CUDA 配置为 stages=3、warps=4，这不是“始终有三块数据同时在途”的运行时保证。[默认配置][s-config]、[Triton 参数定义][triton-config]

**条件与代价。** 更多 stage 可能需要更多缓冲或增加资源占用；更多 warp 也不意味着一定更快。教学算例 gate/up 只有两轮 K、down 只有一轮，流水线能摊销的空间有限。真实 H/I 更大时，K 循环更长，取舍又会变化。

TBO 下还需考虑另一条计算/通信 stream：单 kernel 独占时最优的配置，可能留不出足够资源获得有效重叠。应在单算子与联合运行两种条件下检验，而不是只追求 occupancy 数值最大。

**实现状态。** 编译参数与默认值已有；针对目标 GPU 的更优组合、与 TBO 协同的资源配置仍需基准。本文不声称某个 stage/warp 数已经最优，也不将 TMA 等其他分支的收益归入当前普通路径。

**本层自检：** Compact、BM 和 stages 都可能降低总延迟，但它们分别改变容量工作、计算粒度和执行流水线。拿到一个更快的候选，能否指出究竟改变了哪一个变量，是否同时改变了其他变量？

## Level 5｜TBO 案例：从性能现象到可检验结论

### 5.1 先固定事实：拆批后的单次 GEMM 没有快一半

历史旧 trace 的 B32 样本中，每次 replay 的 routed GEMM 调用数从 52 增至 104；每次 replay 的这类 kernel 总耗时中位数由约 5.981 ms 增至 10.826 ms，按该聚合折算的单次耗时约为 115.0→104.1 μs。它证明了该采样条件下的现象，尚不能单独证明某个专家的 tile 数完全没变。

这份 trace 使用旧的未 compact 适配器，且存在 Nsight 告警。后来的 v5 使用 compact adapter、不同采集条件，不能用旧 kernel 时间替代新版本各项时间。[历史归因报告][local-diagnosis]

同一轮 v5 无 profiler、1024 请求、C256 的服务数据是：

| 模式 | Output tok/s | 平均 TTFT | 平均 TPOT |
|---|---:|---:|---:|
| 不拆批 | 2170.95 | 2671.82 ms | 102.10 ms |
| 双微批顺序执行 | 1396.29 | 5078.98 ms | 153.67 ms |
| TBO | 1427.38 | 5299.88 ms | 150.78 ms |

TBO 相对顺序拆批的吞吐提升约 2.23%，相对不拆批仍下降约 34.25%。这些请求使用自然 EOS，实际输出、路由与批次轨迹不完全相同；它是部署表现对照，不是严格等工作量的算子实验。[原始汇总][local-summary]

基于这些事实，下一步应提出能被测量否定的成本假设，而不是直接归因为“CUDA Graph 固定了 grid”或“GPU 利用率低”。

### 5.2 将现象分解为五个假设

| 假设 | 为什么可能发生 | 要补什么证据 | 什么情况会削弱这个解释 |
|---|---|---|---|
| 有效 tile 重复 | 两个微批都命中同一 expert，各自重新按 BM 对齐 | 固定路由下的每专家 counts、两次 GEMM 的有效配置与 tile 数 | 有效 tile 已明显减少，GEMM 仍同样慢 |
| 容量辅助成本重复 | 每 lane 保留相近输入范围，pack/量化/metadata 不按 counts 缩小 | 每 lane 计算容量、量化与适配耗时、发射规模 | 容量与辅助耗时已缩小，回归仍由别处主导 |
| 权重访问重复 | 相同专家在两微批中各走一次 K 循环 | 专家活跃集合、逻辑访问、L2/DRAM 指标 | 缓存有效复用，且带宽不构成瓶颈 |
| 并行资源竞争 | 两条 compute lane 或通信与计算同时消耗资源 | 顺序拆批与 TBO 的匹配 kernel 时长、资源指标 | 联合运行没有增加 kernel 时长，回归在顺序拆批已存在 |
| 可隐藏成本不足 | 通信较短或依赖限制了可重叠区域 | 含依赖关系的实际时间线与完整阶段跨度 | 存在足够有效重叠且阶段显著缩短 |

“多个 kernel 在时间线上重叠”只说明有并发，不直接证明有效网络进度，更不保证总时间减少。分类 kernel duration 之和也不等于关键路径时间。

### 5.3 用第二、三章的算例推导拆批代价

先固定路由，只把 counts `[70,12]` 等分成 `[35,6]` 和 `[35,6]`。在 BM64 下：

```text
整批：ceil(70/64)+ceil(12/64) = 3 个有效专家 M 块
每半批：ceil(35/64)+ceil(6/64) = 2 个有效专家 M 块
两半批合计：4 个有效专家 M 块
```

有效数学工作量整轮不变，但有效专家 tile 覆盖行数从 192 增为 256；每次 GEMM 的有效 dot program 总数由 6 增为 8。若每 lane 仍保留 S=256 的适配容量，整轮容量行处理还从一份变成两份。能否重叠只决定这些成本的时间安排，不会把新增工作从算术上取消。

对某个只有 24 行的专家，24→12＋12 在 BM64 下则是一块变两块，每半批的有效 tile 数完全没少。这解释了历史现象可能怎样产生；真实实验是否如此，仍需每层 counts，而不能把这个例子当作测量结果。

更一般地，对固定路由 $m_e=a_e+b_e$：

$$
\left\lceil\frac{a_e}{BM}\right\rceil+
\left\lceil\frac{b_e}{BM}\right\rceil
\geq\left\lceil\frac{m_e}{BM}\right\rceil.
$$

各专家独立对齐，因此只知道总 token 数不足以知道总 tile 数。路由偏斜、空专家和两微批的专家活跃集合都会改变结果。

### 5.4 测什么指标，才不会把“利用率”混成一个词？

先记录有效专家 M 行填充率：

$$
\eta_M=\frac{\sum_e m_e}{BM\sum_e\lceil m_e/BM\rceil},
$$

仅在分母非零时使用。本例整批为 82/192≈42.7%，等分后的整轮为 82/256≈32.0%。它是 padding 的解释指标，未覆盖无效组、N/K 尾部或真实指令效率。

| 指标层 | 必须分开的量 | 用途 |
|---|---|---|
| 数据与工作量 | 容量 S、有效行、索引容量 Q、本轮 padded 长度、有效/launch tile 数 | 解释为何工作没有同比减少 |
| GPU 执行 | 各阶段时长、完整 replay 跨度、必要的带宽/缓存/驻留指标 | 找到成本与关键路径 |
| 请求表现 | output tok/s、TTFT、TPOT 及分位数、错误数和输出长度 | 判断用户实际获得什么收益 |

输入只占容量的 25%、M tile 填充率 25%、occupancy 25% 是不同结论。不要将其中任意一个改写为“GPU 浪费了 75%”。

诊断时可以采集 counters；正式计时时不要逐步 `.cpu()`/`.item()` 读取 GPU counters，否则可能额外引入同步，改变要测量的路径。

### 5.5 为什么下一轮固定客户端 C64？

这是已选定的实验点，不是普遍最优值。v5 中，不拆批平均 TTFT 从 C64 的约 299 ms 增至 C256 的约 2672 ms，顺序拆批从约 551 ms 增至 5079 ms。提高并发虽然增加在途工作，也付出了首 token 等待代价。[汇总数据][local-summary]

客户端 C64 不等于每 rank 的 Graph B64，更不等于每专家 64 行。历史模型有 64 个 routed experts、top-k=6；若全局恰有 64 个有效 decode token 且均衡分配，整批每 expert 平均只有 6 行，等分微批平均为 3 行。这是工作量估算，不是实际 counts。

增加总请求数用于延长观测窗口和覆盖启动/收尾之外的运行阶段。它不会自动增大每步 M。输入文件、shuffle、缓存策略和输出规则应固定；简单循环同一批 prompt 还可能增加 prefix-cache 命中，必须记录。

### 5.6 最小对照如何区分“拆批成本”与“重叠收益”？

先固定 backend、模型、dtype、adapter/compact/tight-capacity 版本及硬件，定义三种模式：

| 模式 | 回答的问题 |
|---|---|
| A：不拆批 | 整批执行的基准成本是多少？ |
| B：双微批顺序执行 | 没有跨微批重叠时，拆批本身增加了多少成本？ |
| C：TBO | 并发调度相对 B 回收或新增了多少时间？ |

在匹配工作量的 decode 实验中，可以比较 `T_B−T_A` 与 `T_B−T_C`。后者是调度的净效果，混合了隐藏等待、资源竞争和额外同步，不能直接命名为“隐藏的通信时间”。测量应跨 rank 对齐，按同一步的最慢参与者评估分布式完成时间。

为了分析 tile，应先固定路由和输入做 compute 对照；为了验证真实网络运行，再做完整模型与服务对照。历史 router 的浮点归约曾随 batch 形状改变，因此自然模型输出不能默认满足严格的整批/半批等路由条件。

正式服务结果与 profiler 结果分开。固定输入的局部算子实验解释机制，正常 EOS 的 HTTP 测试检验部署表现，二者不互相替代。

### 5.7 BM sweep 如何形成可提交的结论？

候选以 BM8/16/32 加默认 BM64 参照；BM4 或其他值需先确认编译与正确性。两个 routed GEMM 都应用目标 BM，保留共享 alignment，记录最终 BN/BK/warps/stages、是否切换内部路径。不要把编译失败或准确性失败的点静默删除后只展示胜者。

按以下顺序推进：

1. **先核对执行版本。** 明确有没有 compact/tight-capacity；历史 v6 同时改过容量和 BM，不能作为单变量 BM 结果。
2. **正确性门禁。** 覆盖零 counts、空专家、偏斜、尾块、不同动态 replay 输入和有效输出比较；验证两次 GEMM，并保留约定误差标准。
3. **局部成本。** 固定输入与路由，观察有效 tile、adapter/量化/GEMM 时间；诊断开销不混入正式计时。
4. **C64 服务比较。** 所有模式使用相同请求文件、总请求数、缓存与输出规则；重复并交错运行，报告分布与波动，不挑单次最好成绩。

两类结果都要保留：

- **相同 BM：** 看 TBO 在相同分块设定下的变化。
- **各自最优稳定 BM：** 看两种部署分别调优以后，用户能获得的最佳已测表现；同时检查 TTFT/TPOT 是否可接受。

如最优 BM 不同，先用 counts、容量驱动的配置键和联合执行指标区分原因，再讨论实现职责。给维护者的问题应是“MoE backend 的配置选择是否要考虑微批规模与可获得的专家负载信息”，而不是先要求 TBO 固定写入某个 BM。

PR 的结论应落在明确条件内：哪套 GPU、模型、adapter 和负载下，减少了哪一项成本，端到端改善多少，是否存在回归。若重叠收益不足以抵消拆批成本，也应直接报告。

**本层自检：** 更小 BM 让单次 GEMM 更快，为什么 TBO 仍可能更慢？如果 C64 吞吐增加但 TTFT 恶化，应怎样描述这次取舍，而不是简单宣布优化成功？

## 验证范围与后续边界

- 本次依据固定提交推演索引、配置与地址，并用 CPU 算术检查例子；没有新增 GPU 正确性、BM sweep 或 profiler 实验。
- Level 1 的数学目标保持不变；Level 2 讲后端通用的任务组织，Level 3 对应主线的非 TMA 实现，Level 4 区分已有优化与候选方案。
- 精确索引顺序按同一提交的小批量 alignment 分支推演；实际安装的 native kernel 来源或 override 不同，需重新核对。
- rank-major 只作布局对照，不提供未接通的 rank-major＋Triton 执行示例。
- 历史 compact、tight-capacity 与 TBO 数据分别标明版本，不能归入原始兼容 PR；C64 也不是普遍最优并发结论。

## 参考源码与历史材料

主线链接固定到 `491a30820f6c4a65094eb97ae12604766b2f7847`；rank-major 固定到 `e1f7de17a521dd607e1bec62081f26121cf5c6cc`。后端概览文档核对日期为 2026-09-30；本次新增的 grouped 调度与 GEMM 资源说明核对日期为 2026-10-01。

[s-adapter]: https://github.com/Laceprndpm/sglang/blob/491a30820f6c4a65094eb97ae12604766b2f7847/python/sglang/srt/layers/moe/moe_runner/nccl_ep_triton.py
[s-dispatch]: https://github.com/Laceprndpm/sglang/blob/491a30820f6c4a65094eb97ae12604766b2f7847/python/sglang/srt/layers/moe/token_dispatcher/nccl_ep.py
[s-runner]: https://github.com/Laceprndpm/sglang/blob/491a30820f6c4a65094eb97ae12604766b2f7847/python/sglang/srt/layers/moe/moe_runner/triton.py
[s-fused]: https://github.com/Laceprndpm/sglang/blob/491a30820f6c4a65094eb97ae12604766b2f7847/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py
[s-kernel]: https://github.com/Laceprndpm/sglang/blob/491a30820f6c4a65094eb97ae12604766b2f7847/python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py
[s-config]: https://github.com/Laceprndpm/sglang/blob/491a30820f6c4a65094eb97ae12604766b2f7847/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py
[s-align]: https://github.com/Laceprndpm/sglang/blob/491a30820f6c4a65094eb97ae12604766b2f7847/python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py
[s-rank]: https://github.com/sgl-project/sglang/blob/e1f7de17a521dd607e1bec62081f26121cf5c6cc/python/sglang/srt/layers/moe/token_dispatcher/nccl_ep.py
[cuda-graph]: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
[local-diagnosis]: /home/patchouli/work/docs/profile-v5-c256/Diagnosis.md
[local-v5]: /home/patchouli/work/docs/profile-v5-c256/README.md
[local-v6]: /home/patchouli/work/docs/profile-v6-microbatch/README.md
[local-summary]: /home/patchouli/work/docs/profile-v5-c256/summary.json

[backend-overview]: https://docs.sglang.io/docs/advanced_features/expert_parallelism#backends-for-moe-computation

[s-align-entry]: https://github.com/Laceprndpm/sglang/blob/491a30820f6c4a65094eb97ae12604766b2f7847/python/sglang/kernels/ops/moe/__init__.py
[s-align-native]: https://github.com/Laceprndpm/sglang/blob/491a30820f6c4a65094eb97ae12604766b2f7847/sgl-kernel/csrc/moe/moe_align_kernel.cu
[s-quant]: https://github.com/Laceprndpm/sglang/blob/491a30820f6c4a65094eb97ae12604766b2f7847/python/sglang/kernels/ops/quantization/fp8_kernel.py
[triton-config]: https://triton-lang.org/main/python-api/generated/triton.Config.html
[triton-matmul]: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

[cutlass-grouped]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html
[cutlass-efficient]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
[triton-grouped]: https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html
