---
title: sglang-runtime
tags:
  - SGLang
  - Runtime
  - Intro
  - AI-Infra
categories:
  - 框架
mathjax: true
---
# SGLang Execution Runtime 入门：职责边界、对象资源与动态数据流

**面向读者：** 已初步了解 SGLang 请求处理、调度和模型执行，但还不清楚执行层内部对象如何协作的开发者。本文帮助读者区分对象职责、长期资源与本轮数据，并沿一次普通 decode 追踪它们的关系。

**核心问题与结论：** 一轮请求的数据如何流动，执行所需的资源由谁持有？ModelRunner 组织本地执行环境，runner 准备执行输入，backend 提供具体实现；理解这一层需要同时追踪计算数据、执行元信息和 KV 访问，区分调用关系与资源所有权。

**范围与依据：** 展开 Level 1–3，Level 4–5 保留后续位置。对象拓扑与 Eager decode 主线基于 PR #38683 的固定提交 `537b7a17477a6621efd727f5f088848d754ed3b4`；Init 节沿用原稿未固定 commit 的公开 `main` 走读，单独标明范围。数值与形状是教学示例，未运行模型或完成实卡验证；PR 专用 NCCL EP Graph 资源不代表所有 SGLang 版本的默认行为。

## 引言

初看 SGLang，容易记住一条调用链：Scheduler 把任务交给 ModelRunner，ModelRunner 再让 GPU 执行。但这还不能回答几个具体问题：这一轮的 token 放在哪里？模型权重和 KV cache 是每轮传入，还是提前准备？CUDA Graph 究竟由哪个对象保存？

这些问题都落在 **Execution Runtime，即执行运行时**。理解这一层，不能只看「输入 → 输出」，还要看一个对象在两次调用之间保留了什么，以及它引用的资源实际挂在哪里。

本文沿用「调用关系、数据关系、所有权关系分开看」的方法，先建立对象地图，再沿一次 Eager forward 追踪数据，不展开 kernel 实现。这里的 Level 是教学视角，不是 SGLang 官方层级：先看这一层负责什么，再看它由什么组成，最后进入本轮数据流；执行机制和分布式正确性留作后续章节。

为避免名称误导，本文把 **算子／通信 backend** 作为执行层下方的实现组件；`FullCudaGraphBackend` 虽然也叫 backend，但负责的是图的捕获、执行和清理，仍归入本文的 Execution Runtime。[4][ref-4]

## Level 1：职责边界——这一层具体负责什么？

**Execution Runtime 负责把调度器选定的这一轮 batch，组织成模型在本地设备上的一次执行，并把结果交回调度侧。** 它接收 `ScheduleBatch`，由 `TpModelWorker` 构造执行用的 `ForwardBatch`；随后利用已经加载的模型权重、KV 存储与映射、attention backend 和 CUDA 执行资源，检查本轮能否使用已准备的 Graph，否则走相应的普通执行路径。模型计算通过下层算子和通信实现完成，生成路径再按需要采样、包装成 `GenerationBatchResult`。它通常不重新决定队列里哪些请求应当入选，也不亲自实现 attention 或 NCCL 的通信算法。因此，这一层的输入是**本轮数据与元信息**，不是「Scheduler 这个对象」；输出是**执行结果**，不是「调用 Backend」这个动作。[1][ref-1] [2][ref-2]

## Level 2：对象拓扑——谁持有谁，资源挂在哪里？

### 2.1 先看职责图，再看所有权

下面是以**普通文本生成、full decode CUDA Graph**为主线的职责图。它省略了其他执行分支，不表示每个框都是独立进程，也不表示每条线都是直接函数调用。[1][ref-1] [2][ref-2] [3][ref-3] [4][ref-4]

```text
                        请求接入：API / tokenization
                                      │
                                      ▼
          ┌───────────────────────────────────────────────────────┐
          │ 调度与 KV 管理协作                                    │
          │ Scheduler：选择请求、维护状态、构造 batch             │
          │ cache / allocator / 映射表：提供复用与容量信息        │
          └───────────────────────────────────────────────────────┘
                                      │ ScheduleBatch
                                      ▼
                  TpModelWorker.forward_batch_generation()
                                      │ ForwardBatch.init_new(...)
                                      │ 产生 ForwardBatch
                                      ▼
                            ModelRunner.forward()
                                      │ 检查可用执行路径
                  ┌───────────────────┴───────────────────┐
                  ▼                                       ▼
             EagerRunner                        DecodeCudaGraphRunner
            普通模型执行                      选 bucket、填输入、裁输出
                  │                                       │
                  │                             FullCudaGraphBackend
                  │                          capture / replay / cleanup
                  ▼                                       ▼
        调用 model.forward()                  replay 已捕获的 GPU 操作
                  │                               不重跑图内 Python
                  ▼                                       │
      模型中的 attention / MoE                            │
                  │                                       │
  MoE：dispatch → compute → combine                       │
                  │                                       │
                  └───────────────────┬───────────────────┘
                                      ▼
                          CUDA / 通信后端的设备工作
                                      │ 对应的输出 Tensor
                                      ▼
                              ModelRunnerOutput
                                      │ TpModelWorker 按需采样、包装
                                      ▼
                            GenerationBatchResult
                                      │
                                      └── 返回调度侧处理结果

旁路资源管理：不是模型 Tensor 必须依次经过的计算节点

FullCudaGraphBackend ── 持有并管理 ──> NcclEpGraphResources
NcclEpDispatcher    ── 借用资源   ──> Graph EP group / handle / buffers

```

图中的 Graph 分支要分阶段理解：**准备和 capture 时仍会调用模型代码；replay 时不再逐层重跑这些 Python 调用，而是提交已经捕获的设备操作。** 外围的选图、填输入、包装输出等 Python 代码仍然执行。[3][ref-3] [4][ref-4]

另外，`GenerationBatchResult` 不一定装着新 token。最后一个 PP stage 可以返回 logits 和采样结果；非最后一个 stage 返回交给后续 stage 的中间 Tensor。这里先认识返回对象，不展开流水线并行。[1][ref-1]

### 2.2 阅读对象时，先区分三种东西

以下是本文采用的阅读约定，不是源码中的三种基类：

| 类别      | 具体例子                                       | 阅读时要问什么？           |
| ------- | ------------------------------------------ | ------------------ |
| 本轮数据    | `input_ids`、`positions`、`out_cache_loc`    | 本轮处理什么？这些值是什么格式？   |
| 配置／状态   | `gpu_id`、`forward_pass_id`、`max_bs`、当前借用者  | 对象属于哪里？目前准备到了什么状态？ |
| 资源／资源引用 | model、KV 存储、输入 buffer、CUDA Graph、通信 handle | 谁创建、谁持有、谁使用、谁关闭？   |

**持有引用，不等于复制了一份资源，更不等于独占所有权。** 例如 Worker 和 ModelRunner 可以引用同一个 allocator；一个 GPU Tensor 的 Python 对象保存形状、类型和设备等信息，而数据存储位于对应设备。给另一个对象传这个 Tensor 引用，不等于把整块数据复制过去。[1][ref-1] [5][ref-5]

下面所有形状记号都是阅读辅助：`B` 表示本轮请求数，`T` 表示本轮输入 token 总数，`H` 表示隐藏维度，`V` 表示词表大小。在不含推测解码的普通 decode 中，每个请求输入一个 token，因此 `T = B`；prefill 则不能直接这样等同。[5][ref-5]

### 2.3 先认识两个「本轮数据包」

#### ScheduleBatch：调度侧交来的对象

它不是纯 token 数组，而是包含请求对象、执行模式、Tensor 和调度元信息的 Python 对象。它由 Scheduler 管理；下面只展示与本篇相关的字段。[6][ref-6]

| 字段                 | 格式／装着什么                                    | 用途                           |
| ------------------ | ------------------------------------------ | ---------------------------- |
| `reqs`             | `list[Req]`；每个请求的身份、已输入／已生成 token、采样与结束状态等 | 保留请求层信息                      |
| `forward_mode`     | `ForwardMode` 枚举，如 `DECODE`、`EXTEND`       | 标明这一轮计算的语义                   |
| `input_ids`        | 整型 Tensor，普通路径可按 `[T]` 理解                  | 本轮真正要送进模型的 token 编号          |
| `seq_lens`         | 整型 Tensor，普通路径为 `[B]`                      | 每个请求当前的序列长度                  |
| `req_pool_indices` | 整型 Tensor，普通路径为 `[B]`                      | 每个请求在请求映射池中的行索引              |
| `out_cache_loc`    | 整型 Tensor，普通路径为 `[T]`                      | 本轮输入 token 对应的 KV 写入 slot 编号 |
| `sampling_info`    | 复合对象，包含采样参数及相关 Tensor／元信息                  | 为后续采样提供参数                    |

`ScheduleBatch` 有很多 CPU 侧信息，但**不能把上表全部理解成 CPU 数据**。其中一些核心字段已经是 GPU Tensor，后续转换会直接复用它们。[5][ref-5]

#### ForwardBatch：执行侧使用的表示

`ForwardBatch` 是一个 dataclass。`ForwardBatch.init_new()` 从 `ScheduleBatch` 取出本轮执行需要的字段，再补充位置等执行信息。它不是重新调度，也不是「把整个 CPU batch 全量拷到 GPU」。该提交中，`input_ids`、`req_pool_indices`、`seq_lens`、`out_cache_loc` 等字段会按引用传入。[5][ref-5]

下面是一份**普通 decode 的字段示意**，不是完整构造代码，也不是测试实测值：

```text
ForwardBatch
├─ forward_mode      = DECODE
├─ batch_size        = 2
├─ input_ids         = Tensor([1011, 2022])  # [2]，两个请求本轮的输入 token
├─ seq_lens          = Tensor([5, 3])        # [2]，本轮计入当前 token 的长度
├─ positions         = Tensor([4, 2])        # [2]，从 0 开始的位置示意
├─ req_pool_indices  = Tensor([3, 7])        # [2]，映射池中第 3、7 行
├─ out_cache_loc     = Tensor([101, 205])    # [2]，本轮 KV 写入 slot
└─ sampling_info     = SamplingBatchInfo(...)

```

这里的 `101`、`205` 是 **KV slot 编号，不是 token 编号，也不是内存地址**。具体 backend 如何构造消费用的 KV 索引，将在 Level 3 中展开。[5][ref-5] [7][ref-7]

`ForwardBatch` 携带的是本轮输入与寻址信息；模型权重和整块 KV 存储不需要随着每个 batch 重新传一份。

### 2.4 TpModelWorker：调度侧进入本地执行系统的入口

`TpModelWorker` 的主要调用者是 Scheduler。它接收 `ScheduleBatch`，或者接收已经准备好的 `ForwardBatch`，返回 `GenerationBatchResult`。[1][ref-1]

把这个长期存在的对象展开，重点不是记住所有字段，而是区分**身份配置、下级对象和共享引用**：

| 字段／成员                                         | 格式                                  | 里面实际是什么？                                             |
| -------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------ |
| `server_args`                                      | `ServerArgs` 对象                     | 模型路径、并行规模、Graph 与执行功能开关等配置；不是权重本身 |
| `ps`                                               | `ParallelState` 对象                  | 当前 worker 的并行身份，如 TP／PP rank 和相应规模；不是通信数据 |
| `gpu_id`、`nccl_port`                              | 整数                                  | 设备编号、初始化通信所用的端口信息                           |
| `model_config`                                     | `ModelConfig` 对象                    | 模型结构与类型相关配置                                       |
| `_model_runner`                                    | `ModelRunner` 引用                    | 真正的本地模型执行环境；通过 `model_runner` 属性访问         |
| `req_to_token_pool`、`token_to_kv_pool_allocator`  | 对象引用，也可能在初始化阶段为 `None` | 请求映射池、KV slot 分配器；可由外部传入，不代表 Worker 独占 |
| `pp_group`、`world_group`                          | 通信组对象引用                        | 当前 worker 所属的 PP／world 组及其通信上下文                |
| `tokenizer`、`processor`                           | 处理器对象或 `None`                   | 文本／多模态相关处理工具；持有它们不等于每轮 forward 都重新分词 |
| `enable_overlap`、`enable_spec`、`is_draft_worker` | 布尔值                                | 调度重叠、推测解码、draft 角色等配置状态                     |

以上成员和初始化方式可直接在 Worker 构造函数中核对。[1][ref-1]

一次普通生成调用中，它主要连接下面两个接口：

```text
输入：ScheduleBatch
    │ 构造 ForwardBatch
    ▼
调用：model_runner.forward(forward_batch)
    │ 拿到 ModelRunnerOutput
    ▼
按需调用：model_runner.sample(...)
    │
    ▼
输出：GenerationBatchResult

```

采样并非无条件发生：PP 中间 stage、verify、prefill-only 或延迟采样路径会有不同处理。因此 **Worker 是执行入口和结果适配对象，不是所有模型计算的实现者**。[1][ref-1]

### 2.5 ModelRunner：本 rank 的模型执行环境

在本文讨论的常规模型执行路径中，每个模型执行 rank（执行组中的一个成员）都有自己的 `ModelRunner`。Worker 创建它时，会传入本地 `gpu_id` 和 `ParallelState`。这意味着各 rank 分别引用自己的模型状态、设备资源与通信上下文；不是一个全局 Python 对象装下所有 GPU 的执行环境。推测解码等配置还可能让一个 rank 拥有额外的 ModelRunner。[1][ref-1] [2][ref-2]

#### 它长期持有什么？

| 成员／资源                        | 格式与实际内容                              | 与本轮输入的关系                                |
| ---------------------------- | ------------------------------------ | --------------------------------------- |
| `model`                      | 模型对象，其参数中包含本 rank 使用的权重 Tensor       | 提前加载并跨 forward 使用，不随每个 batch 重新传入       |
| `model_config`、`ps`          | 配置与并行身份对象                            | 提供结构、设备和分片等执行上下文                        |
| `req_to_token_pool`          | 请求到 token 位置的映射池引用                   | 配合 `req_pool_indices` 找到请求的缓存映射         |
| `token_to_kv_pool_allocator` | 管理 KV slot 编号的 allocator 引用          | 与实际 KV 存储关联，但 allocator 本身不等于 K/V 数据    |
| KV 相关配置与存储组件                 | 本地 KV 资源，由 pool／configurator 等组件具体组织 | attention 读取旧缓存、写入新缓存所依赖的长期存储           |
| `forward_stream`             | 设备 Stream 对象                         | 组织异步设备工作；不是模型数据 Tensor                  |
| `attn_backend`               | attention 实现对象                       | 为具体模型执行准备、使用 attention 所需元信息与实现         |
| `eager_runner`               | `EagerRunner` 对象                     | 普通执行路径                                  |
| `decode_cuda_graph_runner`   | Graph runner 对象，或未启用时为空              | 管理 decode 的形状、输入与 Graph backend         |
| `prefill_cuda_graph_runner`  | prefill 执行路径相关对象，依配置存在               | 其他执行分支；本文不展开                            |
| `sampler`                    | 采样组件                                 | 由 `sample()` 等路径使用 logits 和采样信息产生 token |
| `forward_pass_id`            | 整数状态                                 | 记录 forward 轮次相关状态，不是 GPU 资源             |

这是一份关键成员索引，不是完整字段清单；资源是否分配以及具体类型取决于配置。源码中的初始化、KV 配置和执行分流分别可在 ModelRunner 与相关组件中核对。[2][ref-2] [7][ref-7]

#### KV 的三个对象不能混成「KV\_slot」

```text
请求映射池：ReqToTokenPool
    保存「某个请求的某个序列位置，对应哪个 token slot」

KV slot 分配器：TokenToKVPoolAllocator
    管理 slot 编号的分配与回收

KV 存储：KVCache / 具体 pool 实现
    实际保存各层使用的缓存数据

```

例如，`req_pool_indices=[3,7]` 指向映射池中的请求行，`out_cache_loc=[101,205]` 描述本轮要写的 slot；它们都是**索引信息**，不是两份 KV cache。实际缓存布局还依赖 attention 类型、数据类型和存储实现，不能统一画成所有模型都相同的一张 Tensor。[5][ref-5] [7][ref-7]

#### 它接收和返回什么？

输入主要是 `ForwardBatch`，PP 路径还可以带中间 Tensor。返回的 `ModelRunnerOutput` 主要有下面这些字段：[2][ref-2]

```text
ModelRunnerOutput
├─ logits_output
│  ├─ LogitsProcessorOutput：最后一段模型的 logits 等结果
│  └─ 或 PPProxyTensors：传递给后续 PP stage 的中间 Tensor
├─ can_run_graph：本次执行分流记录的布尔值
├─ expert_distribution_metrics：可选专家分布统计
├─ routed_experts_output：可选路由记录
└─ indexer_topk_output：可选索引记录

```

在普通 decode、最后一个 PP stage 的简化例子里，可以把 logits 理解为 `[B, V]` 的 Tensor：每行对应一个请求，每列对应一个词表 token 的分数。后续采样得到 `[B]` 的 token ID；因此 `ModelRunner.forward()` 返回 logits，与 Worker 最后交回采样结果，是两件不同的事。[1][ref-1] [10][ref-10]

#### 为什么在这里选择执行路径？

`ForwardBatch.forward_mode` 说明这一轮是什么计算，例如 decode；`ModelRunner` 再向 runner 检查本轮是否符合 Graph 执行条件。已准备哪些 bucket、当前 capture 配置是什么、对应图存不存在，这些信息在 runner／backend 附近，所以让它们提供可用性判断，可以避免 Worker 重复维护这些内部状态。[2][ref-2] [3][ref-3]

这是一种职责安排，不是说语义上禁止在上层组织决策。并且，**本地执行判断不等于允许通信相关的 rank 任意分叉**；多 rank 的协调约束留到 Level 5。

### 2.6 谁真正持有 CUDA Graph？

先看最重要的一条成员关系：[2][ref-2] [3][ref-3] [4][ref-4]

```text
TpModelWorker
└─ _model_runner：ModelRunner
   ├─ model                         模型对象与参数
   ├─ req_to_token_pool             请求映射池引用
   ├─ token_to_kv_pool_allocator     KV slot 分配器引用
   ├─ forward_stream                设备 Stream
   ├─ attn_backend                  attention 实现对象
   ├─ eager_runner：EagerRunner
   │  └─ _eager_registry            输入 slot / buffer 的注册与访问对象
   └─ decode_cuda_graph_runner：DecodeCudaGraphRunner
      ├─ capture_bs / max_bs        bucket 列表与最大 batch 容量
      ├─ captured_req_width         capture 时每个请求的 token 宽度
      ├─ capture_hidden_mode        捕获哪些 hidden states 的配置
      ├─ buffers                    静态输入 Tensor 的容器
      ├─ buffer_registry            输入 slot 的注册与访问对象
      └─ backend：FullCudaGraphBackend
         ├─ _graphs[shape_key]       各 shape 对应的 CUDAGraph 对象
         ├─ _outputs[shape_key]      相应的输出对象／Tensor 引用
         ├─ _pool                   capture allocator pool 的标识
         └─ _nccl_ep_resources      可选 NCCL EP Graph 资源管理对象

```

这张图表示持有／引用关系，不表示每个节点都独占下面全部资源，也没有画出子对象对 `model_runner` 的反向引用。

**ModelRunner 是间接持有 CUDA Graph；直接保存图字典的是 FullCudaGraphBackend。** `shape_key` 是查找图的键，不是输入 Tensor，也不是设备指针。[4][ref-4]

#### EagerRunner：没有 Graph，也不等于没有 buffer

`EagerRunner` 接收 `ForwardBatch`，组织普通模型调用并返回模型结果。这个固定提交里，它还持有 `_eager_registry`，默认会把本轮输入整理到已有输入缓冲区中；相关缓冲区还可能与 Graph 路径共享底层存储。因此不要把 Eager 简化成「每次现分配一切，Graph 才复用资源」。本篇只认识资源位置，不展开拷贝和同步顺序。[8][ref-8]

#### DecodeCudaGraphRunner：决定选哪张图、怎样组织输入

这里的 **bucket** 可以先理解为提前准备的形状档位。假设普通 decode 的 `capture_bs=[1,2,4,8]`，那么最大档位是 `max_bs=8`。这组数只是示例，不是该 PR 的固定配置。[3][ref-3]

| 成员                | 格式                      | 具体含义                                             |
| ----------------- | ----------------------- | ------------------------------------------------ |
| `capture_bs`      | 整数列表                    | 准备捕获的 batch 档位                                   |
| `max_bs`          | 整数                      | 最大捕获 batch 档位                                    |
| `max_num_token`   | 整数                      | 最大输入 token 容量，与每请求 token 宽度有关                    |
| `buffers`         | `DecodeInputBuffers` 对象 | 持有 `input_ids`、`positions`、`seq_lens` 等输入 Tensor |
| `buffer_registry` | 注册表对象                   | 统一访问、填充这些输入 slot；不必另复制一套存储                       |
| `backend`         | Graph backend 对象        | 保存并提交真正的图                                        |

例如，在不涉及额外并行切分的简化情形中，输入 buffer 可包含 `[max_num_token]` 的 token／position 数组，以及 `[max_bs]` 的请求索引／长度数组。**容器对象在 host 侧，数组可以在 GPU 上，部分字段还会有 CPU 镜像。** 不要把整个 `buffers` 对象理解成一个 GPU Tensor。[3][ref-3]

#### FullCudaGraphBackend：保存图及关联输出

它的三个核心成员分别解决三个问题：[4][ref-4]

```text
_graphs  : dict[shape_key, CUDAGraph]
           这个 shape 的可执行图在哪里？

_outputs : dict[shape_key, output_object]
           图对应的输出对象／Tensor 引用在哪里？

_pool    : allocator pool 标识
           capture 期间使用哪个分配池？

```

其中 `_outputs` 不是「每次请求的历史结果仓库」；它保存与已捕获图对应的输出引用。输出跨调用能保留多久、何时要另存，属于后续的执行与生命周期问题。

到这里，可以把两个类的分工压成一句话：**Runner 组织本轮形状和输入，Backend 保存、执行和清理捕获的图。**

### 2.7 NCCL EP Graph 资源为什么画在旁边？

下面是本篇采用的 PR 专用案例：`NcclEpGraphResources`。它不是所有模型都需要的通用模块，而是连接 full decode Graph 与 NCCL EP 低延迟通信路径的资源适配对象。[9][ref-9] [11][ref-11]

这里先区分两个角色：

```text
NcclEpDispatcher
    做什么：接收 MoE 的 token／路由输入，调用 dispatch 和 combine

NcclEpGraphResources
    保存什么：Graph 路径需要长期有效的 EP group、handle 和缓冲区

```

真实的 expert compute 在 MoE 层的 dispatch 与 combine 之间执行，不由这个资源管理器计算。`ModelRunner` 也不是直接亲自调用 NCCL EP 完成每层通信；具体调用由模型中的 MoE 与 dispatcher 衔接。[12][ref-12] [13][ref-13]

#### 谁创建、谁持有、谁借用？

```text
FullCudaGraphBackend
    └─ 创建并持有 NcclEpGraphResources
         ├─ state.group：Graph 专用 native EP Group
         ├─ handle：持续保留的 LL Handle
         ├─ state 中的发送／接收／合并 buffer
         └─ signature / rows / borrower 等状态

RuntimeContext.resources.buffers["nccl_ep_graph_resources"]
    └─ 登记同一个管理对象，供其他模块查找

NcclEpDispatcher
    └─ 在 warmup / capture 的准备过程中，借用其中的 state 与 handle

```

资源管理对象创建时，native Group 和 Handle 不一定已创建；该实现会在 warmup 的首次准备过程中初始化它们。Backend 负责调用相应清理流程，资源管理对象负责关闭自己管理的 native 资源。[4][ref-4] [9][ref-9]

注册表是查找入口，不是模型数据必经的计算节点；登记的是同一个对象，也不是再复制一套 EP 资源。

#### 缓冲区里实际放什么？

对于这个实现，令 `C` 为本 rank 的最大分发 token 容量，`H` 为隐藏维度，`K` 为每个 token 选择的 expert 数，`P` 为通信 rank 数，`E_local` 为本 rank 的 expert 数。资源可具体展开为：[9][ref-9] [13][ref-13]

| 字段                    | 存储格式                        | 实际内容                       |
| ----------------------- | ------------------------------- | ------------------------------ |
| `state.send_tokens`     | BF16 Tensor `[C, H]`            | 待发送的 token hidden states   |
| `state.topk_ids`        | INT64 Tensor `[C, K]`           | 每个 token 的 expert 路由编号  |
| `state.topk_weights`    | FP32 Tensor `[C, K]`            | 对应的合并权重                 |
| `state.recv_tokens`     | BF16 Tensor `[E_local, P*C, H]` | 为各本地 expert 预留的接收存储 |
| `state.expert_counters` | INT32 Tensor `[E_local]`        | 各 expert 的有效接收计数       |
| `state.combined`        | BF16 Tensor `[C, H]`            | 合并阶段使用的输出 scratch     |

表里的形状是**容量布局**，不表示所有行都有效。有效数据范围需要结合路由、mask 和计数解释；通信数据流见 Level 3 的 MoE 分支。这里的 BF16 是通信缓冲区类型，也不能据此推出真实 expert GEMM 必然使用 BF16。

另外，SGLang 的 `GroupCoordinator`、native NCCL EP `Group` 和 `Handle` 不是一个对象。这个实现使用已有 `ncclComm_t` 创建 EP 资源，所以拥有 Graph 专用 EP Group，不等于额外创建了一套模型 rank 拓扑。[9][ref-9]

### 2.8 把对象地图收拢成四个问题

**这一轮的数据在哪里？** 在 `ScheduleBatch` 和 `ForwardBatch` 的字段中：token ID、长度、位置、KV slot 索引、采样元信息等；部分 Tensor 会被复用引用，部分执行信息会补充生成。[5][ref-5]

**长期计算资源在哪里？** `ModelRunner` 持有模型与下级执行组件，并关联本地 KV、设备和通信资源。具体资源继续由各自组件管理，不是全都直接放在 ModelRunner 一层。[2][ref-2]

**CUDA Graph 在哪里？** 在 `DecodeCudaGraphRunner.backend` 所指向的 `FullCudaGraphBackend._graphs` 中；输入 buffer 则主要由 runner 组织。[3][ref-3] [4][ref-4]

**NCCL EP Graph 的专用资源在哪里？** 在 Backend 持有的 `NcclEpGraphResources` 中，dispatcher 借用；它不负责请求调度，也不是 expert 数学计算的一环。[9][ref-9] [13][ref-13]

本层的完成标准：看到一个字段，能够说明它是**本轮数据、配置／状态，还是资源引用**；看到一项资源，能够找到它的持有者，而不把「间接调用」误认为「直接拥有」。

## Level 3：动态数据流

## Level 3.1 : Decode

Level 2 建立的是静态地图：ModelRunner 持有模型、KV 相关资源、EagerRunner 和 Graph runner。但知道对象的归属，还不能解释一次请求真正执行时发生了什么。

Level 3 换一个观察角度：**固定一轮 forward，沿着接口追踪数据。** 每到一个边界，只问三个问题：传过去什么格式的数据？这份数据是谁准备的？下一个组件拿它做什么？

这里尤其要分开两件事：`input_ids → hidden_states → logits` 是模型计算的数据变化；`ForwardBatch` 和 attention metadata 则把本轮的输入、位置、长度及缓存寻址信息组织起来。**ForwardBatch 包含模型输入，不是与 token 并列的另一种数据，也不会整体「变成」hidden states。**[5][ref-5]

本节沿用前文的固定提交源码。模型子层的顺序补充对照同一提交的 Qwen3 实现；数值都是教学示例，不是模型实测或 NCCL EP 端到端验证结果。

### 3.1 固定范围，先看总览

主线选择普通自回归 **Eager decode**：不展开推测解码、PP、DP attention、PDMux 和 CUDA Graph。模型计算先按单卡、普通 MHA/GQA 理解；KV 索引转换采用 **Triton attention、`page_size=1`、无滑动窗口和额外虚实 slot 转换**的路径。MoE／EP 作为后面的可选接口分支，不把它冒充成普通 dense 模型必经的部分。

记号：`B` 是请求数，`T` 是本轮输入 token 数，`H` 是隐藏维度，`V` 是词表大小。普通 decode 每个请求输入一个 token，所以这里 `T=B`；这不表示每个请求的历史长度相同。

#### 执行纵览图

下图的竖向箭头表示本节范围内的主要先后关系，旁边标明传递的数据。它不是进程图，也不是所有函数的完整调用栈。[1][ref-1] [2][ref-2] [8][ref-8]

```text
                      Scheduler 已选定本轮请求
                                 │ ScheduleBatch
                                 ▼
                 TpModelWorker.forward_batch_generation()
                                 │ ForwardBatch.init_new(...)
                                 ▼
           ┌──────────────────────────────────────────┐
           │ ForwardBatch                             │
           │                                          │
           │ 模型输入：input_ids                      │
           │ 位置／长度：positions、seq_lens          │
           │ KV 寻址：req_pool_indices、out_cache_loc │
           │ 其他信息：forward_mode、sampling_info    │
           └──────────────────────────────────────────┘
                                 │
                                 ▼
                        ModelRunner.forward()
                                 │ 本轮进入 Eager 路径
                                 ▼
                  EagerRunner.execute() → _execute_decode()
                                 │
                                 ▼
                            load_batch()
                       整理本轮输入到执行 buffer
                                 │ 执行用的 ForwardBatch
                                 ▼
               attn_backend.init_forward_metadata(...)
                  准备 KV 索引等 backend 专用元信息
                                 │
                                 ▼
               model.forward(input_ids, positions, batch)
                                 │
                                 ▼
                           Embedding
                                 │ hidden_states [T, H]
                                 ▼
                    Decoder layers：依次执行
                 Attention → MLP 或 MoE → 下一层
                    （此处省略 norm／residual）
                                 │ 最终 hidden_states
                                 ▼
                 logits_processor（使用 lm_head）
                                 │ LogitsProcessorOutput
                                 ▼
                   经 EagerRunner 返回 ModelRunner
                                 │ 包装 ModelRunnerOutput
                                 ▼
                           TpModelWorker
                      按需调用 model_runner.sample()
                      再包装 GenerationBatchResult
                                 │
                                 ▼
                       Scheduler 处理本轮结果
```

**读图时注意两点：**`load_batch()` 与 metadata 初始化是先后关系，不是两个并行分支；Attention 和 MoE 的并列介绍表示两类组件，不表示它们必然并行。这里用普通串行 decoder block 示意，Qwen3 源码对应的是 Attention 后接 MLP。[8][ref-8] [14][ref-14]

后面只展开图中真正改变数据表示的几个位置。

### 3.2 起点：两个请求的一份 ForwardBatch

沿用前文的两个请求示例，假设进入模型前，这一轮已经准备好：

```text
forward_mode      = DECODE
batch_size        = 2
input_ids         = [1011, 2022]
positions         = [4, 2]
seq_lens          = [5, 3]
req_pool_indices  = [3, 7]
out_cache_loc     = [101, 205]
```

这些字段分别告诉执行层不同的信息：[5][ref-5]

| 字段               | 本例格式          | 具体含义                                           |
| ------------------ | ----------------- | -------------------------------------------------- |
| `input_ids`        | 整型 Tensor `[2]` | 请求 A 本轮输入 token 1011，请求 B 输入 token 2022 |
| `positions`        | 整型 Tensor `[2]` | 两个输入 token 在各自序列中的位置，按从 0 开始计数 |
| `seq_lens`         | 整型 Tensor `[2]` | 本轮计入当前输入 token 后的序列长度，分别是 5 和 3 |
| `req_pool_indices` | 整型 Tensor `[2]` | 到请求映射池的第 3、7 行查找各自的缓存映射         |
| `out_cache_loc`    | 整型 Tensor `[2]` | 本轮计算得到的 K/V 分别写到 slot 101、205          |
| `sampling_info`    | 复合对象          | 后续采样需要的参数与相关 Tensor，不是模型权重      |

这里最容易混淆的是：**本轮只算两个输入 token，不等于 attention 只看两个 token。** 两个请求分别有自己的上下文，attention 会结合已有 KV cache 处理长度为 5 和 3 的序列。

再区分三个编号：`1011` 是词表中的 token ID，`3` 是请求映射池的行号，`101` 是 KV slot ID。它们虽然都是整数，但索引的是三种不同的东西。[5][ref-5] [7][ref-7]

#### Token 与 ForwardBatch 的边界在哪里？

它们是包含关系，不是「前一种数据变成后一种数据」的关系：

```text
ForwardBatch
├─ input_ids：模型要处理的 token ID
└─ positions / lengths / cache locations / ...：配套执行信息
```

`ForwardBatch.init_new()` 从 `ScheduleBatch` 取得核心字段，其中一些 Tensor 直接按引用传入，再补充位置等信息。因此这一步不是重新分词，也不是把整个 CPU 对象统一搬到 GPU。[5][ref-5]

**生产者：** 调度侧准备本轮输入与资源分配结果，Worker 调用 `init_new()` 构造执行表示。
**消费者：** ModelRunner 及后续 runner、模型和 backend。

### 3.3 EagerRunner：先准备存储，再准备 backend 元信息

ModelRunner 进入 Eager 分支后，普通 decode 的主要顺序可以用下面的简化伪代码表示。省略了计时、特殊模型准备和其他执行分支，不是完整源码：[8][ref-8]

```python
batch = self.load_batch(forward_batch)

if batch.needs_forward_metadata_init():
    attn_backend.init_forward_metadata(batch)

output = model_runner.model.forward(
    batch.input_ids,
    batch.positions,
    batch,
)
```

#### load_batch()：值不一定变化，存储位置可能变化

在固定提交的默认路径中，EagerRunner 用 `_eager_registry` 把本轮相关输入填到预先准备的 buffer，再提取一个引用这些 buffer 切片的 `ForwardBatch`。没有 Graph，并不表示没有输入 buffer 或存储复用。[8][ref-8]

```text
原 ForwardBatch 中的相关 Tensor
                │ fill_from：复制／填充注册的输入字段
                ▼
Eager 输入 buffer 的本轮有效切片
                │ extract_buffer
                ▼
执行用的 ForwardBatch
```

这一阶段关注的是**执行从哪份存储读输入**，不是把 token ID 算成 hidden states。比如复制前后 `input_ids` 都可以是 `[1011,2022]`，但它所引用的存储位置不同。

#### init_forward_metadata()：通用信息转成专用表示

`ForwardBatch` 给出请求行号、序列长度等通用信息；具体 attention backend 再把它们整理成自己的执行格式。对本节选定的 Triton decode 路径，重要产物包括 `kv_indptr` 和 `kv_indices`，并由 backend 的 `forward_metadata` 保存。[15][ref-15]

这一步**还没有计算 Q/K/V，也不需要复制整份历史 K/V 数据**；它先准备好后面访问缓存所用的索引。

### 3.4 KV 转换：从请求大表到本 batch 的索引流

这里承接前面讨论的核心问题：**CUDA 后端最终消费的 KV 表，与 Runtime 长期维护的请求映射表之间，发生了什么转换？**

#### 原始表示：按请求行号保存的长期映射

在示例中，当前 token 对应的 slot 已经分配并登记。两条有效映射为：

```text
req_pool_indices = [3, 7]
seq_lens         = [5, 3]

req_to_token[3, :5] = [21, 56, 91, 120, 101]
req_to_token[7, :3] = [17, 33, 205]
```

`ReqToTokenPool.req_to_token` 是二维整型映射表，固定提交中使用 `int32` 存储。上面只取本 batch 涉及的两行和各行有效长度，不把其他请求或整行未使用容量带入计算。[7][ref-7]

#### 转换结果：两个数组描述变长列表

首先对长度做前缀和：

```text
seq_lens  = [5, 3]
kv_indptr = [0, 5, 8]
```

再按照 batch 中的请求顺序，收集各请求的有效 slot：

```text
kv_indices = [21, 56, 91, 120, 101, 17, 33, 205]
              └───── 请求 A ─────┘  └─ 请求 B ─┘

请求 A 的索引区间：kv_indices[0:5]
请求 B 的索引区间：kv_indices[5:8]
```

这就是这里说的 **CSR 风格表示**：一条连续索引数组，加一条记录每个请求起止位置的数组。`kv_indptr` 虽然名字带 `ptr`，其中的数值仍是数组偏移，不是设备内存地址。[15][ref-15] [16][ref-16]

源码中的 `_fill_kv_indptr_and_indices()` 填写前缀和，然后调用 `create_flashinfer_kv_indices_triton` 收集索引。**函数名中有 FlashInfer，不代表只有 FlashInfer backend 调它；这里的 Triton backend 同样使用它。**[15][ref-15]

#### 格式要按所选 backend 核对

在本节限定的固定提交、普通 Triton Eager decode 路径中：[7][ref-7] [15][ref-15]

| 数据           | 形状                           | 存储类型 | 保存什么？                              |
| -------------- | ------------------------------ | -------- | --------------------------------------- |
| `req_to_token` | `[请求容量+1, 最大上下文长度]` | `int32`  | 长期维护的 request-position → slot 映射 |
| `kv_indptr`    | `[B+1]`                        | `int32`  | 本 batch 每个请求在索引流中的起止偏移   |
| `kv_indices`   | 有效长度为 `sum(seq_lens)`     | `int64`  | 本 batch 按请求顺序排列的 KV slot ID    |

本例有效索引数为 8。源码在缺少长度总和信息时也可能预留更大的数组；**分配容量与有效区间不是同一个概念**，有效边界仍由长度和 `kv_indptr` 描述。[15][ref-15]

不能把这个表推广为「所有 CUDA attention 都要求这三个格式」。例如 FlashMLA 的 page-granular 路径另有按页取样和 `slot // page_size` 的转换；本节的 `page_size=1` 路径不做这种压页。[16][ref-16]

#### kernel 怎样消费？

在本节无额外 slot 转换的简化布局中，逻辑上相当于：

```python
# 仅表示寻址关系；不是实际 kernel 的并行循环或完整 Tensor 布局。
start = kv_indptr[b]
end = kv_indptr[b + 1]

for j in range(start, end):
    slot = kv_indices[j]
    k = K_cache_of_this_layer[slot]
    v = V_cache_of_this_layer[slot]
    # 与当前请求的 query 参与 attention 计算。
```

**变连续的是索引数组，不是所有 K/V 的物理位置。** `[21,56,91,...]` 仍可能指向分散的缓存 slot；这一步没有把历史 KV 打包成一份连续副本。[7][ref-7] [16][ref-16]

对应的生产者／消费者是：

```text
Runtime 的请求映射 + ForwardBatch 的行号／长度
                  │ attention backend 构造
                  ▼
            kv_indptr + kv_indices
                  │ attention kernel 消费
                  ▼
             本层 K/V cache 的访问
```

### 3.5 进入模型：计算数据与元信息开始共同工作

EagerRunner 调用模型时，传入的是：

```python
model.forward(input_ids, positions, forward_batch)
```

模型既拿到计算输入，也拿到位置和缓存相关元信息。这里不是把 ForwardBatch 交出去之后，Runtime 信息就消失了；模型中的 attention 等组件仍会使用它。[8][ref-8] [14][ref-14]

#### 主计算线：token ID → hidden states

以普通 embedding 输入路径作形状示意：

```text
input_ids [2]，例如 [1011, 2022]
                 │ embedding
                 ▼
hidden_states [2, H]
```

这才是明确的数据性质变化：**离散的词表编号变成浮点向量。** ForwardBatch 不是这个转换的产物，也不是参与 embedding 的整块数值输入。

hidden states 的数学含义由模型定义；本节只关注它经过哪些接口，不展开网络公式。它仍是执行过程中需要分配、传递和保持有效的 Tensor，不能据此说它与 Runtime 完全无关。

#### Attention 同时消费两类输入

对于普通 MHA/GQA，用 `n_q`、`n_kv` 表示本地 Q／KV head 数，`D` 表示 head dimension。下面是逻辑形状，实际实现可能把 head 维展平或使用融合算子：[14][ref-14]

```text
hidden_states [T, H]
       │ QKV projection、位置相关处理
       ├─ Q     [T, n_q,  D]
       ├─ K_new [T, n_kv, D]
       └─ V_new [T, n_kv, D]
```

但光有这三块本轮 Tensor，还不知道每个请求的历史上下文在哪里。于是计算线与寻址线在 attention 处交汇：

```text
模型计算线                         Runtime／metadata 线

Q [T, n_q, D]                     kv_indptr + kv_indices
       │                                  │ 定位可见缓存
       │                                  ▼
       │                           本层 K_cache / V_cache
       │                                  ▲
K_new / V_new ── 按 out_cache_loc 写入 ─────┘
       │
       └────────── 与 Q、可见缓存共同计算 attention
                                          │
                                          ▼
                           attention 输出 → 输出投影
                                          │
                                          ▼
                                hidden_states [T, H]
```

本例 `out_cache_loc=[101,205]` 描述的是：**为本轮输入 token 1011、2022 计算得到的 K/V，写入本层缓存的 slot 101、205。** 不是为还没有采样出来的下一个 token 写 KV。[5][ref-5] [7][ref-7]

因此上一步准备 KV 索引时，可以已经列出 slot 101、205，即使当时还没算出其中本层的 K/V 值。**映射已经有效，不等于该位置的数据已经写好；在消费这些位置前，需要相应的写入先完成。** 具体提交与同步机制留到 Level 4。

位置、长度、索引和 hidden states 也不会互相替代：`positions` 参与位置相关计算；KV metadata 指定缓存访问；hidden states 承载模型中间结果。[14][ref-14] [15][ref-15]

#### Attention 之后继续执行模型子层

普通串行 decoder block 中，attention 之后还有前馈部分。Qwen3 的实现依次组织 attention 与 MLP，并处理 norm、residual 等状态；这里不把每个融合操作展开。[14][ref-14]

```text
Attention 输出
      │ norm／residual 等衔接
      ▼
MLP，或对应模型中的 MoE
      │
      ▼
下一层使用的 hidden_states
```

这部分由具体模型定义，不能把所有模型统一说成「Attention 和 MoE 两个并行分支」。

### 3.6 可选分支：MoE 的 dispatch—compute—combine

这一节只补齐前文关心的 MoE 接口，不改变前面普通 decode 的讲解主线。**Dense 模型不会因为使用 SGLang 就自动经过 MoE；NCCL EP 则需要对应的专家并行环境与配置。**

在 MoE 层，除了 hidden states，还需要 router 给出的 expert 编号和权重。逻辑上，令 `K` 为每个 token 选择的 expert 数：

```text
hidden_states [T, H]
        │ router / top-k
        ▼
TopKOutput
├─ topk_ids     [T, K]：每个 token 交给哪些 expert
└─ topk_weights [T, K]：这些 expert 结果如何加权
```

在已讨论的 `DeepEPMoE.forward_impl()` 非委托路径中，MoE 层组织下面三段，而不是让 dispatcher 包办全部数学计算：[12][ref-12]

```text
hidden_states + TopKOutput
             │ dispatcher.dispatch()
             ▼
       dispatch_output
             │ run_moe_core()：expert compute
             ▼
        combine_input
             │ dispatcher.combine()
             ▼
   按输入 token 顺序恢复的输出 [T, H]
```

对于当前 NCCL EP LL 适配，可以这样认识接口中的数据：[13][ref-13]

| 边界          | 载荷与元信息                         | 生产者 → 消费者                   |
| ----------- | ------------------------------ | --------------------------- |
| dispatch 输入 | token 向量、expert 编号、对应权重        | 模型／router → dispatcher      |
| dispatch 输出 | 按本地 expert 组织的接收数据、有效计数、路由相关字段 | dispatcher → expert compute |
| combine 输入  | expert 计算结果与相应路由／权重            | expert compute → dispatcher |
| combine 输出  | 恢复到原输入 token 顺序的向量 `[T,H]`     | dispatcher → 模型后续计算         |

接收区不是简单的 `[T,H]`。当前 LL 实现按 `[E_local, C_recv, H]` 预留存储，其中 `E_local` 是本 rank 的 expert 数，`C_recv` 是每个 expert 的接收容量；**哪些行有效由计数等信息限定**。分发后，本 rank 要处理的数据还可能来自其他 rank，所以不能用本地输入 token 数直接代替每个 expert 的接收数。[13][ref-13]

还要保留前文已经区分的两个概念：`DeepEPLLDispatchOutput` 是复用的数据格式名称，不等于实际调用 DeepEP 通信；此处 BF16 收发数据还会经过 FP8 转换，不能据此把后续 expert GEMM 一概称为 BF16 GEMM。[13][ref-13]

这段解释接口如何连接，不声称 PR #38683 已通过真实 expert GEMM、完整模型 accuracy 或 serving 性能验收；PR 对这些验证边界有明确说明。[11][ref-11]

### 3.7 返回路径：logits 不是最终生成结果

模型完成各层计算后，通过 logits processor 使用 LM head 产生词表分数。普通 decode 中，每个请求有一行下一 token 的分数，逻辑形状为 `[B,V]`。[14][ref-14] [10][ref-10]

```text
最终 hidden_states [2, H]
             │ logits_processor 使用 lm_head
             ▼
next_token_logits [2, V]
             │
             ▼
LogitsProcessorOutput
```

`LogitsProcessorOutput` 是模型返回的数据对象；随后 ModelRunner 再包装执行结果。普通生成路径中，Worker 按需调用 `model_runner.sample()`，得到 token ID，再交回 Scheduler：[1][ref-1] [2][ref-2]

```text
LogitsProcessorOutput
        │ 经 EagerRunner 返回
        ▼
ModelRunnerOutput
├─ logits_output
├─ can_run_graph
└─ 可选统计／记录
        │ TpModelWorker 读取
        ▼
model_runner.sample(logits_output, forward_batch)
        │ next_token_ids [B]
        ▼
GenerationBatchResult
├─ logits_output
├─ next_token_ids
├─ can_run_cuda_graph
└─ 其他可选结果
        │
        ▼
Scheduler 处理结果，进入后续调度
```

本节范围内，一轮输入 `[2]` 的 token ID，最终可以采样出 `[2]` 的下一 token ID；两者是相邻轮次的 token，不是同一组编号。采样还依赖 `sampling_info`，因此元信息并没有在进入模型后就失去作用。

PP 中间 stage、verify、prefill-only 和延迟采样会改变这条返回路径；这里保留普通生成分支，不把它推广成所有调用都立即采样。[1][ref-1]

### 3.8 收拢：Level 3 要建立的三条线

**第一条是模型计算线。** 它说明数值载荷如何变化：

```text
input_ids → embedding → hidden_states
          → attention / MLP / MoE 等子层
          → logits → sampled token IDs
```

**第二条是执行元信息线。** 它伴随计算，不是模型激活值的下一种表示：

```text
ScheduleBatch
    → ForwardBatch：包含 input_ids，也包含位置／长度／寻址等信息
    → runner 准备执行输入，backend 准备专用 metadata
    → 模型、attention、采样等组件按需消费
```

**第三条是 KV 访问线。** 它把本轮执行与跨轮保存的缓存连接起来：

```text
读：请求行号 + 长度 → 请求映射表 → backend KV 索引 → 本层 K/V cache
写：本轮 K_new / V_new + out_cache_loc → 本层 K/V cache
```

它们是同一次 forward 的不同观察视角，不是三个互斥的模块。Level 3 的重点是：**知道一条边上传的是 token、向量、索引还是执行结果，并能指出生产者与消费者。** 资源如何复用、异步操作如何排序，留到 Level 4；跨 rank 协调和失败恢复，留到 Level 5。

#### 本层自检

| 问题                                                   | 应能说清楚的答案                                             |
| ------------------------------------------------------ | ------------------------------------------------------------ |
| `ForwardBatch` 会整体变成 hidden states 吗？           | 不会。embedding 消费其中的 `input_ids`，其他字段继续提供位置、寻址和执行信息。 |
| `kv_indices` 连续，是否意味着历史 K/V 也变连续了？     | 不意味着。这里收集的是索引，不是重排整份缓存数据。           |
| 本轮为什么只有两个输入 token，却要访问更长的 KV 列表？ | 每个输入 token 的 attention 需要结合自己请求的历史上下文。   |
| `out_cache_loc` 对应本轮输入还是本轮采样出的 token？   | 对应本轮输入 token 的 K/V 写入位置。                         |
| Runtime 的输出是不是「调用了 backend」？               | 不是。调用是动作，返回的是 logits、采样 token 与执行状态等结果对象。 |

## Level 3.2 : Init

建议接在第三层 decode 调用链之后阅读：decode 解释“每轮如何执行”，本节解释“执行所需的对象与资源如何在启动时建立”。

> 范围：8 卡、TP=4、PP=2、EP=4，普通 decode，无 DP attention、无投机解码。参考本次查阅的公开 main 源码，未固定 commit，函数组织可能随版本变化。以下是配置与代码走读，未做实卡验证，也不代表 NCCL EP PR 已验证 PP 支持。

### 1. 具体启动配置

假设单机 8 张 A100 80GB，使用 Qwen3 MoE：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B \
  --tp-size 4 \
  --pp-size 2 \
  --ep-size 4 \
  --moe-a2a-backend none \
  --moe-runner-backend triton \
  --max-running-requests 32 \
  --cuda-graph-bs 1 2 4 8 16 32 \
  --context-length 4096 \
  --mem-fraction-static 0.8
```

`none` 不表示关闭 EP：expert 仍分布在不同 GPU，只是使用 All-Reduce / All-Gather 路线，而非专门的 all-to-all 后端。本例用于理解初始化；它不是 NCCL EP PR 专用启动命令。

来源：[参数文档](https://docs.sglang.io/docs/advanced_features/server_arguments)、[EP 后端说明](https://docs.sglang.io/docs/advanced_features/expert_parallelism)。

### 2. 配置变成 rank 拓扑

此处 DP=1，总 GPU 数为 `TP × PP = 4 × 2 = 8`。EP 在每个 PP stage 的 4 张卡内部划分 expert，不再额外乘 4。

| 全局 rank / GPU | PP rank | TP rank | EP rank |
| --------------- | ------: | ------: | ------: |
| 0               |       0 |       0 |       0 |
| 1               |       0 |       1 |       1 |
| 2               |       0 |       2 |       2 |
| 3               |       0 |       3 |       3 |
| 4               |       1 |       0 |       0 |
| 5               |       1 |       1 |       1 |
| 6               |       1 |       2 |       2 |
| 7               |       1 |       3 |       3 |

对应的逻辑通信组：

| 类型 | 成员                               | 用途                    |
| ---- | ---------------------------------- | ----------------------- |
| TP   | `{0,1,2,3}`、`{4,5,6,7}`           | stage 内张量并行        |
| EP   | `{0,1,2,3}`、`{4,5,6,7}`           | stage 内 expert 分布    |
| PP   | `{0,4}`、`{1,5}`、`{2,6}`、`{3,7}` | 相邻 stage 传递中间结果 |

成员相同，不意味着 TP 和 EP 的逻辑职责相同；逻辑组也不应直接等同于独立 native communicator 的数量。上表按该例的常规 rank 排布展开。

来源：[配置源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py)、[并行组源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/parallel_state.py)。

### 3. 初始化调用链

下面是职责与调用主干，省略包装函数、可选分支及版本相关的中间调用；“启动器”是职责名称。

| 顺序  | 入口 / 调用                                                   | 产生什么                                                |
| --- | --------------------------------------------------------- | --------------------------------------------------- |
| ①   | `prepare_server_args()` → `ServerArgs`                    | 解析参数、补默认值、检查组合                                      |
| ②   | 服务启动器 → `run_scheduler_process(...)`                      | 为各 GPU 启动工作进程，传入配置与 rank 信息                         |
| ③   | 各进程初始化分布式环境与并行组                                           | 建立 TP / PP / EP 通信关系                                |
| ④   | `Scheduler` → `TpModelWorker`                             | 创建本 rank 的模型执行入口                                    |
| ⑤   | `TpModelWorker._init_model_runner()` → `ModelRunner(...)` | 传入模型配置、GPU、并行状态、内存配置                                |
| ⑥   | ModelRunner 初始化模型和执行资源                                    | 本 stage 的模型层、本 rank 的权重分片、KV pool、attention backend |
| ⑦   | decode Graph 初始化                                          | 确定可捕获 bucket，构造 Graph runner 并捕获                    |

Worker 和 ModelRunner 通常是 Scheduler 进程内的对象，不是每个名字又对应一个独立进程。

分布式底座的入口是 `init_distributed_environment()`，随后通过 `initialize_model_parallel()` 或 `ensure_model_parallel_initialized()` 建立并行组。这涉及跨进程协调，属于启动成本。其具体调用位置应以本地分支为准，上表不表示所有步骤都由同一函数直接依次调用。

源码入口：[Engine](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/engine.py)、[Scheduler](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)、[Worker](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tp_worker.py)、[ModelRunner](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py)。

### 4. Graph 在哪里真正生成？

本次查阅源码把这部分拆到了 `model_runner_components/cuda_graph_setup.py`。decode 侧入口：

```python
capture_decode_graph(model_runner=...)
```

内部关键顺序：

1. 检查生成模型、设备及已解析的 decode Graph 配置。
2. 普通 decode 设置 `num_tokens_per_req = 1`。
3. 调用 `get_batch_sizes_to_capture(model_runner, num_tokens_per_req)`。
4. 构造该设备的 Graph runner，进入资源准备和 capture。

这些开关判断读取本地已有配置，不是逐项向其他 rank 发起能力查询。

以 bucket `16` 为例，capture 的含义是：

- 各 rank 准备对应的占位输入和固定 storage。
- 执行本 rank 所负责的模型路径。
- 涉及 TP / EP collective 时，其他参与 rank 需要匹配执行。
- 各 rank 保存自己的 Graph executable。

不是 rank 0 捕获一张覆盖 8 卡的大 Graph，再广播给其他 rank。PP 两个 stage 执行的模型层不同，各自保存本地执行图；stage 间的数据传递仍由 PP 执行流程衔接。PP 传输是否进入某个捕获范围，需要查看具体实现，不能仅凭启用 Graph 推断。

来源：[Graph 初始化源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py)。

### 5. 静态容量的信息来自谁？

| 信息                                    | 提供者                           | 确定时间   |
| --------------------------------------- | -------------------------------- | ---------- |
| 请求并发上限、候选 capture buckets      | 启动配置及运行时解析策略         | 初始化     |
| 实际可用的 capture buckets              | Graph 初始化逻辑结合执行资源约束 | capture 前 |
| 本轮 batch / microbatch 的有效 token 数 | Scheduler 与 PP 调度流程         | 每轮执行前 |
| expert 实际收到的 token 数              | Router 结果与通信                | 本轮执行中 |

普通 decode 每个请求本轮处理 1 个 token。本例最大候选 bucket 为 32；最终以初始化日志中的有效 capture sizes 为准。

最大 Graph bucket 不必等于 Scheduler 的最大 batch。本例把两者设为 32 是为了方便理解；更大的 batch 可以超出 Graph 覆盖范围并走其他执行路径。

EP 接收容量也不能直接照搬本地输入容量：它需要结合参与 rank 数、top-k、路由布局、对齐和后端契约推导。不同通信后端的 buffer 组织并不相同。

这里的 token 数是一次执行的 token 数，不是用户的 `max_new_tokens`，也不是 KV cache 中历史 token 的总数。

### 6. 接回 decode 调用链

假设某个普通 decode microbatch 有 11 个请求，且符合 Graph replay 条件：

| 初始化时已经准备好          | 本轮执行时使用                                  |
| --------------------------- | ----------------------------------------------- |
| TP / PP / EP 通信组         | 执行相应 collective 和 stage 间传输             |
| capture buckets             | 为 11 个请求选取 bucket `16`                    |
| 固定输入 storage            | 写入本轮 token、position、KV 索引，处理 padding |
| 本 rank 的 Graph executable | replay 本 rank 的计算路径                       |
| 输出及工作区                | 消费结果，并保证下次复用的顺序                  |

Graph 不是根据配置凭空生成，而是在初始化阶段用占位输入执行并捕获实际路径。本轮 replay 复用初始化的执行资源。

### 7. 哪些地方有通信 / 同步开销？

| 阶段           | 操作                                    | 开销性质                   |
| -------------- | --------------------------------------- | -------------------------- |
| 初始化         | 建立通信组、加载后的协调                | 启动成本                   |
| 初始化         | capture / warmup 中的 collective 与同步 | 启动成本                   |
| 每轮执行       | TP / EP 通信、PP 传输                   | 分布式执行本身的成本       |
| 每轮执行       | 根据本地配置检查 Graph 条件             | 本地判断，不自动产生通信   |
| 特定分布式路径 | 汇总各 rank 本轮元数据、协调执行模式    | 是否额外通信取决于具体实现 |

本例没有启用 DP attention，不应套用“多个独立 DP batch 每轮汇总 token 数”的解释。Graph 不会消除模型原有的数据通信；也不能仅凭启用了 TP+PP+EP，就断言每轮都有一次额外的 Graph 能力协商。

## Level 4：执行机制

内容：—。**待补：** 本节预留给输入 buffer、bucket、capture／replay 与 stream／event 的执行顺序；现稿尚未提供完整机制分析，不能从对象拓扑直接推导异步执行的正确性。

## Level 5：分布式与生命周期正确性

内容：—。**待补：** 本节预留给多 rank 协调、Graph／eager 资源隔离以及 recapture、cleanup、fallback 和异常处理；现稿尚缺系统分析与相应验证，不能由局部接口说明替代。

---

## 可迁移的方法与未解问题

阅读执行运行时时，先分开调用关系、资源所有权与数据传递，再固定一轮执行，对每个接口记录格式、生产者和消费者。本文贯穿的两个请求示例可用于检查 token ID、请求行号、KV slot 与实际缓存数据是否被混淆。

| 项目 | 内容 | 状态与原因 |
| --- | --- | --- |
| Init 节固定源码版本 | — | 待补：原稿未记录 commit；需核对原始查阅版本，不能把该节直接归入前文固定提交 |
| 实卡执行 trace 与示例对照 | — | 待补：现稿是代码走读与教学示例，没有对应实测记录 |
| 吞吐或延迟基准 | — | 不适用：本文未提出性能收益结论，当前教学范围不要求性能对照 |

执行机制与分布式生命周期问题分别保留在 Level 4、5。

## 参考与复现材料

模型运行复现材料：—。**待补：** 本文未运行模型，现有启动命令属于 Init 节配置示例，不构成已验证的复现流程；下方固定源码支持阅读与核对。


本文的职责地图沿用《从 PR #38683 重新建立 SGLang 架构：六步源码学习路径》的组织方式；字段与数据流说明以原两份文档采用的固定提交源码为基准，模型子层顺序对照同一提交的 Qwen3 实现。对象分层与三条数据线是教学组织方式，不是仓库官方分层标准。数值、形状与伪代码仅用于说明接口，没有运行模型或重新复现实验，不代表 NCCL EP 的完整模型验收。

下列编号源码链接固定到 `537b7a17477a6621efd727f5f088848d754ed3b4`，PR 页面除外。Init 节另列的 `main` 链接没有固定版本，不属于这组编号来源；待补齐版本后再统一核对。相同编号来源只列一次。

1. [tp_worker.py：Worker 的成员、batch 转换和结果包装][ref-1]
2. [model_runner.py：本地执行环境、返回结构和路径选择][ref-2]
3. [decode_cuda_graph_runner.py：bucket、输入缓冲区与 backend][ref-3]
4. [full_cuda_graph_backend.py：图、输出引用与可选 EP 资源][ref-4]
5. [forward_batch_info.py：ForwardBatch 字段和 init_new][ref-5]
6. [schedule_batch.py：请求与调度 batch 的结构][ref-6]
7. [memory_pool.py：请求映射、slot 分配与实际 KV 存储的区别][ref-7]
8. [eager_runner.py：Eager 输入 registry 和执行入口][ref-8]
9. [nccl_ep_graph.py：Graph 专用 EP 资源与所有权][ref-9]
10. [logits_processor.py：LogitsProcessorOutput 与 logits 处理][ref-10]
11. [PR #38683：NCCL EP LL CUDA Graph follow-up 的范围与验证边界][ref-11]
12. [ep_moe/layer.py：dispatch—expert compute—combine 的衔接][ref-12]
13. [nccl_ep.py：dispatcher 与 EP scratch 的具体实现][ref-13]
14. [Qwen3：attention、decoder layer 与 logits 返回；用于核对模型子层顺序][ref-14]
15. [Triton attention backend：KV 索引构造、类型与 forward metadata][ref-15]
16. [kv_indices.py：token 索引收集与另一类按页转换][ref-16]

[ref-1]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/managers/tp_worker.py
[ref-2]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/model_executor/model_runner.py
[ref-3]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
[ref-4]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py
[ref-5]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/model_executor/forward_batch_info.py
[ref-6]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/managers/schedule_batch.py
[ref-7]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/mem_cache/memory_pool.py
[ref-8]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/model_executor/runner/eager_runner.py
[ref-9]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/layers/moe/token_dispatcher/nccl_ep_graph.py
[ref-10]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/layers/logits_processor.py
[ref-11]: https://github.com/sgl-project/sglang/pull/38683
[ref-12]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/layers/moe/ep_moe/layer.py
[ref-13]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/layers/moe/token_dispatcher/nccl_ep.py
[ref-14]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/models/qwen3.py
[ref-15]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/layers/attention/triton_backend.py
[ref-16]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/kernels/ops/kvcache/kv_indices.py