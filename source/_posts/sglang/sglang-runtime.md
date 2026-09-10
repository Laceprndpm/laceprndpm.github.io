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
# SGLang Execution Runtime 入门：从职责边界到对象与资源

> 面向读者：已经初步了解 SGLang 的请求处理、调度和模型执行，但还不清楚执行层内部对象如何协作的开发者。  
> 本篇范围：展开 Level 1「职责边界」与 Level 2「对象拓扑」；Level 3–5 留作后续章节。  
> 源码基准：PR #38683 的固定提交 `537b7a17477a6621efd727f5f088848d754ed3b4`。文中的 NCCL EP Graph 资源适配来自该 PR，不代表所有 SGLang 版本的默认行为。

## 引言

初看 SGLang，容易记住一条调用链：Scheduler 把任务交给 ModelRunner，ModelRunner 再让 GPU 执行。但这还不能回答几个具体问题：这一轮的 token 放在哪里？模型权重和 KV cache 是每轮传入，还是提前准备？CUDA Graph 究竟由哪个对象保存？

这些问题都落在 **Execution Runtime，即执行运行时**。理解这一层，不能只看「输入 → 输出」，还要看一个对象在两次调用之间保留了什么，以及它引用的资源实际挂在哪里。

本文沿用「调用关系、数据关系、所有权关系分开看」的方法，先建立对象地图，不展开 kernel 实现。这里的 Level 是教学视角，不是 SGLang 官方层级：先看这一层负责什么，再看它由什么组成，后续才进入数据流、执行机制和分布式正确性。

为避免名称误导，本文把 **算子／通信 backend** 作为执行层下方的实现组件；`FullCudaGraphBackend` 虽然也叫 backend，但负责的是图的捕获、执行和清理，仍归入本文的 Execution Runtime。[4][graph-backend]

## Level 1：职责边界——这一层具体负责什么？

**Execution Runtime 负责把调度器选定的这一轮 batch，组织成模型在本地设备上的一次执行，并把结果交回调度侧。** 它接收 `ScheduleBatch`，由 `TpModelWorker` 构造执行用的 `ForwardBatch`；随后利用已经加载的模型权重、KV 存储与映射、attention backend 和 CUDA 执行资源，检查本轮能否使用已准备的 Graph，否则走相应的普通执行路径。模型计算通过下层算子和通信实现完成，生成路径再按需要采样、包装成 `GenerationBatchResult`。它通常不重新决定队列里哪些请求应当入选，也不亲自实现 attention 或 NCCL 的通信算法。因此，这一层的输入是**本轮数据与元信息**，不是「Scheduler 这个对象」；输出是**执行结果**，不是「调用 Backend」这个动作。[1][worker] [2][model-runner]

## Level 2：对象拓扑——谁持有谁，资源挂在哪里？

### 2.1 先看职责图，再看所有权

下面是以**普通文本生成、full decode CUDA Graph**为主线的职责图。它省略了其他执行分支，不表示每个框都是独立进程，也不表示每条线都是直接函数调用。[1][worker] [2][model-runner] [3][decode-runner] [4][graph-backend]

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

图中的 Graph 分支要分阶段理解：**准备和 capture 时仍会调用模型代码；replay 时不再逐层重跑这些 Python 调用，而是提交已经捕获的设备操作。** 外围的选图、填输入、包装输出等 Python 代码仍然执行。[3][decode-runner] [4][graph-backend]

另外，`GenerationBatchResult` 不一定装着新 token。最后一个 PP stage 可以返回 logits 和采样结果；非最后一个 stage 返回交给后续 stage 的中间 Tensor。这里先认识返回对象，不展开流水线并行。[1][worker]

### 2.2 阅读对象时，先区分三种东西

以下是本文采用的阅读约定，不是源码中的三种基类：

| 类别           | 具体例子                                             | 阅读时要问什么？                     |
| -------------- | ---------------------------------------------------- | ------------------------------------ |
| 本轮数据       | `input_ids`、`positions`、`out_cache_loc`            | 本轮处理什么？这些值是什么格式？     |
| 配置／状态     | `gpu_id`、`forward_pass_id`、`max_bs`、当前借用者    | 对象属于哪里？目前准备到了什么状态？ |
| 资源／资源引用 | model、KV 存储、输入 buffer、CUDA Graph、通信 handle | 谁创建、谁持有、谁使用、谁关闭？     |

**持有引用，不等于复制了一份资源，更不等于独占所有权。** 例如 Worker 和 ModelRunner 可以引用同一个 allocator；一个 GPU Tensor 的 Python 对象保存形状、类型和设备等信息，而数据存储位于对应设备。给另一个对象传这个 Tensor 引用，不等于把整块数据复制过去。[1][worker] [5][forward-batch]

下面所有形状记号都是阅读辅助：`B` 表示本轮请求数，`T` 表示本轮输入 token 总数，`H` 表示隐藏维度，`V` 表示词表大小。在不含推测解码的普通 decode 中，每个请求输入一个 token，因此 `T = B`；prefill 则不能直接这样等同。[5][forward-batch]

### 2.3 先认识两个「本轮数据包」

#### ScheduleBatch：调度侧交来的对象

它不是纯 token 数组，而是包含请求对象、执行模式、Tensor 和调度元信息的 Python 对象。它由 Scheduler 管理；下面只展示与本篇相关的字段。[6][schedule-batch]

| 字段               | 格式／装着什么                                               | 用途                                    |
| ------------------ | ------------------------------------------------------------ | --------------------------------------- |
| `reqs`             | `list[Req]`；每个请求的身份、已输入／已生成 token、采样与结束状态等 | 保留请求层信息                          |
| `forward_mode`     | `ForwardMode` 枚举，如 `DECODE`、`EXTEND`                    | 标明这一轮计算的语义                    |
| `input_ids`        | 整型 Tensor，普通路径可按 `[T]` 理解                         | 本轮真正要送进模型的 token 编号         |
| `seq_lens`         | 整型 Tensor，普通路径为 `[B]`                                | 每个请求当前的序列长度                  |
| `req_pool_indices` | 整型 Tensor，普通路径为 `[B]`                                | 每个请求在请求映射池中的行索引          |
| `out_cache_loc`    | 整型 Tensor，普通路径为 `[T]`                                | 本轮输入 token 对应的 KV 写入 slot 编号 |
| `sampling_info`    | 复合对象，包含采样参数及相关 Tensor／元信息                  | 为后续采样提供参数                      |

`ScheduleBatch` 有很多 CPU 侧信息，但**不能把上表全部理解成 CPU 数据**。其中一些核心字段已经是 GPU Tensor，后续转换会直接复用它们。[5][forward-batch]

#### ForwardBatch：执行侧使用的表示

`ForwardBatch` 是一个 dataclass。`ForwardBatch.init_new()` 从 `ScheduleBatch` 取出本轮执行需要的字段，再补充位置等执行信息。它不是重新调度，也不是「把整个 CPU batch 全量拷到 GPU」。该提交中，`input_ids`、`req_pool_indices`、`seq_lens`、`out_cache_loc` 等字段会按引用传入。[5][forward-batch]

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

这里的 `101`、`205` 是 **KV slot 编号，不是 token 编号，也不是内存地址**。具体 backend 如何把 slot 转成实际存储位置，本篇不展开。[5][forward-batch] [7][memory-pool]

`ForwardBatch` 携带的是本轮输入与寻址信息；模型权重和整块 KV 存储不需要随着每个 batch 重新传一份。

### 2.4 TpModelWorker：调度侧进入本地执行系统的入口

`TpModelWorker` 的主要调用者是 Scheduler。它接收 `ScheduleBatch`，或者接收已经准备好的 `ForwardBatch`，返回 `GenerationBatchResult`。[1][worker]

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

以上成员和初始化方式可直接在 Worker 构造函数中核对。[1][worker]

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

采样并非无条件发生：PP 中间 stage、verify、prefill-only 或延迟采样路径会有不同处理。因此 **Worker 是执行入口和结果适配对象，不是所有模型计算的实现者**。[1][worker]

### 2.5 ModelRunner：本 rank 的模型执行环境

在本文讨论的常规模型执行路径中，每个模型执行 rank（执行组中的一个成员）都有自己的 `ModelRunner`。Worker 创建它时，会传入本地 `gpu_id` 和 `ParallelState`。这意味着各 rank 分别引用自己的模型状态、设备资源与通信上下文；不是一个全局 Python 对象装下所有 GPU 的执行环境。推测解码等配置还可能让一个 rank 拥有额外的 ModelRunner。[1][worker] [2][model-runner]

#### 它长期持有什么？

| 成员／资源                   | 格式与实际内容                                     | 与本轮输入的关系                                     |
| ---------------------------- | -------------------------------------------------- | ---------------------------------------------------- |
| `model`                      | 模型对象，其参数中包含本 rank 使用的权重 Tensor    | 提前加载并跨 forward 使用，不随每个 batch 重新传入   |
| `model_config`、`ps`         | 配置与并行身份对象                                 | 提供结构、设备和分片等执行上下文                     |
| `req_to_token_pool`          | 请求到 token 位置的映射池引用                      | 配合 `req_pool_indices` 找到请求的缓存映射           |
| `token_to_kv_pool_allocator` | 管理 KV slot 编号的 allocator 引用                 | 与实际 KV 存储关联，但 allocator 本身不等于 K/V 数据 |
| KV 相关配置与存储组件        | 本地 KV 资源，由 pool／configurator 等组件具体组织 | attention 读取旧缓存、写入新缓存所依赖的长期存储     |
| `forward_stream`             | 设备 Stream 对象                                   | 组织异步设备工作；不是模型数据 Tensor                |
| `attn_backend`               | attention 实现对象                                 | 为具体模型执行准备、使用 attention 所需元信息与实现  |
| `eager_runner`               | `EagerRunner` 对象                                 | 普通执行路径                                         |
| `decode_cuda_graph_runner`   | Graph runner 对象，或未启用时为空                  | 管理 decode 的形状、输入与 Graph backend             |
| `prefill_cuda_graph_runner`  | prefill 执行路径相关对象，依配置存在               | 其他执行分支；本文不展开                             |
| `sampler`                    | 采样组件                                           | 由 `sample()` 等路径使用 logits 和采样信息产生 token |
| `forward_pass_id`            | 整数状态                                           | 记录 forward 轮次相关状态，不是 GPU 资源             |

这是一份关键成员索引，不是完整字段清单；资源是否分配以及具体类型取决于配置。源码中的初始化、KV 配置和执行分流分别可在 ModelRunner 与相关组件中核对。[2][model-runner] [7][memory-pool]

#### KV 的三个对象不能混成「KV_slot」

```text
请求映射池：ReqToTokenPool
    保存「某个请求的某个序列位置，对应哪个 token slot」

KV slot 分配器：TokenToKVPoolAllocator
    管理 slot 编号的分配与回收

KV 存储：KVCache / 具体 pool 实现
    实际保存各层使用的缓存数据
```

例如，`req_pool_indices=[3,7]` 指向映射池中的请求行，`out_cache_loc=[101,205]` 描述本轮要写的 slot；它们都是**索引信息**，不是两份 KV cache。实际缓存布局还依赖 attention 类型、数据类型和存储实现，不能统一画成所有模型都相同的一张 Tensor。[5][forward-batch] [7][memory-pool]

#### 它接收和返回什么？

输入主要是 `ForwardBatch`，PP 路径还可以带中间 Tensor。返回的 `ModelRunnerOutput` 主要有下面这些字段：[2][model-runner]

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

在普通 decode、最后一个 PP stage 的简化例子里，可以把 logits 理解为 `[B, V]` 的 Tensor：每行对应一个请求，每列对应一个词表 token 的分数。后续采样得到 `[B]` 的 token ID；因此 `ModelRunner.forward()` 返回 logits，与 Worker 最后交回采样结果，是两件不同的事。[1][worker] [10][logits]

#### 为什么在这里选择执行路径？

`ForwardBatch.forward_mode` 说明这一轮是什么计算，例如 decode；`ModelRunner` 再向 runner 检查本轮是否符合 Graph 执行条件。已准备哪些 bucket、当前 capture 配置是什么、对应图存不存在，这些信息在 runner／backend 附近，所以让它们提供可用性判断，可以避免 Worker 重复维护这些内部状态。[2][model-runner] [3][decode-runner]

这是一种职责安排，不是说语义上禁止在上层组织决策。并且，**本地执行判断不等于允许通信相关的 rank 任意分叉**；多 rank 的协调约束留到 Level 5。

### 2.6 谁真正持有 CUDA Graph？

先看最重要的一条成员关系：[2][model-runner] [3][decode-runner] [4][graph-backend]

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

**ModelRunner 是间接持有 CUDA Graph；直接保存图字典的是 FullCudaGraphBackend。** `shape_key` 是查找图的键，不是输入 Tensor，也不是设备指针。[4][graph-backend]

#### EagerRunner：没有 Graph，也不等于没有 buffer

`EagerRunner` 接收 `ForwardBatch`，组织普通模型调用并返回模型结果。这个固定提交里，它还持有 `_eager_registry`，默认会把本轮输入整理到已有输入缓冲区中；相关缓冲区还可能与 Graph 路径共享底层存储。因此不要把 Eager 简化成「每次现分配一切，Graph 才复用资源」。本篇只认识资源位置，不展开拷贝和同步顺序。[8][eager-runner]

#### DecodeCudaGraphRunner：决定选哪张图、怎样组织输入

这里的 **bucket** 可以先理解为提前准备的形状档位。假设普通 decode 的 `capture_bs=[1,2,4,8]`，那么最大档位是 `max_bs=8`。这组数只是示例，不是该 PR 的固定配置。[3][decode-runner]

| 成员              | 格式                      | 具体含义                                                |
| ----------------- | ------------------------- | ------------------------------------------------------- |
| `capture_bs`      | 整数列表                  | 准备捕获的 batch 档位                                   |
| `max_bs`          | 整数                      | 最大捕获 batch 档位                                     |
| `max_num_token`   | 整数                      | 最大输入 token 容量，与每请求 token 宽度有关            |
| `buffers`         | `DecodeInputBuffers` 对象 | 持有 `input_ids`、`positions`、`seq_lens` 等输入 Tensor |
| `buffer_registry` | 注册表对象                | 统一访问、填充这些输入 slot；不必另复制一套存储         |
| `backend`         | Graph backend 对象        | 保存并提交真正的图                                      |

例如，在不涉及额外并行切分的简化情形中，输入 buffer 可包含 `[max_num_token]` 的 token／position 数组，以及 `[max_bs]` 的请求索引／长度数组。**容器对象在 host 侧，数组可以在 GPU 上，部分字段还会有 CPU 镜像。** 不要把整个 `buffers` 对象理解成一个 GPU Tensor。[3][decode-runner]

#### FullCudaGraphBackend：保存图及关联输出

它的三个核心成员分别解决三个问题：[4][graph-backend]

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

下面是本篇采用的 PR 专用案例：`NcclEpGraphResources`。它不是所有模型都需要的通用模块，而是连接 full decode Graph 与 NCCL EP 低延迟通信路径的资源适配对象。[9][ep-resources] [11][pr]

这里先区分两个角色：

```text
NcclEpDispatcher
    做什么：接收 MoE 的 token／路由输入，调用 dispatch 和 combine

NcclEpGraphResources
    保存什么：Graph 路径需要长期有效的 EP group、handle 和缓冲区
```

真实的 expert compute 在 MoE 层的 dispatch 与 combine 之间执行，不由这个资源管理器计算。`ModelRunner` 也不是直接亲自调用 NCCL EP 完成每层通信；具体调用由模型中的 MoE 与 dispatcher 衔接。[12][moe-layer] [13][dispatcher]

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

资源管理对象创建时，native Group 和 Handle 不一定已创建；该实现会在 warmup 的首次准备过程中初始化它们。Backend 负责调用相应清理流程，资源管理对象负责关闭自己管理的 native 资源。[4][graph-backend] [9][ep-resources]

注册表是查找入口，不是模型数据必经的计算节点；登记的是同一个对象，也不是再复制一套 EP 资源。

#### 缓冲区里实际放什么？

对于这个实现，令 `C` 为本 rank 的最大分发 token 容量，`H` 为隐藏维度，`K` 为每个 token 选择的 expert 数，`P` 为通信 rank 数，`E_local` 为本 rank 的 expert 数。资源可具体展开为：[9][ep-resources] [13][dispatcher]

| 字段                    | 存储格式                        | 实际内容                       |
| ----------------------- | ------------------------------- | ------------------------------ |
| `state.send_tokens`     | BF16 Tensor `[C, H]`            | 待发送的 token hidden states   |
| `state.topk_ids`        | INT64 Tensor `[C, K]`           | 每个 token 的 expert 路由编号  |
| `state.topk_weights`    | FP32 Tensor `[C, K]`            | 对应的合并权重                 |
| `state.recv_tokens`     | BF16 Tensor `[E_local, P*C, H]` | 为各本地 expert 预留的接收存储 |
| `state.expert_counters` | INT32 Tensor `[E_local]`        | 各 expert 的有效接收计数       |
| `state.combined`        | BF16 Tensor `[C, H]`            | 合并阶段使用的输出 scratch     |

表里的形状是**容量布局**，不表示所有行都有效。有效数据范围需要结合路由、mask 和计数解释；本篇暂不展开通信数据流。这里的 BF16 是通信缓冲区类型，也不能据此推出真实 expert GEMM 必然使用 BF16。

另外，SGLang 的 `GroupCoordinator`、native NCCL EP `Group` 和 `Handle` 不是一个对象。这个实现使用已有 `ncclComm_t` 创建 EP 资源，所以拥有 Graph 专用 EP Group，不等于额外创建了一套模型 rank 拓扑。[9][ep-resources]

### 2.8 把对象地图收拢成四个问题

**这一轮的数据在哪里？** 在 `ScheduleBatch` 和 `ForwardBatch` 的字段中：token ID、长度、位置、KV slot 索引、采样元信息等；部分 Tensor 会被复用引用，部分执行信息会补充生成。[5][forward-batch]

**长期计算资源在哪里？** `ModelRunner` 持有模型与下级执行组件，并关联本地 KV、设备和通信资源。具体资源继续由各自组件管理，不是全都直接放在 ModelRunner 一层。[2][model-runner]

**CUDA Graph 在哪里？** 在 `DecodeCudaGraphRunner.backend` 所指向的 `FullCudaGraphBackend._graphs` 中；输入 buffer 则主要由 runner 组织。[3][decode-runner] [4][graph-backend]

**NCCL EP Graph 的专用资源在哪里？** 在 Backend 持有的 `NcclEpGraphResources` 中，dispatcher 借用；它不负责请求调度，也不是 expert 数学计算的一环。[9][ep-resources] [13][dispatcher]

本层的完成标准：看到一个字段，能够说明它是**本轮数据、配置／状态，还是资源引用**；看到一项资源，能够找到它的持有者，而不把「间接调用」误认为「直接拥有」。

## Level 3：动态数据流

TODO：沿一次 Eager forward，追踪 `ForwardBatch → model → attention / MoE → 输出` 的 Tensor 形状、元信息以及生产者和消费者。

## Level 4：执行机制

TODO：解释 KV 寻址、输入 buffer、bucket、capture／replay，以及 stream／event 如何组织设备执行。

## Level 5：分布式与生命周期正确性

TODO：解释多 rank 执行协调、Graph／eager 资源隔离，以及 recapture、cleanup、fallback 和异常处理的正确性边界。

---

## 参考源码与阅读范围

本文的职责地图沿用《从 PR #38683 重新建立 SGLang 架构：六步源码学习路径》的组织方式；字段说明补充核对固定提交源码。文中的数值示例仅用于解释格式，没有运行模型或重新复现实验。除 PR 页面外，下面的源码链接均固定到同一提交；不要用持续变化的 `main` 对照字段。

1. [tp_worker.py：Worker 的成员、batch 转换和结果包装][worker]
2. [model_runner.py：本地执行环境、返回结构和路径选择][model-runner]
3. [decode_cuda_graph_runner.py：bucket、输入缓冲区与 backend][decode-runner]
4. [full_cuda_graph_backend.py：图、输出引用与可选 EP 资源][graph-backend]
5. [forward_batch_info.py：ForwardBatch 字段和 init_new][forward-batch]
6. [schedule_batch.py：请求与调度 batch 的结构][schedule-batch]
7. [memory_pool.py：请求映射、slot 分配与实际 KV 存储的区别][memory-pool]
8. [eager_runner.py：Eager 输入 registry 和执行入口][eager-runner]
9. [nccl_ep_graph.py：Graph 专用 EP 资源与所有权][ep-resources]
10. [logits_processor.py：LogitsProcessorOutput 与 logits 处理][logits]
11. [PR #38683：NCCL EP LL CUDA Graph follow-up 的范围与验证边界][pr]
12. [ep_moe/layer.py：dispatch—expert compute—combine 的衔接][moe-layer]
13. [nccl_ep.py：dispatcher 与 EP scratch 的具体实现][dispatcher]

[worker]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/managers/tp_worker.py
[model-runner]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/model_executor/model_runner.py
[decode-runner]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
[graph-backend]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py
[forward-batch]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/model_executor/forward_batch_info.py
[schedule-batch]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/managers/schedule_batch.py
[memory-pool]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/mem_cache/memory_pool.py
[eager-runner]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/model_executor/runner/eager_runner.py
[ep-resources]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/layers/moe/token_dispatcher/nccl_ep_graph.py
[logits]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/layers/logits_processor.py
[pr]: https://github.com/sgl-project/sglang/pull/38683
[moe-layer]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/layers/moe/ep_moe/layer.py
[dispatcher]: https://github.com/Laceprndpm/sglang/blob/537b7a17477a6621efd727f5f088848d754ed3b4/python/sglang/srt/layers/moe/token_dispatcher/nccl_ep.py