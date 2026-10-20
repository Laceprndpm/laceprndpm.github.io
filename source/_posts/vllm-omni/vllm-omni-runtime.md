---
title: vllm-omni-runtime
date: 2026-10-05 20:00:00 +08:00
tags:
  - vLLM-Omni
  - Runtime
  - AI-Infra
categories:
  - 框架
mathjax: true
---

沿一次 Qwen3-Omni 文本输入、文本与语音输出请求，追踪跨 stage 的登记、提交、结果汇合与终止，理解 Runtime 的对象归属和完成条件。

<!-- more -->

# vLLM-Omni Runtime：一次请求怎样穿过多个 stage？

**面向读者：** 已经了解 SGLang／vLLM 的请求处理、调度与模型执行，希望能在 vLLM-Omni 中定位跨 stage 生命周期代码、判断修改影响并设计验证的开发者。

**核心问题与结论：** 一个用户请求会在几个 stage 中形成不同的执行状态，Runtime 必须同时维护它们的关联。`AsyncOmni` 面向调用者保存输出队列；`AsyncOmniEngine` 连接入口与后台编排；`Orchestrator` 推进逻辑请求并判断何时整体完成；`StageRuntime` 建立 stage 实例；`StagePool` 维护副本与请求绑定。stage 内的调度器仍然决定本轮哪些请求真正执行。最容易出错的地方，是把“已经提交”“输入已经就绪”“某 stage 已结束”和“整个请求已结束”合并成一个状态。

阅读顺序建议先看[架构总览](vllm-omni-arch.md)，再把本文与 [stage 内执行](vllm-omni-stage-execution.md)对照。本文只把 connector 当作生产者与消费者之间的数据通道；传输协议和资源回收细节见 [Connectors](vllm-omni-connectors.md)。Diffusion 的执行循环另见 [Diffusion Runtime](vllm-omni-diffusion-runtime.md)，不要套用到本例的 Code2Wav。

## Level 1：职责边界——三个“调度”分别解决什么问题？

**跨 stage Runtime 负责让一个逻辑请求沿模型流水线推进，并向调用者交付完整的终止语义。** 它接收 prompt、各 stage 的采样参数和目标输出模态，建立请求身份，将工作送入 stage，消费阶段输出，然后决定转交、继续等待、返回或取消。它既不直接计算 attention，也不代替 stage 内 Scheduler 决定 token budget。[4][ref-4] [5][ref-5]

如果只画 `Thinker → Talker → Code2Wav`，能看出计算顺序，却看不出一个关键事实：下游请求可以先登记，真正输入晚一点才到。图中一条逻辑依赖，不等于一次同步 Python 函数调用，更不等于整块中间 Tensor 必须经由 Orchestrator 转发。[5][ref-5] [6][ref-6]

为定位修改位置，先区分下面三项职责。

| 职责            | 回答的问题                              | 主要对象                                                         |
| ------------- | ---------------------------------- | ------------------------------------------------------------ |
| 跨 stage 编排    | 请求要跑到哪里？哪些输出可以返回？所有必要阶段是否结束？       | `AsyncOmni`、`Orchestrator`                                   |
| stage 实例管理与路由 | stage 在哪些设备上启动？有哪些可用副本？这次请求绑定哪个副本？ | `StageRuntime`、`StagePool`、`StageClient`                     |
| stage 内调度     | 输入是否就绪？本轮选择哪些请求？分配多少 token／执行工作？   | stage 的 Scheduler；本例 V1 使用 `OmniChunkTransferAdapter` 配合输入等待 |

这三项职责有协作关系，但没有共用一个全局 batch。比如 Talker 还没收到 Thinker 的下一段输入时，Orchestrator 可以已经保存了 Talker 的提交时间，StagePool 可以已经完成副本绑定，而 Talker 的 Scheduler 仍将请求停在等待输入的状态。只有最后一项决定该请求能否进入本轮执行。[6][ref-6] [7][ref-7] [8][ref-8]

这里的 **stage 是模型流水线中的逻辑计算阶段**。它不是 vLLM 的 PP rank，也不是 GPU 的同义词。本例有三个 stage，但默认部署使用两张 GPU；Talker 与 Code2Wav 在同一张 GPU 上，仍拥有各自的 stage 配置、客户端和执行状态。[1][ref-1] [2][ref-2]

因此，定位问题可以从失败条件开始：输出模态错误看入口和 final stage 选择；请求错投副本看 StagePool；输入没到却执行看 stage 内输入就绪协调；数据已经生成却一直不结束看 Orchestrator 的终态集合。无需一开始就跳进 model runner 或通信后端。

## Level 2：对象与状态——谁持有谁，谁记住哪一部分请求？

### 2.1 先画对象关系，不把每个类画成一个进程

**主例的 Orchestrator 运行在后台线程及其独立 asyncio event loop 中，stage EngineCore 则通过客户端连接。** `AsyncOmniEngine` 与 Orchestrator 之间用 Janus 队列衔接线程；它们不是两个名字不同的 GPU engine。下面是职责与引用关系图，省略 metrics、控制 RPC 和可选分布式组件。[4][ref-4] [5][ref-5] [9][ref-9]

```text
HTTP /v1/chat/completions
        │ prompt、sampling_params_list、output_modalities
        ▼
AsyncOmni
├─ request_states[r] → ClientRequestState → 每请求 asyncio.Queue
├─ final_output_task：将共享输出分发到对应请求队列
└─ engine → AsyncOmniEngine
             ├─ input_processor、stage metadata
             ├─ request_queue / output_queue / rpc_output_queue
             ├─ _runtime → StageRuntime
             │              └─ stage_pools ───────────────┐
             └─ orchestrator_thread                      │ 同一组 pool 引用
                    └─ Orchestrator                      │
                       ├─ request_states[r]              │
                       │   → OrchestratorRequestState    │
                       └─ stage_pools ◄──────────────────┘
                            ├─ StagePool[0] → Thinker client(s)
                            ├─ StagePool[1] → Talker client(s)
                            └─ StagePool[2] → Code2Wav client(s)
                                              │ 提交 / 取结果 / abort
                                              ▼
                                  各 stage 的 EngineCore 与执行资源
```

图中的箭头不都表示独占所有权。例如 engine 保存 `stage_pools`，Orchestrator 也引用这组 pool，并不意味着模型权重或客户端被复制了一份。`stage_clients` 则是 engine 暴露的逻辑 stage 视图，不能把它的长度当成整个服务的副本总数。[9][ref-9]

### 2.2 六个对象分别保存什么？

`AsyncOmni` 保存调用者视角：某次 `generate()` 是否还活着、结果应送到哪个队列、用户提供的请求 ID 是什么。它创建异步生成器，向上层逐个 yield `OmniRequestOutput`；它不亲自连接 Thinker 的输出到 Talker 的输入。[4][ref-4] [10][ref-10]

`AsyncOmniEngine` 保存整个 engine 的配置和入口处理能力。它把 stage 0 prompt 处理成可提交的请求，将携带上下文的 `StageSubmissionMessage` 放入请求队列。其基类 `OmniEngineBase` 建立后台线程、stage runtime 与控制 RPC。名字里有 Async，并不表示每一步都换线程执行：当前 `add_request_async()` 会直接调用 `add_request()`，输入处理和消息构造发生在这条调用路径上。[11][ref-11] [9][ref-9]

`Orchestrator` 保存跨 stage 进度。它要知道该请求需要哪些 final 输出、已经向哪些 stage 提交、哪些 stage 已结束，还要保留供下游构造输入使用的原始 prompt。它同时消费请求消息与 stage 输出，维护的是多个执行实例之间的关系，而不是某张卡的 KV block 表。[5][ref-5]

`StageRuntime` 负责建立可用的 stage 实例图。单机实现解析初始化计划、启动或连接副本、建立客户端和输出处理器，失败时回滚已初始化资源。它不是每次 `generate()` 都重新创建的请求对象。外部已启动引擎、多 API server 和分布式连接还有不同资源归属路径，不能仅凭持有一个 client 就推断当前 frontend 拥有对应进程。[12][ref-12]

`StagePool` 面向一个逻辑 stage，持有其 frontend／head 侧客户端、输出处理器、配置以及请求到副本的绑定。首次提交时选择副本，后续更新与取消复用绑定。单机多个可用副本采用源码中的轮转路径；分布式模式可接入 hub 与负载均衡器。本例每 stage 只有一个副本，但保留 pool 层仍有价值，因为请求状态不需要知道如何选择或寻找实例。[7][ref-7]

`StageClient` 是协议边界，不是一种装着所有权重的统一容器。基础协议约定 stage metadata、健康检查和关闭；`StagePoolLLMClient` 再提供提交请求、异步取结果等能力，diffusion 客户端有自己的接口。这使 Orchestrator 能按 stage 类型与能力工作，而实际模型执行仍在对应 engine 一侧。[13][ref-13]

### 2.3 同一个请求为什么有两份 request state？

**因为入口需要解决“结果交给谁”，编排器需要解决“整个计算进行到哪里”。** 两份状态通过请求 ID 关联，承担不同生命周期，不是相互冗余的缓存。[4][ref-4] [5][ref-5] [14][ref-14]

| 状态 | 长期保存的关键内容 | 不代表什么 |
| --- | --- | --- |
| `ClientRequestState` | 内部与外部请求 ID、结果队列、请求选定的 final stage、统计与输入 pump task | 不保存每个 stage 的 GPU KV cache |
| `OrchestratorRequestState` | 原始 prompt、各 stage 参数、final stage 集合、已提交／已结束 stage、待发布的最终输出 | 不决定每轮 GPU batch |
| StagePool 的绑定 | `request_id → replica_id`，分布式路径还有地址 affinity | 不表示请求已经执行完成 |
| stage 内请求状态 | 本 stage 的输入、进度、等待状态、缓存分配等 | 不自动知道其他 final stage 是否结束 |

入口给用户 ID 加唯一后缀，形成内部追踪 ID；向调用者 yield 时再恢复外部 ID。stage 0 的 InputProcessor 还可能产生它自己的 engine 内部 ID，因此 `external_req_id` 必须维持与 Orchestrator 查表键的对应关系。理解 abort 丢失或输出查不到请求时，至少应同时记录用户 ID、编排 ID 和 stage engine ID，而不是只在一处打印 `request_id`。[4][ref-4] [11][ref-11]

这里用 `r` 表示同一次用户请求的编排 ID。以下关键字段可以这样读：

```text
OrchestratorRequestState(r)
├─ final_stage_id = 2                  # 本请求选定的最远输出阶段 / E2E 口径
├─ final_output_stage_ids = {0, 2}     # text + audio 都要完成
├─ stage_submit_ts = {0: t0, 1: t1, 2: t2}
├─ finished_final_output_stage_ids    # 已到达终态的 final stage 集合
├─ finished_stage_ids                 # 已到达终态的已执行 stage 集合
└─ pending_final_output               # 必要时暂存整体终止消息
```

`final_stage_id` 是一个整数，`final_output_stage_ids` 是一个集合：二者不能互换。前者用于确定运行到哪里与 E2E 统计，后者用于等待请求要求的全部 final 输出。对于本例 `text + audio`，Thinker 虽然是 final text stage，也仍然是 Talker 的上游生产者。[1][ref-1] [15][ref-15]

### 2.4 哪些队列跨边界，哪些资源留在 stage？

**队列传递消息和结果，stage 保存模型执行资源。** 一个请求消息可能引用 prompt、参数、结果对象或传输元信息；这不意味着 frontend 队列保存了各 stage 的全部模型状态。[9][ref-9] [13][ref-13]

| 通道／资源 | 主要用途 | 负责其生命周期的层次 |
| --- | --- | --- |
| engine 的 `request_queue` | 提交、abort、控制等消息进入 Orchestrator | engine 与后台编排循环 |
| engine 的 `output_queue` | 阶段输出、统计和错误返回 frontend | engine；入口分发器消费 |
| engine 的 `rpc_output_queue` | 将带关联 ID 的 abort／控制结果交给等待者 | engine 的 correlated RPC 客户端 |
| `ClientRequestState.queue` | 某一个 `generate()` 消费自己的结果 | AsyncOmni 请求生命周期 |
| StagePool 输出处理器 | 维护 stage 输出聚合、请求身份转换等状态 | 对应逻辑 stage 的 pool |
| 权重、KV、执行 buffer | stage 内计算与跨轮复用 | stage EngineCore／Worker／ModelRunner |
| 跨 stage payload 与传输资源 | 生产者向下游提供条件、codec 等数据 | connector 与其 stage 侧适配层 |

一个实用检查是：如果把 frontend 中的 `request_states[r]` 删除，GPU 任务会自动停止吗？不会。删除结果路由状态与向 stage engine 发出 abort 是两件事。反过来，engine 已停止计算，也不意味着等待 `queue.get()` 的 Python 生成器已经收到可以退出的终态。Level 5 的真实修复恰好落在后一侧。[10][ref-10] [16][ref-16]

## Level 3：沿一次真实请求走读——文本进入，文本与语音返回

### 3.1 固定模型、部署和请求，先消除分支歧义

**主线使用官方示例的 text 输入分支，显式请求 `text` 与 `audio`，HTTP 响应采用非 streaming。** 示例脚本支持这组参数；它向 `/v1/chat/completions` 发送带 system／user messages 的请求，并为三个 stage 提供各自的 sampling 参数。[3][ref-3]

下面的命令坐标以该固定版本的 vLLM-Omni checkout 根目录为工作目录，假定依赖已经按官方 online recipe 指向的安装说明准备，模型权重已可访问，并有符合部署配置的 GPU 环境。服务器启动形式沿用 recipe，显式指定仓库自带 deploy 文件，便于把后文行为固定到同一配置；recipe 的默认启动也会自动解析这份配置。[28][ref-28]

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --deploy-config vllm_omni/deploy/qwen3_omni_moe.yaml \
  --port 8091
```

等服务准备完成后，在同一 checkout 的另一个终端发送请求：

```bash
bash examples/online_serving/qwen3_omni/run_curl_multimodal_generation.sh \
  text '["text", "audio"]'
```

脚本的用户文本是一个简短架构说明请求。本文不依赖模型实际生成哪一句话，只沿相同 prompt、stage 拓扑和输出模态追踪生命周期。运行命令仅用来定位官方例子，不表示本文已经启动服务；脚本默认向本机 `8091` 端口发请求。[3][ref-3]

模型拓扑与部署资源分别来自两个文件。`pipeline.py` 声明计算阶段及其语义，`vllm_omni/deploy/qwen3_omni_moe.yaml` 声明设备、容量、connector 与默认采样配置。对于本文 CUDA 配置：[1][ref-1] [2][ref-2]

| stage | 模型职责 | execution type | final 输出 | 设备 |
| --- | --- | --- | --- | --- |
| 0 Thinker | 理解输入并生成文本，同时生产 Talker 所需条件 | `LLM_AR` | text | GPU 0 |
| 1 Talker | 根据文本相关条件生成 RVQ codec codes | `LLM_AR` | 无 | GPU 1 |
| 2 Code2Wav | 将 codec codes 解码为音频波形 | `LLM_GENERATION` | audio | GPU 1 |

默认部署设置 `async_chunk: true`，两条下游边使用 `SharedMemoryConnector`；deploy 未改写的 model runner 默认是 v1。Code2Wav 是音频生成 stage，不是 diffusion；把它叫“最后一个生成阶段”并不会改变它的 execution type。HTTP 层的音频编码和 response 组装也不是第四个模型 stage。[1][ref-1] [2][ref-2] [17][ref-17]

```text
                           ┌──── final text ───────────────┐
HTTP prompt → Thinker(0) ──┤                              │
                           └─条件 chunk→ Talker(1)         │
                                            │ codec chunk │
                                            ▼             ▼
                                       Code2Wav(2) → response 汇合
                                            │             ▲
                                            └─final audio─┘
```

这里有两个不同的 streaming 维度。HTTP `stream=false` 决定响应如何交付给客户端；`async_chunk=true` 决定 stage 之间能否增量交换数据。非 streaming HTTP 请求仍然可以在内部以 chunk 推进，最终再由 serving 层收集、编码并返回一份 response。[2][ref-2] [18][ref-18]

### 3.2 HTTP 层决定请求语义，AsyncOmni 登记结果接收者

`OpenAIServingChat._create_chat_completion()` 完成模型检查、chat template／prompt 准备，验证请求模态，再构造 `sampling_params_list`。它按 HTTP `stream` 调整输出种类，随后调用 `engine_client.generate(...)`，传入 prompt、请求 ID、各 stage 参数和 `output_modalities`。在本例的非 streaming 分支，它将这个异步结果迭代器交给 `chat_completion_full_generator()`。[18][ref-18]

进入 `AsyncOmni.generate()` 后，第一项生命周期工作是建立内部身份和请求状态。它启动共享结果分发器，解析各 stage 参数，计算本请求的最远 final stage 和全部 final stage 集合，再创建 `ClientRequestState`。在调用 engine 提交之前，`request_states[r]` 与专属队列就已经存在，因此快速返回的输出有明确接收者。[4][ref-4]

本例的两个选择结果是：

```python
final_stage_id_for_e2e = 2
final_output_stage_ids = [0, 2]
```

这两行是从固定拓扑和 `output_modalities=["text", "audio"]` 推导的示意，不是运行日志。若请求只要 text，选择最远 stage 的逻辑会停在 Thinker；若请求需要 audio，就必须推进到 Code2Wav。所谓 final stage 是请求相关的边界，不是始终等于 stage 0，也不是始终等于配置数组末尾。[15][ref-15] [26][ref-26]

### 3.3 AsyncOmniEngine 处理 stage 0 输入，再交给编排线程

**这条边上传的是“已准备好提交的 stage 0 请求，加上整个流水线所需上下文”。** `_build_add_request_message()` 为 stage 0 调用 input processor，必要时补充 Omni 字段、全局请求身份和 final stage 信息，同时保留 `original_prompt`。它返回 `StageSubmissionMessage`，随后 `add_request()` 将消息放入 Janus 请求队列。[11][ref-11]

```text
StageSubmissionMessage
├─ request_id                 → 编排查表键 r
├─ prompt                     → stage 0 EngineCore 请求
├─ original_prompt            → 后续 stage 所需原始上下文
├─ sampling_params_list       → [Thinker参数, Talker参数, Code2Wav参数]
├─ final_stage_id              → 2
├─ final_output_stage_ids      → [0, 2]
└─ request_timestamp / enqueue_ts / preprocess_ms
```

为什么还要保留原始 prompt？因为 stage 0 的 tokenized request 不等于后续 stage 的完整输入来源。流水线可能需要原始多模态材料、附加信息或模型定义的转换结果。当前文本主例不需要把图片重新送下游，但 Runtime 的消息边界不能假设所有模型都只依赖一串 token ID。[11][ref-11]

这一步完成只意味着消息已提交到编排入口。`AsyncOmni.generate()` 记录提交相关时间后开始消费自己的结果队列；不能把它当成“Thinker 已经执行第一轮”。请求仍可能在编排、stage engine 或输入就绪队列中等待。[4][ref-4] [11][ref-11]

### 3.4 Orchestrator 登记跨 stage 状态，StagePool 提交 Thinker

后台请求处理循环读到 add 消息，进入 `Orchestrator._handle_add_request()`。它建立 `OrchestratorRequestState`，登记运行请求，记录 stage 0 提交时间，随后调用 `StagePool[0].submit_initial()`。[5][ref-5]

对本例的 LLM stage，`submit_initial()` 有两个不可省略的动作：先选择并保存副本绑定、向本 stage 的输出处理器登记请求，再调用对应 client 的 `add_request_async()`。输出处理器要知道请求身份，后续原始 engine output 才能转成能被 Orchestrator 正确路由的结果。若登记或提交失败，pool 会回滚相应绑定／输出处理状态，避免只留下半个登记记录。[7][ref-7]

```text
OrchestratorRequestState(r)
       │ stage 0 submit_initial
       ▼
StagePool[0]
       ├─ 选择 replica 0，保存 r → replica 0
       ├─ output_processor.add_request(...)
       └─ StagePoolLLMClient.add_request_async(request)
                    │
                    ▼
             Thinker EngineCore
```

本例单副本时选择几乎没有竞争，但记录绑定仍然必要。取消时要找到当初接收请求的 engine，输出转换也要保存身份关联。未来把 Talker 扩成多个副本，只改变 pool 的实例选择空间，不应要求入口修改请求的语义。[7][ref-7] [13][ref-13]

### 3.5 下游预提交：先创建接收者，再等真实输入

由于部署启用 `async_chunk`，stage 0 提交成功后，Orchestrator 调用 `_prewarm_async_chunk_stages()`。该函数遍历本请求最远 final stage 之前的下游，仅对**通过 connector 接收 runtime 输入**的 stage 预提交。本例的 Talker、Code2Wav 都满足这条路径。[6][ref-6]

对 LLM 类型的下游，它建立带占位 `prompt_token_ids` 的请求、填入上游 payload sender 信息，然后通过该 stage 的 pool 提交。占位长度由专用逻辑计算，模型也可声明覆盖；这是一份能登记接收和调度状态的请求骨架，不是给 Talker 或 Code2Wav 计算答案的真实条件。[6][ref-6]

“预提交”与 GPU 的 warmup／CUDA Graph capture 无关。它在这里的作用，是让请求、输出处理器和下游接收路径提前存在。真正的输入就绪门控在 stage 内：本例 V1 的 Scheduler 通过 `OmniChunkTransferAdapter` 根据 chunk 接收状态让请求等待或重新进入可执行队列。不能因为 placeholder 已经到 engine，就让模型使用一串零当成有效条件。[6][ref-6] [8][ref-8] [29][ref-29]

```text
控制消息：  submit Thinker → pre-submit Talker → pre-submit Code2Wav
                                     │                    │
                                     ▼                    ▼
                                等待首个输入          等待首个输入

数据到达：  Thinker 条件 chunk ───────► Talker
                                        │ codec chunk
                                        └────────────────► Code2Wav
```

因此，程序调用顺序和模型执行的时间线不同。三个请求可以先后登记，之后各 stage 随输入到达交错推进。它们仍受真实数据依赖约束，不是三个完全独立的并行模型调用。

### 3.6 阶段执行：Runtime 关心身份与完成，载荷内容由模型契约决定

Thinker 产生文本 token，同时为 Talker 提供模型要求的中间条件。固定源码的 `thinker2talker_async_chunk()` 处理 token、指定 hidden states／embedding 等信息；Talker 输出的是 codec codes，由 `talker2code2wav_async_chunk()` 组织给 Code2Wav 的输入。下游不是拿到 Thinker 的自然语言字符串后再走一次普通 chat。[19][ref-19]

本例默认 model runner v1，Scheduler 持有的 `OmniChunkTransferAdapter` 负责把 chunk 接收状态接入调度等待机制；`_process_pending_omni_inputs()` 调用 adapter 的 `process_pending_chunks()`，随后具体 Scheduler 选择本轮工作，Worker／ModelRunner 执行模型。原生 MRv2 的 readiness 协调是另一分支，不能接到这条默认调用链上。Runtime 只需在本层确认四件事：数据属于请求 `r`；来自正确上游；对应的下游接收者已登记；终态不会在数据还未处理时过早清除请求。[8][ref-8] [29][ref-29]

本篇不展开 Talker code predictor、Code2Wav 图捕获或共享内存序列化。那些细节改变单个 stage 的执行成本或传输生命周期，但不能用来替代这里的请求完成条件。进一步追踪一轮输入如何变成执行 Tensor，接着读 [stage 内执行](vllm-omni-stage-execution.md)。

### 3.7 结果返回：先按 stage 解释，再判断整个请求是否完成

Orchestrator 的输出循环从各 pool 取原始结果，经 `_process_llm_stage_outputs()` 与 pool 的输出处理器转换，进入 `_handle_processed_outputs()`、`_route_output()`。默认配置下的本例使用轮询路径；源码另有可切换的 event-driven 路径，但二者复用主要结果处理逻辑，不能据此推断请求终态规则不同。[20][ref-20]

Thinker 是 final text stage，因此它的输出会被包装成 `OutputMessage(stage_id=0, ...)` 返回入口。Talker 不是 final stage，通常只通过统计消息报告这层所需信息；Code2Wav 的 audio 输出再作为另一组 final 输出返回。**一个 final 输出可以结束，而逻辑请求仍然继续。**[5][ref-5]

这里必须区分两个 `finished`：

```text
OutputMessage.finished           → 编排器允许这个请求的结果循环退出吗？
OutputMessage.engine_outputs
             .finished          → 这个 stage 的输出是否已经结束？
```

例如 text 已完整生成、audio 还在生成时，可以看到内部 stage 输出 `finished=True`，外层消息 `finished=False`。入口会 yield 这份完整文本，但继续读取下一条消息。若把内层 finished 直接用作生成器退出条件，音频就会丢失。[5][ref-5] [10][ref-10]

### 3.8 两路 final 输出怎样汇合，何时清理？

`_route_output()` 将完成的 final stage 加入 `finished_final_output_stage_ids`，检查请求要求的 `{0, 2}` 是否全部满足。对于普通 async-chunk turn，还要处理另一个时序：下游最终输出到达 Orchestrator 时，上游的终态／统计可能尚未到达。此时最终消息保存在 `pending_final_output`，等所有已提交 stage 的终态被观察到后再发布。[5][ref-5]

这不意味着音频依赖了尚不存在的输入。数据通道与结果／统计通道可以有不同的到达顺序：Code2Wav 已消费最后一段 codec，Talker 的最终统计消息仍可能在另一条路径上。延后整体终止是为了不丢 usage、stop reason 和其他阶段收尾信息。[5][ref-5] [21][ref-21]

final 消息经 engine 共享输出队列进入 `AsyncOmniBase._final_output_handler()`，再按请求 ID 送到专属队列。`_process_orchestrator_results()` 包装并 yield 输出，最后根据外层 `OutputMessage.finished` 退出。`generate()` 的 `finally` 路径随后清掉入口请求状态。[10][ref-10] [4][ref-4]

HTTP 非 streaming 路径一直收集这些输出。`chat_completion_full_generator()` 按请求模态挑出已结束的 text 和 audio，构造对应 choice，再由 `_merge_audio_choices()` 按相同 output index 将音频并入文本 choice。这里的“汇合”是 response 对象组合，不是又执行一次模型 forward。[18][ref-18]

与此同时，Orchestrator 的 `_cleanup_request_ids()` 释放各 pool 的请求绑定、发起跨 stage 传输资源释放、移除逻辑请求状态并更新运行计数。正常完成清理与 frontend 的 `finally` 清理分别结束各自的状态生命周期。请求结束不会卸载整份模型权重；connector 资源释放还有异步与幂等边界，另见 [Connectors](vllm-omni-connectors.md)。[16][ref-16] [27][ref-27]

### 3.9 回看启动：请求依赖的对象什么时候建好？

**这些 stage 实例和通道在接收普通请求前建立，请求只在已有系统上登记自己的状态。** `OmniEngineBase` 解析 pipeline、deploy 与覆盖参数，创建 Janus 队列并启动 `orchestrator_thread`。后台线程建立自己的 event loop，通过 StageRuntime 初始化 stage，最后构造 Orchestrator 并进入运行循环；启动 future 用来把准备完成或初始化异常交给调用者。[9][ref-9]

```text
pipeline.py：阶段语义与拓扑
           +
deploy yaml：设备、容量、connector、默认参数
           +
显式配置覆盖
           │
           ▼
OmniEngineBase 解析最终 stage_configs
           │
           ▼
后台线程 / event loop
           │ StageRuntime.initialize()
           ├─ stage / replica 初始化计划
           ├─ 启动或连接 EngineCore，建立 client
           ├─ 建立 StagePool 与输出处理器
           └─ 将 metadata / pools 提供给 engine 和 Orchestrator
                       │
                       ▼
                 普通请求可以进入
```

StageRuntime 会按设备重叠关系安排初始化。默认 `parallel_stage_init=False` 时，共用物理设备的阶段依次初始化，避免 profiling 相互干扰；不重叠的设备组可并行。因此本例的 Talker 与 Code2Wav 共用 GPU 1，是需要协调初始化的一组，不能把 `stage_id` 不同当作资源独立的证据。[12][ref-12]

初始化失败时，StageRuntime 收集已创建的客户端进行回滚，后台线程把失败传播给启动等待者。对于由 serving parent 预启动、多 frontend 再 attach 的路径，进程资源归属另有管理对象；本文图示的核心关系仍成立，但不能将本节简图当成所有部署模式的逐行启动顺序。[12][ref-12]

## Level 4：执行机制——哪些条件允许推进，哪些条件必须继续等待？

### 4.1 输入就绪是一道独立门槛

**提交成功只证明接收者存在，不证明它可以运行。** 本例 V1 async-chunk 路径由 `OmniChunkTransferAdapter.process_pending_chunks()` 处理下游请求。adapter 检查已接收 chunk、`requests_with_ready_chunks` 和上游是否耗尽等状态；需要接收时登记 `load_async()`，让请求等待，接收完成后再恢复到适当的等待／运行队列，由 Scheduler 决定何时选中。[8][ref-8] [29][ref-29]

```text
请求已登记
    │ 无可消费 chunk
    ▼
WAITING_FOR_CHUNK
    │ adapter 后台接收完成 / 上游输入结束
    ▼
恢复可调度状态
    │ Scheduler 本轮选中
    ▼
执行并消费当前输入
```

adapter 的后台接收循环执行实际 connector I/O，调度线程消费已经准备好的接收状态、更新请求与等待队列。上图概括已经登记接收的请求；启用 adapter 的 active-stream admission 时，尚未准入的 placeholder 也会先移出可调度队列，不能将所有等待都简化为已经执行过一次 `WAITING_FOR_CHUNK` 状态切换。[29][ref-29]

源码还有两条需要单独标明的路径。原生 MRv2 async-chunk 配置使用 `OmniSchedulingCoordinator`，消费 model runner 回传的 `chunk_ready_req_ids` 等 readiness 信息，本身不执行 connector I/O；支持完整 payload 输入的下游在非 async-chunk 模式下，也可通过 coordinator 等待整份输入，状态为 `WAITING_FOR_INPUT`。它们复用了“输入先就绪、然后调度”的原则，但不能把其对象和字段套到本例默认 V1 adapter 路径。[8][ref-8] [30][ref-30]

### 4.2 跨 stage 提交有两条路线，不能把它们叠加执行

async-chunk 主线提前提交接收者，后续数据由 connector 输送；普通顺序转交路径则在上游完成后，由 `_forward_to_next_stage()` 组织下游输入并提交。`_route_output()` 明确检查目标 stage 是否接收 async chunks，以及下一 stage 是否已经提交，以避免在同一请求上再次创建下游工作。[5][ref-5] [6][ref-6]

这也是 `_prewarm_async_chunk_stages()` 不能无条件预提交所有 stage 的原因。若某个 stage 只有发送能力、实际输入必须由 Orchestrator 的 bridge 提供，先送 placeholder 可能让它在真正条件到达前执行。源码会根据接收能力跳过这类预提交，不能把 pipeline 的全局 `async_chunk` 当作每条边都采用同一种输入机制。[6][ref-6]

修改这层时，检查对象应是“一条有方向的输入边”：谁提供内容、谁登记接收、谁报告 ready、谁触发第一次执行。只增加一个全局开关而不核对边上的生产者／消费者，会使本来正确的请求骨架变成无条件输入。

### 4.3 整体完成包含两个集合条件

对于本例普通 turn，可以用两个集合表达源码中的关键条件：[5][ref-5]

$$
F_{\mathrm{required}} = \{0,2\}, \qquad
F_{\mathrm{required}} \subseteq F_{\mathrm{finished}}
$$

第一条保证请求要求的 final text 和 final audio 都已结束。对于 async-chunk 的整体终态，还需确认：

$$
S_{\mathrm{submitted}} \subseteq S_{\mathrm{finished}}
$$

第二条保证已经投入执行的阶段都已给出终态，避免下游终态先到就清理上游的统计接收者。集合名称是教学记号，分别对应 `final_output_stage_ids`、`finished_final_output_stage_ids`、`stage_submit_ts` 的键与 `finished_stage_ids`。[5][ref-5]

一个具体到达顺序能说明它们为什么不能合并：

| 到达事件 | final 完成集合 | 全部阶段完成集合 | 应有行为 |
| --- | --- | --- | --- |
| Thinker text 结束 | `{0}` | `{0}` | 可以交付完整文本，继续等音频 |
| Code2Wav audio 结束，Talker 终态未到 | `{0,2}` | `{0,2}` | 暂存整体最终消息 |
| Talker 终态与统计到达 | `{0,2}` | `{0,1,2}` | 发布整体终态，允许清理 |

表格是允许出现的教学时序，不是断言 GPU 总按这个顺序完成。测试 `test_terminal_metrics_precede_final_audio_for_either_arrival_order` 刻意覆盖上游终态先到、下游终态先到两种情况，说明该契约需要对消息到达顺序保持正确。[21][ref-21]

### 4.4 Abort 同时结束计算和等待者

正常完成靠结果推进；abort 则由控制消息驱动。`AsyncOmniBase._abort()` 调用 engine 的 `abort_async()`，后者通过带 `rpc_id` 的 `AbortRequestMessage` 等待 Orchestrator 确认。Orchestrator 将请求交给各 stage pool，pool 根据自己的绑定定位 engine，并在适用时从输出处理器收集已生成的部分结果。[10][ref-10] [16][ref-16]

```text
调用者 abort
    │
    ▼
AsyncOmniBase._abort()
    │ engine.abort_async()，等待关联确认
    ▼
Orchestrator._handle_abort()
    ├─ 各 pool 按绑定取消 engine 请求
    ├─ 收集可用 final-stage 部分输出
    ├─ 释放绑定 / 清理编排状态 / 发起传输资源释放
    └─ AbortResultMessage
            │
            ▼
入口将终态放入 ClientRequestState.queue
            │
            ▼
generate() 消费后退出，finally 清理入口状态
```

如果 engine 没有返回可用终态，入口仍需给活跃等待者合成一份 abort terminal，否则其 `queue.get()` 可能一直等待。这里不能立即删掉 `ClientRequestState`：消费者要先收到终态，再执行自身清理。控制动作成功和用户得到可观察的结束结果，是两个都必须满足的条件。[10][ref-10]

当前 Orchestrator 可能收集多个 final stage 的 abort 输出，只将该请求最后一条的**外层** `finished` 标为 true；每条阶段结果本身都可为 finished。这样可以先交付已有部分文本等结果，再退出生成器。代码中的确认表示该控制清理路径已执行，跨 stage 传输资源释放是额外调度的 best-effort 工作，不能把收到确认解释成所有物理资源已同步回收。[16][ref-16]

客户端断开还涉及 Python cancellation。`generate()` 在取消异常路径用受保护且有时间界限的 abort 清理，再重新抛出取消；最终仍由 `finally` 清掉本地状态。该路径的目标是收束请求生命周期，不是承诺立即中断已经提交到 GPU 的任意设备指令。[4][ref-4] [10][ref-10]

### 4.5 用状态边界定位问题，而不是先归因模型

如果某次请求有文本却没有音频，先回答最后一次可见事件是什么：Talker 是否已登记、是否出现输入 ready、是否生成 codec、Code2Wav 是否有输出、音频是否被 final stage／模态过滤、生成器是否提前退出。每一步对应不同拥有者，日志和测试应沿同一请求身份串起这些事件。[4][ref-4] [5][ref-5] [8][ref-8] [18][ref-18]

如果 GPU 仍在执行但客户端已拿到完整响应，检查是否把 stage finished 当成 request finished；如果所有 stage 看似结束但请求仍挂起，检查是否遗漏一条终态、错误消息或等待者通知。不要仅凭某一层队列为空就判断请求已清理，因为 frontend 队列、Orchestrator 状态与 stage 请求不具有相同生命周期。

## Level 5：真实问题分析——PR #7006 为什么必须保留 final stage 身份？

### 5.1 问题不是 abort 没执行，而是终态没有到达消费者

**PR #7006 修复了合成 abort 结果的错误路由身份。** PR 已于 2026-09-16 合并；本文核对的 head 是 `cd6960e6eaa70e620411d6d76a3f32f69fcfde3a`。最终 patch 涉及 `async_omni.py`、`async_omni_base.py`、`client_request_state.py` 与入口测试。历史描述仍可能把逻辑统称为 AsyncOmni，但最终 abort 实现在 base 文件中。[22][ref-22] [23][ref-23]

缺陷的前提是：一个 live `generate()` 正在等待结果，`abort_async()` 没有返回引擎侧终态，入口走合成结果分支；该请求的合法 final 输出并不在 stage 0。旧代码却固定构造：

```python
OutputMessage(stage_id=0, finished=True, ...)
OmniRequestOutput(stage_id=0, final_output_type="text", finished=True, ...)
```

这并非只有显示标签错误。`_process_single_result()` 会检查对应 stage 的 `final_output`；若 stage 0 不是 final，这个结果会返回 `None`。外层 `_process_orchestrator_results()` 随后仍看到 `OutputMessage.finished=True`，直接结束循环。因此调用者得到的是“生成器正常结束，但没有 yield abort terminal”。[22][ref-22] [23][ref-23]

```text
旧 synthetic abort：stage 0 / text / outer finished=True
                        │
                        ▼
入口 final-stage 检查：stage 0 不是 final → 不 yield
                        │
                        ▼
外层 finished=True → break
                        │
                        ▼
消费者收集结果：[]
```

这个机制揭示了一个接口契约：终态不仅要说明“已经结束”，还要以消费者认可的 stage 身份结束。内容为空是合法的 abort 表达方式，路由身份错误却会让结果消失。

### 5.2 为什么用两 stage 测试，而不直接搬完整 Qwen3-Omni？

PR 的回归测试采用最小两 stage 配置：stage 0 非 final，stage 1 是 final audio。它发起 `output_modalities=["audio"]` 的生成任务，等待 fake engine 记录已提交，再 abort，然后收集输出。这样能稳定触发“硬编码 stage 0 被 final 过滤器丢弃”的确切失败条件。[24][ref-24]

必须把它与本文贯穿的 Qwen3-Omni 区分开：Qwen3-Omni 的 stage 0 **本来就是 final text**，所以不能声称原始两 stage 的“非 final 过滤后 `outputs == []`”会不变地发生在同一主例上。Qwen3-Omni 若请求 audio，硬编码 text 身份仍违背请求输出语义，但具体在哪层被忽略、怎样形成 response，需要沿其多 final 配置继续判断。[1][ref-1] [18][ref-18]

这正是用小型回归测试的价值：去掉模型权重、GPU 和音频计算后，仍保留真实入口生成器、结果队列、路由过滤及清理路径。测试锁住的是 Runtime 的终态契约，不是语音质量。

### 5.3 修复把请求选择保存在它所属的生命周期中

修复首先在创建 `ClientRequestState` 时保存已经算出的 `final_stage_id_for_e2e`。它不是临时在取消时重新猜测输出阶段，而是沿用该请求提交时选定的边界。随后 `_abort()` 从这个状态取 stage ID，再从入口构造时缓存的 `_stage_meta_list` 取 `final_output_type`，传给 `_synthetic_abort_output_message()`。[23][ref-23]

```text
generate() 已算出请求的 final stage
          │ 保存
          ▼
ClientRequestState.final_stage_id
          │ engine 没有返回 abort 输出时读取
          ▼
cached stage metadata → 声明的 final_output_type
          │
          ▼
外层 OutputMessage 与内层 OmniRequestOutput 使用相同 stage_id
```

保存在入口状态里的好处是，构造 fallback 时无需依赖 engine 正处于正常服务状态，尤其适合 shutdown 附近的路径。它同时保留兼容分支：手工构造、没有 `final_stage_id` 的 request state 仍回退 stage 0；找不到声明输出类型时回退 text。这是为旧调用形状保留的 fallback，不应被当作正常多 stage 请求的选择策略。[23][ref-23]

修复没有取消 final-stage 过滤，也没有让任意 stage 输出都能作为合法终态。正确位置是终态生产者：让它满足现有消费者契约，保证外层身份、内层身份与请求选定的 final stage 一致。

### 5.4 验证应证明消费者收到什么

PR 新增的 `test_async_omni_abort_yields_terminal_from_requested_final_stage` 断言：恰好收到一份结果；stage ID 为 1；输出类型为 audio；结果 finished；完成原因为 abort；请求状态已清理。测试还用带 timeout 的等待约束生成器退出，避免只验证入队却漏掉消费者仍挂起的情况。[24][ref-24]

相比只单测 `_synthetic_abort_output_message()` 返回的字段，这个测试跨过了真正发生丢弃的过滤边界。即使工厂函数正确，如果 dispatcher 没有送入队列、生成器错误地 break、清理提前删除状态，测试仍会暴露可观察结果不符合预期。[24][ref-24]

| 要验证的契约 | 本 PR 已有证据 | 继续修改时应考虑的覆盖 |
| --- | --- | --- |
| 合成终态通过合法 final stage 返回 | 两 stage final-audio 回归测试 | text、audio 以外 final 类型 |
| 消费者确实观察到终态 | 收集 live generate 输出并断言数量 | 结果与 abort 并发到达时的行为 |
| 外层／内层身份一致 | final patch 同时设置两层 stage ID | 直接断言两层原始消息，便于缩小失败位置 |
| 不会等不到结果或遗留入口状态 | timeout 与状态清理断言 | 多 final stage、真实部分输出与 fallback 组合 |
| 不破坏已有兼容入口 | 旧状态保留 stage-0/text fallback | 没有 final-stage 字段的手工构造状态 |

右列是由接口契约推导的后续验证建议，不是声称本 PR 全部实现了这些测试。当前主线还包含多 final abort 输出仅在最后一条设置外层 finished 的测试，以及最终音频与上游统计顺序测试；它们可作为扩展修改时的相邻回归入口。[21][ref-21] [25][ref-25]

PR body 记录作者当时运行相关 CPU suites 得到 **78 passed、2 deselected**；rebase 后的 2026-09-16 评论更新为 **85 passed、2 deselected**。这是两个代码时点的作者报告，不是本文运行结果，也不能据此声称音频模型或 GPU 取消路径已经实测通过。[22][ref-22]

## 自检：能否用同一个请求回答这些问题？

读完后，应能离开图示，直接在源码中指出每个状态由谁建立、谁推进、谁销毁。下面的问题刻意围绕同一个 Qwen3-Omni 请求，而不是考查类名记忆。

| 问题 | 应能说清楚的答案 |
| --- | --- |
| 三个 stage 为什么不需要三张 GPU？ | stage 是逻辑阶段；deploy 把 Talker 与 Code2Wav 放到 GPU 1。 |
| Talker 已提交为什么仍不能执行？ | placeholder 只完成登记，Scheduler 还需等真实输入 readiness。 |
| Thinker finished 为什么请求没有结束？ | 它只完成 final text；请求还要求 final audio，且 async-chunk 收尾还需观察已提交阶段终态。 |
| `AsyncOmniEngine.add_request_async()` 返回是否代表首轮完成？ | 不是，它完成输入处理与编排消息提交，后续执行异步推进。 |
| 没有音频时，首先查哪一个对象？ | 先确定最后可见的状态边界，再分别检查 pool 绑定、输入 ready、阶段输出和 final 路由。 |
| abort 后为什么保留 frontend request state？ | 结果等待者还要消费真实或合成终态，再由 generate 的 finally 清理。 |
| stage ID 为什么属于正确性信息？ | 消费者据此判断合法 final 输出；错误身份可让终态被过滤却仍结束生成器。 |
| Code2Wav 的音频 response 编码是不是第四 stage？ | 不是；模型 stage 到 2 结束，HTTP 层再做结果编码与组合。 |

本文的阅读方法是固定一条请求、同时画控制流和数据到达，再把每个“完成”归属到具体层次。它适用于分析 Runtime 改动，但不自动证明每个模型的 payload 兼容、connector 回收或 GPU 执行正确性；这些边界需要对应的模型与传输测试。

当前没有未核实的主线函数归属；尚未覆盖的是实卡执行 trace、压力下的取消竞态与端到端性能。本文也没有把 duplex session、流式输入、PD disaggregation、CFG companion 或多机弹性副本路径推广成本例默认行为。需要进入这些分支时，应重新固定配置与请求形态，再检查同样的身份、输入就绪和终态契约。

## 参考与复现材料

编号源码链接均固定到本文主提交，PR 分析专用链接另固定到 PR head；PR 页面与评论是历史验证报告来源。文件级参考索引及整组教程的版本记录见 [references.md](references.md)。

1. [Qwen3-Omni pipeline：三个 stage、执行类型与 final 输出][ref-1]
2. [Qwen3-Omni CUDA 默认部署：设备、async chunk 与 SHM][ref-2]
3. [官方 online text／multimodal 请求脚本][ref-3]
4. [AsyncOmni.generate：登记、提交、结果消费与取消清理][ref-4]
5. [Orchestrator：请求状态、add、结果路由与整体完成][ref-5]
6. [Orchestrator._prewarm_async_chunk_stages：下游预提交][ref-6]
7. [StagePool：副本绑定、提交和输出处理器登记][ref-7]
8. [OmniSchedulerMixin：V1 adapter 与原生 MRv2 coordinator 分支选择][ref-8]
9. [OmniEngineBase：队列、后台线程与 stage 启动][ref-9]
10. [AsyncOmniBase：结果分发、生成器退出和 abort fallback][ref-10]
11. [AsyncOmniEngine：stage 0 输入处理与提交消息][ref-11]
12. [StageRuntime：实例初始化、设备协调及失败回滚][ref-12]
13. [StageClient 协议：基础生命周期及 LLM／diffusion 能力][ref-13]
14. [ClientRequestState：入口请求状态][ref-14]
15. [OmniBase：final-stage 集合、结果过滤与入口清理][ref-15]
16. [Orchestrator：abort、请求清理与传输资源释放][ref-16]
17. [stage_config.py：默认 model runner 与解析][ref-17]
18. [OpenAI serving_chat：生成器接入和 text／audio response 汇合][ref-18]
19. [Qwen3-Omni stage input processors：模型条件与 codec 转换][ref-19]
20. [Orchestrator：原始结果处理与轮询／事件驱动输出循环][ref-20]
21. [最终音频与上游终态统计的顺序测试][ref-21]
22. [PR #7006：修复说明、讨论与作者验证报告][ref-22]
23. [PR head 的 AsyncOmniBase：合成终态最终实现][ref-23]
24. [PR head 的回归测试：请求选定 final audio stage 的 abort][ref-24]
25. [当前 Orchestrator 测试：多 final-stage abort 结束标记][ref-25]
26. [final stage 选择辅助函数][ref-26]
27. [Orchestrator._cleanup_request_ids：统一请求清理][ref-27]
28. [Qwen3-Omni online recipe：安装入口与服务器启动][ref-28]
29. [OmniChunkTransferAdapter：V1 chunk 接收状态与等待队列][ref-29]
30. [OmniSchedulingCoordinator：原生 MRv2／full-payload 输入就绪分支][ref-30]

[ref-1]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/model_executor/models/qwen3_omni/pipeline.py#L22-L89
[ref-2]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/deploy/qwen3_omni_moe.yaml#L1-L85
[ref-3]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/examples/online_serving/qwen3_omni/run_curl_multimodal_generation.sh
[ref-4]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/entrypoints/async_omni.py#L114-L347
[ref-5]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/engine/orchestrator.py
[ref-6]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/engine/orchestrator.py#L2639-L2801
[ref-7]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/engine/stage_pool.py
[ref-8]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/core/sched/omni_scheduler_mixin.py#L135-L413
[ref-9]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/engine/omni_engine_base.py#L130-L503
[ref-10]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/entrypoints/async_omni_base.py#L222-L545
[ref-11]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/engine/async_omni_engine.py#L246-L620
[ref-12]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/engine/stage_runtime.py
[ref-13]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/engine/stage_client.py#L26-L120
[ref-14]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/entrypoints/client_request_state.py
[ref-15]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/entrypoints/omni_base.py#L434-L768
[ref-16]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/engine/orchestrator.py#L608-L729
[ref-17]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/config/stage_config.py#L564-L609
[ref-18]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/entrypoints/openai/serving_chat.py
[ref-19]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/model_executor/stage_input_processors/qwen3_omni.py#L409-L820
[ref-20]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/engine/orchestrator.py#L807-L1100
[ref-21]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/tests/engine/test_terminal_stage_metrics_order.py
[ref-22]: https://github.com/vllm-project/vllm-omni/pull/7006
[ref-23]: https://github.com/vllm-project/vllm-omni/blob/cd6960e6eaa70e620411d6d76a3f32f69fcfde3a/vllm_omni/entrypoints/async_omni_base.py#L446-L541
[ref-24]: https://github.com/vllm-project/vllm-omni/blob/cd6960e6eaa70e620411d6d76a3f32f69fcfde3a/tests/entrypoints/test_omni_entrypoints.py#L638-L672
[ref-25]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/tests/engine/test_orchestrator.py#L1057-L1088
[ref-26]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/entrypoints/utils.py#L149-L191
[ref-27]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/engine/orchestrator.py#L1627-L1690
[ref-28]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/examples/online_serving/qwen3_omni/README.md#L5-L39
[ref-29]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py#L1113-L1619
[ref-30]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/core/sched/omni_scheduling_coordinator.py
