---
title: vLLM-Omni Diffusion Runtime：从完整请求到逐步去噪
date: 2026-10-05
tags:
  - vLLM-Omni
  - AI-Infra
categories:
  - 框架
mathjax: true
---

沿一次 Qwen-Image 文生图请求追踪状态、批次与去噪循环，再用 MiniMax-H3 continuous batching 的实际 PR 检查隔离、失败同步和性能边界。

<!-- more -->

# vLLM-Omni Diffusion Runtime：从完整请求到逐步去噪

**面向读者：** 已了解 SGLang／vLLM 的调度与模型执行，具备 CUDA 和分布式推理基础，希望进入 vLLM-Omni 定位 diffusion 代码、分析修改影响并设计验证的开发者。

**核心问题与结论：** 将完整去噪循环交给 pipeline，与每次只推进一个去噪步，改变的是 scheduler 重新取得控制权的位置。要让后者安全地合并不同请求，必须把 latent、进度和模型私有状态绑定到请求，并让每个执行 backend 遵守相同的隔离边界；支持 continuous batching 并不自动意味着吞吐更高。

本文接续[架构](vllm-omni-arch.md)、[运行时](vllm-omni-runtime.md)和[阶段执行](vllm-omni-stage-execution.md)。全书的 Qwen3-Omni Thinker／Talker／Code2Wav 主线不等于 diffusion pipeline，因此这里另用一条合法的文生图请求，而不把语音生成流程强行解释为 AR + DiT。跨执行阶段的数据交接见 [Connectors](vllm-omni-connectors.md)。

## Level 1：职责边界——去噪循环何时把控制权交回来？

### 1.1 两种执行模式切开的是同一个请求的生命周期

一次 diffusion 请求通常先取得条件表示，再准备 latent，反复调用去噪模型更新 latent，最后将其解码成图像、视频或音频。
对服务系统而言，重要的区别不是「有没有循环」，而是循环由谁控制、循环期间是否能重新选择请求。

**Request execution 把完整生成过程留在一次 pipeline `forward()` 中；step execution 则把状态保留下来，让一次执行波次只推进一个去噪步。**
前者完成当前波次后才返回服务 scheduler，后者在相邻去噪步之间返回，使新请求有机会入场、完成请求有机会离场，取消也有明确的步边界。[1][ref-1] [2][ref-2]

```text
request execution
调度一次 → 准备／编码 → 去噪 0 → 去噪 1 → … → 去噪 K-1 → decode → 返回

step execution
调度一次 → 新请求准备／编码 → 去噪 0 → 返回
调度一次 →                  去噪 1 → 返回
    …
调度一次 →                  去噪 K-1 → decode → 返回最终结果
```

上图按本文 Qwen-Image 的最终输出路径展开。
第一波次可以同时完成准备和第一步去噪，最后一波次可以同时完成最后一步与 decode

完整请求模式也可以做 batch。
当前 Qwen-Image 声明了 request-batch 能力，多个兼容请求能够一起进入完整 `forward()`；它们只是在这次执行期间不能像 step 模式那样于每一步重新组合。
因此，比较模式时应分别记录**调度粒度**和**本波次请求数**，不能把 request mode 简化为「永远一个请求」，也不能把 step mode 简化为「一定多个请求」。[1][ref-1] [9][ref-9]

### 1.2 Denoise step 不是 Omni 的 pipeline stage

Omni 的执行阶段描述服务拓扑中的执行单元及其上下游关系；denoise step 描述某个 diffusion 请求在模型内部迭代了多少次。
同一 diffusion 执行阶段可以在同一组设备上反复运行许多步，每次都复用已加载的 DiT 权重，而不是每一步向下一个服务阶段发送一次新的模型输入。

此外，`prepare_encode → denoise_step → step_scheduler → post_decode` 是 pipeline 的执行协议。

### 1.3 两类 scheduler 决定不同的事情

服务侧 `StepScheduler` 决定本轮让哪些请求进入执行、记录它们的步数和结束状态。
pipeline 里的 noise scheduler 决定给定当前 latent、模型预测和 timestep 后，下一份 latent 怎么算。
前者面对请求队列，后者面对数值积分状态；修改其中一个，未必需要修改另一个。[4][ref-4] [10][ref-10]

| 本文称呼 | 源码中的例子 | 它回答的问题 |
| --- | --- | --- |
| 服务 scheduler | `StepScheduler`、`RequestScheduler` | 谁参与这一波执行，谁继续等待，谁已结束？ |
| noise scheduler | Qwen-Image 使用的 `FlowMatchEulerDiscreteScheduler` | 这一预测在当前时间位置应如何更新 latent？ |
| pipeline 的更新方法 | `step_scheduler(state, noise_pred)` | 如何用本请求的数值规则推进并保存状态？ |

步边界是一种控制机会，不是 GPU kernel 内的抢占点。
若当前一次 DiT forward 已经很长，或者新请求的编码需要较长时间，开启 step execution 仍不能保证很低的入场延迟。
这个区别会在 Level 5 的负向性能结果中变得具体。

## Level 2：对象拓扑——谁持有状态，谁拼出本轮输入？

### 2.1 从 engine 到 pipeline，分开看控制与模型语义

`DiffusionEngine` 根据配置选择执行模式，初始化服务 scheduler 与 executor，并把执行函数接到 `execute_step` 或 request-batch 路径。
executor 负责把工作送到实际 worker；worker 的 `DiffusionModelRunner` 负责本地模型及执行状态；具体 pipeline 才知道条件编码、latent 布局、去噪模型和解码规则。[2][ref-2] [5][ref-5] [6][ref-6]

```text
DiffusionEngine
├─ 服务 Scheduler
│    waiting / running / finished，选择兼容请求
│                  │ DiffusionSchedulerOutput
├─ DiffusionExecutor
│                  │ 分派到 worker；具体方式依 uni / mp / ray 配置
│                  ▼
│   DiffusionWorker → DiffusionModelRunner
│                      ├─ pipeline：权重与模型方法
│                      ├─ state_cache[request_id]：长期请求状态
│                      └─ input_batch：当前波次的连续输入表示
│                                   │
│                                   ▼
│                      pipeline.denoise_step(...)
│                                   │ 本轮预测
│                      各请求 step_scheduler(...)
│                      完成请求 post_decode(...)
│                                   │
└────────────── 返回各请求的进度／结果，更新服务调度状态
```

图中各框是职责边界，不保证每个框都有独立进程。
例如该版本单 GPU 默认可以选择 uni executor，多 GPU 默认可以选择 mp；调度输出的语义不因二者不同而变成另一种 latent 更新规则。[5][ref-5]

服务 scheduler 先保留正在运行的请求，再在容量允许时接纳等待请求。
接纳还受兼容键约束：分辨率、帧数、guidance、输出数量、LoRA 等会影响能否合批。
当前 step 兼容键没有把 `num_inference_steps` 作为所有请求必须相同的字段，因此协议有容纳不同进度、不同总步数的空间；这不意味着任意尺寸、任意模型状态都能直接混在一起。[15][ref-15] [26][ref-26]

### 2.2 StepRequestState 是跨步的事实来源

假设请求 A 在第 7 步结束后暂时回到 engine，下一轮又与请求 B 一起执行。
不能依赖上次 Python 函数的局部变量恢复 A：局部调用已经返回，下一次 batch 的成员和位置也可能改变。
runner 因此用 `state_cache[request_id]` 保存 `StepRequestState`，pipeline 在这个请求对象上初始化并更新所需数据。[6][ref-6] [7][ref-7]

| 数据                     | 放在哪里                           | 生命周期与含义                                         |
| ---------------------- | ------------------------------ | ----------------------------------------------- |
| 请求身份、prompt、sampling   | `StepRequestState`             | 跨步存在，指明是谁、要生成什么                                 |
| 正／负条件 embedding 与 mask | state 的对应字段                    | 编码后跨步保留，提供条件信息                                  |
| 当前 latent              | `state.latents`                | 每一步更新；始终属于该请求                                   |
| 时间序列与当前位置              | `state.timesteps`、`step_index` | 前者保存 timestep schedule，后者指出下次使用哪一项              |
| noise scheduler        | `state.scheduler`              | Qwen-Image 的请求私有 noise scheduler；不是服务 scheduler |
| 模型私有状态                 | `state.extra`                  | 例如 H3 的 audio latent、sigma schedule 和布局         |
| 本轮模型预测                 | `denoise_step()` 的返回值          | 通常只在本轮消费，再按请求分回更新方法                             |

这里尤其要区分三种容易都被称为 noise 的东西。
初始随机 noise 用来构造起点 latent；latent 是当前迭代状态；`noise_pred` 则是本轮模型输出的通用变量名。
该名字不能保证模型实际采用 epsilon 预测参数化，H3 的相应返回值就是 video／audio velocity。
分析数值错误时，应回到 pipeline 的更新方法看它如何解释预测，不能直接套用另一类 diffusion 模型的公式。[9][ref-9] [18][ref-18]

同理，`step_index=7` 是数组位置，不代表 `timestep=7`。
`current_timestep` 从本请求的时间序列取当前项；不同请求即使步索引相同，也可能拥有不同的时间值。
这也是合批时不能只向整个模型传一个未经检查的全局 timestep 的原因。[7][ref-7]

### 2.3 InputBatch 是本轮视图，不接管请求身份

`InputBatch.make_batch()` 将选中的 state 组合成当前步的连续输入，包括 latents、timesteps、条件 embedding、mask 和布局信息。
其职责是满足本次模型调用，不是保存请求从入场到结束的唯一状态。
当 batch 成员保持一致时，它可以复用输入存储与静态条件，仅刷新动态 latent 和 timestep；成员变化时则重建相应组合。[8][ref-8]

把两者理解为「持久状态」与「本轮执行表示」，就能解释为什么既有 `state.latents` 又有 `input_batch.latents`。
前者回答 A 当前处于什么状态，后者回答这一次 forward 的连续张量是什么。
这里的组装可能涉及复制、拼接或 padding，不能把“view”这个语义称呼理解为所有字段都必定是零拷贝 Tensor view。

另一个边界是维度语义。
Qwen-Image 的 latent 可以按 `[图像数, 图像 patch 数, 通道]` 阅读；H3 则把 runner 可见的 video latent 表示为 `[video rows, 96]`。
通用 runner 按各 state 的 `latents.shape[0]` 切分本轮返回值，所以不能给这个维度统一起名为“请求数”。
pipeline 必须保证返回的第一维排列与 runner 的切分约定一致。[6][ref-6] [18][ref-18]

### 2.4 四方法协议让职责可以单独验证

`SupportsStepExecution` 要求 pipeline 提供四个方法，并声明支持 step execution。
协议不是让通用 runner 接管模型数学，而是把生命周期中可以稳定协作的边界暴露出来。[3][ref-3]

| 方法 | 读取什么 | 留下什么 |
| --- | --- | --- |
| `prepare_encode(state)` | prompt、sampling、模型配置 | 条件、初始 latent、时间序列和私有状态 |
| `denoise_step(input_batch, ...)` | 当前 batch 的 latent、条件、时间 | 当前步预测；必要时暂存模型私有分支结果 |
| `step_scheduler(state, prediction)` | 本请求的预测与 timestep / sigma schedule | 更新后的 latent、递增的步索引 |
| `post_decode(state)` | 已完成去噪的 latent 与布局 | 解码后的 `DiffusionOutput` |

只有第一项通常对一个请求执行一次；第二、三项按步重复；最后一项在本文的最终输出路径执行一次。
这种拆分使“输入准备错了”“预测合批错了”“积分状态串了”和“decode 解释错了布局”可以分别定位，而不是只看到最终图片不对。

## Level 3：一次 Qwen-Image 请求如何走完？

### 3.1 先固定外部请求，避免从参数名猜 API

以下是本文源码走读采用的启动方式和请求，取自当前执行模式文档所支持的配置。
`max-num-seqs=1` 用于先理解一条请求的状态转换；后面再讨论提高容量后的多请求问题。
这些命令是可复现入口说明，本文没有实际执行模型。[1][ref-1]

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --step-execution \
  --max-num-seqs 1
```

```bash
curl -X POST http://localhost:8091/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen-Image",
    "prompt": "a cat sitting on a windowsill",
    "size": "1024x1024",
    "n": 1,
    "num_inference_steps": 50,
    "seed": 42,
    "response_format": "b64_json"
  }'
```

这里 HTTP 使用 `size` 和 `n`；内部 sampling 使用 `height／width` 和 `num_outputs_per_prompt`。
`generate_images()` 将 `size` 按 `WIDTHxHEIGHT` 解析，并显式将 `request.n` 写入 `OmniDiffusionSamplingParams`。
当前名为 `openai_chat_client.py` 的示例也 POST 到 `/v1/images/generations`，因此不能根据文件名把这条请求改写成 Chat Completions，也不能把示例 CLI 的 `--height` 直接当作 Images API JSON 字段。[11][ref-11] [12][ref-12] [13][ref-13]

本文没有设置 negative prompt。
在该 pipeline 的判断中，没有负条件时 `do_true_cfg` 为 false；即使默认 `true_cfg_scale` 大于 1，也不能据此认定每步必定做正负两次计算。
这一选择让主线保持为单条件文生图，并不构成生成质量的推荐。[9][ref-9]

### 3.2 请求先变成内部参数，再进入 diffusion 调度

API handler 构造带有 `modalities=["image"]` 的 `OmniTextPrompt`，复制尺寸、步数和 seed，生成 `img_gen-...` 请求 ID，随后通过 `_generate_with_async_omni()` 提交。
这里先完成的是服务表示的转换；尚未生成图像 latent，也没有让 API handler 自己跑 50 次 DiT。[12][ref-12]

当请求到达 diffusion engine 时，服务 scheduler 接纳它并生成 `DiffusionSchedulerOutput`。
step 执行函数经 executor、worker 进入 runner；runner 为新请求建立 state，复制 sampling 配置，再执行准备。
若 `generator` 尚未存在而 seed 为 42，`_initialize_generator()` 创建适用设备上的 `torch.Generator` 并调用 `manual_seed(42)`。[2][ref-2] [6][ref-6]

固定 seed 将随机起点纳入验证条件，却不能单独证明不同 backend、dtype、并行配置和执行次序下所有结果都逐位相同。
本文只据源码说明 generator 如何产生，不虚构本机生成图片的具体像素或随机 Tensor 数值。

### 3.3 prepare_encode：条件只编码一次，latent 和 timestep schedule 属于请求

`QwenImagePipeline.prepare_encode()` 调用共享的 `_prepare_generation_context()`。
这一准备逻辑也被 request 模式使用，依次检查输入、取得条件 embedding、准备 latent，并创建 timestep schedule。
读到这里应把“共享准备函数”与“共享可变请求状态”分开：前者减少两种模式的语义漂移，后者如果处理不当会造成串请求。[9][ref-9]

文本首先套入 Qwen-Image 自己的编码模板，经过 tokenizer，再由 `Qwen2_5_VLForConditionalGeneration` 的 text encoder 取得最后一层 hidden states。
实现按照 attention mask 提取有效部分、移除模板的固定前缀，并生成 embedding 与对应 mask。
这是 diffusion 的条件编码路径，不能把此 text encoder 当作全书 Qwen3-Omni 请求里的 Thinker 阶段。[9][ref-9]

用符号表示，本请求的条件是 `[1, L, D]` 的 `prompt_embeds` 和 `[1, L]` 的 mask；`L` 由实际 tokenizer 与模板处理结果决定，`D` 来自模型配置。

接下来 `prepare_latents()` 用 generator 产生随机 Tensor，再把空间上的 `2×2` latent patch 合并。
设 VAE 空间压缩因子为 `s`、latent 通道为 `C=transformer.in_channels/4`，则输入 1024×1024 对应的空间尺寸由源码计算为：

$$
h_\ell=w_\ell=2\left\lfloor\frac{1024}{2s}\right\rfloor,
\qquad
P=\frac{h_\ell w_\ell}{4}.
$$

原始随机布局为 `[1, 1, C, h_l, w_l]`，packing 后为 `[1, P, 4C]`。
**仅作形状教学设定**，若取 `s=8、C=16`，便得到 `[1,1,16,128,128] → [1,4096,64]`；这不是本文从实卡打印出的结果。
这里的 4096 是同一张图的 latent patch 数，不是 4096 个独立服务请求。[9][ref-9]

`prepare_timesteps()` 在未给自定义 sigmas 时先构造 50 个 sigma 采样点，再依据图像序列长度计算 shift 参数，并让 noise scheduler materialize 实际时间序列。
因此不能把推理简化为“时间从整数 50 递减到 1”，也不应把初始 sigma 列表未经变换就标成真实模型 timestep。

准备完成后，pipeline 把条件、latent、timesteps、图像布局写入 state，并把 `step_index` 设为 0。
特别值得注意的是，它在 timestep schedule 准备之后 `deepcopy(self.scheduler)`，再将副本写入 `state.scheduler`。
这样复制的不仅是静态配置，也包括刚刚准备的动态 shifting 状态；后续 A 的数值更新不会推进 B 的 noise scheduler 游标。[9][ref-9] [10][ref-10]

### 3.4 第一步：InputBatch 提供输入，DiT 返回本轮预测

runner 通过 `InputBatch.make_batch()` 组装本轮输入。
本例只有一个请求、一张图片，故 latent 第一维为 1；时间输入来自 `state.timesteps[0]`，文本条件复用准备阶段的结果。
`denoise_step()` 随后构造 transformer 的参数：latent 是 `hidden_states`，条件 embedding 是 `encoder_hidden_states`，mask 和图像形状共同描述本轮输入。[8][ref-8] [9][ref-9]

Qwen-Image 在 `_build_denoise_kwargs()` 中将 timestep 扩展到 latent 的 batch 维，并以 `t/1000` 传给 transformer。
随后 `predict_noise_maybe_with_cfg()` 根据是否启用 true CFG 选择相应预测路径；本请求没有负条件，沿无 true CFG 的路径执行。
若修改 guidance 或添加 negative prompt，这个分支和条件缓存都需要重新纳入验证。[9][ref-9] [10][ref-10]

返回值仍处在 latent 表示空间，尚不是 PNG，也不是已完成的一张图片。
runner 按各请求的 latent 第一维切出对应预测，调用 `pipeline.step_scheduler(state, prediction)`。
该方法传入 `state.scheduler` 完成本次数值更新，替换 `state.latents`，最后将 `step_index` 加一。[6][ref-6] [9][ref-9]

```text
第一轮开始：state.step_index = 0，state.latents = x_0
      │ t_0 = state.timesteps[0]
      ▼
模型预测：prediction_0 = DiT(x_0, t_0, conditioning)
      ▼
数值更新：x_1 = 本请求 noise scheduler 的更新结果
      ▼
第一轮结束：state.step_index = 1，state.latents = x_1
```

上图的 `x_0／x_1` 按程序执行顺序命名，而非特指某篇 diffusion 论文中的时间方向。
第一轮尚未完成去噪，runner 返回带新步索引的 `RunnerOutput`，最终媒体结果仍为空；服务 `StepScheduler.update_from_output()` 将此进度记回请求。[4][ref-4]

### 3.5 中间步：保留请求状态，允许下一轮重新组成 batch

第二次调度不再把这个请求当作新的 prompt 重编码。
runner 从 `state_cache` 找回状态，InputBatch 刷新 latent 与 timestep，再执行下一步。
若调度容量大于 1，下一轮可以同时出现刚入场的 B：A 用自己的第 `k` 个 timestep，B 用自己的第 0 个 timestep，静态条件仍分别属于 A、B。[6][ref-6] [8][ref-8]

因此，“共享一次模型调用”与“共享一个 noise scheduler 游标”是两回事。
step batching 要共享合适的计算资源，同时保留每个请求独立的演化状态。
检查此类改动时，单请求图片看起来正常只能验证第一部分路径；不能替代跨请求进度隔离验证。

### 3.6 最后一步：decode 与响应编码处于不同层

当状态的步索引达到本请求 `total_steps`，runner 在这轮调用 `post_decode()`。
Qwen-Image 的 `_decode_latents()` 先还原 packed latent，按 VAE 配置的 mean／std 恢复其表示，再调用 VAE decode，并取出图像的时间维切片。
此处产生的是 `DiffusionOutput` 中的图像 Tensor；函数参数名中的默认 `"pil"` 不意味着 VAE 本身直接返回了 HTTP 可发送的 PIL 对象。[6][ref-6] [9][ref-9]

Qwen-Image 注册的后处理使用 `VaeImageProcessor.postprocess()` 把图像 Tensor 转为后续服务层可以处理的图像表示。
API 最后从结果抽取图像，`_build_image_generation_response()` 按输出格式编码，放进 `data[0].b64_json`，再返回 JSON。
于是一个完整闭环是：**内部条件与 latent → 解码图像 → 图像后处理 → PNG／base64 响应**，不能把其中某个 Tensor 提前当成完成响应。[9][ref-9] [12][ref-12] [14][ref-14]

本例的 step execution 也不意味着客户端收到 50 张中间图片。
调度粒度与输出流粒度由不同能力决定；这条非流式 Images API 请求等待最终输出后才构造响应。[1][ref-1]

## Level 4：真实改动——PR #5810 怎样把 MiniMax-H3 接入 step execution？

### 4.1 问题是循环封装边界，而不是少了一个 batch 参数

PR #5810 的历史起点是：MiniMax-H3 把整个 denoise loop 留在一个 `forward()` 中，无法参与逐步调度。
最终改动实现四方法协议，并将 request／step 两条路径的准备、去噪输入和结果 unpack 逻辑抽成共享函数。
这样，完整请求执行与逐步执行是在改变控制权交回的位置，而不是分别维护两套音视频生成算法。[16][ref-16] [18][ref-18]

以下机制分析固定在该 PR 最终 head，不能直接套用当前 MiniMax-H3 的所有默认配置。
特别是该模型后续已有部署拆分等变化；读者若在当前提交上找代码，应按同名方法确认行为，不能从这段历史反推出今天只有一个服务阶段。

H3 比 Qwen-Image 多出一个容易暴露抽象边界的条件：video 与 audio rows 宽度不同，分别为 96 和 32。
通用协议每步返回一个供 runner 切分的预测 Tensor，因此最终实现把 video rows 放进 `state.latents`，audio rows、各自 sigma schedule 和分支布局放进 `state.extra`。
这保留了通用 runner 的切分约定，也把模型特有的耦合留在 pipeline 内。[18][ref-18] [19][ref-19]

### 4.2 Packing 不只是把若干 Tensor 接起来

H3 的 DiT 本来就消费 packed sequence。
PR 的 `batched_packing.py` 将若干请求的布局合为一次 forward，但必须同时重建位置索引、document 边界、timestep 映射及文本 refiner 的边界。
否则即使最终 Tensor 的形状正确，模型仍可能把 A 的位置、时间或条件用于 B。[17][ref-17]

考虑以下**教学布局**，数值用于展示边界，不代表某个真实视频：A 有 60 行有效数据，补齐到 64 行；B 有 64 行有效数据，恰好不需要 padding。
实现保留每个请求的 real document 和 alignment tail，所以本轮累计边界为：

```text
packed rows: [ A 的有效 60 行 | A 的 pad 4 行 | B 的有效 64 行 | B 的 pad 0 行 ]
边界位置:    0              60             64              128           128
cu_seqlens = [0, 60, 64, 128, 128]
```

这里重复的 `128` 表示空 padding tail；恰好 64 行时仍需保持约定一致，不能因为“没有补齐”就漏掉布局分支。
`max_seqlen` 对应各 document 的最大长度，而不是整个拼接序列的总长。
实际 attention 必须把这些分段视为独立 document，才允许 A 与 B 共享一次 forward。[17][ref-17]

位置也有两个概念。
用于把 rows 写入合并 Tensor 的索引需要加本请求的 `seq_offset`；每个请求自己用于 RoPE 的图像坐标则保留原有语义。
将二者都当作“位置”并机械加偏移，会让物理排布正确而模型的位置解释错误。
文本 token refiner 另外重建 text-only 的累计边界，不能借用包含音视频 rows 的那一组界限。[17][ref-17]

### 4.3 不同 timestep 必须下沉到对应 rows

连续入场后，A 可能已推进多步，B 才开始。
H3 还为 video 和 audio 保留各自的 sigma schedule，条件 rows 也可能使用固定条件时间。
因此该 helper 先按请求和 row 类型赋值 timestep，再构造 `unique_timesteps` 与 `inverse_indices`：前者允许复用相同时间的 embedding，后者把每行映回自己应使用的时间。[17][ref-17] [18][ref-18]

这解释了为什么 `step_index` 不能直接当全 batch 的进度。
对于依赖全局 denoise 进度开关的 attention 优化，最终实现只在所有请求共享同一进度点时发布该值；若混合步索引或 sigma，则发布空进度，让相关门控保守地保持 dense 路径。
这是一项显式的兼容取舍，不是遗漏进度后可以随意补上 `states[0].step_index`。[18][ref-18] [19][ref-19]

一次 DiT forward 返回 video 与 audio velocity 后，video 直接交给通用 runner，按各请求 video row 数切分。
audio 则由 pipeline 按 audio row 数拆分，暂存到对应 state；`step_scheduler()` 用 `pop()` 取走本步 audio 预测，防止第二次更新误用旧结果。
两个分支各按本请求 sigma 更新；固定的条件 rows 按 anchor 恢复，最后推进同一个请求的步索引。[18][ref-18]

H3 的数值循环消费相邻 sigma 对，所以 sigma schedule 的点数比实际更新次数多一个。
step 状态中的 `timesteps` 使用 video sigma schedule 去掉最后一个边界后的值，并按 `1-sigma` 转换；audio sigma schedule 留在私有状态中。
这说明服务参数、时间边界数量和 DiT 更新次数需要分别核对，不能看到 `num_inference_steps` 就直接把它代入所有 pipeline 的循环次数。[18][ref-18] [19][ref-19]

### 4.4 提前离场改变下一轮布局，不改变其他请求的数学结果

假设 A 需要更多步，B 更早完成。
B 在最后一次更新后解码并退出，下一轮就只剩 A；packing 应当能够从多请求布局退回单请求布局。
该 PR 的 batch-of-one 路径直接复用原来的单请求 `forward_kwargs()`，既减少两套逻辑的差异，也保留单请求布局附带的优化提示。[17][ref-17]

这里的正确性目标不是“最终 shape 都对”，而是：在相同初始条件下，A 与 B 合批、分开跑、其中一个提前离开，仍各自得到应有的 latent 演化。
如果请求退出会改变另一个请求的时间、条件或 attention 可见范围，continuous batching 的隔离就没有成立。

### 4.5 测试为什么要让输出依赖 document 和 timestep？

最终测试采用 `_SegmentMeanModel` 作为无需权重的替身 DiT。
每一行的输出同时依赖本行输入、自己 document 的均值和本行 timestep；因此错误的 `cu_seqlens`、偏移或时间映射会产生数值差异。
若替身只返回输入或全零，即使把所有请求放在同一 document，测试也可能照样通过。[19][ref-19]

测试先比较完整 denoise loop 与逐步执行，再比较两个独立请求和它们合批执行的最终 video／audio latent。
合批测试特意使用不同步数，让其中一个先退出，并设置一个恰好 64 packed rows、没有 padding tail 的请求。
这两项边界同时覆盖了成员变化和空 tail，而不是只检验一次静态拼接。

还要区分 pipeline 能力测试和服务 admission 策略。
该测试直接构造不同布局的 state 来验证 packing 数学，不等于线上 scheduler 已承诺把所有异形请求放进同一 batch；Level 2 的兼容键仍是服务侧边界。
CPU 替身也没有验证真实 FlashAttention kernel、设备 collective 或模型视觉质量，相关结论需要下一层证据。

## Level 5：如何判断改动正确、兼容，而且值得开启？

### 5.1 Attention 隔离是 backend 的能力，不是名字

review 首先指出，只检查 backend 名称为 `FLASH_ATTN` 不足以证明请求隔离。
启用 Ring sequence parallelism 时，实际调用可经过不消费 packed `cu_seqlens` 的 Ring attention；同一个 backend 名字在不同设备平台上，也可能落到只支持单个 real document 加 padding 的实现。
这类错误危险在于 Tensor 尺寸仍可能完全合法，只是 A 已能读取 B 的内容。[20][ref-20]

最终改动新增 `supports_multi_doc_packed_varlen()` 能力探测，并检查 transformer 中各 attention，包括 text refiner。
当实际 attention 使用 Ring、backend 不支持多 document 隔离，或同批请求对应不同 task-specific transformer 时，H3 step 路径退回逐请求 forward。
服务层仍可以逐步调度这些请求，只是该波次不再实现一次 DiT forward 的计算合并。[18][ref-18] [21][ref-21]

还必须检查 attention 的其他捷径是否覆盖了新语义。
单请求的“有效 KV 是一个前缀”和一维 padding mask，都无法表示多个相互隔离的 real documents。
最终 transformer 改动在多请求布局下取消这些旧假设，保留真正的分段边界；否则传入了 `cu_seqlens`，也可能在后续层被更简单的 mask 路径误解释。[22][ref-22]

审查类似优化时，可以先问一个可执行的问题：**改变 B 的输入，保持 A 不变，A 的输出是否发生超出约定数值误差的变化？**
这个测试比检查 backend 字符串更接近需要保护的语义，还应覆盖 batch 重排、B 退出及不同 attention role 的 backend 配置。

### 5.2 分布式失败必须发生在一致的控制流位置

PR 的另一个问题发生在正常算子之前：H3 有 rank 0 独占的参考媒体准备。
若 rank 0 读取无效输入时抛错，其他 rank 却继续等待后面的 broadcast／collective，仅在最外层捕获异常就会留下不一致的参与者，甚至挂到通信超时。[20][ref-20]

最终实现分两层处理。
pipeline 内部在 rank-0-only 工作之后、后续数据 collective 之前广播失败信息，让其他 rank 在同一逻辑位置得知请求失败；runner 的准备路径再用 `_dit_any_rank_failed()` 汇总本地失败标志，使各 rank 一致地剔除请求。
generator 初始化和 profiler 准备也被纳入相同的 try 范围，避免某个 rank 在失败汇总之前先退出。[18][ref-18] [23][ref-23]

两层不能互相替代。
如果某 rank 已经卡在 pipeline 内部 collective，外层“准备结束后的 all-reduce”来不及救它；反过来，pipeline 处理一种参考媒体异常，也不能覆盖 runner 的所有本地准备错误。
错误同步必须贴着可能导致控制流分叉的位置设计。

最终 CPU 测试检查 rank-0 错误元信息传播，并检查混合 batch 中失败请求被结束、其他请求仍可推进；这为控制逻辑提供了证据。
实际多 rank 验证还应注入无效参考媒体、单 rank 准备失败，确认所有 rank 及时返回，并能处理下一条正常请求。
collective 使用的 flag Tensor 还必须与实际 process-group backend 的设备要求兼容；该 PR 最终 helper 按平台放置 flag，不能把模拟广播测试当作所有通信后端均已验证的证明。[19][ref-19] [23][ref-23] [27][ref-27]

### 5.3 缓存兼容需要追踪状态所有者

同一请求的相邻 denoise steps 可以复用一些中间计算，但这并不自动支持两个请求交错复用同一份缓存。
review 发现 H3 的 `quality=high` 能在请求准备时安装 Cache-DiT profile，其上下文和 buffers 挂在共享 transformer 上，而不属于 `StepRequestState`。
即使启动时没显式设置 cache backend，后来的请求也可能遇到前一个请求留下的不兼容缓存。[20][ref-20]

最终实现对 H3 step mode 的 `quality=high` 给出明确错误；通用 step runner 同时拒绝已配置的 diffusion cache backend。
这不是说明缓存永远无法与连续调度结合，而是当前实现尚未为这类缓存提供 request-local 状态、正确的切换恢复和合批语义。
若要放开限制，验证应包括请求进入、退出、重排、不同步数，以及缓存命中与不命中的等价性，而不只是在单请求循环上确认“缓存能用”。[18][ref-18] [23][ref-23]

接口改动还可能破坏旧模式。
PR 曾给 transformer 内部 `_embed()` 增加请求数参数，TeaCache 的 H3 extractor 沿旧调用方式进入，导致 request-mode 测试失败。
最终 patch 更新 extractor 以传递相应 packed 元信息，说明一个看似只为 step batching 服务的参数，会沿调用者影响未开启该功能的缓存路径。[24][ref-24] [28][ref-28]

### 5.4 负向性能结果反驳的是成本假设

该 PR 作者报告真实权重下，相同 seed 的 request／step 输出逐位一致，两个合批请求各自输出也与独立运行一致。
这是作者的特定配置验证结果，本文未复测，也不将它推广为所有设备和 backend 都必定逐位等价。[16][ref-16]

同一 PR 同时记录了以下负向性能结果：2×H100 80GB、TP2、BF16、672×384、209 frames、30 steps，4 个请求以 concurrency 4 同时提交。[16][ref-16] [25][ref-25]

| 配置 | 总完成时间 | 平均延迟 | 峰值显存 |
| --- | ---: | ---: | ---: |
| request mode | 174.8 s | 111.5 s | 72.4 GB |
| step，`max-num-seqs=1` | 179.0 s | 113.8 s | 72.4 GB |
| step，`max-num-seqs=4` | 182.1 s | 175.7 s | 78.3 GB |

首先比较 request 与 step1，可观察拆出每步执行后的整体变化；再比较 step1 与 step4，才能看到提高并发容量与 packing 共同产生的变化。
这里 step4 的总时间更长，平均延迟也明显升高，因此“请求更早参加计算”不能直接推出“请求更早完成”。
多个长请求共享算力，会让它们同时缓慢推进，并使完成时刻更集中。

作者随后还报告了不同的开放到达实验：4×H100、TP2／USP2、text encoder TP4、VAE patch parallelism 4，10 个请求每 5 秒到达一次，672×384、4 秒视频、24 FPS、20 inference steps。
该实验 request／step4 总时间分别为 60.8／67.9 秒，平均延迟为 11.7／23.6 秒；step4 确实达到 packed compute overlap 4，却仍没有带来收益。
这是一项独立硬件与负载配置，不能把它写成上一张表的同硬件补测。[16][ref-16]

为什么这个结果值得保留在教程里？因为它迫使我们检查“合批节省了什么”。
在自回归 decode 中，单请求一步可能只提供很少计算 rows；H3 的一次长视频去噪却已经带来大量 dense 计算。
把调用次数从 N 次压成 1 次，并没有把 N 个请求的矩阵乘法工作量也压成一份。
是否有足够的 launch、权重访问或低利用率成本可摊薄，必须由该模型和该尺寸的测量决定。

作者的额外算子对照比较了单 GPU 上 3 次独立 forward 与 1 次三请求 fused forward，报告 wall time 差异接近运行间波动。
这支持“该配置中合并 forward 的收益很有限”的观察，却不能单凭端到端表格确定剩余退化究竟由 Python、packing、内存局部性还是其他部分导致。
作者给出了 profiler 分项和解释；若要在自己的配置下采用同样归因，应重新测量，而不能把其解释写成本文已证明的普遍瓶颈。[16][ref-16]

### 5.5 从源码结论走向验证计划

对这类修改，验证应沿依赖关系逐层增加成本。
先用能感知 document／timestep 的小模型验证状态与 packing，再验证真实 backend 的隔离，随后检查多 rank 失败恢复，最后评估完整服务负载。
否则一次昂贵的真实模型生成即使成功，也可能没有触及关键分支。

| 待验证的主张 | 有针对性的实验 | 不能被它替代的下一层证据 |
| --- | --- | --- |
| step 数学与 request 一致 | 相同初始 latent、条件、timestep / sigma schedule，对比每步及最终状态 | 真实权重与真实 kernel |
| 请求互不污染 | 改 B 不改 A、交换顺序、混合步数、提前退出 | backend 确实消费分段边界 |
| 失败只结束目标请求 | 准备／更新／decode 注入失败，检查错误与状态回收 | 多 rank collective 顺序与恢复 |
| backend 与缓存兼容 | 覆盖能力探测、回退、拒绝及旧模式回归 | 受支持配置的端到端生成 |
| batching 值得开启 | 同时到达和错峰到达；大小请求；延迟分位数、吞吐、显存 | 目标线上到达分布下的长期表现 |

测性能时也应分开记录排队时间、每步服务时间、完成时间、batch 成员数及有效 rows。
只看到 GPU 上请求数增加，无法判断吞吐是否提高；只看到队列变短，也无法判断最终用户延迟是否下降。
PR #5810 给出的价值正是：调度能力、数值正确性和性能收益是三个需要分别建立证据的主张。

## 阅读后的自检与适用边界

读完后，应能在源码中回答下面几项，而不只是复述四个接口名：

1. 新请求只在哪一轮做 `prepare_encode()`，旧请求如何取回状态？
2. `step_index`、模型 timestep、服务 scheduler 和 noise scheduler 分别由谁维护？
3. 某 pipeline 的 `latents.shape[0]` 表示什么，runner 如何把预测交回正确请求？
4. 合批后哪些索引应加偏移，哪些位置坐标和 timestep schedule 仍保持请求私有？
5. backend 不理解多 document 时，代码选择回退、拒绝还是可能静默串请求？
6. 请求在某 rank 准备失败时，其他 rank 会停在哪个通信边界？
7. 性能数字覆盖了什么硬件、请求尺寸和到达分布，哪些归因仍待测量？

本文没有展开所有 diffusion 模型接入、所有并行策略或 kernel 实现，也没有把 Qwen-Image 的形状约定推广到 H3。
定位改动时，先用 engine／runner 确认生命周期，再进入具体 pipeline 和 backend 检查语义；跨服务阶段的传输继续阅读 [Connectors](vllm-omni-connectors.md)，全套版本与证据索引见 [references](references.md)。

## 参考资料

正文的源码链接固定到当前走读提交或 PR 最终 head。文件锚点用于帮助定位，历史 review 中提到的中间版本问题需结合最终 patch 阅读。

- 执行模式与对象入口：[执行模式][ref-1]、[Engine][ref-2]、[step 协议][ref-3]、[Runner][ref-6]。
- 当前请求主线：[Images API handler][ref-12]、[Qwen-Image pipeline][ref-9]、[InputBatch][ref-8]、[状态定义][ref-7]。
- 历史机制与测试：[PR #5810][ref-16]、[packing helper][ref-17]、[H3 四方法实现][ref-18]、[数值契约测试][ref-19]。
- 兼容与结果：[关键 review][ref-20]、[多 document 能力协议][ref-21]、[历史 recipe 测量][ref-25]。

[ref-1]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/docs/user_guide/diffusion/execution_modes.md
[ref-2]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/diffusion/diffusion_engine.py#L332
[ref-3]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/diffusion/models/interface.py#L90
[ref-4]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/diffusion/sched/step_scheduler.py
[ref-5]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/diffusion/executor/abstract.py#L27
[ref-6]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/diffusion/worker/diffusion_model_runner.py#L1090
[ref-7]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/diffusion/worker/utils.py#L58
[ref-8]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/diffusion/worker/input_batch.py#L594
[ref-9]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py
[ref-10]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/diffusion/distributed/cfg_parallel.py#L549
[ref-11]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/entrypoints/openai/protocol/images.py#L54
[ref-12]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/entrypoints/openai/api_server.py#L2042
[ref-13]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/examples/online_serving/text_to_image/openai_chat_client.py
[ref-14]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/diffusion/diffusion_engine.py#L554
[ref-15]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/diffusion/sched/interface.py#L49
[ref-16]: https://github.com/vllm-project/vllm-omni/pull/5810
[ref-17]: https://github.com/vllm-project/vllm-omni/blob/d8dfe6bd1d2a21f8061b9035890b73274f64733e/vllm_omni/diffusion/models/minimax_h3/batched_packing.py
[ref-18]: https://github.com/vllm-project/vllm-omni/blob/d8dfe6bd1d2a21f8061b9035890b73274f64733e/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py#L2271
[ref-19]: https://github.com/vllm-project/vllm-omni/blob/d8dfe6bd1d2a21f8061b9035890b73274f64733e/tests/diffusion/models/minimax_h3/test_minimax_h3_step_execution.py
[ref-20]: https://github.com/vllm-project/vllm-omni/pull/5810#pullrequestreview-4896171782
[ref-21]: https://github.com/vllm-project/vllm-omni/blob/d8dfe6bd1d2a21f8061b9035890b73274f64733e/vllm_omni/diffusion/attention/backends/abstract.py#L40
[ref-22]: https://github.com/vllm-project/vllm-omni/blob/d8dfe6bd1d2a21f8061b9035890b73274f64733e/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py
[ref-23]: https://github.com/vllm-project/vllm-omni/blob/d8dfe6bd1d2a21f8061b9035890b73274f64733e/vllm_omni/diffusion/worker/diffusion_model_runner.py
[ref-24]: https://github.com/vllm-project/vllm-omni/pull/5810#discussion_r3787763490
[ref-25]: https://github.com/vllm-project/vllm-omni/blob/d8dfe6bd1d2a21f8061b9035890b73274f64733e/recipes/MiniMaxAI/MiniMax-H3.md
[ref-26]: https://github.com/vllm-project/vllm-omni/blob/5ca23106fd6a0f500deea818bc2b7089306a210f/vllm_omni/diffusion/sched/base_scheduler.py#L166
[ref-27]: https://github.com/vllm-project/vllm-omni/blob/d8dfe6bd1d2a21f8061b9035890b73274f64733e/tests/diffusion/test_diffusion_step_pipeline.py
[ref-28]: https://github.com/vllm-project/vllm-omni/blob/d8dfe6bd1d2a21f8061b9035890b73274f64733e/vllm_omni/diffusion/cache/teacache/extractors.py
