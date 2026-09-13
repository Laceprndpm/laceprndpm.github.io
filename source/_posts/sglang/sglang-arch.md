---
title: sglang架构概览
tags:
  - SGLang
  - Intro
  - AI-Infra
categories:
  - 框架
mathjax: true
---

# SGLang 架构概览

**面向读者：** 已了解基本 LLM 推理流程，希望初步定位 SGLang 请求接入、调度、执行和底层实现职责的开发者。本文提供阅读导航，帮助读者判断一个问题应从哪个模块开始查。

**核心问题与结论：** 从请求到模型结果，主要职责如何划分？下图按 Frontend、Scheduler、Execution Runtime 和 Backend 组织输入、输出及组件，作为后续源码阅读的地图。

**范围与依据：** 本文保留现有职责提纲，不展开完整调用栈或所有配置分支。固定源码版本：—。**待补：** 原提纲未记录 commit，当前分层应作为阅读组织方式，不能据此认定所有版本具有相同接口。

## 职责地图

```
1. Frontend / Tokenizer
   输入：文本 / API Request
   输出：token ids / Request
   包含：tokenizer、multimodal preprocessing、detokenization

2. Scheduler
   输入：Request
   输出：ScheduleBatch
   包含：
   - waiting/running request
   - batching
   - prefix cache
   - KV slot allocation
   - scheduling policy

3. Execution Runtime
   输入：ScheduleBatch / ForwardBatch
   输出：logits / sampled token
   包含：
   - TpModelWorker
   - ModelRunner
   - EagerRunner
   - DecodeCudaGraphRunner
   - execution path selection

4. Backend
   输入：tensor + execution metadata
   输出：tensor
   包含：
   - Attention backend
   - MoE dispatcher / expert compute
   - GEMM / Attention kernels
   - NCCL / communication
```


## 可迁移的方法与未解问题

先确定问题属于请求接入、请求选择、执行组织还是底层计算与通信，再沿该层的输入输出查找生产者与消费者。继续阅读时，可用 [Execution Runtime 入门](sglang-runtime.md) 细化执行层对象与动态数据流。

| 项目 | 内容 | 状态与原因 |
| --- | --- | --- |
| 贯穿请求示例与机制展开 | — | 不适用：本文作为导航页保留简图；执行层的详细示例放在 Runtime 文章中 |
| 精确调用与所有权关系 | — | 待补：原提纲仅列职责，尚未逐项关联固定版本源码 |
| 正确性实验与性能基准 | — | 不适用：本文没有修改方案或性能收益主张，不需要实验对照 |

## 参考与复现材料

- 延伸阅读：[SGLang Execution Runtime 入门](sglang-runtime.md)。该文分别标明固定提交主线与 Init 节的版本范围。
- 本图对应的固定源码引用：—。**待补：** 原提纲没有独立的版本及引用记录，需要逐项核对后补齐。
- 实验复现材料：—。**不适用：** 本文是职责导航，没有需要复现的实验结果。
