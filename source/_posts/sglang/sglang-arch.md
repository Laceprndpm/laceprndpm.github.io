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
