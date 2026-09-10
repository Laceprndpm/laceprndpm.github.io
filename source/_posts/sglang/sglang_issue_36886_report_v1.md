---
title: SGLang Issue #36886
tags:
  - SGLang
  - DCP
  - AI-Infra
categories:
  - 框架
mathjax: true
---

# SGLang Issue #36886 调查报告（第一版）

**主题：DCP 下 index-K 容量与主 c-KV 寻址约定失配**

| 项目 | 范围 |
|---|---|
| 面向读者 | 已了解 SGLang 基本 serving/runtime 架构，希望理解 DCP、DSA 与 KV 内存管理的工程师 |
| 原始问题 | SGLang Issue #36886 |
| 问题版本 | `033446bb05f35c0943aed2750c443077ffc0b92c`，GLM-5.3-Flash 支持分支 |
| 修复参考 | PR \#36989；本次查阅的 head 为 `017d5cb6c92954a0126a80412b193c93e3a94cce` |
| 文档版本 | v0.1，2026-09-08 |
| 证据性质 | 原始 issue、修复 diff 与固定版本源码的整理；尚未独立运行 GPU 复现或验证 |

本文先梳理已经有依据的**症状、架构、根因和修复**。第 2 节区分公开材料中的排查线索与仍待补齐的调试过程；第 6 节中的实验结果均注明为原作者报告，不作为本文作者的实测结果。

截至本次查阅，PR #36989 为 **closed、未通过该 PR 合并**。本文讨论的是该修复方案，不据此判断当前 `main` 是否已包含其他等价修复。其端到端验证范围是 **H100 + TileLang + BF16 KV**，不能直接外推到其他 attention backend。[[1]][r1] [[2]][r2]

> 核心结论：两个缺陷不能混为一种。**index-K 应该保留 global virtual loc，但 buffer 容量不足；主 c-KV 应该分片并转换索引，却漏掉了 owner filter 与 global → local 映射。**

---

## 1. 症状：崩溃、静默数据破坏与状态相关故障

### 1.1 原始环境

原作者在单节点 **8×H100 80GB** 上运行 GLM-5.3-Flash，使用 TP8、DCP8、BF16 KV、TileLang DSA prefill/decode backend，并开启 EAGLE/MTP。报告中的模型 revision 为 `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`。软件环境包括 Python 3.12、PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA 13.0 和驱动 590.48.01。[[1]][r1]

这些信息用于限定问题版本，不是当前版本的通用部署建议。

### 1.2 两类表面症状

| 表现 | 原作者观察 | 对应的主要缺陷 |
|---|---|---|
| **Crash** | 长请求使分配位置越过某一边界后，出现设备断言或非法访存；报错可能落在不同 kernel | Defect 1：index-K buffer 容量不足 |
| **Quality degradation / silent corruption** | 相同提示词在刚重启时正常，经过长请求后出现数字混乱、翻译遗漏、推理过长直至输出预算耗尽 | Defect 2：主 c-KV 读写未执行 DCP localization |

这是 issue 对主要现象的归类，不表示某种越界只能产生一种症状。[[1]][r1]

例如，作者观察到：比较小数时结论可能仍对，但解释混乱；翻译中遗漏“公园”；简单应用题的推理异常膨胀，最终没有正常答案。关键不是某个提示词答错，而是**同一服务在不同运行状态下，对同一组提示词表现不同**。[[1]][r1]

### 1.3 Watermark：为什么运行一段时间后才暴露

作者给出的状态对照是：刚重启服务，三个提示词正常；再发送两个不同的约 300K-token 请求，使其报告的累计 virtual allocation 达到约 640K；随后重试同一组提示词，质量下降再次出现。该次实验使用 `--mem-fraction-static 0.68`，`max_total_num_tokens` 为 455,744。[[1]][r1]

这里的 watermark 指向**实际访问的 slot 编号逼近或超过 buffer 可寻址边界**，不是模型上下文长度上限。

需要保留两个边界：

- `max_total_num_tokens` 是容量配置，不是当前活跃 token 数；累计处理量也不能直接替代实际 `loc`。
- 真实 buffer 包含页对齐和保留空间。主 c-KV 的行数为 `size + page_size`，index-K 还按页组织。因此本文不会把 `loc == max_total_num_tokens` 写成所有路径共同的精确首个越界点。[[1]][r1] [[5]][r5] [[6]][r6]

后文用简化容量模型解释因果；精确复现时仍需记录实际 slot 与 tensor shape。

## 2. 排查：从“哪里崩”转向“什么 invariant 被破坏”

### 2.1 公开材料已经提供的线索

原作者报告，报错位置可能是 ATen `index_put`、`causal_conv1d`、DeepGEMM、`chunk_gla` 或 embedding，但故障与累计分配越过容量边界有稳定关系。作者还修正过最初“DCP 始终导致质量下降”的判断：**新启动的 DCP 服务正常，越过 watermark 后才劣化**。[[1]][r1]

这支持一种排查方向：报错 kernel 可能是被污染数据的后续使用者，应优先检查先前的 KV/index-K 写入、索引范围和 buffer 容量，而不是只修改 traceback 最后一帧。

另外，模型最前面的三层是 linear attention/KDA；该报告列出的 DSA 层为 `[3, 7, 11, ..., 43]`。只比较 layer 0–2 的输出，尚未覆盖本问题中的 DSA paged-KV 路径，不能排除其后发生的错误。[[1]][r1]

### 2.2 可以重建的逻辑排查链

以下是**本文根据公开证据整理的排查逻辑，不是原作者完整的操作时间线**：

```text
同一提示词：重启后正常，经过长请求后异常
    ↓
把“运行状态 / 分配位置”作为关键变量
    ↓
报错位置不固定，检查更早的数据生产者
    ↓
追踪 allocator 发出的 loc
    ↓
对照每个 consumer 的索引约定与实际容量
    ├─ index-K：用 global loc，容量却只有 local size
    └─ c-KV：本应按 owner 存 local shard，却直接用 global loc
    ↓
提出对应修复，再用同一类跨 watermark 负载对照
```

这个方向的核心 invariant 是：

> **一个 consumer 接到的索引，必须属于其约定的索引空间；该索引经过必要转换后，必须落在实际分配的存储范围内。**

### 2.3 尚未补齐的 debug 链

当前材料不足以还原作者怎样从第一个异常逐步锁定具体写入点。第二版应补充以下证据，而不是编写一段事后推测的“真实调试经历”：

| 待补内容 | 需要回答的问题 |
|---|---|
| 分配与访问记录 | 第一个异常 batch 各 rank 的 `out_cache_loc` 范围、pool shape、页大小分别是多少？ |
| 首次异常定位 | 最早出现错误的是哪个写入点？使用了哪些检查方法？原始日志是什么？ |
| 补丁隔离实验 | 只修 index-K、只修 c-KV 写侧、再补读侧时，各自改变了什么现象？ |
| 逐层对照 | 首个 DSA 层的 index-K、c-KV、top-k、attention output 从哪一步开始分歧？ |

**本版已经能解释故障机制，但还没有独立完成“最小复现 → 首个错误操作 → 隔离修复”的调试闭环。**

## 3. 架构裁定：先说明每个索引和 buffer 的含义

### 3.1 SGLang 有请求到 KV slot 的映射

这里不是缺少类似 block table 的机制。相关数据关系可以概括为：

```text
request 的池内编号 + 请求内部 token position
    ↓
ReqToTokenPool.req_to_token
    ↓
global virtual slot / loc
    ↓
根据所访问的 cache 选择解释方式
    ├─ index-K：保留 global loc
    └─ DCP-sharded c-KV：owner filter + local loc
    ↓
具体层的 KV / index-K tensor
```

`req_to_token` 中保存的是 token 对应的存储位置，不是词表 token ID；在本 issue 的 DCP 配置下，该位置属于 global virtual slot space。`page_table_1` 中的“1”表示按单 token slot 表达索引，不意味着索引已自动转换成 rank-local。[[1]][r1]

### 3.2 容量规划、存储分配与 slot 分配是不同职责

| 组件 | 本问题中的职责 |
|---|---|
| `KVCacheConfigurator` | 结合模型配置、容量配置、DCP 和 target/draft 身份，构造 pool 与 allocator |
| `HybridLinearKVPool` | 包装混合模型的存储，并将 DSA 参数传给内部 `full_kv_pool` |
| `DSATokenToKVPool` / `MLATokenToKVPool` | 持有主 c-KV 及 DSA 相关 cache 对象 |
| `IndexKeyCache` | 根据 `index_buf_size` 与页布局分配、访问 index-K 存储 |
| `PagedTokenToKVPoolAllocator` | 从预先规划的 slot 空间中分配位置编号，而不是每来一个 token 就重新分配整块 KV tensor |

配置器先解析容量，再创建这些对象；`IndexKeyCache` 中可以直接看到按 shape 创建 `torch.zeros` buffer 的代码。[[1]][r1] [[4]][r4] [[5]][r5] [[6]][r6]

`index_head_dim` 表达每个 index key 的特征维度；`index_buf_size` 表达能够容纳多少 slot。**本缺陷修复的是后者，而不是模型的特征维度。**

### 3.3 DCP 的 global → local 约定

令 `C` 表示简化后的每 rank 主 c-KV 容量，`D` 表示 DCP group 大小。忽略页对齐与保留 slot 时，逻辑 slot 空间为 `[0, C × D)`。本例采用交错分片：[[1]][r1] [[3]][r3]

```python
owner_rank = loc % D
local_loc = loc // D
```

以 `D = 4` 为例：

```text
global loc    0  1  2  3  4  5  6  7  8  9 10 11
owner rank    0  1  2  3  0  1  2  3  0  1  2  3
local loc     0  0  0  0  1  1  1  1  2  2  2  2
```

例如 `loc = 10` 对应 rank2 的 local slot 2。这里的 `loc` 是软件 slot 编号，`local_loc` 也仍是 tensor 下标；它们不是 CUDA 虚拟地址。

`C` 是容量，不是某个 rank 当前使用的 token 数。假设四个 rank 都有 100-slot 容量，即使当前占用分别为 99、100、100、100，也不能用 99×4 推导全局容量。**最终 sizing 应覆盖 allocator 可能发出的有效索引范围，而不是由当前占用量推算。**

### 3.4 同一个 global loc，对不同 cache 的解释不同

| 对象                         | 本例的存储策略           | 索引约定                              | 简化容量                 |
| -------------------------- | ----------------- | --------------------------------- | -------------------- |
| Target 主 c-KV              | DCP 分片            | 只处理本 rank 拥有的 loc，再使用 `loc // D`  | 每 rank `C`           |
| DSA index-K                | DCP ranks 内保留完整副本 | 使用 global loc，不做 owner filter 和除法 | 每 rank `C × D`       |
| 本例中的 replicated draft pool | 完整副本              | 使用 global loc                     | 覆盖共享 allocator 的逻辑范围 |

index-K 用于选择历史 token，主 c-KV 用于真正的 sparse attention。两者对应同一个历史 token，但不是同一套特征表示，也不具有相同的存储策略。上述表格限于 #36886 涉及的设计。[[1]][r1] [[2]][r2]

因此不能给所有 KV 相关索引统一加一次 `// D`，也不能让所有 buffer 都默认使用同一个 `size`。

### 3.5 Extend 临时工作集与持久 pool 不同

本报告中的 ordinary extend 是对已有 prefix 追加一批待计算 token 的路径。`[prefix_i ; extend_i]` 中的 `i` 是 **request 编号**，不是 rank 编号。

#36989 的 sparse extend 修复读取各 rank 的 prefix c-KV 分片，通过通信收集，再按 request 顺序与当前 extend KV 拼接，构成临时工作集：[[3]][r3] [[8]][r8]

```text
[prefix_0 ; extend_0 | prefix_1 ; extend_1 | ...]
```

这里 **RAGGED 是布局，RAGGED top-k transform 是适配该布局的索引操作**。PAGED transform 则将位置映射到 paged cache slot。
## 4. 根因：两个不同的约定失配

### 4.1 Defect 1：index-K 索引正确，但容量不足

原代码中，`DSATokenToKVPool` 支持可选的 `index_buf_size`，未提供时使用如下默认值：[[5]][r5]

```python
if index_buf_size is None:
    index_buf_size = size
```

而本例的 index-K 写入使用 raw `forward_batch.out_cache_loc`，读取所用的 metadata 也保留 global virtual loc。**保留 global loc 本身符合 index-K 的 replicated 设计，错误是其存储容量没有跟随这个索引空间扩大。**[[1]][r1]

```text
allocator：可发出覆盖 C × D 范围的 loc
    ↓
index-K：正确地直接使用 global loc
    ↓
index_buf_size 未单独推导和传入
    ↓
默认退回 c-KV 的 local size C
    ↓
较大的合法 global loc 超出 index-K 实际存储范围
    ↓
越界写入 → 后续数据或 kernel 异常
```

修复 diff 显示两处缺口：配置器需要推导独立的 index-K 容量；GLM-5.3-Flash 使用的 `HybridLinearKVPool` 包装层还需要新增参数并转发给内部 DSA pool。这不是一个已经计算正确的数值在途中偶然丢失，而是**容量推导与参数传递通道都需要补齐**。[[1]][r1] [[3]][r3]

### 4.2 Defect 2：主 c-KV 漏掉 DCP owner/local 映射

**写侧。** 当 `qk_rope_head_dim == 0` 时，wrapper 选择 `set_mla_kv_buffer_kernel_norope`。基线源码直接以 raw loc 计算目的地址：[[7]][r7]

```python
loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs
```

同文件的 rope kernel 已有 owner filter 和 `loc // D`，norope 分支却没有。于是本应分片的 c-KV 在各 rank 上按 global loc 写入，违反本地 pool 的寻址约定。

相关写入链可概括为：[[1]][r1] [[3]][r3] [[5]][r5] [[7]][r7]

```text
主 c-KV 写入调用
    → MLATokenToKVPool.set_mla_kv_buffer
    → _write_mla_kv_buffer
    → set_mla_kv_buffer_triton
    → set_mla_kv_buffer_kernel_norope
    → raw loc 直接参与本地 buffer 寻址
```

**读侧。** Issue 描述，decode metadata 从 `req_to_token` 复制出的 `page_table_1` 保留 virtual loc；fused top-k v2 从 `real_page_table` 重建的 slot 也仍属于该空间。随后 sparse decode/verify/extend 将这些索引用于 per-rank c-KV，读侧同样没有满足 consumer 的映射约定。[[1]][r1]

```text
本应：global loc → owner filter → local loc → local c-KV
实际：global loc ──────────────────────────→ local c-KV
```

低水位时，错误的读写约定可能互相匹配，形成意外复制；高水位时，则越出 buffer 范围。这解释了为什么“先前输出正常”不能证明分片已正确实现。[[1]][r1] [[2]][r2]

### 4.3 后续发现：容量断言使用了错误的比较边界

2026-08-29 的 issue 更新还记录了一个相关问题：`init_forward_metadata` 对一份包含 virtual loc 的 flattened page table，使用主 c-KV 的 per-rank 容量做上界检查。作者报告遇到 `max_idx = 2,937,855`、比较上界为 382,976 的失败。[[1]][r1]

该表的 consumer 是需要覆盖 virtual space 的 index-K；因此，**超过主 c-KV 的 local size，并不自动说明这些 index-K 索引非法**。这里是容量检查本身误判，不能与真实的 OOB 混为一谈。

## 5. 修复：分别恢复写入、读取与容量约定

以下内容描述 #36989 的方案。代码片段标为“示意”的部分用于说明语义，不是完整可应用补丁。

### 5.1 Write：owner filter 后转换 local loc

norope kernel 增加 rank 和 DCP size 参数，然后按照所属 rank 决定是否写入：[[3]][r3]

```python
is_valid = loc % DCP_WORLD_SIZE == DCP_RANK
safe_loc = tl.where(is_valid, loc, 0) // DCP_WORLD_SIZE
dst_ptr = kv_buffer_ptr + safe_loc * buffer_stride + offs
tl.store(dst_ptr, src, mask=mask & is_valid)
```

`is_valid` 决定“该不该写”；`safe_loc` 决定“写到本地哪里”。非 owner 条目被 mask 掉，不能只除以 DCP size 而不做过滤。

同时，pool 增加 `dcp_localized_writes`，wrapper 接收 `dcp_localize`：分片 target 使用 localization，replicated draft 不使用。PyTorch 写入分支也从单纯过滤改为过滤后再除以 DCP size。[[3]][r3]

### 5.2 Decode / target verify：读侧执行相同的空间转换

补丁在 top-k transform 后增加 `_dcp_localize_page_table()`，保留本 rank 拥有的条目并转成本地 slot；其余条目变为 `-1`，由这条 sparse kernel 路径屏蔽。[[3]][r3]

```python
# 示意；实际实现使用 tensor 运算。
local_index = loc // D if loc >= 0 and loc % D == rank else -1
```

例如 `D = 4`、选中 global slots `[2, 5, 8, 10]`，rank0 得到 `[-1, -1, 2, -1]`。它只读取本地 slot 2，对应 global slot 8。

各 rank 得到所持 context 子集的 attention 结果，再通过既有的 LSE merge 合并。这个修复改变读地址的解释，而不是把主 c-KV 扩成每 rank 完整副本。`-1` 的处理也属于具体 kernel 接口约定，不能未经检查就照搬到其他 backend。[[1]][r1] [[3]][r3]

### 5.3 Extend：收集 prefix，再匹配临时 KV 布局

原作者明确说明，**该版本的这条 sparse extend 路径**没有 decode/verify 那样的跨 rank LSE merge。因此补丁不让每个 rank 只消费残缺 context，而是复用 `all_gather_kv_cache_for_mha_extend()`。[[1]][r1] [[3]][r3]

其操作是：

```text
用 dcp_local_prefix_kv_indices 读取本 rank 的历史 c-KV
    ↓
收集各 rank 的 prefix 分片，并重组顺序
    ↓
按 request 拼入本轮 extend KV
    ↓
形成 [prefix_0; extend_0 | prefix_1; extend_1 | ...]
    ↓
使用指向该临时工作集的 top-k 索引执行 sparse attention
```

源码中既有 collective，也有 reshape/transpose 和按 request 拼接；**不能把 all-gather 这个通信原语本身等同于自动生成最终 RAGGED 布局。** helper 才是完成这些步骤的整体。这里收集的是相关请求的 prefix KV，不能将其描述为“只传输 top-k 最终选中的 KV”。[[8]][r8]

`get_topk_transform_method()` 同时为该分支选择 `RAGGED`。概念上，request 内位置 `p` 对应临时工作集中的 `cu_seqlens_k[req] + p`；原作者特别指出，这条路径的 unfused top-k 输出并非可以直接如此相加的纯 sequence position，因此采用既有 fused RAGGED transform 处理 `row_start` 等偏移。[[1]][r1] [[3]][r3]

**数据收集与索引生成需要遵守同一布局约定，但不要求代码一定先 gather、再执行 top-k transform。** 关键是最终交给 kernel 的 buffer 和 indices 配套。

可以把它简记为“计算时临时恢复完整 context”，但不是关闭 DCP，也不是永久恢复完整 KV pool。更不能据此宣称所有 extend 都不能做 LSE merge，或这份修复已经证明某种通信策略性能最优。

### 5.4 Index-K sizing：扩大应当 replicated 的辅助存储

配置器新增：[[3]][r3]

```python
def _dsa_index_buf_size(self, size: int) -> Optional[int]:
    scale = get_parallel().attn_dcp_size // self.loc_space_scale
    return size * scale if scale > 1 else None
```

对于 target，`loc_space_scale = 1`，因此将 index-K 容量扩到 DCP global space。对于本例的 replicated draft，传入的 pool size 已由 `_derive_pool_sizes()` 扩大，`loc_space_scale = D`，不再重复乘 D。[[4]][r4]

计算结果通过两条构造路径传入：

```text
KVCacheConfigurator
    ├─ _build_dsa_kv_pool
    │    → DSATokenToKVPool(index_buf_size=...)
    │
    └─ _build_hybrid_linear_kv_pool
         → HybridLinearKVPool(index_buf_size=...)
         → 内部 DSATokenToKVPool
         → IndexKeyCache 分配存储
```

因此修复同时增加 wrapper 构造参数、向内转发以及对外的 `index_buf_size` 属性。**它没有改变 index-K 的 global indexing，而是让容量与这一设计一致。**[[3]][r3]

该扩容有真实显存代价：作者报告其 H100 配置每 GPU 额外开销约 5 GB，并降低 `mem-fraction-static` 留出空间。这个数值是具体配置下的报告值，不是所有 DSA+DCP 配置的固定成本。[[1]][r1]

### 5.5 容量断言：按被检查索引的空间选择上界

补丁将相关检查中的容量主项由 `size` 调整为 DCP 开启时的 `size × D`，保留对应页空间项。这与 flattened page table 在该处仍包含 virtual loc 的语义一致。[[3]][r3]

这一改动不应概括成“所有 capacity assert 都乘 D”。如果检查的是转换后的 local c-KV index，上界仍应属于本地 buffer；检查前必须先确定 consumer。

### 5.6 修改位置总览

| 文件 / 对象 | 修复内容 |
|---|---|
| `kernels/ops/kvcache/mla_buffer.py` | norope 写入增加 owner/local 转换；wrapper 接收 localization 开关 |
| `srt/mem_cache/memory_pool.py` | 区分 target/draft 写入约定；修正 torch 写分支；补 wrapper 参数和属性 |
| `srt/layers/attention/dsa_backend.py` | decode/verify 读索引转换；extend gather + RAGGED 接入；容量断言修正 |
| `srt/mem_cache/kv_cache_configurator.py` | 推导并传递 index-K 容量，配置 replicated draft 的行为 |

以上为与两个 defect 直接相关的修改，依据 PR diff 整理。[[3]][r3]

## 6. 验证：区分因果解释、作者结果与待做实验

### 6.1 为什么低水位时可能正常

对 **Defect 1**，index-K 的 global indexing 本来正确。只要实际访问的 global loc 仍落在已分配范围内，读写就可以正常；问题是未来可分配的范围大于当前 buffer 容量。

对 **Defect 2**，主 c-KV 的 raw-global 读写约定是错的，但低水位时它们可能互相匹配：每 rank 意外存下相同历史 KV，读侧也访问相同位置。作者将早期正确输出归因于这种意外 replication，以及对相同结果进行合并仍可得到正确输出。[[1]][r1] [[2]][r2]

因此不能只写“reserve > use，所以正常”。更准确的条件是：

> **相关读写取到了相互匹配的数据，且实际访问的 slot 仍在相应 buffer 的可寻址范围内。**

高水位使后一条件失效后，就可能发生数据破坏或崩溃。累计处理 token 数只是原复现中推进分配状态的方式，不是一个可以代替 `max(loc)` 的通用 OOB 判据。

### 6.2 作者报告的修复前后结果

以下均为公开报告结果，**本文未独立复现**。[[1]][r1] [[2]][r2]

| 验证项 | 修复前 | 修复后 / 作者报告 |
|---|---|---|
| 3 个并发 300K-token 请求 | 5 次实验均在越过报告边界附近崩溃 | 多轮同类请求运行正常 |
| 重启 → 提示词检查 → 长请求 → 再检查 | 越过 watermark 后出现质量下降 | 前后检查均正常 |
| 4K、150K-token needle retrieval | — | 各自 3/3；作者注明 150K 用例经过 18 个 prefill chunks |
| 2×300K 单请求及两轮 3×253K 并发请求 | — | 后续提示词与 needle 检查仍正常 |
| 896K-token 三针检索 | — | 3/3 |
| 日志与进程状态 | 崩溃或输出异常 | 作者报告无相应 NaN、OOB、OOM、断言或重启异常 |

后续 graphwalks 子集结果也被记录为：≤128K 的 120 个样本，BFS/parents F1 分别为 99.1/98.6；约 256K 的 40 个样本，分别为 100.0/97.6。**这些是仅针对成功解析答案的 F1，解析失败样本被排除，不能当作完整评测集准确率。**[[1]][r1]

日志未出现异常不等于已经证明不存在所有越界；上述结果支持指定配置下的回归修复，不构成对全部 backend、dtype 和并行组合的保证。

### 6.3 必须控制的独立干扰项

作者另报告：`SGLANG_OPT_DG_MASKED_M_CAP=1` 在其 GLM-5.3-Flash BF16 + DeepGEMM 配置下会独立造成与提示词相关的 NaN。作者移除了该开关，并保留 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 处理其遇到的碎片问题。[[1]][r1]

这应作为**作者环境中的干扰项**记录，而不是归为 DCP 根因。对 `expandable_segments` 如何影响错误表现，原报告提出了解释，但本文没有独立验证其具体内存机制，也不将它推广成所有越界行为的规律。

### 6.4 本地待补的验证记录

第二版的首要工作不是继续扩充推测，而是建立可核验记录：固定模型与代码版本，记录每个相关 tensor 的容量和实际索引范围；用同一类负载比较基线和修复；再隔离 index-K 扩容、c-KV 写入、decode/verify 读取、extend 重组及容量断言各自的作用。

记录格式可先固定为：

```text
代码 / 模型 revision：
硬件、DCP/TP、backend、dtype：
本轮请求、缓存状态、执行阶段：
各 rank 的 global loc 范围：
local index 范围与真实 tensor shape：
首个异常操作及日志：
启用的补丁子集：
复现结果与输出对照：
```

这些是后续实验要求，不是本版已经完成的测试。

## 7. 经验：不要把不同 index space 混成一种

### 7.1 给索引附上 consumer，而不只看变量名

同一个 `loc` 在 index-K 中可以是合法的 global index，在主 c-KV 中却需要 owner/local 转换。`page_table`、`physical slot` 这样的名字也不能证明它已经是某块本地 tensor 的有效下标。

本例值得保留的检查顺序是：**来源 → 索引空间 → consumer → 转换 → 实际容量**。

### 7.2 分配、写入、读取和断言必须维护同一约定

只扩大 buffer 可以掩盖错误分片，只修写地址会使读写不一致，只修读写又可能被旧容量断言阻断。#36886 的修复跨越多个模块，是因为约定在多个 consumer 上展开，而不是因为 attention 公式本身发生了变化。[[1]][r1] [[3]][r3]

### 7.3 表示与操作分开讲

RAGGED 是变长序列紧凑存储的表示；top-k transform 是生成适用于该表示的索引。all-gather 是通信，helper 中的重组才生成目标布局。将这些层次分开，才能解释为什么 persistent pool 的旧 mapping 不能原样指向 gathered tensor。

这也不意味着只能选 RAGGED：本版材料证明的是作者采用了这条实现，并未比较所有其他映射方案的性能。

### 7.4 因果链清楚，不等于 debug 过程已还原

本文已能将两个 defect 的不变量失配与修复位置对应起来，但仍缺少独立复现、首个错误操作以及逐步排除假设的原始记录。面向后续工程读者，必须保留这一证据边界。

### 因果链总览

```text
DCP 扩展 allocator 的逻辑 slot 空间
    │
    ├─ index-K 继续 replicated，并用 global loc
    │      ↓
    │  index_buf_size 仍默认等于 local c-KV size
    │      ↓
    │  合法 global loc 超过实际 index-K 容量
    │      ↓
    │  OOB / 后续 kernel 异常
    │      ↓
    │  修复：独立推导容量，补齐配置与 wrapper 传参
    │
    └─ 主 c-KV 应按 DCP owner 分片
           ↓
       norope 写入和相关读路径继续使用 raw global loc
           ↓
       低水位：读写自洽，形成意外复制，输出可能正常
           ↓
       高水位：访问越出本地 buffer，数据破坏 / 质量下降
           ↓
       修复：write 与 decode/verify 做 owner/local 转换；
             extend 收集 prefix 并匹配 RAGGED 索引；
             相关容量断言使用正确的索引空间
```

---

## 附录：DCP 与 DSA 背景补充

**DCP（Decode Context Parallelism）** 沿上下文维度分片保存 KV，并在 decode 时对各 rank 持有的 context 子集计算 attention，再利用包含归一化信息的 LSE 合并结果。本例的 ownership 按 `loc % dcp_size` 决定，本地 slot 为 `loc // dcp_size`。这里讨论的是 #36886 的具体实现约定，不把某个 backend 的 extend 策略当成 DCP 的普遍定义。[[1]][r1] [[8]][r8]

**DSA（DeepSeek Sparse Attention）** 在主 attention 前增加 indexer，利用独立的 index-K 为历史 token 打分并选择 top-k，再访问对应的 compressed KV（c-KV）进行 sparse attention。index-K 负责选择位置，c-KV 保存真正参与主 attention 的表示。在本 issue 涉及的 DCP 方案中，index-K 保持 replicated/global-indexed，而主 c-KV 应为 sharded/local-indexed；这两种存储约定的差异是理解两个 defect 的起点。[[1]][r1] [[2]][r2]

## 参考资料

1. **[SGLang Issue #36886][r1]**：原始故障报告、两个 defect、复现环境、验证结果，以及 2026-08-29 的容量断言更新。原作者：`junliu-mde`。
2. **[SGLang PR #36989][r2]**：修复方案、验证范围和提交状态。标题：*fix(dcp): GLM-5.3-Flash norope c-KV sharding under DCP (Hopper/tilelang only)*。
3. **[PR #36989 Files changed][r3]**：各文件修改。本文查阅时 head 为 `017d5cb6c92954a0126a80412b193c93e3a94cce`；[对应固定 head 源码][r3head]。
4. **[基线 `kv_cache_configurator.py`][r4]**：`configure()`、`loc_space_scale`、`_derive_pool_sizes()` 与 pool 构造职责。
5. **[基线 `memory_pool.py`][r5]**：pool 类型、主 c-KV 分配、`DSATokenToKVPool.index_buf_size` 默认值与包装层。默认值位于 [L4800–L4845][r5size]。
6. **[基线 `index_key_cache.py`][r6]**：index-K 的页数、buffer shape 与分配操作。
7. **[基线 `mla_buffer.py`][r7]**：rope/norope 两条写入路径与 wrapper 分派。
8. **[基线 `layers/dcp/comm.py`][r8]**：prefix KV 收集与重组、按 request 拼接及 LSE 合并相关 helper。

**版本约定：** 参考资料 4–8 固定在问题基线 commit `033446bb05f35c0943aed2750c443077ffc0b92c`，避免将持续变化的 `main` 与原故障混读。Issue 和 PR 正文可能继续更新；本版查阅日期为 2026-09-08。示意图和简化容量模型为本文整理，不是实测日志。

[r1]: https://github.com/sgl-project/sglang/issues/36886
[r2]: https://github.com/sgl-project/sglang/pull/36989
[r3]: https://github.com/sgl-project/sglang/pull/36989/files
[r3head]: https://github.com/StarDuster/sglang/tree/017d5cb6c92954a0126a80412b193c93e3a94cce
[r4]: https://github.com/sgl-project/sglang/blob/033446bb05f35c0943aed2750c443077ffc0b92c/python/sglang/srt/mem_cache/kv_cache_configurator.py
[r5]: https://github.com/sgl-project/sglang/blob/033446bb05f35c0943aed2750c443077ffc0b92c/python/sglang/srt/mem_cache/memory_pool.py
[r5size]: https://github.com/sgl-project/sglang/blob/033446bb05f35c0943aed2750c443077ffc0b92c/python/sglang/srt/mem_cache/memory_pool.py#L4800-L4845
[r6]: https://github.com/sgl-project/sglang/blob/033446bb05f35c0943aed2750c443077ffc0b92c/python/sglang/srt/mem_cache/index_key_cache.py
[r7]: https://github.com/sgl-project/sglang/blob/033446bb05f35c0943aed2750c443077ffc0b92c/python/sglang/kernels/ops/kvcache/mla_buffer.py
[r8]: https://github.com/sgl-project/sglang/blob/033446bb05f35c0943aed2750c443077ffc0b92c/python/sglang/srt/layers/dcp/comm.py
