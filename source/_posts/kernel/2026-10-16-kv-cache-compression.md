---
title: KV Cache 压缩的执行路径与性能
date: 2026-10-16
tags:
  - AI-Infra
  - CUDA
  - KV-Cache
  - 推理优化
  - 量化
categories:
  - 算子优化
mathjax: true
---

TurboQuant 优化 attention 对压缩 KV 的读取与计算；KVTC、nvCOMP-ANS 优化缓存存储和回载。本文保留 15 张讲座原图，集中记录数据布局、执行流水线和性能结果。

图表数字按讲座及其引用的公开实验口径表述。

<!-- more -->

# KV Cache 压缩的执行路径与性能

- [压缩放在哪条路径](#paths)
- [TurboQuant：页内布局与融合 decode](#turboquant)
- [KVTC：存储压缩与流水化回载](#kvtc)
- [nvCOMP-ANS：无损压缩传输](#nvcomp)
- [三种方案的结果对照](#comparison)

<a id="page-01"></a>

![原图 01：讲座议程](01.jpg)

<a id="paths"></a>
<a id="recap-kv"></a>

## 压缩放在哪条路径

<a id="page-02"></a>

![原图 02：KV Cache 的容量、带宽与迁移成本](02.jpg)

KV 占用随历史长度增长，影响 GPU 可容纳的请求数、decode 读取量，以及卸载和回载的数据量。

<a id="page-03"></a>

![原图 03：KV Cache 压缩方法分类](03.jpg)

**量化**减少每个元素的位数；**存储稀疏**移除部分 KV 条目；**计算稀疏**保留缓存，只选取部分条目参与本步 attention。三者分别改变表示大小、保存范围和访问范围。

<a id="page-04"></a>

![原图 04：在线 attention 与缓存后端](04.jpg)

本讲的两条执行路径是：

```text
TurboQuant：HBM 中的压缩 KV → attention 内解码 → 计算
KVTC / ANS：后端压缩载荷 → 回载并恢复标准 KV → attention
```

前者可直接减少每步 decode 的 HBM 读取；后者主要改变缓存容量和跨层级搬运。TTFT 指首 token 延迟，TPOT 指后续每 token 的平均生成时间。

<a id="turboquant"></a>

## TurboQuant：页内布局与融合 decode

<a id="page-05"></a>
<a id="recap-compression"></a>

### 量化表示

![原图 05：旋转、码本与低位编码](05.jpg)

图中的算法先随机旋转向量，使坐标分布更适合量化，再用预计算的 Lloyd–Max 码本编码。向量保存中心编号、范数，以及可选的 QJL 残差修正；旋转矩阵与码本共享。[TurboQuant 论文](https://arxiv.org/abs/2504.19874)

图中 **50.06 的 LongBench-V1 分数、至少 4.5× 名义压缩率**，对应论文中 Llama-3.1-8B-Instruct 的平均 3.5-bit 配置。下面的 kernel 实验使用 **K4V4-NC**，是另一组配置。

<a id="page-06"></a>
<a id="recap-layout"></a>

### 按页聚合字段，提高 sector 利用率

![原图 06：token 交错布局改为页内字段分区](06.jpg)

在 $D=128$、K/V 各 4 bit 时，每 token 为 **134 B**：

```text
K codes 64 B | K norm 2 B | V codes 64 B | V scale 2 B | V zero 2 B
```

原布局中，同类字段跨 token 相隔 134 B，V codes 从偏移 **66 B** 开始，协作读取容易跨越更多 32 B sector。新布局将 **16 个 token** 的同类字段聚在一起：

| 页内字段 | 字节数 | 起始偏移 |
| --- | ---: | ---: |
| K codes | 1024 | 0 |
| V codes | 1024 | 1024 |
| K norms | 32 | 2048 |
| V scales | 32 | 2080 |
| V zeros | 32 | 2112 |

总量仍为 **2144 B/page/KV head**，但字段起点相对页基址按 32 B 对齐。量化、打包时直接写入新布局，decode 无需再做布局转换。

图中每个 32 B load sector 的有效字节从 **3.68 B → 29.75 B**，利用率约 **11.5% → 93.0%**；这是访存指标，不是端到端加速倍数。

<a id="page-07"></a>
<a id="recap-attention"></a>

### 在 attention 内恢复 MMA 操作数

![原图 07：4-bit KV 的分块 MMA decode](07.jpg)

每次处理 **64-token KV tile**，双缓冲重叠当前块计算与下一块加载。共享同一 KV head 的 GQA 查询最多打包 **16 个 Q heads**，通过匹配的 FWHT（快速 Walsh–Hadamard 变换）生成 FP16 Q tile，复用同一批 KV。

K 与 V 的恢复路径不同：

```text
K codes → 码本查表 → MMA fragment → QKᵀ → K norm 修正
V codes + scale/zero → FP16 共享内存 tile → PV
```

压缩 KV 在片上恢复后直接参与计算，省去“完整解码到 HBM，再读回 attention”的中转。4 bit 是存储编码，MMA 使用恢复后的 FP16 操作数。

每个 CTA 处理一个 KV split，以 FP32 保存运行最大值、指数和与加权和；跨 tile 更新最大值时，同时重缩放旧状态。若各非空 split 的状态为 $(m_s,\ell_s,u_s)$，其中 $u_s$ 是未归一化加权和，则最终合并为：

$$
m=\max_s m_s,\qquad
O=\frac{\sum_s e^{m_s-m}u_s}{\sum_s e^{m_s-m}\ell_s}.
$$

这保留了跨全部历史的归一化，不能直接平均各 split 的输出。

上述优化来自 [vLLM PR #53060](https://github.com/vllm-project/vllm/pull/53060)，整理时尚未合并。其快速路径面向单 token causal decode、BF16 Q/output、K4V4-NC 等支持组合。

<a id="page-08"></a>

### Decode 性能

![原图 08：不同上下文长度下的 decode 性能](08.jpg)

条件：**Qwen3-8B-Base、batch=32、CUDA Graph、生成 128 tokens，仅测 decode**。以下倍率均相对**相同 K4V4-NC 配置的原 vLLM TurboQuant 实现**。

| 缓存长度 | Ampere | Hopper | Blackwell |
| --- | ---: | ---: | ---: |
| 128 | 1.27× | 1.25× | 1.28× |
| 4K | 2.74× | 2.52× | 3.27× |
| 16K | 3.99× | 3.71× | 4.54× |
| 32K | 4.42× | 4.03× | 4.84× |

32K 点的实际缓存长度为 **32640**，为输出预留 128 个位置。测量包含模型的 decode 执行，排除 prefill、TTFT 和首 token；图中 FA OOM 表示因容量不足跳过。各架构独立运行，应在各自配置内比较。[A100/B100 数据](https://github.com/vllm-project/vllm/pull/53060#issuecomment-5364963055)、[H100 数据](https://github.com/vllm-project/vllm/pull/53060#issuecomment-5352630554)

<a id="kvtc"></a>

## KVTC：存储压缩与流水化回载

<a id="page-09"></a>

### 跨层变换与混合位宽

![原图 09：KVTC 的压缩与恢复路径](09.jpg)

KVTC 以一个 token 为压缩单位，将跨层、跨 head 的特征拼接，经 PCA 去相关，再按主成分重要性分配位宽；K、V 分开处理。低位量化和舍弃分量产生误差，可选的 **DEFLATE** 只对量化后的字节串做无损压缩。[KVTC 论文](https://arxiv.org/html/2511.01815v1)

```text
压缩：跨层/head 特征 → PCA → 混合位宽量化 → 可选 DEFLATE
回载：CPU 压缩载荷 → GPU 可选 DEFLATE 解码 → 解量化、逆 PCA → 标准 KV 写入 HBM → attention
```

跨层压缩与逐层 attention 的读取粒度不同，且逆变换有成本，因此本讲将它用于存储和回载。报告压缩范围为 **8×–64×**；在 **16×** 下，三种模型、五项基准的最大评测分数下降不超过 **0.8 分**。

<a id="page-10"></a>

### 分摊逆变换，并重叠传输与解压

![原图 10：KVTC 的流水化解压与集成](10.jpg)

原方案在八卡上重复执行完整逆变换 GEMM。新方案沿 **M 维分片，每卡计算 M/8**，再 AllGather 得到完整 `[M,D]`，以通信换取减少重复计算。

稳态时，不同 chunk 的三个阶段交叠：

```text
下一块：H2D 传输
当前块：解量化 + TP-GEMM
上一块：AllGather
```

图中已有集成为 **vLLM + LMCache**；SGLang、TensorRT-LLM、HiCache、Dynamo 标为候选适配。

<a id="page-11"></a>
<a id="recap-transfer"></a>

### 单请求 TTFT

![原图 11：KVTC 单请求首 token 延迟](11.jpg)

条件：**GLM-5.1 FP8、8 张 Hopper、PCIe Gen5、TP=8、vLLM + LMCache，压缩比 4.7×**。KVTC 回载耗时已包含解压成本。

| 上下文 | 未压缩 TTFT | KVTC TTFT | 图中降幅 |
| --- | ---: | ---: | ---: |
| 32K | 204 ms | 182 ms | 10.8% |
| 64K | 407 ms | 377 ms | 7.4% |
| 128K | 591 ms | 558 ms | 5.5% |
| 192K | 843 ms | 773 ms | 8.4% |

例如 128K 的回载部分从 390 ms 降到 355 ms，后者已包含传输与解压成本。图中的灰色 “Prefill” 实际定义为 **TTFT − LMCache load latency**，包含回载之外的剩余耗时。32K 按柱顶的 10.8% 保留，原图副标题的 10.6% 与之不一致。

<a id="page-12"></a>

### 容量压力下保住缓存命中

![原图 12：KVTC 的主机容量实验](12.jpg)

该实验同样使用 GLM-5.1-FP8、八卡 Hopper、TP=8、vLLM + LMCache。在 **96 GiB 主机容量、并发 16** 时：

| 指标 | 未压缩 | KVTC | 变化 |
| --- | ---: | ---: | ---: |
| 输出吞吐 | 50.0 tok/s | 124.7 tok/s | 2.49× |
| LMCache 命中率 | 11.9% | 74.7% | +62.8 个百分点 |
| 平均 TTFT | 108.8 s | 39.1 s | 约 −64% |

压缩让更多 KV 留在缓存中，减少驱逐和未命中后的重新计算，因而避免了吞吐陡降。容量充足的 **400 GiB** 组，吞吐变化仅 **−2.1%～+3.6%**。

该页配置写 **CR=8×**，实际整体主机占用约缩小 **4.7×**；载荷压缩率与整体内存占用是两个口径。

<a id="nvcomp"></a>

## nvCOMP-ANS：无损压缩传输

<a id="page-13"></a>

### 实际载荷与固定槽位

![原图 13：ANS 的 GPU、CPU 与 SSD 数据路径](13.jpg)

GPU 对原始 KV 做 ANS 无损编码，CPU/SSD 保存压缩载荷及每个 chunk 的长度；回载后精确恢复编码前的 KV。批量 GPU codec 与双缓冲让编解码和传输交叠。

**这份 FlexKV 实现仍保留固定大小的 CPU/SSD 槽位，CPU 按未压缩大小预留空间。** 长度表决定实际传输多少字节，因此它减少的是有效读写和搬运量，不能按压缩率增加可分配的缓存槽位。[FlexKV PR #183](https://github.com/taco-project/FlexKV/pull/183)

<a id="page-14"></a>
<a id="recap-metrics"></a>

### 低并发 TTFT 改善，吞吐基本持平

![原图 14：ANS 压缩率与服务性能](14.jpg)

报告的无损压缩率为 **BF16 KV 约 1.4×、FP8 KV 约 1.1×**。服务测试使用 **SGLang + FlexKV（CPU 缓存）、GLM-5.1-FP8、BF16 KV、128K 输入、8 张 Hopper**。

| 并发数 | 平均 TTFT speedup | P90 TTFT speedup |
| --- | ---: | ---: |
| 1 | +5.7% | +16.9% |
| 2 | +8.2% | +18.7% |
| 3 | +1.8% | +4.2% |
| 4 | +3.2% | +3.0% |
| 8 | −5.6% | +2.1% |

这里 $\mathrm{speedup}=T_{\mathrm{baseline}}/T_{\mathrm{ANS}}-1$，所以 **+18.7% 对应延迟降低约 15.8%**。并发 8 的平均 TTFT 已变慢，P90 为 **+2.1% ± 21.9%**，波动较大；整个并发区间的输出吞吐仅变化 **−1.9%～+2.4%**。收益主要体现在部分低并发请求的首 token 等待。[基准数据](https://github.com/taco-project/FlexKV/pull/183)

<a id="comparison"></a>
<a id="page-15"></a>

## 三种方案的结果对照

![原图 15：讲座总结](15.jpg)

| 方案 | 主要改变 | 本讲观察到的效果 |
| --- | --- | --- |
| 优化后的 TurboQuant | HBM 中的压缩布局与 attention 内解码 | 长上下文 decode 加速 |
| KVTC | 后端缓存载荷、回载与解压流水线 | 容量紧张时保住命中率和吞吐 |
| nvCOMP-ANS | 无损减少实际传输字节 | 部分低并发 TTFT 改善，吞吐基本持平 |
