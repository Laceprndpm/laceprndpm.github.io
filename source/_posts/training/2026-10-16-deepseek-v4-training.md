---
title: Megatron 中的 DeepSeek-V4-Pro 训练优化
date: 2026-10-16
tags:
  - AI-Infra
  - CUDA
  - 分布式训练
  - MoE
  - 长上下文
categories:
  - AI-Infra
mathjax: true
---

NVIDIA 在 Megatron Core 中训练 DeepSeek-V4-Pro 的优化案例：64K 主要减少运行时开销，1M 主要优化注意力与 CP 负载均衡，累计吞吐分别提升 59% 和 32%。

根据 2026 年 10 月 16 日 NVIDIA 中国开发者日讲座整理。配置与性能数字均为讲座报告值。

<!-- more -->

# Megatron 中的 DeepSeek-V4-Pro 训练优化

- [配置与瓶颈](#configuration)
- [融合与 CUDA Graph](#runtime)
- [混合注意力的三个改动](#attention)
- [MoE 通信重叠与缓冲复用](#moe)
- [CP 的 zigzag 负载均衡](#cp)
- [性能结果与代码入口](#results)

<a id="page-01"></a>
## 概览：
![原图 01：总体收益](01.jpg)

<a id="configuration"></a>

## 配置与瓶颈

<a id="page-02"></a>

![原图 02：训练容量与性能目标](02.jpg)

<a id="page-03"></a>

![原图 03：DeepSeek-V4-Pro 模型概览](03.jpg)

模型为 **1.6T 总参数、每 token 激活 49B 参数**；实验使用 **256 张 Blackwell GPU**，以 **PP4、EP64** 为初始布局。PP 切分模型层，EP 切分专家，CP 切分序列；这些通信组可以使用同一批 GPU，不能再把 CP 数直接乘进设备总数。

<a id="page-04"></a>

![原图 04：模型特征与优化对象](04.jpg)

<a id="page-05"></a>

![原图 05：训练时间组成](05.jpg)

64K 中 MoE 占比明显；1M 中 **Indexer + CP communication** 成为最大的部分。因此，MoE 优化在 1M 的整步收益较小；CP 均衡的收益则来自后文展示的 rank 间失衡。图中各柱为归一化占比，4K 是稀疏代理配置。

<a id="page-06"></a>

![原图 06：并行配置](06.jpg)

| 上下文 | CP | 每 rank token 数 | 激活内存策略 |
| --- | ---: | ---: | --- |
| 64K | 16 | 4,096 | 细粒度重计算 + offload，为 CUDA Graph 留空间 |
| 1M | 64 | 16,384 | full recompute |

<a id="page-07"></a>

![原图 07：上下文长度与内存策略](07.jpg)

序列长度增为 16 倍，CP 只增为 4 倍，**每 rank 的 token 仍增为 4 倍**。因此，1M 配置采用更激进的重计算；两组实验的执行与显存策略并不相同。

<a id="runtime"></a>
<a id="page-08"></a>
## 融合与 CUDA Graph

![原图 08：融合与图重放前后的时间线](08.jpg)

64K 的时间线有大量短 kernel 和提交空隙，改动包括：

- 融合简单数学操作和数据搬运，减少 kernel 启动与中间结果。
- 对 packed sequences 做 padding，使捕获区域的形状固定，再重放静态模块。
- 减少 CSA 注意力路径保存的张量，并加入更细粒度的重计算与 offload，为 Graph 所需缓冲腾出显存。

Graph 减少重复提交的开销，融合还减少中间数据读写；讲座将这组改动一起测量。**1M 在这一步仅报告了部分融合的收益**，没有把它归因于 Graph。

<a id="page-09"></a>

![原图 09：STEP 1 的内存取舍与收益](09.jpg)

<a id="attention"></a>
<a id="page-10"></a>

## 混合注意力的三个改动

![原图 10：稀疏索引、Indexer 与 Group-O 投影](10.jpg)

### 压紧有效索引
这条公开路径省掉的是**矩形矩阵中的无效区域及其存储、处理开销**，并没有把所有中间评分都留在片上。候选缓冲仍随有效 QK 对数增长，长序列下仍是二次规模。
将每行索引中的有效项移到前部，并传入有效长度 $L$：

```text
原列表：[12, -1, 7, 40, -1, 3, 4, -1, 6]
有效前缀：[12, 7, 40, 3, 4, 6]，L = 6
```

稀疏 attention 只扫描前 $L$ 项，省去对无效槽位的遍历。这里压紧的是索引列表，KV 本体仍由索引寻址。

### 融合 Indexer 打分与 Top-K

图 10 中间主要做三件事：

- **少算**：跳过因果 mask 中无效的 QK 区域。
- **算快**：用 MXFP8 执行 QK 乘法。
- **少存**：只保存因果有效分数，去掉矩形矩阵中的无效区域。

公开实现仍是：

```
评分 kernel → 全部有效分数写入显存 → Top-K kernel
```

因此，**并非“单 kernel、只写最终 Top-K”**；仍有中间显存读写，分数缓冲仍是二次规模。

### 合并 Group-O 投影

将分组的小 GEMM 合成一次 batched GEMM，并启用 FP8。原图以八组投影为例：从八次独立启动改成一次批量执行。

<a id="moe"></a>
<a id="page-11"></a>

## MoE 通信重叠与缓冲复用

![原图 11：两种 EP 通信重叠方式](11.jpg)

讲座中 **EP 通信占 MoE 部分约 30%–50% 的时间**，给出两种重叠方式：

| 方式 | 重叠对象 | 讲座中的取舍 |
| --- | --- | --- |
| Cross-batch EP overlap | 一批的计算与另一批的通信 | 更适合跨 NVLink 域，但增加在途数据与显存占用 |
| 细粒度 megakernel | 同一算子内不同 chunk 的通信与计算 | NVLink 域内更易实现，也便于复用缓冲 |

<a id="page-12"></a>

![原图 12：Mixture-of-Kittens](12.jpg)

采用 Cursor 的 **Mixture-of-Kittens（MoK）**，将专家计算与通信融合，按 chunk 推进流水线。固定大小的 ring buffer 循环承载在途 token，减少缓冲占用，并避免动态分配所需的 CPU 同步；槽位在旧数据消费完成后才可覆盖。[MoK 代码](https://github.com/cursor/mixture-of-kittens)

讲座称 MoE 局部获得 **30%–40% speedup**；整步吞吐增益见末表。1M 的 MoE 占比较低，所以同类优化对整步的帮助较小。

<a id="cp"></a>
<a id="page-13"></a>

## CP 的 zigzag 负载均衡

![原图 13：512K、CP32 下各 rank 的耗时](13.jpg)

在 **512K、CP32** 的诊断实验中，CSA forward 从 rank 0 的 **18.4 ms** 增至 rank 31 的 **53.1 ms**，Indexer 打分也随 rank 增长。连续均分 token 后，后段 query 需要处理更长的有效历史，导致后面的 rank 更慢。

Zigzag 将序列分成 **2×CP 段**，让每个 rank 取得首尾对称的两段。CP=4 时分配为：

```text
rank0: 段 0 + 段 7
rank1: 段 1 + 段 6
rank2: 段 2 + 段 5
rank3: 段 3 + 段 4
```

早期 query 的轻工作与后期 query 的重工作配对，降低最慢 rank 的耗时。

用 **8 个 token、2 个 rank 的稠密因果注意力例子**看这个差别：位置 $i$ 有 $i$ 个有效键，连续分配的工作量为 $1+2+3+4=10$ 与 $5+6+7+8=26$；首尾配对后变成 $1+2+7+8=18$ 与 $3+4+5+6=18$。这说明分配原理，实际混合注意力的成本仍由各分支决定。

<a id="page-14"></a>

![原图 14：zigzag 布局与收益](14.jpg)

Zigzag 也增加重排与通信开销。讲座中它在 1M 有明显增益，在 64K 没有可测收益。

<a id="results"></a>
<a id="page-15"></a>

## 性能结果与代码入口

![原图 15：逐步优化后的累计吞吐](15.jpg)

下表中，**增量相对上一步配置，累计相对初始基线**。累计倍率按原图舍入值保留。

| 改动 | 64K 增量 | 64K 累计 | 1M 增量 | 1M 累计 |
| --- | ---: | ---: | ---: | ---: |
| 初始基线 | — | 1.00× | — | 1.00× |
| STEP 1：融合、Graph 与显存优化 | +32.5% | 1.33× | +2.8%（部分融合） | 1.03× |
| STEP 2：混合注意力 | 未单列 | 1.39× | 未单列 | 1.12× |
| STEP 3：MoE | +14.0% | 1.59× | +4.6% | 1.17× |
| STEP 4：CP 均衡 | 无可测收益 | 1.59× | +12.2% | 1.32× |

64K 最大的改善来自第一步；1M 最大的单步增量来自 CP 均衡。**这些数字比较的是 NVIDIA 这套实现优化前后，不是相对 DeepSeek 内部原始训练系统。**

公开实现可从 [Megatron-LM 的 V4 支持跟踪](https://github.com/NVIDIA/Megatron-LM/issues/4468)和 [Megatron Bridge 的 V4 训练示例](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/deepseek_v4)进入。讲座照片没有展开 rank group 映射、Graph 捕获边界、Indexer kernel 布局和 MoK 接入细节，这些仍需查代码。

<a id="page-16"></a>

![原图 16：讲座总结](16.jpg)
