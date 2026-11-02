---
title: FlashMLA 详解
date: 2026-10-31 19:13:46 +08:00
tags:
  - AI-Infra
  - CUDA
  - FlashMLA
  - MLA
  - Blackwell
categories:
  - 算子优化
mathjax: true
---

以 FlashMLA 的 Blackwell 稀疏 decode 实现为例，串联 MLA、分块 softmax、KV 搬运、反量化与矩阵计算，追踪内核的数据流和执行流水线。

<!-- more -->

# FlashMLA 详解

本文以 FlashMLA 的固定提交 [`15f13e5`](https://github.com/deepseek-ai/FlashMLA/tree/15f13e5030374295491c5ce31b02d7e63a7772c6) 为例，精读 Blackwell SM100 上的稀疏 decode 实现 [`csrc/sm100/decode/head64/kernel.cuh`](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/csrc/sm100/decode/head64/kernel.cuh)。讲解以 `V32` 分支为主，沿着 KV 搬运、反量化、矩阵乘法和 online softmax 逐步展开；相关类型、调度与分片合并代码在需要时补充引用。

## 1.前置知识

FlashMLA 的计算可以分成两个层面理解：**MLA 决定 attention 在什么空间里计算，分块 softmax 决定这些计算如何逐块完成。** Attention sink 则在最终归一化时，为注意力权重增加一个额外的去向。

本文先介绍 MLA 如何把计算转移到压缩后的 KV 空间，再推导分块 softmax、并行分片合并，以及 attention sink 的加入方式。

全文使用行向量记法：粗体小写字母表示向量，例如 $\mathbf q$、$\mathbf v$、$\mathbf o$；大写字母表示矩阵，例如 $Q$、$K$、$V$。$m$、$\ell$、$L$、$w$ 等表示标量。上标 ${}^\top$ 表示转置。公式按实数运算推导，实际 kernel 使用有限精度计算。

### 1.从普通 attention 开始

先考虑一个请求中的一个 query token，并固定一个 attention head。

设这个 query 需要关注 $N$ 个历史 token，query、key 和 value 的形状分别为：

$$
\mathbf q\in\mathbb R^{1\times d_k},
\qquad
K\in\mathbb R^{N\times d_k},
\qquad
V\in\mathbb R^{N\times d_v}
$$

第一步计算 query 与每个 key 的匹配分数：

$$
\mathbf s=\beta\mathbf qK^\top
\in\mathbb R^{1\times N}
$$

其中，$\beta$ 是 attention 的缩放系数。

第二步沿 token 方向执行 softmax：

$$
a_j=\frac{e^{s_j}}{\sum_{t=1}^{N}e^{s_t}}
$$

所有权重组成行向量：

$$
\mathbf a=[a_1,\ldots,a_N]\in\mathbb R^{1\times N}
$$

第三步，用这些权重对 value 求加权和：

$$
\boxed{
\mathbf o=\mathbf aV
=\sum_{j=1}^{N}a_j\mathbf v_j
\in\mathbb R^{1\times d_v}
}
$$

因此，attention 权重和 attention 输出是两个不同的对象：

$$
\underbrace{\mathbf a}_{1\times N}
\underbrace{V}_{N\times d_v}
=
\underbrace{\mathbf o}_{1\times d_v}
$$

即使只有一个 query 和两个历史 token，最终输出仍然是 $d_v$ 维向量；中间的注意力权重才是 $[1,2]$。

### 2.MLA 如何压缩 KV

普通多头 attention 为每个历史 token 保存各个 head 的 key 和 value。MLA 则先把输入压缩成一个较小的潜在向量，让多个 head 共享这份缓存。

设一个 token 的输入为：

$$
\mathbf h\in\mathbb R^{1\times d_{\text{model}}}
$$

为了突出投影关系，下面省略归一化操作。MLA 的 query 和 KV 分别经过两条压缩支路：

$$
\mathbf c^Q=\mathbf hW^{DQ}
\in\mathbb R^{1\times d_{cQ}}
$$

$$
\mathbf c^{KV}=\mathbf hW^{DKV}
\in\mathbb R^{1\times d_c}
$$

这里，$d_{cQ}$ 和 $d_c$ 是独立的模型参数。它们控制两条支路的压缩维度，并不要求相等。

以 DeepSeek-V3 的公开配置为例：

$$
d_{cQ}=1536,\qquad d_c=512
$$

也就是说，一个 token 对应的两个压缩向量分别为：

$$
\mathbf c^Q:[1,1536],
\qquad
\mathbf c^{KV}:[1,512]
$$

这些维度来自模型配置；FlashMLA kernel 接收到的是后续处理好的 query 和 KV cache。[官方配置](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/configs/config_671B.json)

接下来固定第 $h$ 个 head。它的内容 query、内容 key 和 value 可以写成：

$$
\mathbf q_h^C=\mathbf c^QW_h^{UQ}
$$

$$
\mathbf k_{j,h}^C=\mathbf c_j^{KV}W_h^{UK}
$$

$$
\mathbf v_{j,h}=\mathbf c_j^{KV}W_h^{UV}
$$

其中，上标 $C$ 表示内容部分；$W^{UK}$、$W^{UV}$ 分别把压缩后的 KV 展开成这个 head 的 key 和 value。

这给出了 MLA 的模型定义。不过，decode 时可以利用矩阵乘法的结合律，直接在压缩空间里完成主要计算。

### 3.把 key 的展开投影移到 query 一侧

先看内容部分的点积：

$$
\mathbf q_h^C(\mathbf k_{j,h}^C)^\top
$$

代入 key 的展开关系：

$$
\begin{aligned}
\mathbf q_h^C(\mathbf k_{j,h}^C)^\top
&=
\mathbf q_h^C
\left(\mathbf c_j^{KV}W_h^{UK}\right)^\top\\
&=
\mathbf q_h^C(W_h^{UK})^\top
(\mathbf c_j^{KV})^\top
\end{aligned}
$$

定义变换后的 query：

$$
\boxed{
\widetilde{\mathbf q}_h
=
\mathbf q_h^C(W_h^{UK})^\top
\in\mathbb R^{1\times d_c}
}
$$

于是，内容部分的分数就变成：

$$
\boxed{
\mathbf q_h^C(\mathbf k_{j,h}^C)^\top
=
\widetilde{\mathbf q}_h(\mathbf c_j^{KV})^\top
}
$$

这样，每个历史 token 只需要提供压缩向量 $\mathbf c_j^{KV}$。当前 query 经过一次变换后，就可以与整段历史缓存进行点积。

在本文讨论的 FlashMLA kernel 中，**这一步 query 变换已经由上游完成**。kernel 直接读取准备好的 query。

加入解耦的位置编码部分后，完整分数写为：

$$
s_{h,j}
=
\beta\left(
\widetilde{\mathbf q}_h(\mathbf c_j^{KV})^\top
+
\mathbf q_h^R(\mathbf k_j^R)^\top
\right)
$$

其中，$\mathbf q_h^R$ 和 $\mathbf k_j^R$ 是已经完成位置编码处理的 query、key 部分。

### 4.把 value 的展开投影移到加权求和之后

value 一侧也可以调整计算顺序：

$$
\begin{aligned}
\sum_j a_{h,j}\mathbf v_{j,h}
&=
\sum_j a_{h,j}
\left(\mathbf c_j^{KV}W_h^{UV}\right)\\
&=
\left(\sum_j a_{h,j}\mathbf c_j^{KV}\right)W_h^{UV}
\end{aligned}
$$

定义压缩空间中的 attention 输出：

$$
\boxed{
\mathbf o_h
=
\sum_j a_{h,j}\mathbf c_j^{KV}
\in\mathbb R^{1\times d_c}
}
$$

那么展开后的 value 加权结果就是：

$$
\mathbf o_hW_h^{UV}
$$

这解释了为什么同一份压缩 KV 会被使用两次：

- 作为 key，与变换后的 query 计算分数；
- 作为 value，被 attention 权重加权求和。

FlashMLA 在压缩空间中完成这两次计算，然后返回 $\mathbf o_h$。

后续还有 attention 的输出投影。把输出投影矩阵按 head 分块写成 $W_h^O$，整个 attention 的输出为：

$$
\mathbf y
=
\sum_h\mathbf o_hW_h^{UV}W_h^O
$$

因此，可以把这两个投影合并为：

$$
W_h^{\text{合并}}=W_h^{UV}W_h^O
$$

再计算：

$$
\mathbf y=\sum_h\mathbf o_hW_h^{\text{合并}}
$$

这里合并的是 attention 内部的线性投影：先在压缩空间完成加权求和，再回到模型的输出空间。MLA 的模型定义见 [DeepSeek-V2 技术报告](https://arxiv.org/html/2405.04434v5#S2.SS1.SSS2)。

### 5.分块 online softmax 保存什么

下面重新固定一个 head，用 $\mathbf v_j$ 表示 kernel 实际参与加权求和的 value 向量。

为了对应代码中的 `exp2`，把 attention 分数换成以 2 为底的形式：

$$
x_j=s_j\log_2e
$$

于是：

$$
2^{x_j}=e^{s_j}
$$

最终输出可以写成：

$$
\boxed{
\mathbf o
=
\frac{\displaystyle\sum_j2^{x_j}\mathbf v_j}
{\displaystyle\sum_j2^{x_j}}
}
$$

历史 token 被分成多个块，kernel 依次处理这些块。处理过程中，保存三个状态：

- 标量 $m$：指数计算使用的参考值；
- 标量 $\ell$：已经处理的 token 的指数权重之和；
- 向量 $\mathbf u$：已经处理的 token 的 value 加权和。

它们满足：

$$
\ell=\sum_{\text{已处理 }j}2^{x_j-m}
$$

$$
\mathbf u=
\sum_{\text{已处理 }j}2^{x_j-m}\mathbf v_j
$$

这里的 $\mathbf u$ 尚未归一化。所有块完成后，再计算：

$$
\mathbf o=\frac{\mathbf u}{\ell}
$$

在数学上，初始累积量为 $\ell=0$、$\mathbf u=\mathbf 0$；指定 kernel 将初始参考值设为 $m=-10^{30}$。

### 6.新块到来时，怎样更新已有结果

设新块包含的 token 集合为 $\mathcal B$。处理新块时，决定使用的新参考值记为 $m'$。

旧结果是以 $m$ 为参考计算的，现在要转换到 $m'$，因此定义缩放系数：

$$
\alpha=2^{m-m'}
$$

对于旧分母：

$$
\begin{aligned}
\alpha\ell
&=
2^{m-m'}
\sum_{\text{已处理 }j}2^{x_j-m}\\
&=
\sum_{\text{已处理 }j}2^{x_j-m'}
\end{aligned}
$$

加入新块后：

$$
\boxed{
\ell'
=
\alpha\ell+
\sum_{j\in\mathcal B}2^{x_j-m'}
}
$$

输出分子执行相同的尺度变换：

$$
\begin{aligned}
\alpha\mathbf u
&=
2^{m-m'}
\sum_{\text{已处理 }j}2^{x_j-m}\mathbf v_j\\
&=
\sum_{\text{已处理 }j}2^{x_j-m'}\mathbf v_j
\end{aligned}
$$

再加入当前块的贡献：

$$
\boxed{
\mathbf u'
=
\alpha\mathbf u+
\sum_{j\in\mathcal B}2^{x_j-m'}\mathbf v_j
}
$$

这两条式子就是 online softmax 的递推：

$$
\boxed{
\begin{aligned}
\ell'&=\alpha\ell+\text{当前块的指数权重之和}\\
\mathbf u'&=\alpha\mathbf u+\text{当前块的 value 加权和}
\end{aligned}
}
$$

其中，分母的更新是标量运算；输出分子的更新是向量运算。

将当前块的指数权重排列成行向量 $\mathbf p$，将 value 排列成矩阵 $V_{\mathcal B}$，当前块的加权和就是：

$$
\underbrace{\mathbf p}_{1\times|\mathcal B|}
\underbrace{V_{\mathcal B}}_{|\mathcal B|\times d_v}
=
\underbrace{\sum_{j\in\mathcal B}2^{x_j-m'}\mathbf v_j}_{1\times d_v}
$$

### 7.参考值为什么可以延迟更新

常见的 online softmax 每次选择：

$$
m'=\max\left(m,\max_{j\in\mathcal B}x_j\right)
$$

本文 kernel 则允许暂时保留旧参考值，以减少对输出累加器的缩放。

其依据是：只要分子和分母使用相同的参考值，最终相除时就会约掉这个尺度：

$$
\begin{aligned}
\frac{\mathbf u}{\ell}
&=
\frac{\sum_j2^{x_j-m}\mathbf v_j}
{\sum_j2^{x_j-m}}\\
&=
\frac{2^{-m}\sum_j2^{x_j}\mathbf v_j}
{2^{-m}\sum_j2^{x_j}}\\
&=
\frac{\sum_j2^{x_j}\mathbf v_j}
{\sum_j2^{x_j}}
\end{aligned}
$$

这份实现使用阈值 $6$：当当前块最大分数相对旧参考值的增长超过 $6$ 时，触发参考值更新。触发结果通过 warp 内投票共享。

未触发时：

$$
m'=m,\qquad \alpha=1
$$

此时允许未归一化指数权重大于 $1$，其上界对应：

$$
2^6=64
$$

触发后则更新参考值，并同时缩放旧分母与旧输出。阈值的直接作用，是用一定的指数动态范围换取更少的 TMEM 输出读改写；第 2 章第 10 节会把这个判断与线程分工对应起来。[online softmax 实现](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/csrc/sm100/decode/head64/kernel.cuh)

### 8.多个并行分片怎样合并

前面讨论的是一个计算单元顺序处理多个 token 块。为了增加并行度，同一个请求还可以沿 token 方向分成多个分片，每个分片内部继续执行 online softmax。

下面用 $r$ 表示分片编号，先推导各分片均含有效 token 的情形。每个分片使用参考值 $m_r$，计算：

$$
\ell_r=\sum_{j\in r}2^{x_j-m_r}
$$

分片输出已经在内部完成归一化：

$$
\mathbf o_r=
\frac{\displaystyle\sum_{j\in r}2^{x_j-m_r}\mathbf v_j}
{\ell_r}
$$

同时保存：

$$
L_r=m_r+\log_2\ell_r
$$

对这个式子取以 2 为底的指数：

$$
\begin{aligned}
2^{L_r}
&=2^{m_r+\log_2\ell_r}\\
&=2^{m_r}\ell_r\\
&=2^{m_r}\sum_{j\in r}2^{x_j-m_r}\\
&=\sum_{j\in r}2^{x_j}
\end{aligned}
$$

因此：

$$
\boxed{
2^{L_r}=\text{分片 }r\text{ 的指数权重总和}
}
$$

再乘以分片输出：

$$
\begin{aligned}
2^{L_r}\mathbf o_r
&=
2^{m_r}\ell_r
\cdot
\frac{\displaystyle\sum_{j\in r}2^{x_j-m_r}\mathbf v_j}
{\ell_r}\\
&=
\sum_{j\in r}2^{x_j}\mathbf v_j
\end{aligned}
$$

把所有分片的分子和分母分别相加，得到：

$$
\boxed{
\mathbf o=
\frac{\displaystyle\sum_r2^{L_r}\mathbf o_r}
{\displaystyle\sum_r2^{L_r}}
}
$$

这里，$\sum_{j\in r}$ 是分片内部对 token 求和，$\sum_r$ 是对所有分片求和。

合并 kernel 先取：

$$
M=\max_rL_r
$$

由于：

$$
\sum_r2^{L_r}
=
2^M\sum_r2^{L_r-M}
$$

对两边取 $\log_2$：

$$
\begin{aligned}
\log_2\left(\sum_r2^{L_r}\right)
&=
\log_2\left(2^M\sum_r2^{L_r-M}\right)\\
&=
M+\log_2\left(\sum_r2^{L_r-M}\right)
\end{aligned}
$$

因此，全局 LSE 为：

$$
\boxed{
L_{\text{总}}
=
M+\log_2\left(\sum_r2^{L_r-M}\right)
}
$$

这里加回 $M$，就是对乘数 $2^M$ 取对数的结果。它满足：

$$
2^{L_{\text{总}}}=\sum_r2^{L_r}
$$

代回输出：

$$
\begin{aligned}
\mathbf o
&=
\frac{\displaystyle\sum_r2^{L_r}\mathbf o_r}
{2^{L_{\text{总}}}}\\
&=
\sum_r2^{L_r-L_{\text{总}}}\mathbf o_r
\end{aligned}
$$

于是，每个分片的合并权重为：

$$
\boxed{
w_r=2^{L_r-L_{\text{总}}}
=
\frac{\displaystyle\sum_{j\in r}2^{x_j}}
{\displaystyle\sum_j2^{x_j}}
}
$$

它表示该分片占全体指数权重的比例。这个标量乘上局部输出向量后：

$$
\begin{aligned}
w_r\mathbf o_r
&=
\frac{\displaystyle\sum_{j\in r}2^{x_j}}
{\displaystyle\sum_j2^{x_j}}
\cdot
\frac{\displaystyle\sum_{j\in r}2^{x_j}\mathbf v_j}
{\displaystyle\sum_{j\in r}2^{x_j}}\\
&=
\frac{\displaystyle\sum_{j\in r}2^{x_j}\mathbf v_j}
{\displaystyle\sum_j2^{x_j}}
\end{aligned}
$$

最后对所有分片求和：

$$
\boxed{
\sum_rw_r\mathbf o_r
=
\frac{\displaystyle\sum_j2^{x_j}\mathbf v_j}
{\displaystyle\sum_j2^{x_j}}
=
\mathbf o
}
$$

这就是分片合并能恢复整体 attention 输出的原因。

### 9.Attention sink 怎样加入最终分母

固定一个 head，设传入的 sink 分数为标量 $a$。转换到以 2 为底的尺度：

$$
a_2=a\log_2e
$$

因此：

$$
2^{a_2}=e^a
$$

sink 可以看成一个分数为 $a$、value 为零向量的虚拟 token。它对分母贡献 $2^{a_2}$，对分子贡献为零。

加入 sink 后，目标输出变成：

$$
\boxed{
\mathbf o_{\text{sink}}
=
\frac{\displaystyle\sum_j2^{x_j}\mathbf v_j}
{\displaystyle\sum_j2^{x_j}+2^{a_2}}
}
$$

如果一个计算单元直接完成全部 token 的计算，最终归一化时使用：

$$
\boxed{
\mathbf o_{\text{sink}}
=
\frac{\mathbf u}{\ell+2^{a_2-m}}
}
$$

其中，sink 项也减去同一个参考值 $m$，使它与已有分母处于相同尺度。

多个分片并行计算时，各分片仍然先输出不含 sink 的 $\mathbf o_r$ 和 $L_r$。sink 在合并阶段加入一次：

$$
\mathbf o_{\text{sink}}
=
\frac{\displaystyle\sum_r2^{L_r}\mathbf o_r}
{\displaystyle\sum_r2^{L_r}+2^{a_2}}
$$

合并程序先按前面的流程得到：

$$
2^{L_{\text{总}}}=\sum_r2^{L_r}
$$

加入 sink 后的分母是：

$$
2^{L_{\text{总}}}+2^{a_2}
$$

将 $2^{L_{\text{总}}}$ 提出来：

$$
2^{L_{\text{总}}}+2^{a_2}
=
2^{L_{\text{总}}}
\left(1+2^{a_2-L_{\text{总}}}\right)
$$

再取 $\log_2$：

$$
\begin{aligned}
\log_2\left(2^{L_{\text{总}}}+2^{a_2}\right)
&=
\log_2\left[
2^{L_{\text{总}}}
\left(1+2^{a_2-L_{\text{总}}}\right)
\right]\\
&=
L_{\text{总}}
+\log_2\left(1+2^{a_2-L_{\text{总}}}\right)
\end{aligned}
$$

因此，包含 sink 的归一化量为：

$$
\boxed{
L_{\text{sink}}
=
L_{\text{总}}
+\log_2\left(1+2^{a_2-L_{\text{总}}}\right)
}
$$

它满足：

$$
2^{L_{\text{sink}}}
=
\sum_r2^{L_r}+2^{a_2}
$$

分片权重随之变为：

$$
\boxed{
w_r
=
2^{L_r-L_{\text{sink}}}
=
\frac{\displaystyle\sum_{j\in r}2^{x_j}}
{\displaystyle\sum_j2^{x_j}+2^{a_2}}
}
$$

乘回局部输出并对所有分片求和：

$$
\begin{aligned}
\sum_rw_r\mathbf o_r
&=
\sum_r
\frac{\displaystyle\sum_{j\in r}2^{x_j}\mathbf v_j}
{\displaystyle\sum_j2^{x_j}+2^{a_2}}\\
&=
\frac{\displaystyle\sum_j2^{x_j}\mathbf v_j}
{\displaystyle\sum_j2^{x_j}+2^{a_2}}
\end{aligned}
$$

所以：

$$
\boxed{
\mathbf o_{\text{sink}}=\sum_rw_r\mathbf o_r
}
$$

真实分片的权重之和此时为：

$$
\sum_rw_r
=
\frac{\displaystyle\sum_j2^{x_j}}
{\displaystyle\sum_j2^{x_j}+2^{a_2}}
$$

剩余权重属于 sink：

$$
w_{\text{sink}}
=
\frac{2^{a_2}}
{\displaystyle\sum_j2^{x_j}+2^{a_2}}
$$

两者相加仍然为 $1$。由于 sink 的 value 是零向量：

$$
\mathbf o_{\text{sink}}
=
\sum_rw_r\mathbf o_r+w_{\text{sink}}\mathbf 0
=
\sum_rw_r\mathbf o_r
$$

这对应仓库中的 [分片合并实现](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/csrc/smxx/decode/combine/combine.cu)。该实现对外返回的 LSE 使用自然对数、记录真实 token 的总量；包含 sink 的归一化量用于计算最终输出权重。

### 10.Attention sink 对输出起什么作用

令真实 token 的指数权重总量为：

$$
Z=\sum_j e^{s_j}
$$

没有 sink 时：

$$
\mathbf o=\frac{\sum_j e^{s_j}\mathbf v_j}{Z}
$$

加入 sink 后：

$$
\begin{aligned}
\mathbf o_{\text{sink}}
&=
\frac{\sum_j e^{s_j}\mathbf v_j}{Z+e^a}\\
&=
\frac{Z}{Z+e^a}\mathbf o
\end{aligned}
$$

因此：

$$
\boxed{
\mathbf o_{\text{sink}}
=
\frac{Z}{Z+e^a}\mathbf o
}
$$

sink 保持真实 token 之间的权重比例，同时减少分配给它们的总权重。

当真实 token 的总量 $Z$ 远大于 $e^a$ 时，输出基本保持原样；当 $Z$ 相对较小时，更多权重分给零 value 的 sink，这个 head 的输出就会被压低。

从行为上看，sink 给一个 head 增加了“少输出一些”的自由度。普通 softmax 即使面对一组都不匹配的 token，也必须把权重总和归一化为 1；加入零 value 的 sink 后，一部分权重可以不落在真实 token 上。

例如两个真实 token 的分数都是 $0$，则 $Z=2$。如果 $a=\ln 2$，sink 的指数权重也是 $2$，两个 token 的权重便从各 $1/2$ 变成各 $1/4$，输出恰好缩小一半。如果真实 token 的分数一起提高，分给 sink 的比例就会下降。因此，它保留了 token 之间的相对选择，同时使 head 的输出幅度能反映真实分数相对 sink 分数的大小。

对 kernel 而言，这个功能只需要一个逐 head 的 sink 标量，以及最终归一化时增加的一项分母；不需要实际读取一个虚拟 KV token。

## 2.代码详解

本章讨论 decode 阶段，以每个请求本轮只有一个 query token，即 `s_q = 1` 为例。这个 query 同时包含 64 个 attention head；64 个 head 构成矩阵的行，稀疏选中的 KV token 构成分块遍历的方向。

为避免混淆，后文统一使用以下称呼：

| 名称 | 本文含义 |
| --- | --- |
| CTA / 线程块 | 一组 384 个 CUDA 线程 |
| KV 数据块 | 一轮计算处理的 64 个候选 KV token，对应 `block_idx` |
| KV page | cache 中的一页，容量由 `page_block_size` 指定 |
| split / 分片 | 同一请求分配给某个 CTA 的连续若干 KV 数据块 |
| partition / 调度分区 | 一个 CTA 的完整工作清单，可以跨越多个请求 |

下面沿 `V32` 分支展开，在布局或同步条件发生变化的位置对照 `MODEL1`。

### 1.本文所讨论 kernel 的维度

上面的压缩空间计算，对应本文源码中 `V32` 分支的典型形式：

$$
Q=[\widetilde Q,\ Q_R],
\qquad
K=[C,\ K_R],
\qquad
V=C
$$

同一请求、同一 query token 的 64 个 head 可以按行排列，得到：

$$
Q\in\mathbb R^{64\times576}
$$

其中，576 维由 512 维内容部分和 64 维位置部分组成。对应的 value 和每个 head 的输出都是 512 维。

指定版本还包含 `MODEL1` 分支，其数据布局如下：

| 配置               | `V32` |                         `MODEL1` |
| ------------------ | ----: | -------------------------------: |
| Query / Key 总维度 |   576 |                              512 |
| NoPE 部分          |   512 |                              448 |
| RoPE 部分          |    64 |                               64 |
| Value 维度         |   512 |                              512 |
| Value 使用的数据   |  NoPE | NoPE 与 RoPE 合起来的全部 512 维 |

因此，`V32` 的位置部分通过打分影响权重；`MODEL1` 的位置部分还会进入 value 加权求和。后面推导的 softmax 和分片合并过程对这两种布局都适用。[维度配置](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/csrc/sm100/decode/head64/config.h)

这个 kernel 接收已经选好的稀疏 token 索引。下文中的“全部 token”，指本次参与 attention 的有效 token；分块与合并都在这组 token 上进行。

### 2.代码中每个warp的具体工作

先给出后续会用到的分工表。一个线程块有 384 个线程，分成三个 warpgroup；下面的 warp 编号是线程块内的编号：

| Warpgroup | Warp | 主循环职责                                                   |
| --------- | ---- | ------------------------------------------------------------ |
| WG0       | 0–3  | 读取 QK 结果、合并部分点积、掩码、指数与分母更新、缩放旧输出、最终写回 |
| WG1       | 4    | 发起 Q 的加载和 SMEM→TMEM 复制；发射 QK 与 SV 矩阵乘法       |
| WG1       | 5    | 发起 NoPE 数据的 TMA gather                                  |
| WG1       | 6    | 发起 RoPE 数据的 TMA gather                                  |
| WG1       | 7    | 读取稀疏索引和 scale，生成搬运坐标与有效性掩码               |
| WG2       | 8–11 | 将 SMEM 中的 FP8 NoPE 读入寄存器，反量化后写回 SMEM          |

Warp 4、5、6 的这些发射工作，各自由其中选出的一个线程执行；数据搬运和矩阵乘法由相应硬件异步执行。

数据沿下面的路径流动：

```
Q：
GMEM → SMEM → TMEM
                 └─ 在当前分片的多个 KV 块之间复用

NoPE：
GMEM → SMEM 中的 FP8 缓冲
     → 寄存器反量化
     → SMEM 中的 BF16 缓冲
                 ├─ 供 QK 使用
                 └─ 供 SV 使用

RoPE：
GMEM → SMEM
       └─ 供 QK 使用；MODEL1 中也供 SV 使用

QK 部分点积：
TMEM → WG0 寄存器
     → 经 SMEM 交换、合并
     → 在寄存器中计算指数权重
     → 将 BF16 权重写入 SMEM

SV：
SMEM 中的权重 × SMEM 中的 V
     → 累加到 TMEM 中的输出

最终输出：
TMEM → 寄存器归一化 → SMEM → GMEM
```

下面用一次完整的计算，把这几条路径连起来。固定为 V32：**一个 CTA 负责一个请求的一个 query token，包含 64 个 Q head，本次关注 70 个有效 KV token。** 使用主 KV cache，`topk=128` 表示索引数组预留 128 个位置，`topk_length=70` 表示其中前 70 项有效；当前 CTA 完成全部计算，直接写最终输出。这个例子不使用 extra cache 和 attention sink。

我们始终跟踪两个对象：**head 0 的 query，以及候选序号 69 对应的 KV token**。编号从 0 开始，所以候选 69 是第 70 个有效候选。设 `indices[69]=913`，即它实际指向 cache 中的 token 913。其余候选也都给出了有效的 token ID。

**① 先确定 tile：64 个 head 同时看 64 个 KV token。**

Q 的整体形状为 `[64,576]`，每一行是一个 head；每个 KV token 提供 512 维 latent 和 64 维 RoPE。70 个候选沿 token 方向分成两块：

| KV 数据块 | 对应的候选位置 | 有效位置 | KV 槽位 `buf_idx` | 辅助槽位 `index_buf_idx` | 两类 phase |
| --- | --- | --- | ---: | ---: | ---: |
| 第 0 块 | 0–63 | 全部 64 个 | 0 | 0 | 均为 0 |
| 第 1 块 | 64–127 | 前 6 个，即候选 64–69 | 1 | 1 | 均为 0 |

每块执行的逻辑矩阵乘法是：

```text
QK：Q [64 head,576 feature] × K块ᵀ [576 feature,64 token]
                                      ↓
                              分数 [64 head,64 token]

SV：S [64 head,64 token] × V块 [64 token,512 feature]
                                      ↓
                              输出贡献 [64 head,512 feature]
```

head 0 对应分数矩阵的第 0 行；token 913 位于第二块的第 `69-64=5` 行，对应这次 QK 结果的第 5 列。我们关注的点积就是第二块的 `(head=0, token=5)`，但硬件会随整个 tile 一起计算它。

![任务范围与两块 KV](01_scope_map.gif)

**② 把 Q 放到 TMEM，供两块 KV 共同使用。**

warp4 先发起 Q 的 GMEM→SMEM 搬运：前 512 维进入 `plan.u.qo.q`，后 64 维进入 `q_sw64`。head 0 的 576 个 BF16 元素也随这两个部分进入共享内存。

等 Q 搬入完成，warp4 再把它从 SMEM 复制到 TMEM。为了让后面的双路 GEMM 使用上下两半 datapath，512 维 NoPE 按 64 维切成八段：段 0、2、4、6 放在上一路，段 1、3、5、7 放在下一路；64 维 RoPE 则按两个 32 维段分到两路。head 0 的 query 因而分布在 TMEM 的上下两半，仍然表示同一个 head。

复制完成后，`bar_q_utccp` 放行。接下来两块 KV 都直接读取 TMEM 中的 Q；SMEM 中原先保存 Q 的空间可以交给 KV 使用。当前实现的 warp5、warp6 都在这个放行点之后才开始搬 KV。

![Q 的搬运与同址复用](02_q_copy_map.gif)

![TMEM 列布局](02b_tmem_column_layout.png)

**③ 第一块先算起来，第二块进入另一个缓冲槽。**

对第 0 块，warp7 先把索引转成搬运坐标，读取 scale，并生成有效性位图，写入辅助槽位 0。它的 32 个线程各自完成后通知 `bar_valid_coord_scale_ready[0]`。消费者的 `.wait(0)` 随之通过，开始使用这批辅助信息。

warp5 将 64 个 token 的 FP8 NoPE 搬入 `raw_nope[0]`，warp6 将 BF16 RoPE 搬入 `dequant[0].rope`。WG2 等 FP8 就绪后反量化，写入 `dequant[0].nope`。这样，槽位 0 中形成了供 QK 使用的 `[64,576]` BF16 K，以及供 SV 使用的 `[64,512]` BF16 V；V 就是这份 K 的 NoPE 部分。

warp4 用 TMEM 中的 Q 和 SMEM 中的 K 计算 QK，WG0 接过分数计算指数权重，warp4 再执行 SV。第一块结束时，TMEM 中的 O 保存了前 64 个 token 的未归一化加权和，WG0 的寄存器保存对应的参考值和分母。

与此同时，准备后续数据的角色可以沿着槽位 1 推进第二块。这使第二块的搬运、反量化有机会与第一块的计算重叠；各条流水线通过 ready/free 屏障控制进度。这里从槽位 0 换到槽位 1 时，phase 仍为 0：换的是缓冲位置，还没有绕回同一个槽位的新一轮。

![KV 双缓冲与 SMEM 位置复用](03_double_buffer_map.gif)

**④ 索引决定读谁，mask 决定谁参与计算。**

warp7 将选中的 token 索引转换为搬运坐标，读取 scale 并生成有效性 mask。候选 69 指向 token 913，放入尾块的第 5 行；warp5、6 随后发起搬运，WG2 完成 NoPE 反量化。尾块只有前 6 行有效，其余位置在计算权重时由 WG0 屏蔽。

**⑤ 从分数生成权重，再更新累计结果。**

warp4 发射 QK，Tensor Core 读取 TMEM 中的 Q 和 SMEM 中的 K，将部分点积 P 写回 TMEM。WG0 读出并合并分数、屏蔽无效位置，生成写入 SMEM 的权重 S，同时更新寄存器中的分母。**此时的 S 尚未归一化。**

加入下一块时，WG0 必要时同步缩放旧 O 和分母，使新旧贡献处于同一尺度。S 就绪且旧 O 处理完成后，warp4 发射 SV；Tensor Core 读取 SMEM 中的 S 和 V，将当前块的贡献累加到 TMEM 的 O。**O 在这里仍是未归一化的累计分子。**

**⑥ 最后一次 SV 完成后，统一归一化并写回。**

WG0 等待最后一次 SV 完成，从 TMEM 读取 O，除以合并后的总分母，转成 BF16 写入 SMEM 输出暂存区。warp0、warp1 随后发射 TMA，将结果写回 GMEM。最终输出仍为 `[64,512]`：尾块的 6 个有效 token 改变参与计算的贡献数量，不改变输出形状。[内核实现](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/csrc/sm100/decode/head64/kernel.cuh)

![FlashMLA 整体数据流](04_full_flow.gif)

后文沿这个实例的顺序展开源码：先看 warp4 将 Q 搬到 TMEM，再说明任务参数与缓冲布局，随后依次介绍索引准备、KV 搬运、反量化、QK、softmax、SV，以及最终归一化写回。实际执行时，各角色通过屏障协作并行推进。

TMEM 在 CTA 启动时一次申请 512 列，V32 中的固定分区如下。一列跨 128 个 datapath，每个位置为 32 bit；BF16 数据可以在同一位置中打包两个元素。

| TMEM 列区间（左闭右开） | 内容             | 有效期                |
| -------------- | -------------- | ------------------ |
| `[0,256)`      | FP32 输出累加器 O   | 当前请求分片的全部 KV 块     |
| `[256,384)`    | Q 的 512 维 NoPE | 当前请求分片的全部 KV 块     |
| `[384,400)`    | Q 的 64 维 RoPE  | 当前请求分片的全部 KV 块     |
| `[400,464)`    | 双路 QK 的部分点积 P  | 当前 KV 块，随后由 WG0 读取 |
| `[464,512)`    | 未用于上述张量的列      | 属于本次分配的容量          |

例如 Q 占用 `64 × 576 × 2 = 73728 B`，等价于 `144 × 128 × 4 B`；O 占用 `64 × 512 × 4 = 131072 B`，等价于 256 列。这里的 TMEM 是 Tensor Core 使用的独立存储，不计入后面计算的共享内存容量。

### 3.warp4：搬运 Q（GMEM → SMEM → TMEM）

warp4 中选出的线程负责提交 Q 的搬运、QK 和 SV 指令。QK 的结果留在 TMEM 中交给 WG0；WG0 生成 S 并完成旧 O 的缩放后，warp4 再提交 SV。它主要管理地址描述符和依赖关系，大矩阵的数据并不装进这个发射线程的寄存器。

Q 在整个分片内保持不变，先从 GMEM 搬到 SMEM，再复制到 TMEM，供每个 KV 块重复使用。本节先跟踪这条搬运路径；代码中的任务参数与共享内存布局在下一节集中说明。

```cpp
{
    Tensor gQ =
        tma_params.tma_Q_SW128.get_tma_tensor(tma_params.shape_Q_SW128)(_, _, s_q_idx, args.batch_idx);
    Tensor sQ = make_tensor(make_smem_ptr(plan.u.qo.q.data()), SmemLayoutQ_SW128{});
    ku::launch_tma_copy(tma_params.tma_Q_SW128, gQ, sQ, plan.bar_q_tma, TMA::CacheHintSm90::EVICT_FIRST);
}
if constexpr (D_Q_SW64 > 0) {
    cute::SM90_TMA_LOAD_5D::copy(&tma_params.tensor_map_q_sw64, (uint64_t *)&plan.bar_q_tma,
                                 (uint64_t)TMA::CacheHintSm90::EVICT_FIRST, plan.u.qo.q_sw64, 0, 0, 0,
                                 s_q_idx, args.batch_idx);
}
plan.bar_q_tma.arrive_and_expect_tx(B_H * D_Q * sizeof(bf16));
plan.bar_q_tma.wait(args.bar_phase_batch_rel);
ku::tcgen05_after_thread_sync();
```

`get_tma_tensor` 返回按 `[head, feature, query_position, request]` 编排的逻辑视图，其物理地址仍由 `params.stride_q_*` 决定。`(_, _, s_q_idx, args.batch_idx)` 固定后两维，取得当前请求、当前 query 的 Q。

`launch_tma_copy` 的第一个参数是包含 tensor map 和拷贝规则的 TMA 对象；`gQ`、`sQ` 分别提供源、目标视图。它先搬前 512 维。V32 再通过 5D tensor map 搬 RoPE，box 为 `[32,64,2,1,1]`：32 个 BF16、64 个 head、两个 RoPE 分段，以及各一个 query 和请求。

两部分共享 `bar_q_tma`，预期字节数为 `64 × 576 × 2 = 73728 B`。等待结束后，`tcgen05_after_thread_sync()` 建立线程同步与后续 tcgen05 操作之间的顺序，再发起 SMEM→TMEM 复制。

```cpp
{
    UMMA::SmemDescriptor sQ_desc = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(make_smem_ptr(plan.u.qo.q.data()),
                    tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<B_H * 2>, Int<64>>{}
                                  // *2 to leverage dual GEMM
                                  )));
    static_assert(D_Q_SW128 % 128 == 0);
    CUTE_UNROLL
    for (int tile_idx = 0; tile_idx < D_Q_SW128 / 128; ++tile_idx) {
        // Each tile: 64 x (64*2) logically, 128 x 64 bf16 on TMEM
        CUTE_UNROLL
        for (int subtile_idx = 0; subtile_idx < 64 / 16; ++subtile_idx) {
            // Each subtile: 64 x (16*2) logically, 128 x 16 bf16 (128dp256b) on TMEM
            SM100_UTCCP_128dp256bit_1cta::copy(sQ_desc +
                                                   (tile_idx * (B_H * 128) + subtile_idx * 16) * 2 / 16,
                                               tmem_cols::Q + tile_idx * 32 + subtile_idx * 8);
        }
    }
}
// Issue Q (SW64) UTCCP
if constexpr (D_Q_SW64 > 0) {
    UMMA::SmemDescriptor sQ_SW64_desc = UMMA::make_umma_desc<UMMA::Major::K>(
        make_tensor(make_smem_ptr(plan.u.qo.q_sw64),
                    tile_to_shape(UMMA::Layout_K_SW64_Atom<bf16>{}, Shape<Int<B_H * 2>, Int<32>>{}
                                  // *2 to leverage dual GEMM
                                  )));
    static_assert(D_Q_SW64 % 64 == 0);
    CUTE_UNROLL
    for (int tile_idx = 0; tile_idx < D_Q_SW64 / 64; ++tile_idx) {
        // Each tile: 64 x (32*2) logically, 128 x 32 bf16 on TMEM
        CUTE_UNROLL
        for (int subtile_idx = 0; subtile_idx < 32 / 16; ++subtile_idx) {
            // Each subtile: 64 x (16*2) logically, 128 x 16 bf16 (128dp256b) on TMEM
            SM100_UTCCP_128dp256bit_1cta::copy(
                sQ_SW64_desc + (tile_idx * (B_H * 64) + subtile_idx * 16) * 2 / 16,
                tmem_cols::Q + (B_H * D_Q_SW128 / 2 / 128) + tile_idx * 16 + subtile_idx * 8);
        }
    }
}
ku::umma_arrive_noelect(plan.bar_q_utccp);
```

这次复制还为 dual GEMM 安排了 Q 的行映射。把 NoPE 的 512 维按 64 维切成 8 段，记作 q0 到 q7。每对相邻的 `[64 head,64 feature]` 在 SMEM 中相邻，描述符把它们解释成 `[128,64]`：上 64 行对应偶数段，下 64 行对应奇数段。

```text
原始逻辑列： q0  q1  q2  q3  q4  q5  q6  q7
TMEM 上半： q0      q2      q4      q6
TMEM 下半：     q1      q3      q5      q7
```

这个安排把特征维的两路部分数据放到不同 datapath，不是把整个 Q 做转置。RoPE 用同样的方法把两个 32 维段分到上下两半。

每条 `SM100_UTCCP_128dp256bit_1cta::copy` 复制 128 个 datapath 上各 256 bit，也就是 `[128,16]` 个 BF16。因此一个 `[128,64]` 块发起四条复制，512 维 NoPE 总共发起 `4 × 4 = 16` 条；RoPE 再发起两条。

SMEM 描述符的地址增量以 16 B 为单位，所以元素偏移先乘 `sizeof(bf16)=2`，再除以 16。TMEM 每个 datapath 的一列宽 4 B，16 个 BF16 占 8 列，所以 `subtile_idx` 每增加 1，目标 TMEM 地址增加 8。

`umma_arrive_noelect(bar_q_utccp)` 将前面的异步复制与完成屏障关联起来。等待它完成以后，warp4 可以用 TMEM 中的 Q，warp5、6 以及 WG2 可以复用原先的 Q 暂存区。

### 4.参数详解

Q 的搬运路径中已经用到了当前请求的参数和共享内存地址，下面说明它们从哪里来。先看 CTA 如何取得任务。这是一个按调度元数据持续处理任务的 kernel：CTA 申请一次共享内存和 TMEM，在内部循环处理分配到的请求及分片。

启动配置是 `grid = (params.s_q, params.num_sm_parts, 1)`。`blockIdx.x` 选择 query 位置，`blockIdx.y` 选择调度分区；后者不等于 batch 大小。本文取 `s_q = 1`，所以 x 方向只有一个 CTA。共享内存和 TMEM 的大额分配使它按每 SM 一个 CTA 的资源规模组织，但源码中的任务编号仍然是 CTA 的 partition 编号，不是硬件 SM ID。

```cpp
struct __align__(4 * 8) DecodingSchedMeta {
    int begin_req_idx, end_req_idx;     // Both inclusive
    int begin_block_idx, end_block_idx; // Inclusive, exclusive
    int begin_split_idx;
    int is_first_req_splitted, is_last_req_splitted;
    int _pad[1];
};
```

`DecodingSchedMeta` 是某个 partition 的任务边界。调度器先将所有请求的 KV 数据块按请求顺序串起来，再切成若干段。一个 partition 可以包含前一个请求的尾部、中间若干完整请求，以及后一个请求的头部。

```cpp
partition_idx = blockIdx.y;

sched_meta = tile_scheduler_metadata_ptr[partition_idx];
```

各字段的含义如下：

| 字段 | 含义 |
| --- | --- |
| `begin_req_idx, end_req_idx` | 第一个与最后一个请求编号，二者都包含 |
| `begin_block_idx` | 第一个请求从哪个 KV 数据块开始 |
| `end_block_idx` | 最后一个请求处理到哪个 KV 数据块之前 |
| `begin_split_idx` | 第一个请求在当前 CTA 上的局部分片编号 |
| `is_first_req_splitted` | 第一个请求是否还由其他 CTA 共同完成 |
| `is_last_req_splitted` | 最后一个请求是否还由其他 CTA 共同完成 |
| `_pad[1]` | 补足 32 B，配合 32 B 对齐与整条元数据的向量读取 |

中间请求完全落在当前 partition 内，无须再存单独的起止块号和 split 标志。

`num_splits_ptr` 虽然名字带有“数量”，实际保存的是分片数量的前缀和，长度为 `batch_size + 1`。例如 A、B、C 分别有 2、1、2 个分片，前缀和为 `[0,2,3,5]`：A 对应全局槽位 `[0,2)`，B 对应 `[2,3)`，C 对应 `[3,5)`。当前片段的槽位为 `prefix[request] + local_split_idx`。

假设 CTA0 处理 A 的第一片，CTA1 处理 A 的第二片、整个 B 和 C 的第一片，CTA2 处理 C 的第二片。那么 CTA1 的 `begin_split_idx = 1`，A 写槽位 1；CTA2 的 C 写槽位 `3 + 1 = 4`。B 虽然在前缀和中占一个编号，却走直接输出路径，不必实际写入分片缓冲；combine 发现它只有一个 split 后也会跳过。

这解释了为什么只有 `begin_split_idx`：对于同一 CTA 后续遇到的请求，当前 CTA 都从该请求的第 0 片开始处理。调度逻辑见 [元数据生成实现](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu)。

```cpp
struct MainLoopArgs {
    int batch_idx, start_block_idx, end_block_idx;
    bool is_no_split;
    int n_split_idx;
    bool bar_phase_batch_rel;
    int topk_length, extra_topk_length, num_orig_kv_blocks;
    bool is_last_batch;
};
```

`MainLoopArgs` 把上述跨请求边界展开成“当前这个请求”的参数：

| 字段 | 当前请求中的含义 |
| --- | --- |
| `batch_idx` | 请求编号 |
| `[start_block_idx, end_block_idx)` | 当前 CTA 实际处理的 KV 数据块范围 |
| `is_no_split` | 当前 CTA 是否独立完成这个请求 |
| `n_split_idx` | 分片输出在全局缓冲中的槽位 |
| `bar_phase_batch_rel` | 每个请求使用一次的屏障所对应的 phase |
| `topk_length, extra_topk_length` | 主 cache、extra cache 各自有效的候选索引数量 |
| `num_orig_kv_blocks` | 主 cache 占多少个 KV 数据块，也就是 extra 块的起点 |
| `is_last_batch` | 是否为当前 CTA 工作清单中的最后一个请求 |

`topk_length` 限定索引数组中有效的前缀，其中仍可能存在值为 `-1` 的无效项。主 cache 和 extra cache 分别补齐到 64 个 token 一块，再拼接它们的块序列。例如主 cache 有 70 项、extra 有 20 项，总共是 `2 + 1 = 3` 块，而不是把 90 项混装成两块。主 cache 即使长度为 0，也保留一个全无效块，让主循环正常执行初始化和收尾。

`is_last_batch` 在 WG0 收尾时控制 `cudaTriggerProgrammaticLaunchCompletion()` 的调用，向后续 kernel 发出可以提前启动的信号。后续 combine 通过 `cudaLaunchKernelEx` 和 `cudaLaunchAttributeProgrammaticStreamSerialization` 开启 PDL，可提前处理索引、形状与地址等准备工作，再通过 `cudaGridDependencySynchronize()` 等待主 kernel 完成并使结果可见，之后读取分片输出。主 kernel 使用普通启动语法，并不阻止它作为 PDL 的前驱；其启动处的“不要使用 PDL”注释不能据此扩展为整条主 kernel → combine 链路禁用 PDL。实际能否重叠仍由资源和调度决定。参见 [combine 的同步与启动配置](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/csrc/smxx/decode/combine/combine.cu)。

```cpp
auto run_main_loop = [&](auto f) {
    // NOTE Putting the following code outside the warpgroup specialization switch results in register spilling.
    // [[maybe_unused]] int begin_req_idx, end_req_idx, sched_begin_block_idx, sched_end_block_idx, begin_n_split_idx, is_first_req_splitted, is_last_req_splitted;
    DecodingSchedMeta sched_meta;
    KU_LDG_256(params.tile_scheduler_metadata_ptr + partition_idx, &sched_meta, ".nc", "no_allocate",
               "evict_normal", "256B");

    if (sched_meta.begin_req_idx >= params.b) {
        return;
    }

    bool bar_phase_batch_rel = 0;
#pragma unroll 1
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx;
         ++batch_idx, bar_phase_batch_rel ^= 1) {
        int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
        int orig_topk_padded = max(ku::ceil(topk_length, (int)B_TOPK), (int)B_TOPK);
        int extra_topk_length =
            params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
        int total_topk_padded =
            orig_topk_padded + ku::ceil(extra_topk_length, (int)B_TOPK); // % B_TOPK == 0
        int start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        int end_block_idx =
            batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / B_TOPK;
        bool is_split =
            batch_idx == sched_meta.begin_req_idx
                ? sched_meta.is_first_req_splitted
                : (batch_idx == sched_meta.end_req_idx ? sched_meta.is_last_req_splitted : false);
        int n_split_idx = batch_idx == sched_meta.begin_req_idx
                              ? (__ldg(params.num_splits_ptr + batch_idx) + sched_meta.begin_split_idx)
                              : __ldg(params.num_splits_ptr + batch_idx);

        MainLoopArgs args = {batch_idx,
                             start_block_idx,
                             end_block_idx,
                             !is_split,
                             n_split_idx,
                             bar_phase_batch_rel,
                             topk_length,
                             extra_topk_length,
                             orig_topk_padded / B_TOPK,
                             batch_idx == sched_meta.end_req_idx};

        f(args);
        NamedBarrier(NUM_THREADS, NamedBarriers::everyone_sync).arrive_and_wait_unaligned();
    }
};
```

`run_main_loop` 在每个专职角色内部调用，所有角色读取同一份调度清单，各自执行 `f(args)`。这样既保证各条流水线处理同一组请求，也把调度变量的存活范围留在专职分支内部；源码注释指出，将它们提到分支外会导致 register spilling。

`KU_LDG_256` 一次读取 32 B 的调度元数据，参数的含义在第 3 章第 3 节展开。若 `begin_req_idx >= params.b`，当前 partition 没有任务，直接退出。

循环中需要注意四处边界：

1. 有逐请求长度指针时读取该指针，否则使用统一的 `params.topk` 或 `params.extra_topk`。
2. `ku::ceil(n,64)` 在这里表示向上补到 64 的倍数；除以 64 后才是块数。
3. 只有首尾请求需要裁剪起止块号，中间请求使用完整范围。若首尾是同一个请求，调度器已经统一了两个 split 标志。
4. 每轮末尾的 `everyone_sync` 要求 CTA 的 384 个线程全部到达，之后才能进入下一个请求。warp4、5、6 中未被选为发射者的线程，也会走空的 `run_main_loop` 分支参与这个同步。

`bar_phase_batch_rel` 每完成一个请求翻转一次，对应 Q 搬入、Q 复制到 TMEM、上一请求输出源缓冲释放等一次性事件。逐 KV 块的双缓冲与四槽辅助缓冲则由 `RingState` 跟踪。

```cpp
struct RingState {
    int buf_idx = 0;
    bool bar_phase = 0;
    int index_buf_idx = 0;
    bool index_bar_phase = 0;
    CUTE_DEVICE void update() {
        bar_phase ^= (buf_idx == NUM_BUFS - 1);
        buf_idx = (buf_idx + 1) % NUM_BUFS;
        index_bar_phase ^= (index_buf_idx == NUM_INDEX_BUFS - 1);
        index_buf_idx = (index_buf_idx + 1) % NUM_INDEX_BUFS;
    }
};
```

`buf_idx` 选择两组 KV 缓冲，`index_buf_idx` 选择四组坐标、scale 和有效性标记缓冲。phase 只在对应环形缓冲绕回时翻转：

| 连续处理的块序号 | `buf_idx` | `bar_phase` | `index_buf_idx` | `index_bar_phase` |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 1 | 0 |
| 2 | 0 | 1 | 2 | 0 |
| 3 | 1 | 1 | 3 | 0 |
| 4 | 0 | 0 | 0 | 1 |

每个角色有自己的 `rs`，可以以不同速度推进；ready/free 屏障限制生产者和消费者之间的距离。`rs` 建在请求循环外，因此跨请求继续使用当前槽位和 phase，而不是在每个请求开始时重新归零。

接下来是共享内存的布局定义：

```cpp
template <int NUM_TILES>
using SmemLayoutQTiles =
    decltype(coalesce(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{},
                                    Shape<Int<B_H>, Int<NUM_TILES * 64>>{}, Step<_1, _2>{}),
                      Shape<_1, _1>{}));
using SmemLayoutQ_SW128 = SmemLayoutQTiles<D_Q_SW128 / 64>;
using SmemLayoutOBuf =
    decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<B_H>, Int<D_V>>{}));
using SmemLayoutOAccumBuf =
    Layout<Shape<Int<B_H>, Int<D_V>>,
           Stride<Int<520>, _1> // We use stride = 520 here to avoid bank conflict
           >;
using SmemLayoutS = decltype(tile_to_shape(UMMA::Layout_K_INTER_Atom<bf16>{},
                                           Shape<Int<B_H>, Int<B_TOPK>>{}, Step<_1, _2>{}));
// V32: B_H = B_TOPK = 64, NUM_BUFS = 2, NUM_INDEX_BUFS = 4, D_Q_SW128 = 512.
struct SharedMemoryPlan {
    union {
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutQ_SW128>> q;
            bf16 q_sw64
                [B_H *
                 D_Q_SW64]; // NOTE D_Q_SW64 may be 0 but array_aligned<bf16, 0> will have a size of 16, so we use array here. The former tensor (`q`) promises its alignment.
            union {
                array_aligned<bf16, cosize_v<SmemLayoutOBuf>> o_buf;
                array_aligned<float, cosize_v<SmemLayoutOAccumBuf>> o_accum_buf;
            } o;
        } qo;
        struct {
            struct {
                array_aligned<bf16, B_H * D_NOPE> nope; // NoPE part, dequantized
                array_aligned<bf16, B_H * D_ROPE>
                    rope; // RoPE part, dequantized. SW64 in v32 mode, SW128 in MODEL1 mode
            } dequant[NUM_BUFS];
            static_assert(sizeof(dequant) >=
                          sizeof(bf16) * (B_H * D_Q));            // So that Q does not covers raw_nope
            array_aligned<e4m3, B_H * D_NOPE> raw_nope[NUM_BUFS]; // Raw (quantized) NoPE part
        } kv;
    } u;
    union {
        float4 p_exchange_buf[4][16 * B_TOPK / 4];
        array_aligned<bf16, cosize_v<SmemLayoutS>> s;
    } s_p;
    CUTE_ALIGNAS(16) float rowwise_max_buf[128];
    char is_token_valid[NUM_INDEX_BUFS][B_TOPK / 8];
    int tma_coord[NUM_INDEX_BUFS][B_TOPK];
    e8m0 scales[NUM_INDEX_BUFS][B_TOPK][NUM_SCALES_EACH_TOKEN];
    array_aligned<uint32_t, 1> tmem_start_addr;
    transac_bar_t bar_last_store_done;
    transac_bar_t bar_q_tma, bar_q_utccp;
    transac_bar_t bar_rope_ready[NUM_BUFS];
    transac_bar_t bar_nope_ready[NUM_BUFS];
    transac_bar_t bar_raw_ready[NUM_BUFS], bar_raw_free[NUM_BUFS];
    transac_bar_t bar_valid_coord_scale_ready[NUM_INDEX_BUFS],
        bar_valid_coord_scale_free[NUM_INDEX_BUFS];
    transac_bar_t bar_qk_done[NUM_BUFS], bar_so_ready[NUM_BUFS], bar_sv_done[NUM_BUFS];
};
```

`SharedMemoryPlan` 同时描述“空间怎样排列”和“同一片空间在不同时刻保存什么”。读这个结构体时，可以先抓住两个 union。

`u.qo` 中的 Q 暂存区在开头接收 BF16 query，随后 Q 被复制到 TMEM。V32 的前 512 维放在 `q`，后 64 维放在 `q_sw64`；MODEL1 的 512 维全部放在 `q`。`o` 是结束时的输出中转区：完整请求使用 BF16 `o_buf`，分片使用 FP32 `o_accum_buf`，两条路径共享地址。

`u.kv` 在主循环期间使用这块空间，保存两组 FP8 `raw_nope`、BF16 `dequant.nope` 和 BF16 `dequant.rope`。Q 在 SMEM 中的生命周期结束后，KV 才能覆盖这片区域；KV 计算结束后，输出暂存区再覆盖它。因此内存容量按 `max(sizeof(qo), sizeof(kv))` 计算。

![V32 下 u.qo 与 u.kv 的 SMEM 字节布局](05_smem_union_layout.png)

CuTe 的 tensor 在这里是“指针 + 布局”的视图。`tile_to_shape` 把基本布局铺成目标形状；`Step<_1,_2>` 指定铺排顺序；`coalesce(..., Shape<_1,_1>{})` 将每个逻辑维度内部能合并的层级压平，保留二维访问接口。这些操作定义地址映射，本身不搬运或初始化数据。

`SmemLayoutQTiles` 使用 SW128：一个基本块覆盖 8 行，每行 64 个 BF16；先铺满当前 64 列的全部 head，再铺下一个 64 列区间。swizzle 在块内重排地址。`cosize_v` 统计布局实际覆盖的存储位置数；例如 FP32 输出布局的偏移为 `520 × row + col`，因此覆盖 `63 × 520 + 512 = 33272` 个 float。

下面这条断言把 Q 暂存地址限制在 `dequant` 覆盖范围内，确保它不会伸入后面的 `raw_nope`：

```cpp
static_assert(sizeof(dequant) >= sizeof(bf16) * (B_H * D_Q));
```

它是布局约束；当前搬运路径还会通过 `bar_q_utccp` 等待 Q 复制结束，才复用 KV 缓冲。

另一个 union `s_p` 则在每个数据块内部复用：先存双路 QK 的交换数据 `p_exchange_buf`，WG0 读完并同步，再存 BF16 指数权重 `s`。`rowwise_max_buf` 先交换两半 token 的最大值，收尾时再交换两半分母。

四槽辅助数组各有明确的消费者：`tma_coord` 供 warp5、6 发起搬运，`scales` 供 WG2 反量化，`is_token_valid` 供 WG0 屏蔽无效分数。`tmem_start_addr` 只保存 TMEM 分配返回的地址；各类 barrier 则分别标记“已准备好”和“已经消费完”。完整容量与偏移表放在第 3 章第 2 节。

### 5.warp7：生成TMA搬运索引和有效性索引

warp7 是整个 KV 流水线的入口。它把稀疏索引转成 TMA 坐标，同时读取 scale、生成有效性位图。32 个线程各处理两个候选 token，恰好覆盖一个 64-token 数据块。

```cpp
// Indices transformation warp
// Responsible for generating: TMA coordinates, scale factors, and valid masks
static_assert(B_TOPK == 64);
static constexpr int tma_coords_step_per_token =
    MODEL_TYPE == ModelType::V32 ? 656 / TMA_K_STRIDE : 576 / TMA_K_STRIDE;
int tma_coords_step_per_block =
    params.stride_kv_block / TMA_K_STRIDE; // must < 2G since k_batch_stride < 1T and TMA_K_STRIDE > 512
int tma_coords_step_per_extra_block = params.stride_extra_kv_block / TMA_K_STRIDE;
uint8_t *k_scales_ptr = MODEL_TYPE == ModelType::V32
                            ? (uint8_t *)params.kv + D_NOPE
                            : (uint8_t *)params.kv + params.page_block_size * (D_NOPE + 2 * D_ROPE);
uint8_t *extra_k_scales_ptr =
    MODEL_TYPE == ModelType::V32
        ? (uint8_t *)params.extra_kv + D_NOPE
        : (uint8_t *)params.extra_kv + params.extra_page_block_size * (D_NOPE + 2 * D_ROPE);
```

先把两种 KV cache 的字节布局分清楚：

| 配置 | page 内的布局 |
| --- | --- |
| V32 | 每个 token 依次为 512 B FP8 NoPE、16 B FP32 scales、128 B BF16 RoPE，共 656 B |
| MODEL1 | 先连续保存所有 token 的 448 B NoPE + 128 B RoPE，每 token 576 B；再放每 token 8 B 的 scale 区，页尾可有填充 |

所以 V32 的 scale 基址是 `kv + 512`，之后按 page stride 和 token row stride 访问；MODEL1 的 scale 区从 `kv + page_block_size × 576` 开始，区内每 token 前进 8 B。

TMA 坐标不是字节地址。这里的 tensor map 第二维分别以 656 B 或 576 B 为步长，故 `tma_coords_step_per_token` 在当前两种配置下都为 1。公式仍写成字节长度除以 `TMA_K_STRIDE`，是为了明确“实际 token 间距”和“坐标单位”的换算。

page 之间则可能有填充，需要使用 `stride_kv_block / TMA_K_STRIDE`。这个 stride 单位还关系到可寻址范围：TMA 坐标为有符号 32 bit，以数百字节而非 1 B 为一格，可以用较小的坐标覆盖更大的 cache；启动代码会检查 page stride 是该步长的整数倍。

V32 的四个 scale 在 GMEM 中为 FP32，在这里转成四个 E8M0 字节，再由 WG2 转成 BF16。E8M0 保存指数，适合表示 2 的整数次幂。仓库的 [量化实现](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/tests/quant.py) 先将 scale 向上取到 2 的整数次幂，再量化 NoPE，因此这里可以用紧凑的 E8M0 编码传递 scale，减少辅助缓冲空间。

```cpp
auto process_one_block = [&](int block_idx, auto is_extra_block_t) {
                    static constexpr bool IS_EXTRA_BLOCK = std::is_same_v<decltype(is_extra_block_t), IsExtraBlock>;
                    int cur_block_size = IS_EXTRA_BLOCK ? params.extra_page_block_size : params.page_block_size;
                    int64_t cur_k_block_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_block : params.stride_kv_block;
                    [[maybe_unused]] int cur_k_row_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_row : params.stride_kv_row;
                    uint8_t* cur_k_scales_ptr = IS_EXTRA_BLOCK ? extra_k_scales_ptr : k_scales_ptr;
                    int cur_tma_coords_step_per_block = IS_EXTRA_BLOCK ? tma_coords_step_per_extra_block : tma_coords_step_per_block;

                    int abs_pos, my_indices[2];
                    if (!IS_EXTRA_BLOCK) {
                        abs_pos = block_idx*B_TOPK + lane_idx*2;
                        *(int2*)my_indices = __ldg((int2*)(indices + abs_pos));
                    } else {
                        abs_pos = (block_idx-args.num_orig_kv_blocks)*B_TOPK + lane_idx*2;
                        *(int2*)my_indices = __ldg((int2*)(extra_indices + abs_pos));
                    }
                    plan.bar_valid_coord_scale_free[rs.index_buf_idx].wait(rs.index_bar_phase^1);

                    int tma_coords[2];
                    e8m0 scales[2*NUM_SCALES_EACH_TOKEN];
                    char valid_mask = 0;
                    CUTE_UNROLL
                    for (int i = 0; i < 2; ++i) {
                        int block_idx, idx_in_block;
                        block_idx = (unsigned int)my_indices[i] / cur_block_size;
                        idx_in_block = (unsigned int)my_indices[i] % cur_block_size;
                        bool is_token_valid = my_indices[i] != -1 && (abs_pos+i < (IS_EXTRA_BLOCK?args.extra_topk_length:args.topk_length));
                        valid_mask |= is_token_valid << i;
                        tma_coords[i] = is_token_valid ? block_idx*cur_tma_coords_step_per_block + idx_in_block*tma_coords_step_per_token : -1; // If the token is invalid because it topk position exceeds topk_length, we must manually fill tma_coords with -1 to avoid copying-in NaN.
                        if constexpr (MODEL_TYPE == ModelType::V32) {
                            int64_t offset = is_token_valid ? block_idx*cur_k_block_stride + idx_in_block*cur_k_row_stride : 0;
                            float4 cur_scale_fp32 = __ldg((float4*)(cur_k_scales_ptr + offset));
                            e8m0 res[4];
                            *(__nv_fp8x2_storage_t*)(res+0) = __nv_cvt_float2_to_e8m0x2(float2{cur_scale_fp32.x, cur_scale_fp32.y}, __NV_NOSAT, cudaRoundZero);
                            *(__nv_fp8x2_storage_t*)(res+2) = __nv_cvt_float2_to_e8m0x2(float2{cur_scale_fp32.z, cur_scale_fp32.w}, __NV_NOSAT, cudaRoundZero);
                            if (!is_token_valid) *(uint32_t*)res = (uint32_t)0;
                            *(uint32_t*)(scales+i*NUM_SCALES_EACH_TOKEN) = *(uint32_t*)(res);
                        } else {
                            int64_t offset = block_idx*cur_k_block_stride + idx_in_block*8; // Each token has 7 scale factors with an extra 1B padding
                            uint64_t scalesx8 = is_token_valid ? __ldg((uint64_t*)(cur_k_scales_ptr + offset)) : 0;
                            *(uint64_t*)(scales+i*NUM_SCALES_EACH_TOKEN) = scalesx8;
                        }
                    }
```

外层 `block_idx` 是“本次计算的第几个 64-token 数据块”；内层同名变量则是某个 token 所在的 cache page 编号。阅读时可以把后者记为 `page_idx`：

$$
\mathrm{page\_idx}=\left\lfloor\frac{\mathrm{token\_id}}{\mathrm{page\_size}}\right\rfloor,
\qquad
\mathrm{row\_in\_page}=\mathrm{token\_id}\bmod\mathrm{page\_size}.
$$

于是搬运坐标为：

$$
\mathrm{coord}=\mathrm{page\_idx}\,
\frac{\mathrm{page\_stride}}{\mathrm{TMA\_K\_STRIDE}}
+\mathrm{row\_in\_page}\,\mathrm{token\_coord\_step}.
$$

例如 V32 每 page 存 64 个 token，page stride 设为 `65 × 656 B`，末尾留一个坐标单位的填充。token 137 位于 page 2、页内第 9 行，对应 TMA 坐标 `2 × 65 + 9 = 139`，字节偏移为 `139 × 656 = 91184 B`。因此有 page padding 时，不能直接把 token ID 当成 TMA 坐标。

`abs_pos` 是当前候选在索引数组中的位置，`my_indices[i]` 则是 cache 中的 token ID。主 cache 用 `block_idx*64` 作为索引块起点；extra cache 先减掉主 cache 的块数，再在自己的索引数组中定位。

一项有效需要同时满足：索引不是 `-1`，且 `abs_pos+i` 小于对应的有效长度。上游分配的索引容量按块补齐，所以线程可以先成对读取，再按有效长度判断。无效项的 TMA 坐标设为 `-1`，让搬运路径走越界填零；不能只在 softmax 中屏蔽分数，因为如果 V 搬入 NaN，后面的 `0 × NaN` 仍会污染输出。

V32 无效项暂用 offset 0 读取一组合法地址上的 scale，随后将它的四个编码字节清零；MODEL1 通过条件读取直接令八个编码字节为零。有效性最终仍由位图决定。

处理完每线程的两个 token 后，还需要把有效位打包成整块位图，并发布辅助信息：

```cpp
valid_mask <<= lane_idx % 4 * 2;
valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x1);
valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x2);
if constexpr (MODEL_TYPE == ModelType::V32) {
    *(uint64_t *)(plan.scales[rs.index_buf_idx] + lane_idx * 2) = *(uint64_t *)scales;
} else {
    *(__int128_t *)(plan.scales[rs.index_buf_idx] + lane_idx * 2) = *(__int128_t *)scales;
}
*(int2 *)(plan.tma_coord[rs.index_buf_idx] + lane_idx * 2) = *(int2 *)tma_coords;
if (lane_idx % 4 == 0)
    plan.is_token_valid[rs.index_buf_idx][lane_idx / 4] = valid_mask;
plan.bar_valid_coord_scale_ready[rs.index_buf_idx].arrive();
rs.update();
```

每四个 lane 处理八个 token。四个 lane 先把自己的两位分别移到 bit `[0,1]`、`[2,3]`、`[4,5]`、`[6,7]`，再用 XOR 1、XOR 2 的 shuffle 做两级 OR 合并。这样四个 lane 都得到同一个完整字节，只让编号为 4 的倍数的 lane 写出。整个 warp 最终写 8 B，正好覆盖 64 个 token 的有效性。

每个 lane 同时写出两个 32-bit 坐标，以及两个 token 的 scale：V32 为 8 B，MODEL1 为 16 B。32 个线程全部到达 ready 屏障后，warp5、6、WG2、WG0 才能消费这组辅助信息。

`process_one_block` 的第二个参数使用 `IsOrigBlock` / `IsExtraBlock` 类型标签，让主 cache 与 extra cache 的指针、stride 选择在模板实例化时确定。外层分别遍历两段块区间，结束后回到请求级 `everyone_sync`。

接下来的各节沿着一块数据的生命周期展开：warp7 发布坐标、scale 与有效位；warp5、6 搬入 KV；WG2 发布 BF16 NoPE；warp4 计算 QK；WG0 生成 S 并缩放 O；warp4 完成 SV。各角色在最后一次使用某项资源后单独释放它，于是后续块可以沿不同缓冲槽提前推进。

### 6.warp5：搬运nope数据

warp5 选出一个线程，按照 warp7 生成的坐标，将 64 个候选 token 的 FP8 NoPE 搬入当前 `raw_nope` 槽位。

```cpp
if (warp_idx == 5 && elect_one_sync()) {
    // Raw KV NoPE retrieval warp
    run_main_loop([&](const MainLoopArgs &args) {
        plan.bar_q_utccp.wait(args.bar_phase_batch_rel);
        plan.bar_last_store_done.wait(args.bar_phase_batch_rel);
        CUTE_NO_UNROLL
        for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
            plan.bar_valid_coord_scale_ready[rs.index_buf_idx].wait(rs.index_bar_phase);
            plan.bar_raw_free[rs.buf_idx].wait(rs.bar_phase ^ 1);
            int4 cur_indices = *(int4 *)(plan.tma_coord[rs.index_buf_idx] + 0);
            int4 nxt_cur_indices;
            CUTE_UNROLL
            for (int row = 0; row < B_TOPK; row += 4) {
                if (row + 4 < B_TOPK)
                    nxt_cur_indices = *(int4 *)(plan.tma_coord[rs.index_buf_idx] + row + 4);
                ku::tma_gather4(
                    block_idx >= args.num_orig_kv_blocks ? &tma_params.tensor_map_extra_kv_nope
                                                         : &tma_params.tensor_map_kv_nope,
                    plan.bar_raw_ready[rs.buf_idx], plan.u.kv.raw_nope[rs.buf_idx].data() + D_NOPE * row,
                    0, cur_indices, (int64_t)TMA::CacheHintSm90::EVICT_LAST);
                cur_indices = nxt_cur_indices;
            }
            plan.bar_raw_ready[rs.buf_idx].arrive_and_expect_tx(B_TOPK * D_NOPE * sizeof(e4m3));
            plan.bar_valid_coord_scale_free[rs.index_buf_idx].arrive();
            rs.update();
        }
    });
}
```

先看请求级的两个等待。`bar_q_utccp` 表示 Q 已从 SMEM 复制到 TMEM，可以开始复用 `u` 中的 KV 区域。`bar_last_store_done` 由 WG0 在 `tma_store_wait<0>()` 后通知，表示上一请求的 TMA store 已读完 SMEM 源数据，该源空间可以覆盖。这里的等待使用 `.read` 语义，关注的是源缓冲能否复用。

每个 KV 块再等待两件事：

| 等待 | 解决的问题 |
| --- | --- |
| `bar_valid_coord_scale_ready[index]` | warp7 已经写好这一块的搬运坐标 |
| `bar_raw_free[buf]` | WG2 已经读完这个 raw 槽位的上一轮 FP8 数据 |

`wait(phase)` 等待该 phase 完成，即观察到屏障已经翻到另一相位。ready 等当前轮；free 等同一槽位的上一轮，所以传入 `rs.bar_phase ^ 1`。首次使用时屏障初始为 phase 0，free 等待 phase 1 会直接通过，表示初始槽位本来就是空的。第二次使用槽位 0 时，`rs.bar_phase = 1`，free 等待 phase 0，只有 WG2 完成首轮消费后才能通过。

`cur_indices` 一次读取四个 token 坐标，`tma_gather4` 一次发起四行搬运，64 行共发起 16 次。第一个坐标参数 `0` 表示从 NoPE 的第一列开始。NoPE 的 tensor map 把连续 8 个 FP8 看成一个 `int64` 元素，从而把 V32 的横向 box 长度由 512 压到 64；搬运仍然原样复制位模式，不做数值转换。

V32 这一块总共搬入 `64 × 512 × 1 = 32768 B`。`arrive_and_expect_tx` 把预期字节数登记到 `bar_raw_ready`；到达计数和搬运字节计数都满足后，WG2 才能通过 ready 等待。

最后的 `bar_valid_coord_scale_free.arrive()` 释放的是 warp5 对坐标的使用权。此时 TMA 已取得坐标，辅助槽位中保存的坐标值不必继续保留；FP8 数据是否已经搬完，由另一条 `bar_raw_ready` 追踪。

辅助槽位的 free 屏障初始化为 `128 + 128 + 1 + 1 = 258`：WG0 的 128 个线程读完有效性掩码，WG2 的 128 个线程用完 scale，warp5、warp6 的两个发射线程用完坐标。四方全部到达后，warp7 才能覆盖这个槽位。与之配对的 ready 屏障计数为 32，因为辅助信息由 warp7 的 32 个线程共同生产。

### 7.warp6：搬运rope数据

warp6 的搬运流程与 warp5 相同，但它读取的 RoPE 已经是 BF16，直接写入 `dequant[buf].rope`，省去反量化阶段。

```cpp
if (warp_idx == 6 && elect_one_sync()) {
    // KV RoPE retrieval warp
    run_main_loop([&](const MainLoopArgs &args) {
        plan.bar_q_utccp.wait(args.bar_phase_batch_rel);
        plan.bar_last_store_done.wait(args.bar_phase_batch_rel);
        CUTE_NO_UNROLL
        for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
            plan.bar_valid_coord_scale_ready[rs.index_buf_idx].wait(rs.index_bar_phase);
            if constexpr (MODEL_TYPE == ModelType::V32) {
                plan.bar_qk_done[rs.buf_idx].wait(rs.bar_phase ^ 1);
            } else {
                plan.bar_sv_done[rs.buf_idx].wait(rs.bar_phase ^ 1);
            }
            int4 cur_indices = *(int4 *)(plan.tma_coord[rs.index_buf_idx] + 0);
            int4 nxt_cur_indices;
            CUTE_UNROLL
            for (int row = 0; row < B_TOPK; row += 4) {
                if (row + 4 < B_TOPK)
                    nxt_cur_indices = *(int4 *)(plan.tma_coord[rs.index_buf_idx] + row + 4);
                CUTE_UNROLL
                for (int t = 0; t < D_ROPE / (K_ROPE_SW / 2); ++t) {
                    ku::tma_gather4(
                        block_idx >= args.num_orig_kv_blocks ? &tma_params.tensor_map_extra_kv_rope
                                                             : &tma_params.tensor_map_kv_rope,
                        plan.bar_rope_ready[rs.buf_idx],
                        plan.u.kv.dequant[rs.buf_idx].rope.data() + (K_ROPE_SW / 2) * row +
                            t * B_TOPK * (K_ROPE_SW / 2),
                        t * (K_ROPE_SW / 2), cur_indices, (int64_t)TMA::CacheHintSm90::EVICT_LAST);
                }
                cur_indices = nxt_cur_indices;
            }
            plan.bar_rope_ready[rs.buf_idx].arrive_and_expect_tx(B_TOPK * D_ROPE * sizeof(bf16));
            plan.bar_valid_coord_scale_free[rs.index_buf_idx].arrive();
            rs.update();
        }
    });
}
```

两个差别决定了这里的等待和地址计算。

首先是槽位的释放时机。V32 的 RoPE 只参加 QK，所以等上一轮 `bar_qk_done` 即可覆盖；MODEL1 的 RoPE 还构成 V 的最后 64 维，因此必须等上一轮 `bar_sv_done`。相同的内存区域，最后一个消费者不同，可复用的时刻也随之不同。

其次是 tensor map 的搬运宽度。V32 的 RoPE 使用 SW64，代码为每行选择 64 B、即 32 个 BF16 的 box。64 维需要分两段，列坐标分别为 0 和 32。MODEL1 使用 SW128，一次覆盖整行 64 个 BF16。

V32 的目标布局先放 64 个 token 的前 32 维，再放后 32 维，因此第二段的起点要加 `B_TOPK × 32` 个 BF16。代码中的 `t*B_TOPK*(K_ROPE_SW/2)` 正是这个跨段偏移。一个四行 gather 的起点再加 `(K_ROPE_SW/2)*row`。

两种配置的 RoPE 总数据量都是 `64 × 64 × 2 = 8192 B`。`bar_rope_ready` 在这些字节搬入后完成；warp4 可以据此开始读取 RoPE。

### 8.warp group 2：Dequant

WG2 的 128 个线程把 FP8 NoPE 读入寄存器，乘以分组 scale，并按 QK、SV 共用的 SW128 布局写出 BF16。

```cpp
constexpr int GROUP_SIZE = 8, NUM_GROUPS = 128 / 8, ROWS_PER_GROUP = B_TOPK / NUM_GROUPS,
              COLS_PER_GROUP = D_NOPE / (GROUP_SIZE * 8);
int group_idx = idx_in_warpgroup / GROUP_SIZE, idx_in_group = idx_in_warpgroup % GROUP_SIZE;
Tensor nope0 =
    make_tensor(make_smem_ptr(plan.u.kv.dequant[0].nope.data()), SmemLayoutKTiles_SW128<D_NOPE / 64>{});
bf16 *nope0_base = &nope0(group_idx, idx_in_group * 8);
bf16 *nope1_base = nope0_base + (plan.u.kv.dequant[1].nope.data() - plan.u.kv.dequant[0].nope.data());
e4m3 *raw_nope0_base = plan.u.kv.raw_nope[rs.buf_idx].data() + group_idx * D_NOPE + idx_in_group * 8;
e4m3 *raw_nope1_base = raw_nope0_base + B_H * D_NOPE;
```

每 8 个线程组成一个 group，共 16 组。一组负责 token 行 `group_idx + 16 × local_row_idx`，`local_row_idx` 取 0、1、2、3，恰好覆盖 64 行。

组内每个线程一次处理相邻 8 个 FP8；8 个线程合起来覆盖 64 列。V32 的 512 列需要 8 轮，MODEL1 的 448 列需要 7 轮。因此：

- `ROWS_PER_GROUP = 4`：一组负责的 token 数。
- `COLS_PER_GROUP`：每个 token 沿特征维展开的轮数，V32 为 8，MODEL1 为 7。
- `idx_in_group*8`：当前线程在一轮中的列起点。

```cpp
template <int NUM_TILES>
using SmemLayoutKTiles_SW128 =
    decltype(coalesce(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{},
                                    Shape<Int<B_H>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}),
                      Shape<_1, _1>{}));
```

`nope0` 是反量化目标空间的 CuTe 视图，`nope0(group_idx, idx_in_group*8)` 已经包含 SW128 的块内地址重排。`nope1_base` 在这个地址上加两槽起点的距离，保留同样的布局内位置。

原始 FP8 区域没有 swizzle，按 token 行连续存放。初始化时 `rs.buf_idx = 0`，因此 `raw_nope0_base`、`raw_nope1_base` 分别预先保存两个槽位中属于当前线程的基址。后面每块只需要根据 `buf_idx` 二选一。

```cpp
plan.bar_valid_coord_scale_ready[rs.index_buf_idx].wait(rs.index_bar_phase);
plan.bar_raw_ready[rs.buf_idx].wait(rs.bar_phase);
plan.bar_sv_done[rs.buf_idx].wait(rs.bar_phase ^ 1);
uint32_t cur_nope_base_uint_addr =
    cute::cast_smem_ptr_to_uint(rs.buf_idx == 0 ? nope0_base : nope1_base);
e4m3 *raw_nope_base = rs.buf_idx == 0 ? raw_nope0_base : raw_nope1_base;
auto st_128b = [&](int local_row_idx, int local_col_idx, __int128_t &data) {
    asm volatile("st.weak.shared::cta.b128 [%0], %1;\n"
                 :
                 : "r"(cur_nope_base_uint_addr +
                       2 * (local_row_idx * NUM_GROUPS * 64 + local_col_idx * B_TOPK * 64)),
                   "q"(data) // 2 for sizeof(bf16)
    ); // We have this `asm volatile` here, otherwise the compiler generates ST.E instead of STS
};
```

三个等待分别保证 scale 已准备好、原始 FP8 已搬入、目标 BF16 槽位已被上一轮 SV 消费完。NoPE 同时充当 K 与 V，必须把生命周期延长到 SV 结束。

`st_128b` 一次写出 8 个 BF16，即 16 B。地址计算可以直接与线程负责的元素对应：

$$
\text{row}=\text{group\_idx}+16\,\text{local\_row\_idx},\qquad
\text{col}=8\,\text{idx\_in\_group}+64\,\text{local\_col\_idx}.
$$

基址已经处理第一组 row、col 的 swizzle。每增加一轮 token，沿当前 64 列条带前进 16 行，增加 `16 × 64 × 2 = 2048 B`；每增加一轮特征列，跨过完整的 `[64,64]` BF16 条带，增加 `64 × 64 × 2 = 8192 B`。于是得到代码中的

`2 × (local_row_idx × NUM_GROUPS × 64 + local_col_idx × B_TOPK × 64)`。

例如 group 0、组内线程 0 在 `local_row_idx=1, local_col_idx=1` 时，处理 token 16 的第 64–71 列，写入地址相对自己的初始基址增加 `2048 + 8192 = 10240 B`。这里显式使用 shared store 指令，是为了让编译器确定生成共享内存写入；源码注释指出普通写法曾生成 `ST.E` 而非 `STS`。

V32 每 128 维共用一个 scale，反量化主体如下：

```cpp
auto get_raw_fp8 = [&](int local_row_idx, int local_col_idx) -> uint64_t {
    return *(uint64_t *)(raw_nope_base + local_row_idx * NUM_GROUPS * D_NOPE +
                         local_col_idx * (GROUP_SIZE * 8));
};
// The following code suffers from a 2-way bank conflict when reading from SMEM.
if constexpr (MODEL_TYPE == ModelType::V32) {
    CUTE_UNROLL
    for (int local_row_idx = 0; local_row_idx < ROWS_PER_GROUP; ++local_row_idx) {
        int row_idx = local_row_idx * NUM_GROUPS + group_idx;
        bf16 scales[4];
        e8m0 scales_e8m0[4];
        *(uint32_t *)scales_e8m0 = *(uint32_t *)plan.scales[rs.index_buf_idx][row_idx];
        *(__nv_bfloat162_raw *)(scales + 0) =
            __nv_cvt_e8m0x2_to_bf162raw(*(unsigned short *)(scales_e8m0 + 0));
        *(__nv_bfloat162_raw *)(scales + 2) =
            __nv_cvt_e8m0x2_to_bf162raw(*(unsigned short *)(scales_e8m0 + 2));

        uint64_t cur_data_fp8x8 = get_raw_fp8(local_row_idx, 0);
        CUTE_UNROLL
        for (int local_col_idx = 0; local_col_idx < COLS_PER_GROUP; ++local_col_idx) {
            ku::nve4m3x2 data_fp8[4];
            ku::nvbf16x2 data_bf16[4];
            *(uint64_t *)data_fp8 = cur_data_fp8x8;
            if (local_col_idx + 1 < COLS_PER_GROUP)
                cur_data_fp8x8 = get_raw_fp8(local_row_idx, local_col_idx + 1);
            bf16 scale = scales[local_col_idx / (D_NOPE / (GROUP_SIZE * 8) / 4)];
            CUTE_UNROLL
            for (int i = 0; i < 4; ++i) {
                data_bf16[i] = fp8x2_to_bf16x2_with_scale(data_fp8[i], *(ku::nvbf16 *)(&scale));
            }
            st_128b(local_row_idx, local_col_idx, *(__int128_t *)data_bf16);
        }
    }
}
```

`get_raw_fp8` 一次取 8 B，随后解释成四对 FP8。四个 E8M0 scale 也一次按 32 bit 取出，再两两转换成 BF16。每个 `local_col_idx` 覆盖 64 列，所以 `local_col_idx / 2` 选择 scale：第 0、1 轮用 scale 0，第 2、3 轮用 scale 1，依此类推。

`fp8x2_to_bf16x2_with_scale` 将一对 FP8 转换并乘以同一个 scale，结果保存在一对 BF16 中。四对合起来正好对应 `st_128b` 的 16 B 写出。

循环在计算当前 8 个元素前，先读取下一轮的 8 个 FP8。这是把普通 SMEM load 提前安排的软件流水：让下一次读取与当前转换、乘法和存储之间存在可调度空间。它没有使用 `cp.async` 或 TMA，也没有独立的异步完成屏障；后续使用加载结果仍受寄存器数据依赖约束。

MODEL1 的主体沿用相同分工：

```cpp
else {
    CUTE_UNROLL
    for (int local_row_idx = 0; local_row_idx < ROWS_PER_GROUP; ++local_row_idx) {
        int row_idx = local_row_idx * NUM_GROUPS + group_idx;
        bf16 scales[8];
        e8m0 scales_e8m0[8];
        *(uint64_t *)scales_e8m0 = *(uint64_t *)plan.scales[rs.index_buf_idx][row_idx];
        *(__nv_bfloat162_raw *)(scales + 0) =
            __nv_cvt_e8m0x2_to_bf162raw(*(unsigned short *)(scales_e8m0 + 0));
        *(__nv_bfloat162_raw *)(scales + 2) =
            __nv_cvt_e8m0x2_to_bf162raw(*(unsigned short *)(scales_e8m0 + 2));
        *(__nv_bfloat162_raw *)(scales + 4) =
            __nv_cvt_e8m0x2_to_bf162raw(*(unsigned short *)(scales_e8m0 + 4));
        *(__nv_bfloat162_raw *)(scales + 6) =
            __nv_cvt_e8m0x2_to_bf162raw(*(unsigned short *)(scales_e8m0 + 6));

        uint64_t cur_data_fp8x8 = get_raw_fp8(local_row_idx, 0);
        CUTE_UNROLL
        for (int local_col_idx = 0; local_col_idx < COLS_PER_GROUP; ++local_col_idx) {
            ku::nve4m3x2 data_fp8[4];
            ku::nvbf16x2 data_bf16[4];
            *(uint64_t *)data_fp8 = cur_data_fp8x8;
            if (local_col_idx + 1 < COLS_PER_GROUP)
                cur_data_fp8x8 = get_raw_fp8(local_row_idx, local_col_idx + 1);
            bf16 scale = scales[local_col_idx];
            CUTE_UNROLL
            for (int i = 0; i < 4; ++i) {
                data_bf16[i] = fp8x2_to_bf16x2_with_scale(data_fp8[i], *(ku::nvbf16 *)(&scale));
            }
            st_128b(local_row_idx, local_col_idx, *(__int128_t *)data_bf16);
        }
    }
}
```

区别集中在 scale 粒度：MODEL1 每 64 维一个 scale，七轮分别使用 `scales[0]` 到 `scales[6]`，因此索引直接是 `local_col_idx`。每个 token 存 8 个 E8M0 字节，其中最后一个是填充；它会随向量 load 一起读入，但不参与这七轮计算。

完成当前块后，源码还有三次关键通知：

```cpp
cutlass::arch::fence_view_async_shared();
plan.bar_nope_ready[rs.buf_idx].arrive();
plan.bar_raw_free[rs.buf_idx].arrive();
plan.bar_valid_coord_scale_free[rs.index_buf_idx].arrive();
rs.update();
```

shared fence 让普通线程写出的 BF16 对后续异步 MMA 可见；`bar_nope_ready` 通知 warp4 可以读取它。`bar_raw_free` 通知 warp5 原始 FP8 已经读完，可以搬下一轮；辅助 free 则通知 warp7 本轮 scale 已使用完。三者释放的是不同的资源，不能用一个“本块完成”事件概括。

源码在原始 FP8 的共享内存读取处标出了 2-way bank conflict。这段实现的重点是把行连续的 FP8 转为两次 GEMM 可直接使用的 BF16 布局，并让转换工作与其他角色重叠。

### 9.warp4：QK 矩阵乘法

QK 的 MMA 类型如下：

```cpp
using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, B_H, B_TOPK * 2, UMMA::Major::K,
                                    UMMA::Major::K>{})); // *2 for dual gemm
```

这定义的是 MMA 操作及其布局规则。`TS` 表示 A 来自 TMEM、B 来自 SMEM；两个输入为 BF16，累加结果为 FP32。`B_H=64`，`B_TOPK*2=128`，对应名义上的 `M=64,N=128`。`WS` 使用 `tcgen05.mma.ws`，`NOELECT` 表示包装内部不再选举发射线程，调用者已经选好了。

SV 的类型也在同一处定义：

```cpp
using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H, 256, UMMA::Major::K, UMMA::Major::MN>{}));
```

`SS` 表示两个输入都来自 SMEM；N 维为 256，512 维输出用两个 N tile 覆盖。随后为结果构造 TMEM fragment：

```cpp
TiledMMA tiled_mma_P = TiledMMA_P{};
TiledMMA tiled_mma_O = TiledMMA_O{};
// NOTE These tXXX tensors are only for a forged layout (so that CuTe is able to generate correct address in cute::gemm)
Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, _128>{});
Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<B_H>, Int<D_V>>{});
tP.data().get() = tmem_cols::P;
tO.data().get() = tmem_cols::O;

// Wait for UTCCP
plan.bar_q_utccp.wait(args.bar_phase_batch_rel);
ku::tcgen05_after_thread_sync();
```

`partition_fragment_C` 生成结果 fragment 的形状和寻址规则，`tP.data().get()`、`tO.data().get()` 再将其绑定到已经分配好的 TMEM 列。这里没有新分配存储。`tP` 保存 QK 的两路部分点积，`tO` 保存 SV 的输出累加值。

dual GEMM 的目的，是让 64 个 head 的计算使用 TMEM 上下两半的 128 个 datapath。对于 NoPE，上一路负责特征段 0、2、4、6，下一路负责 1、3、5、7；K 也按相同规则解释。两路分别计算：

$$
P^{(0)}_{h,j}=\sum_{d\in\mathcal D_0}Q_{h,d}K_{j,d},\qquad
P^{(1)}_{h,j}=\sum_{d\in\mathcal D_1}Q_{h,d}K_{j,d}.
$$

完整分数是 $P_{h,j}=P^{(0)}_{h,j}+P^{(1)}_{h,j}$。这里每一路都覆盖同样的 64 个 KV token，拆分的是点积的特征维，不是把候选 token 扩成了 128 个。

在 `M=64,N=128` 的 WS 映射下，名义结果的两段 `[64,64]` 分别落到 TMEM 的上、下 64 行。常规使用要求这两半 A 提供同一份 Q；本文利用这个地址映射，让两半 A 保存不同的特征分片，并让 B 的相应两半与之匹配。因此 CuTe 中 A 的 `[64,K/2]` fragment 主要服务于寻址，实际的两路数据已经由前面的 UTCCP 排好。包装实现见 [MMA 类型与 traits](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/csrc/kerutils/include/kerutils/device/sm100/gemm.cuh)。

V32 先执行 RoPE 部分：

```cpp
if constexpr (MODEL_TYPE == ModelType::V32) {
    // V3.2: RoPE behaves like an extra block with size 64, so we can do RoPE first
    // QK RoPE
    plan.bar_rope_ready[rs.buf_idx].wait(rs.bar_phase);
    ku::tcgen05_after_thread_sync();
    Tensor tQ_rope = tiled_mma_P.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<D_ROPE / 2>>{}));
    tQ_rope.data().get() = tmem_cols::Q_Tail;
    Tensor sK_rope = make_tensor(make_smem_ptr(plan.u.kv.dequant[rs.buf_idx].rope.data()),
                                 SmemLayoutKTiles_DualGemm_SW64<2 / 2>{});
    ku::utcmma_ts(tiled_mma_P, tQ_rope, sK_rope, tP, true);

    // QK NoPE
    plan.bar_nope_ready[rs.buf_idx].wait(rs.bar_phase);
    ku::tcgen05_after_thread_sync();
    Tensor tQ_nope = tiled_mma_P.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<D_NOPE / 2>>{}));
    tQ_nope.data().get() = tmem_cols::Q;
    Tensor sK_nope = make_tensor(make_smem_ptr(plan.u.kv.dequant[rs.buf_idx].nope.data()),
                                 SmemLayoutKTiles_DualGemm_SW128<512 / 64 / 2>{});
    ku::utcmma_ts(tiled_mma_P, tQ_nope, sK_nope, tP, false);
```

RoPE 直接以 BF16 搬入，NoPE 还需要 WG2 反量化。先等 `bar_rope_ready` 并计算 RoPE，可以把这部分 Tensor Core 工作放在等待 NoPE 的前面。

RoPE 的两路各负责 32 维；NoPE 的两路各负责 256 维。`utcmma_ts(..., true)` 只在第一条 K 子迭代忽略旧累加器，后续子迭代自动切换为累加；后面的 NoPE 调用传 `false`，把内容点积继续加到已有的位置点积上。

```
tiled_mma_P：使用哪套 MMA 配置
tQ_rope：   A，位于 TMEM
sK_rope：   B，位于 SMEM
tP：        结果累加器，位于 TMEM
true：      开始本次计算时忽略旧累加值
```

包装的单条 BF16 MMA 在 K 方向处理 16 个元素，即 256 bit。因此 RoPE 每路 32 维展开成两次 K 迭代，NoPE 每路 256 维展开成 16 次。两路部分点积留在 TMEM 中，随后由 WG0 合并。

MODEL1 的 RoPE 则是完整 512 维中的最后一个 64 维段：

```cpp
else {
    // MODEL1: RoPE is the last 64 dims within the full 512 dim, which couples with the last 64 dim from the NoPE part when performing dual GEMM. i.e.
    //
    // logical view: |0|1|2|3|4|5|6|7| (where 7 is the RoPE part)
    // dual gemm's view:
    // |0|2|4|6|
    // |1|3|5|7|
    //
    // So we must wait for both the NoPE and the RoPE part, and then perform dual GEMM
    plan.bar_rope_ready[rs.buf_idx].wait(rs.bar_phase);
    plan.bar_nope_ready[rs.buf_idx].wait(rs.bar_phase);
    ku::tcgen05_after_thread_sync();

    Tensor tQ = tiled_mma_P.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<D_Q / 2>>{}));
    tQ.data().get() = tmem_cols::Q;
    Tensor sK = make_tensor(make_smem_ptr(plan.u.kv.dequant[rs.buf_idx].nope.data()),
                            SmemLayoutKTiles_DualGemm_SW128<512 / 64 / 2>{});
    ku::utcmma_ts(tiled_mma_P, tQ, sK, tP, true);
}
```

它与 NoPE 的最后 64 维组成最后一对 dual-GEMM 数据段。上下两路分别取 `[0,2,4,6]` 和 `[1,3,5,7]`，其中第 7 段是 RoPE，因此必须同时等 NoPE 与 RoPE 就绪，再执行完整的 QK。

两种分支最后都调用 `umma_arrive_noelect(plan.bar_qk_done[rs.buf_idx])`。WG0 等待此屏障完成后才能读取 P，V32 的 warp6 也据此判断当前 RoPE 槽位何时可以复用。

### 10.warpgroup0：softmax 与旧输出缩放

WG0 的 128 个线程处理 64 个 head，每个 head 由相隔 64 的两个线程协作。线程 `t` 的 head 编号是 `t % 64`：前 64 个线程负责 token 0–31，后 64 个线程负责 token 32–63。

```cpp
float mi = MAX_INIT_VAL;
float li = 0.0f;
float real_mi = -CUDART_INF_F;
```

每个线程维护三个标量：

| 变量 | 意义 |
| --- | --- |
| `mi` | 当前指数运算采用的参考值，已经处于以 2 为底的尺度 |
| `li` | 当前线程所负责半边 token 的累计指数和 |
| `real_mi` | 已处理有效分数的真实最大值，用来识别全无效输入 |

由于参考值采用延迟更新，`mi` 不一定等于历史最大值；但同一个 head 的两个线程必须始终使用同一个 `mi`。`mi` 初始化为有限的 `-1e30`，让无效块中的 `-inf - mi` 仍为 `-inf`，避免 `-inf - (-inf)` 产生 NaN。

#### 10.1 合并双路 GEMM 计算的结果

每块开始时，WG0 先同步自己的 128 个线程，确认上一轮的临时区已经用完，再等待辅助信息 ready 和 QK done。随后从 TMEM 读取两路部分点积：

```cpp
float p[B_TOPK / 2], p_peer[B_TOPK / 2];
if (warp_idx < 2) {
    ku::tmem_ld_32dp32bNx<B_TOPK / 2>(tmem_cols::P, p);
    ku::tmem_ld_32dp32bNx<B_TOPK / 2>(tmem_cols::P + 32, p_peer);
} else {
    ku::tmem_ld_32dp32bNx<B_TOPK / 2>(tmem_cols::P, p_peer);
    ku::tmem_ld_32dp32bNx<B_TOPK / 2>(tmem_cols::P + 32, p);
}
cutlass::arch::fence_view_async_tmem_load();
ku::tcgen05_before_thread_sync();

// Reduce within shared mem
{
    // Store
    // Warp 0, 1 store their right (col 32 ~ 63) part, while warp 2, 3 store their left (row 0 ~ 31) part
    CUTE_UNROLL
    for (int i = 0; i < (B_TOPK / 2) / 4; ++i)
        plan.s_p.p_exchange_buf[warp_idx ^ 2][i * 32 + lane_idx] = *(float4 *)(p_peer + i * 4);
    NamedBarrier::arrive_and_wait(
        64, NamedBarriers::wg0_warp02_sync +
                (warp_idx & 1)); // Synchronize between warp 0 and warp 2, as well as warp 1 - warp 3
    // Load
    CUTE_UNROLL
    for (int i = 0; i < (B_TOPK / 2) / 4; ++i) {
        float2 t[2];
        *(float4 *)t = plan.s_p.p_exchange_buf[warp_idx][i * 32 + lane_idx];
        float2 *cur_p = (float2 *)(p + i * 4);
        cur_p[0] = ku::float2_add(cur_p[0], t[0]);
        cur_p[1] = ku::float2_add(cur_p[1], t[1]);
    }
}
```

TMEM 行与 warp 的对应关系是：

| Warp | head 范围 | 手中 QK 结果的来源 | 最终负责的 token |
| --- | --- | --- | --- |
| 0 | 0–31 | 特征分片 0 | 0–31 |
| 1 | 32–63 | 特征分片 0 | 0–31 |
| 2 | 0–31 | 特征分片 1 | 32–63 |
| 3 | 32–63 | 特征分片 1 | 32–63 |

每个线程最初读到同一特征分片上全部 64 个 token 的点积，把自己最终负责的 32 个放入 `p`，另 32 个放入 `p_peer`。

以 head 0 为例：warp0 的 lane0 保留分片 0 对 token 0–31 的贡献，把 token 32–63 的贡献发给 warp2；warp2 的 lane0 保留分片 1 对 token 32–63 的贡献，把 token 0–31 的贡献发给 warp0。交换后各自相加，就获得自己负责的 32 个完整点积。

`warp_idx ^ 2` 给出配对 warp，因此配对为 0↔2、1↔3。交换先写到对方的 SMEM 区，再用只覆盖对应 64 个线程的 named barrier 同步，最后从自己的区域读取对方写来的数据。`i*32 + lane_idx` 以 `float4` 为单位安排相邻 lane 的访问。

这里合并的是特征维上的部分和，必须先完成再做 softmax；分别对两路点积做 softmax 后相加，得不到同一个 attention 结果。

#### 10.2 屏蔽无效 token 并更新最大值，计算 S

```cpp
uint32_t valid_mask =
    *((uint32_t *)plan.is_token_valid[rs.index_buf_idx] + (idx_in_warpgroup >= 64 ? 1 : 0));
CUTE_UNROLL
for (int i = 0; i < B_TOPK / 2; i += 1) {
    if (!(valid_mask >> i & 1))
        p[i] = -CUDART_INF_F;
}
```

一块 64 个 token 的有效性由 64 个 bit 表示。每个线程取与自己 token 半区对应的 32 bit，将无效位置的分数设成负无穷，这些位置的指数权重随后变成 0。

```cpp
float cur_pi_max = -CUDART_INF_F;
CUTE_UNROLL
for (int i = 0; i < (B_TOPK / 2); i += 1) {
    cur_pi_max = max(cur_pi_max, p[i]);
}
cur_pi_max *= params.sm_scale_div_log2;
```

先在本线程的 32 个分数中取最大值，再乘 `sm_scale_div_log2`，转为以 2 为底的指数尺度：

$$
\text{sm\_scale\_div\_log2}=\beta\log_2e=\frac{\beta}{\ln2}.
$$

同一 head 的另半边最大值通过 SMEM 交换：

```cpp
plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
NamedBarrier::arrive_and_wait(
    128, NamedBarriers::wg0_sync); // This also separates "reading p_exchange_buf" and "writing S"
plan.bar_valid_coord_scale_free[rs.index_buf_idx].arrive();
cur_pi_max = max(cur_pi_max, plan.rowwise_max_buf[idx_in_warpgroup ^ 64]);
real_mi = max(real_mi, cur_pi_max);
bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);
```

`idx_in_warpgroup ^ 64` 找到协作线程，二者取 max 后得到同一行完整的块最大值。WG0 的同步还隔开了对 `p_exchange_buf` 的读取和随后对 S 的写入，因为这两者共用一个 union。有效性掩码读取完毕后，WG0 的 128 个线程各到达一次辅助 free 屏障。

`__any_sync` 并不是让整个 warp 共用一个最大值，而是让整个 warp 共用“是否执行 O 缩放代码”的决定。后续的 TMEM load/store 是 warp 协同指令，需要 warp 内一致进入这段路径：只要 32 个 head 中任意一个超过阈值 6，整个 warp 都执行缩放分支，但每个线程仍按自己 head 的分数计算 `new_max` 和缩放系数。

成对 warp 处理同一组 head，交换后各 lane 的块最大值相同，旧 `mi` 也相同，因此它们的投票结果一致，同一 head 的两半输出会按同一尺度调整。

```cpp
float new_max, scale_for_old;
if (!should_scale_o) {
    // Don't scale O
    scale_for_old = 1.0f;
    new_max = mi;
} else {
    new_max = max(cur_pi_max, mi);
    scale_for_old = exp2f(mi - new_max);
}
mi = new_max; // mi is still identical within each row
```

不需要缩放时保持 `mi`，令 `scale_for_old=1`；需要缩放时令 `new_max=max(cur_pi_max,mi)`，旧分母和旧输出统一乘上 `2^(mi-new_max)`。某个 lane 即使本身没超过阈值，也可以随整个 warp 更新参考值；如果它的 `new_max` 没有变，缩放系数就是 1。

```cpp
__nv_bfloat162 s[(B_TOPK / 2) / 2];
float2 neg_new_max = float2{-new_max, -new_max};
float2 cur_sum = float2{0.0f, 0.0f};
CUTE_UNROLL
for (int i = 0; i < (B_TOPK / 2) / 2; i += 1) {
    float2 d = ku::float2_fma(float2{p[i * 2], p[i * 2 + 1]}, scale, neg_new_max);
    d.x = exp2f(d.x);
    d.y = exp2f(d.y);
    cur_sum = ku::float2_add(cur_sum, d);
    s[i] = __float22bfloat162_rn(d);
}
li = fma(li, scale_for_old, (cur_sum.x + cur_sum.y));
```

对当前 32 个 token，先用 FP32 计算 `2^(p[i] × sm_scale_div_log2 - new_max)`。`cur_sum` 累计这些 FP32 指数值；S 则转为 BF16 保存，用于后面的 BF16×BF16、FP32 累加的 SV。

每个线程的分母递推为：

$$
\ell_{\mathrm{half}}'=
\alpha\ell_{\mathrm{half}}+
\sum_{j\in\mathrm{当前半块}}2^{x_j-m'}.
$$

因此，分母来自 FP32 指数和，分子中的 SV 使用 BF16 化后的 S。第 1 章给出的是实数运算下的递推关系，这里则对应其混合精度实现。

#### 10.3 将 S 保存到 SMEM 中

```cpp
CUTE_UNROLL
for (int i = 0; i < B_TOPK / 2 / 8; i += 1) {
    *(uint128_t *)(sS_base + B_H * 8 * i) = *(uint128_t *)(s + i * 4);
}
```

```cpp
using SmemLayoutS = decltype(tile_to_shape(UMMA::Layout_K_INTER_Atom<bf16>{},
                                           Shape<Int<B_H>, Int<B_TOPK>>{}, Step<_1, _2>{}));
```

`SmemLayoutS` 将 `[64 head,64 token]` 按每 8 个 token 一条带铺排。一条带包含 64 个 head，每个 head 有 8 个 BF16，因此条带之间相差 `64 × 8 × 2 = 1024 B`。

`sS_base` 已定位到当前 head 与 token 半区的第一条带：线程 0–63 从 token 0 开始，线程 64–127 从 token 32 开始。每个线程写四次，每次 8 个 BF16，覆盖自己负责的 32 个 token。

源码用 `uint128_t` 一次写 16 B；`s` 数组的元素是 BF16 二元组，因此源地址每轮增加 `i*4` 个二元组，恰好也是 8 个 BF16。目标地址增加 `B_H*8*i` 个 BF16，把四次写入落在四条相邻的 token 条带中。

#### 10.4 缩放 O

本轮参考值变化时，WG0 先缩放旧输出，warp4 再加入当前块的 SV：

$$
O'=\underbrace{2^{m-m'}O}_{\text{WG0}}
+\underbrace{SV}_{\text{warp4}}.
$$

这里的 O 仍是未归一化的分子。

```cpp
if (block_idx != args.start_block_idx && should_scale_o) {
    float2 scale_for_old_float2 = float2{scale_for_old, scale_for_old};
    ku::tcgen05_after_thread_sync();

    static constexpr int CHUNK_SIZE = 64;
    float2 o[CHUNK_SIZE / 2];
    CUTE_UNROLL
    for (int chunk_idx = 0; chunk_idx < (D_V / 2) / CHUNK_SIZE; ++chunk_idx) {
        // Load O
        ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(tmem_cols::O + chunk_idx * CHUNK_SIZE, o);
        cutlass::arch::fence_view_async_tmem_load();

        // Mult
        for (int i = 0; i < CHUNK_SIZE / 2; ++i) {
            o[i] = ku::float2_mul(o[i], scale_for_old_float2);
        }

        // Store O
        ku::tmem_st_32dp32bNx<CHUNK_SIZE>(tmem_cols::O + chunk_idx * CHUNK_SIZE, o);
        cutlass::arch::fence_view_async_tmem_store();
    }
    ku::tcgen05_before_thread_sync();
}

fence_view_async_shared();
plan.bar_so_ready[rs.buf_idx].arrive();
```

逻辑 O 的形状是 `[64,512]`，在 WS MMA 的 TMEM 布局中占 `[128 datapath,256 column]`。每个 WG0 线程负责其中一个 datapath，需处理 256 个 FP32；`CHUNK_SIZE=64` 将它分成四次读改写，避免把整行 256 个数同时放进寄存器。

第一块没有旧输出需要缩放，因为 warp4 会以清空累加器的方式执行首次 SV。后续只有 `should_scale_o` 为真时，才发起这四轮 TMEM load、乘法和 store。这正是延迟更新参考值能省下的工作。

TMEM load/store 后的 fence 用来等待相应异步访问满足后续使用要求；`tcgen05_before_thread_sync()` 将这些操作与接下来的线程通知衔接。最后，shared fence 发布 S，128 个线程共同完成 `bar_so_ready`，warp4 才能执行本轮 SV。

### 11.warp4：SV 矩阵乘法

```cpp
plan.bar_so_ready[rs.buf_idx].wait(rs.bar_phase);
ku::tcgen05_after_thread_sync();
Tensor sS = make_tensor(make_smem_ptr(plan.s_p.s.data()), SmemLayoutS{});
Tensor sV = make_tensor(
    make_smem_ptr(plan.u.kv.dequant[rs.buf_idx].nope.data()),
    SmemLayoutKTilesTransposed_SW128<D_V / 64>{}); // NOTE: For MODEL1, it "expands" to the RoPE part.
ku::utcmma_ss(tiled_mma_O, sS, sV, tO, block_idx == args.start_block_idx);
ku::umma_arrive_noelect(plan.bar_sv_done[rs.buf_idx]);

rs.update();
```

WG0 写入的 S 是当前参考值下的指数权重，尚未除以总分母。矩阵乘法计算的是当前块对分子的贡献：

$$
\underbrace{S}_{64\times64}\;
\underbrace{V}_{64\times512}
=\Delta O\in\mathbb R^{64\times512}.
$$

CuTe 的 MMA B 操作数使用 `[N,K]` 视图，因此将 cache 的 `[token,feature]` 解释成 `[feature,token]`，再由 `Major::MN` 描述相应布局。`SmemLayoutKTilesTransposed_SW128` 组合了坐标映射；底层仍读取同一份 BF16 数据，不需要额外运行转置 kernel。

V32 的 V 只使用 NoPE。MODEL1 的 448 维 NoPE 与 64 维 RoPE 在 `dequant` 中相邻，512 维视图自然延伸到 RoPE。

第一块传入 `true`，清除上一请求留下的 O；其余块在已有 O 上累加。`bar_so_ready` 同时承诺两件事：S 已写好，旧 O 已完成本轮所需的尺度调整。SV 完成后，`bar_sv_done` 通知 WG2 可以覆盖对应的 BF16 NoPE 槽位；MODEL1 的 warp6 也要等它才能覆盖 RoPE。

warp4 按 `QK(i) → SV(i) → QK(i+1)` 的顺序提交指令。WG0 对下一轮 `bar_qk_done` 的等待覆盖该提交序列此前的 MMA，因此再次改写单份 S 缓冲、读取旧 O 时，上一轮 SV 已经结束。最后一块没有下一次 QK 作为衔接，WG0 会在收尾时显式等待最后的 `bar_sv_done`。

#### 11.1 一轮计算后的累计状态

上述 QK、softmax 与 SV 在每个 KV 数据块上重复执行，逐步维护同一份 attention 状态：

| 状态 | 保存位置 | 内容 |
| --- | --- | --- |
| O | TMEM | 当前分片累计的未归一化输出向量 |
| `mi` | 每个 WG0 线程的寄存器 | 当前 head 使用的指数参考值；协作线程一致 |
| `li` | 每个 WG0 线程的寄存器 | 本线程负责的半边 token 的累计分母 |
| `real_mi` | 每个 WG0 线程的寄存器 | 有效分数的历史最大值 |

数学上，它们满足：

$$
O_h=\sum_{j\in\mathrm{已处理块}}2^{x_{h,j}-m_h}V_j,
\qquad
\ell_h=\ell_{h,\mathrm{left}}+\ell_{h,\mathrm{right}}.
$$

`li` 在主循环中不必每块跨线程求和，因为两半始终共享 `mi`，可以各自递推，到末尾再相加。O 的更新已经被拆成 WG0 的尺度变换与 warp4 的 SV 累加。

最后一轮 WG0 刻意不立即调用 `rs.update()`，让槽位和 phase 停留在最后一次 SV 上，便于在收尾时等待它。此时统计量已经算完，最终 O 仍可能正在由 Tensor Core 写入。

### 12.warpgroup0：最终归一化与写回

这个小节完成分片内部的归一化，并根据是否拆分，选择直接输出还是写中间结果。

```cpp
if (real_mi == -CUDART_INF_F) {
    // real_mi == -CUDART_INF_F <=> No valid TopK indices
    // We set li to 0 to fit the definition that li := exp(x[i] - mi)
    li = 0.0f;
    mi = -CUDART_INF_F;
}
```

`real_mi == -inf` 表示从未遇到有效 token。把 `li` 置 0、`mi` 置负无穷，使空分片的统计量与后续合并一致；输出路径会用零缩放系数产生零输出。

```cpp
plan.rowwise_max_buf[idx_in_warpgroup] = li;
NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
li += plan.rowwise_max_buf[idx_in_warpgroup ^ 64];
```

同一 head 的两个线程交换并相加 `li`。此后两者都持有整个 head 在当前分片上的分母，可以分别归一化自己负责的输出列。

```cpp
if (idx_in_warpgroup < B_H) {
    if (args.is_no_split) {
        float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
        //cur_lse = mi * ln(2) + ln(li);
        cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
        //认为当有效Token为空时，cur_lse为正无穷。
        float *gSoftmaxLse = (float *)params.lse + args.batch_idx * params.stride_lse_b +
                             s_q_idx * params.stride_lse_s_q + idx_in_warpgroup;
        *gSoftmaxLse = cur_lse;
    } else {
        float cur_lse = log2f(li) + mi;
        float *gSoftmaxLseAccum = (float *)params.lse_accum +
                                  args.n_split_idx * params.stride_lse_accum_split +
                                  s_q_idx * params.stride_lse_accum_s_q + idx_in_warpgroup;
        *gSoftmaxLseAccum = cur_lse;
    }
}
plan.bar_sv_done[rs.buf_idx].wait(rs.bar_phase);
rs.update();
```

这里由前 64 个线程（`idx_in_warpgroup < B_H`）各自写出对应 head 的 LSE。由 $\ell=\sum_j2^{x_j-m}$ 可得：

$$
L_2=m+\log_2\ell,\qquad L_{\ln}=m\ln2+\ln\ell.
$$

完整请求直接写自然对数 LSE，空输入按接口约定写 `+inf`。分片写以 2 为底的 LSE，空分片保留 `-inf`，这样合并时 `exp2(-inf)` 产生零权重。LSE 在加入 sink 之前计算，记录真实 token 的指数总量。

接着等待最后的 `bar_sv_done`，确认 O 已完成，才推进 `rs` 并读取 TMEM。

**完整请求的输出路径：**

```cpp
if (args.is_no_split) {
    Tensor tma_gO =
        flat_divide(tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx, args.batch_idx),
                    Shape<Int<B_H>, Int<64>>{})(_, _, _0{}, _);
    auto thr_tma = tma_params.tma_O.get_slice(_0{});
    Tensor tma_sO = flat_divide(sO, Shape<Int<B_H>, Int<64>>{})(_, _, _0{}, _);

    float o_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, li + exp2f(attn_sink - mi));
    float2 o_scale_float2 = {o_scale, o_scale};
    float2 o[B_EPI / 2];
    __nv_bfloat162 o_bf16[B_EPI / 2];
    CUTE_UNROLL
    for (int i = 0; i < (D_V / 2) / B_EPI; ++i) {
        // Load
        ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + i * B_EPI, o);
        cutlass::arch::fence_view_async_tmem_load();
        // Scale & Convert
        CUTE_UNROLL
        for (int j = 0; j < B_EPI / 2; ++j) {
            o[j] = ku::float2_mul(o[j], o_scale_float2);
            o_bf16[j] = __float22bfloat162_rn(o[j]);
        }
        // Store
        int col_base = (i * B_EPI >= D_V / 4 ? D_V / 2 : 0) + (i * B_EPI % (D_V / 4));
        CUTE_UNROLL
        for (int j = 0; j < B_EPI / 8; ++j)
            *(__int128_t *)(sO_bases[j] + col_base * B_H) = *(__int128_t *)(&o_bf16[j * 4]);
        // Sync
        fence_view_async_shared();
        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
        // S -> G
        if (warp_idx == 0 && elect_one_sync()) {
            cute::copy(tma_params.tma_O, thr_tma.partition_S(tma_sO(_, _, col_base / 64)),
                       thr_tma.partition_D(tma_gO(_, _, col_base / 64)));
        }
        if (warp_idx == 1 && elect_one_sync()) {
            cute::copy(tma_params.tma_O,
                       thr_tma.partition_S(tma_sO(_, _, col_base / 64 + (D_V / 4) / 64)),
                       thr_tma.partition_D(tma_gO(_, _, col_base / 64 + (D_V / 4) / 64)));
        }
    }
    cute::tma_store_arrive();
}
```

`flat_divide` 把逻辑 `[64,512]` 输出切成八个 `[64,64]` tile：

```
[块内 head, 块内输出维度, head 方向块号, 输出维度方向块号]
[   64,        64,             1,                8       ]
```

`(_, _, _0{}, _)` 固定唯一的 head 方向块号，留下 `[64,64,8]` 的视图。`thr_tma = tma_O.get_slice(_0{})` 获取 CuTe 中这次 TMA 拷贝的参与者视图；`partition_S` 和 `partition_D` 据此组织源、目标 tile。实际发射线程由外层 `warp_idx` 和 `elect_one_sync()` 选择。

有 sink 时，输出乘以：

$$
\mathrm{o\_scale}=\frac{1}{\ell+2^{a\log_2e-m}}.
$$

未传入 sink 时，代码令 `attn_sink=-inf`，额外项自然为零。`li==0` 则直接选零系数。

剩下的关键是将 TMEM 列还原为逻辑输出列。令 `g = idx_in_warpgroup / 64`，`i` 为四次读出中的轮次，则当前线程的逻辑列起点是：

$$
128g+
\begin{cases}
64i,&i<2,\\
256+64(i-2),&i\ge2.
\end{cases}
$$

| TMEM 读出轮次 `i` | 上半 datapath 的输出列 | 下半 datapath 的输出列 |
| ---: | --- | --- |
| 0 | 0–63 | 128–191 |
| 1 | 64–127 | 192–255 |
| 2 | 256–319 | 384–447 |
| 3 | 320–383 | 448–511 |

直接输出路径把 `128g` 预先计入 `sO_bases`，循环中的 `col_base` 只计算剩下的一项。每轮各线程读 64 个 FP32，完成归一化并转换为 BF16，再按 SW128 写入 SMEM。WG0 同步后，warp0、warp1 各选一个线程，发出本轮相隔 128 列的两块 TMA store；四轮合起来覆盖 512 维。

`tma_store_arrive()` 提交这批异步存储。下一请求开始时，WG0 的 `tma_store_wait<0>()` 等待这些操作读完共享内存源数据，再通过 `bar_last_store_done` 允许 KV 覆盖输出暂存区。

**分片请求的输出路径：**

```cpp
else {
    float o_scale =
        li == 0.0f
            ? 0.0f
            : __fdividef(
                  1.0f,
                  li); // Here we leave attn_sink to the combine kernel, otherwise attn_sink will take effect for multiple times
    float2 o_scale_float2 = {o_scale, o_scale};
    constexpr int B_EPI = 64;
    float2 o[B_EPI / 2];
    Tensor sO = make_tensor(make_smem_ptr(plan.u.qo.o.o_accum_buf.data()), SmemLayoutOAccumBuf{});
    CUTE_UNROLL
    for (int i = 0; i < (D_V / 2) / B_EPI; ++i) {
        // Load
        ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + i * B_EPI, o);
        cutlass::arch::fence_view_async_tmem_load();
        // Scale & Convert
        CUTE_UNROLL
        for (int j = 0; j < B_EPI / 2; ++j)
            o[j] = ku::float2_mul(o[j], o_scale_float2);
        // Store
        int col_base = (idx_in_warpgroup / 64) * 128 + (i * B_EPI >= D_V / 4 ? D_V / 2 : 0) +
                       (i * B_EPI % (D_V / 4));
        CUTE_UNROLL
        for (int j = 0; j < B_EPI / 4; ++j)
            *(__int128_t *)&sO(idx_in_warpgroup % 64, col_base + j * 4) = *(__int128_t *)(&o[j * 2]);
    }
    fence_view_async_shared();
    NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
    if (elect_one_sync()) {
        CUTE_UNROLL
        for (int local_row = 0; local_row < B_H / 4; ++local_row) {
            int smem_row = local_row * 4 + warp_idx;
            SM90_BULK_COPY_S2G::copy(
                &sO(smem_row, _0{}),
                (float *)params.o_accum + args.n_split_idx * params.stride_o_accum_split +
                    s_q_idx * params.stride_o_accum_s_q + smem_row * params.stride_o_accum_h_q,
                D_V * sizeof(float));
        }
        cute::tma_store_arrive();
    }
}
```

分片同样需要归一化，但只除以本片的 `li`，保存 $\mathbf o_r=\mathbf u_r/\ell_r$，并保留 FP32 精度。sink 留给 combine 加入一次，否则每个 split 都加入一份 sink，会改变整体分母。

这里的 `col_base` 显式包含 `128g`，与上表完全对应。目标 `sO(row,col)` 使用行跨度 520 的普通 FP32 布局；每次 16 B 写入四个连续 float。与 BF16 路径相比，改变的是目标布局与输出类型，逻辑列映射相同。

128 个线程写完后，四个 warp 各选一个线程，将自己的 16 行从 SMEM 搬到 `o_accum`。warp `w` 负责行 `w, w+4, ..., w+60`，每行复制 `512 × 4 = 2048 B`。全局地址中的 `n_split_idx` 选择前面计算的分片槽位。

随后由 [combine kernel](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/csrc/smxx/decode/combine/combine.cu) 读取各片的 FP32 输出和 base-2 LSE，按第 1 章推导的权重合并，加入一次 sink，并写出最终结果。当前 CTA 的全部请求处理完后，warp0 释放最初申请的 512 列 TMEM。

## 3.额外的讨论

前面已经说明计算如何完成，这一章继续讨论几个决定资源组织方式的实现细节：调度元数据的宽度、共享内存容量、加载指令参数，以及 warpgroup 之间的寄存器分配。

### 1.DecodingSchedMeta大小设为32B的作用

`DecodingSchedMeta` 包含七个有意义的 `int`，共 28 B；加上一个 4 B 的 `_pad`，成员总长度变成 32 B。声明中的 `__align__(4*8)` 又要求对象按 32 B 对齐，因此数组中第 i 项的起点是 `base + 32*i`，每一项都满足 32 B 对齐。

| 字节区间（左闭右开） | 字段 |
| --- | --- |
| `[0,4)` | `begin_req_idx` |
| `[4,8)` | `end_req_idx` |
| `[8,12)` | `begin_block_idx` |
| `[12,16)` | `end_block_idx` |
| `[16,20)` | `begin_split_idx` |
| `[20,24)` | `is_first_req_splitted` |
| `[24,28)` | `is_last_req_splitted` |
| `[28,32)` | `_pad[0]` |

这个尺寸直接配合 `KU_LDG_256`。宏展开为一条带 `.v4.u64` 的 global load：四个 64-bit 量合计 256 bit，正好覆盖一整条元数据。读出的四个值通过寄存器输出约束填入本线程的 `sched_meta`，随后按 `int` 字段使用。

即使删去显式 `_pad`，保留 32 B 对齐时，结构体仍会因为尾部填充保持 32 B；显式成员使这段固定宽度在定义中清楚可见。它不增加一项调度信息，也不需要参与判断。

因此，32 B 的设计除了让数组寻址偏移可以通过位移计算外，更核心的作用是让元数据的大小与对齐要求匹配 256-bit 向量化加载指令。定义见 [params.h](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/csrc/params.h)，读取宏见 [SM100 intrinsics](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh)。

### 2.不同轨迹下SharedMemoryPlan占用空间与对齐标准

`SharedMemoryPlan` 描述一个 CTA 的共享内存布局。本文分别计算 V32 和 MODEL1 配置，重点使用 V32 解释。

同一配置中的 BF16 最终输出路径与 FP32 分片输出路径使用同一个结构体。运行时选择哪条输出路径，不会改变 `sizeof(SharedMemoryPlan)`。

以下结果依据源码的类型、布局与对齐规则推导；单位为字节，1 KiB = 1024 B。

#### 1 配置参数与 scale 类型

| 参数                    |  V32 |              MODEL1 |
| ----------------------- | ---: | ------------------: |
| `B_H`                   |   64 |                  64 |
| `B_TOPK`                |   64 |                  64 |
| `D_Q`                   |  576 |                 512 |
| `D_Q_SW128`             |  512 |                 512 |
| `D_Q_SW64`              |   64 |                   0 |
| `D_NOPE`                |  512 |                 448 |
| `D_ROPE`                |   64 |                  64 |
| `D_V`                   |  512 |                 512 |
| `NUM_BUFS`              |    2 |                   2 |
| `NUM_INDEX_BUFS`        |    4 |                   4 |
| `NUM_SCALES_EACH_TOKEN` |    4 | 8，包含一个填充位置 |

V32 中，每个 token 的 512 维 NoPE 分为四个 128 维量化组。GMEM 中保存四个 FP32 scale，每个 4 B；warp7 将其转换为 E8M0 编码后存入 SMEM，每个编码 1 B；WG2 再转换为 BF16，用于反量化。

MODEL1 中，每个 token 的 448 维 NoPE 分为七个 64 维量化组。scale 使用 E8M0 编码，七个有效编码加一个填充字节，共 8 B。

因此，计算 `SharedMemoryPlan.scales` 时，两种配置均按每个 E8M0 存储位置 1 B 计算。

#### 2 基本类型与对齐规则

| 类型或声明                    |                                                 大小 |      对齐要求 |
| ----------------------------- | ---------------------------------------------------: | ------------: |
| `bf16`                        |                                                  2 B |           2 B |
| `e4m3`                        |                                                  1 B |           1 B |
| `e8m0`                        |                                                  1 B |           1 B |
| `char`                        |                                                  1 B |           1 B |
| `float`                       |                                                  4 B |           4 B |
| `int`                         |                                                  4 B |           4 B |
| `uint32_t`                    |                                                  4 B |           4 B |
| `float4`                      |                                                 16 B |          16 B |
| `transac_bar_t`               |                                                  8 B |           8 B |
| `array_aligned<T,N>`          | 本文非零长度实例按元素存储大小向上补齐到 16 B 的倍数 |     默认 16 B |
| `CUTE_ALIGNAS(16)` 修饰的成员 |                               根据成员类型和数量计算 | 显式要求 16 B |

计算规则：

1. 普通结构体按声明顺序排列成员，各成员起点满足自身对齐要求。
2. `union` 的成员从同一地址开始，大小取最大成员，再按 union 的对齐要求补齐。
3. 结构体或 union 的对齐要求取所有成员中的最大值。
4. 对象总大小必须是其对齐值的整数倍，因此末尾可能有填充。
5. 数组元素间隔为元素类型的 `sizeof`；数组对齐由其元素类型决定。

SW64、SW128 是 swizzle 排列模式；成员的 C++ 对齐要求仍由声明类型决定。

#### 3 `u.qo`：Q 与输出暂存区

```cpp
struct {
    array_aligned<bf16, cosize_v<SmemLayoutQ_SW128>> q;
    bf16 q_sw64[B_H * D_Q_SW64];
    union {
        array_aligned<bf16, cosize_v<SmemLayoutOBuf>> o_buf;
        array_aligned<float, cosize_v<SmemLayoutOAccumBuf>> o_accum_buf;
    } o;
} qo;
```

`q` 和 `q_sw64` 暂存从 GMEM 搬入的 BF16 Q，随后复制到 TMEM。V32 中二者分别保存 NoPE 与 RoPE；MODEL1 的完整 Q 放在 `q` 中。

`o_buf` 保存归一化后的 BF16 最终输出；`o_accum_buf` 保存归一化后的 FP32 分片输出，供后续 combine 使用。逐块 SV 的累积发生在 TMEM 中。

##### 3.1 布局覆盖的元素数

`cosize_v<Layout>` 表示该布局需要覆盖的元素存储位置数，包含布局中的空隙。

- `SmemLayoutQ_SW128`：`64 × 512 = 32768` 个元素。
- `SmemLayoutOBuf`：`64 × 512 = 32768` 个元素。
- `SmemLayoutOAccumBuf`：形状为 `[64,512]`，行跨度为 520。

FP32 输出布局的元素偏移为：

$$
\operatorname{offset}(i,j)=520i+j
$$

最后一个元素位于 `63 × 520 + 511 = 33271`，因此需要：

$$
\operatorname{cosize}=33272
$$

其中包含 63 处行间空隙，最后一行之后没有额外添加八个元素的空隙。

##### 3.2 成员大小与对齐

| 成员            |             V32 大小/B | MODEL1 大小/B | 对齐要求 |
| --------------- | ---------------------: | ------------: | -------: |
| `q`             | `64 × 512 × 2 = 65536` |         65536 |     16 B |
| `q_sw64`        |   `64 × 64 × 2 = 8192` |             0 |      2 B |
| `o.o_buf`       | `64 × 512 × 2 = 65536` |         65536 |     16 B |
| `o.o_accum_buf` |   `33272 × 4 = 133088` |        133088 |     16 B |
| 输出 union `o`  |                 133088 |        133088 |     16 B |

`q_sw64` 的类型要求是 2 B 对齐。由于它紧跟在大小为 65536 B、按 16 B 对齐的 `q` 后面，实际起点也满足 16 B 对齐。

MODEL1 使用 CUDA 编译环境支持的零长度数组写法，使 `q_sw64` 占 0 B。源码特意没有使用 `array_aligned<bf16,0>`，因为后者会占 16 B。

两个输出数组共用空间：

$$
\operatorname{sizeof}(o)=\max(65536,133088)=133088\ \text{B}
$$

相对于 `qo` 起点的偏移：

| 成员                        | V32 偏移/B | MODEL1 偏移/B |
| --------------------------- | ---------: | ------------: |
| `q`                         |          0 |             0 |
| `q_sw64`                    |      65536 |         65536 |
| `o`，以及其中两个数组的起点 |      73728 |         65536 |

总大小：

$$
\operatorname{sizeof}(qo)_{\mathrm{V32}}=65536+8192+133088=206816\ \text{B}
$$

$$
\operatorname{sizeof}(qo)_{\mathrm{MODEL1}}=65536+133088=198624\ \text{B}
$$

`qo` 按 16 B 对齐，成员间及末尾无须额外填充。

#### 4 `u.kv`：KV 双缓冲

```cpp
struct {
    struct {
        array_aligned<bf16, B_H * D_NOPE> nope;
        array_aligned<bf16, B_H * D_ROPE> rope;
    } dequant[NUM_BUFS];
    static_assert(sizeof(dequant) >= sizeof(bf16) * (B_H * D_Q));
    array_aligned<e4m3, B_H * D_NOPE> raw_nope[NUM_BUFS];
} kv;
```

`raw_nope` 保存刚搬入的 FP8 NoPE；WG2 使用 scale 反量化后写入 `dequant[i].nope`。RoPE 已经是 BF16，由 warp6 直接搬入 `dequant[i].rope`。

这部分数组中的 64 行对应 64 个 KV token；源码使用 `B_H`，本配置中它与 `B_TOPK` 都等于 64。

##### 4.1 成员大小与对齐

| 成员                   |             V32 大小/B |          MODEL1 大小/B | 对齐要求 |
| ---------------------- | ---------------------: | ---------------------: | -------: |
| 一个 `dequant[i].nope` | `64 × 512 × 2 = 65536` | `64 × 448 × 2 = 57344` |     16 B |
| 一个 `dequant[i].rope` |   `64 × 64 × 2 = 8192` |                   8192 |     16 B |
| 一个 `dequant[i]`      |                  73728 |                  65536 |     16 B |
| 整个 `dequant[2]`      |                 147456 |                 131072 |     16 B |
| 一个 `raw_nope[i]`     |     `64 × 512 = 32768` |     `64 × 448 = 28672` |     16 B |
| 整个 `raw_nope[2]`     |                  65536 |                  57344 |     16 B |

所有成员均已满足 16 B 对齐，无须额外插入填充。

##### 4.2 成员偏移

相对于 `kv` 起点：

| 成员              | V32 偏移/B | MODEL1 偏移/B |
| ----------------- | ---------: | ------------: |
| `dequant[0].nope` |          0 |             0 |
| `dequant[0].rope` |      65536 |         57344 |
| `dequant[1].nope` |      73728 |         65536 |
| `dequant[1].rope` |     139264 |        122880 |
| `raw_nope[0]`     |     147456 |        131072 |
| `raw_nope[1]`     |     180224 |        159744 |

总大小：

$$
\operatorname{sizeof}(kv)_{\mathrm{V32}}=147456+65536=212992\ \text{B}
$$

$$
\operatorname{sizeof}(kv)_{\mathrm{MODEL1}}=131072+57344=188416\ \text{B}
$$

`kv` 的对齐要求为 16 B。

`static_assert` 是编译期检查，不占空间。它要求 Q 暂存区完全落在 `dequant` 覆盖的范围内，避免延伸至 `raw_nope`：

| 配置   | Q 暂存区/B | 整个 `dequant`/B |
| ------ | ---------: | ---------------: |
| V32    |      73728 |           147456 |
| MODEL1 |      65536 |           131072 |

该断言约束地址布局。当前实现中，warp5、warp6 仍然等待 Q 复制到 TMEM 完成后才开始搬运 KV；这条断言不代表已实现 Q 与原始 KV 的并行搬入。

#### 5 外层 union `u`

`qo` 与 `kv` 共用起始地址，大小取二者最大值：

| 配置   | `sizeof(qo)`/B | `sizeof(kv)`/B | `sizeof(u)`/B | 对齐要求 |
| ------ | -------------: | -------------: | ------------: | -------: |
| V32    |         206816 |         212992 |    **212992** |     16 B |
| MODEL1 |         198624 |         188416 |    **198624** |     16 B |

V32 由 KV 缓冲区决定这段空间的大小；MODEL1 由 Q 与输出暂存区决定大小。

#### 6 `s_p`：分数交换与指数权重

```cpp
union {
    float4 p_exchange_buf[4][16 * B_TOPK / 4];
    array_aligned<bf16, cosize_v<SmemLayoutS>> s;
} s_p;
```

两个模型分支都使用 dual GEMM。WG0 先使用 `p_exchange_buf` 交换部分分数并相加；读取完成、同步后，再把同一段内存作为 `s`，保存交给 SV 的 BF16 指数权重。

`B_TOPK = 64`，因此交换区为 `float4[4][256]`。`SmemLayoutS` 覆盖 `[64,64]` 个 BF16 元素。

| 成员             |                 大小/B |                      对齐要求 |
| ---------------- | ---------------------: | ----------------------------: |
| `p_exchange_buf` | `4 × 256 × 16 = 16384` |        16 B，由 `float4` 决定 |
| `s`              |   `64 × 64 × 2 = 8192` | 16 B，由 `array_aligned` 决定 |
| union `s_p`      |              **16384** |                          16 B |

两个成员相对于 `s_p` 的偏移均为 0，union 无须额外填充。

#### 7 行统计与四槽辅助信息

```cpp
CUTE_ALIGNAS(16) float rowwise_max_buf[128];
char is_token_valid[NUM_INDEX_BUFS][B_TOPK / 8];
int tma_coord[NUM_INDEX_BUFS][B_TOPK];
e8m0 scales[NUM_INDEX_BUFS][B_TOPK][NUM_SCALES_EACH_TOKEN];
```

| 成员              | 用途                                         |          V32 大小/B |       MODEL1 大小/B | 声明要求的对齐 |
| ----------------- | -------------------------------------------- | ------------------: | ------------------: | -------------: |
| `rowwise_max_buf` | WG0 交换局部最大值；收尾时复用来交换局部分母 |     `128 × 4 = 512` |                 512 | 16 B，显式指定 |
| `is_token_valid`  | 每个 token 一位有效性标记                    |        `4 × 8 = 32` |                  32 |            1 B |
| `tma_coord`       | 每槽 64 个 TMA 搬运坐标                      | `4 × 64 × 4 = 1024` |                1024 |            4 B |
| `scales`          | NoPE 反量化 scale 编码                       | `4 × 64 × 4 = 1024` | `4 × 64 × 8 = 2048` |            1 B |

后三个数组由 warp7 准备，共同使用 `index_buf_idx` 选择槽位。坐标供 warp5、warp6 使用，scale 供 WG2 使用，有效性标记供 WG0 使用。

虽然后三个成员自身未要求 16 B 对齐，但在当前完整布局中，它们的实际起点均落在 16 B 边界上。成员之间无额外填充。

#### 8 `tmem_start_addr`

```cpp
array_aligned<uint32_t, 1> tmem_start_addr;
```

有效数据是一个 4 B 的 TMEM 地址，但对象要求 16 B 对齐，因此 `sizeof` 为 16 B，其中含 12 B 尾部填充。

这个成员只保存地址。TMEM 中 Q、分数和输出累积值占用的空间单独分配，不计入 `SharedMemoryPlan`。

#### 9 Transaction barrier

`transac_bar_t` 为 `cutlass::arch::ClusterTransactionBarrier`。底层为一个 `uint64_t`，每个占 8 B、按 8 B 对齐。

| 成员                          |   个数 |  大小/B | 对齐要求 | 含义                           |
| ----------------------------- | -----: | ------: | -------: | ------------------------------ |
| `bar_last_store_done`         |      1 |       8 |      8 B | 上次输出存储已读完 SMEM 源数据 |
| `bar_q_tma`                   |      1 |       8 |      8 B | Q 从 GMEM 搬入 SMEM 完成       |
| `bar_q_utccp`                 |      1 |       8 |      8 B | Q 从 SMEM 复制到 TMEM 完成     |
| `bar_rope_ready`              |      2 |      16 |      8 B | 对应槽位的 RoPE 搬运完成       |
| `bar_nope_ready`              |      2 |      16 |      8 B | 对应槽位的 NoPE 反量化完成     |
| `bar_raw_ready`               |      2 |      16 |      8 B | 对应槽位的原始 FP8 搬运完成    |
| `bar_raw_free`                |      2 |      16 |      8 B | 对应原始 FP8 槽位可以复用      |
| `bar_valid_coord_scale_ready` |      4 |      32 |      8 B | 对应辅助信息槽位已准备好       |
| `bar_valid_coord_scale_free`  |      4 |      32 |      8 B | 对应辅助信息槽位可以复用       |
| `bar_qk_done`                 |      2 |      16 |      8 B | 对应块的 QK 完成               |
| `bar_so_ready`                |      2 |      16 |      8 B | S 已写好，O 所需的缩放已完成   |
| `bar_sv_done`                 |      2 |      16 |      8 B | 对应块的 SV 完成               |
| **合计**                      | **25** | **200** |          |                                |

这些 barrier 连续排列；起点已满足 8 B 对齐，彼此之间无额外填充。数组内相邻 barrier 间隔为 8 B。

Named barrier 使用硬件命名屏障资源，不是这个结构体中的存储成员，不计入以上容量。

#### 10 完整偏移表

所有偏移相对于 `SharedMemoryPlan` 的起点，单位为 B：

| 成员                          | V32 偏移 |   V32 大小 | MODEL1 偏移 | MODEL1 大小 |
| ----------------------------- | -------: | ---------: | ----------: | ----------: |
| `u`                           |        0 |     212992 |           0 |      198624 |
| `s_p`                         |   212992 |      16384 |      198624 |       16384 |
| `rowwise_max_buf`             |   229376 |        512 |      215008 |         512 |
| `is_token_valid`              |   229888 |         32 |      215520 |          32 |
| `tma_coord`                   |   229920 |       1024 |      215552 |        1024 |
| `scales`                      |   230944 |       1024 |      216576 |        2048 |
| `tmem_start_addr`             |   231968 |         16 |      218624 |          16 |
| `bar_last_store_done`         |   231984 |          8 |      218640 |           8 |
| `bar_q_tma`                   |   231992 |          8 |      218648 |           8 |
| `bar_q_utccp`                 |   232000 |          8 |      218656 |           8 |
| `bar_rope_ready`              |   232008 |         16 |      218664 |          16 |
| `bar_nope_ready`              |   232024 |         16 |      218680 |          16 |
| `bar_raw_ready`               |   232040 |         16 |      218696 |          16 |
| `bar_raw_free`                |   232056 |         16 |      218712 |          16 |
| `bar_valid_coord_scale_ready` |   232072 |         32 |      218728 |          32 |
| `bar_valid_coord_scale_free`  |   232104 |         32 |      218760 |          32 |
| `bar_qk_done`                 |   232136 |         16 |      218792 |          16 |
| `bar_so_ready`                |   232152 |         16 |      218808 |          16 |
| `bar_sv_done`                 |   232168 |         16 |      218824 |          16 |
| 结构体尾部填充                |   232184 |          8 |      218840 |           8 |
| **总大小**                    |          | **232192** |             |  **218848** |

结构体最大成员对齐要求为 16 B，因此 `alignof(SharedMemoryPlan) = 16`。两个配置在最后一个 barrier 后均需补 8 B，保证对象总大小为 16 的倍数。

#### 11 总容量与实际申请

V32：

$$
\begin{aligned}
\operatorname{sizeof}(\text{SharedMemoryPlan})
&=212992+16384+512+32+1024+1024+16+200+8\\
&=232192\ \text{B}\\
&=226.75\ \text{KiB}
\end{aligned}
$$

MODEL1：

$$
\begin{aligned}
\operatorname{sizeof}(\text{SharedMemoryPlan})
&=198624+16384+512+32+1024+2048+16+200+8\\
&=218848\ \text{B}\\
&=213.71875\ \text{KiB}
\end{aligned}
$$

这两个结果都包含成员对象内部的必要填充以及整个结构体的尾部填充。union 中重叠的数组没有重复计数。

源码使用该大小申请每个 CTA 的动态共享内存：

```cpp
constexpr size_t smem_size = sizeof(SharedMemoryPlan);
static_assert(smem_size < 227 * 1024);
```

随后将 `smem_size` 作为 kernel 启动时的动态共享内存参数。V32 的 226.75 KiB 满足“小于 227 KiB”的检查。

### 3.KU_LDG_256的参数作用

本文中的调用是：

```cpp
KU_LDG_256(params.tile_scheduler_metadata_ptr + partition_idx, &sched_meta, ".nc", "no_allocate",
           "evict_normal", "256B");
```

代入宏定义后，关键 PTX 指令为：

```ptx
ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v4.u64
    {r0, r1, r2, r3}, [global_addr];
```

逐项拆开即可读懂：

| 参数或限定符 | 作用 |
| --- | --- |
| `global_addr` | 当前 partition 的调度元数据地址 |
| `result` / `&sched_meta` | 接收加载结果的本线程对象；宏把它解释成四个 `uint64_t` |
| `.global` | 从 global memory 地址空间读取 |
| `.nc` | 使用非一致性只读缓存路径，适合执行期间不变的调度元数据 |
| `.L1::no_allocate` | 提示不为本次读取在 L1 中分配新缓存行 |
| `.L2::evict_normal` | 使用普通的 L2 驱逐优先级 |
| `.L2::256B` | 256 B 的 L2 预取大小提示 |
| `.v4.u64` | 指令实际返回四个 64-bit 值，共 32 B |

这里有两个不同的 256：宏名中的 `256` 是读取宽度，单位为 bit；最后参数中的 `256B` 是预取提示，单位为 byte。目标对象只接收 32 B，不会被写入 256 B，也不会因此同时得到八个 partition 的任务。

L1、L2 选项描述缓存策略，不承担跨线程同步。调度元数据在前序步骤生成，本 kernel 执行期间只读；加载结果可以用于当前线程的循环边界和地址计算。宏中的 `asm volatile` 保留这条显式汇编操作，输出约束表达读出值到后续使用之间的依赖。

代码将这次读取放在各个 warpgroup 的专职路径内部。这样做还影响寄存器生命周期：任务边界只在对应路径中保持存活，避免外提后跨越整个分工分支。缓存限定符的定义见 [PTX 的 ld 与 ld.global.nc 说明](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ld-nc)。

### 4.各warpgroup寄存器数目的变化以及作用

三个 warpgroup 在进入自己的分支时，分别执行：

```cpp
// WG0：softmax、输出缩放与写回
cutlass::arch::warpgroup_reg_alloc<224>();

// WG1：MMA/TMA 发射、索引与 scale 准备
cutlass::arch::warpgroup_reg_dealloc<72>();

// WG2：FP8 反量化
cutlass::arch::warpgroup_reg_alloc<208>();
```

这些数字表示每线程的寄存器目标上限。`alloc<224>()` 把上限调整到 224，`dealloc<72>()` 把上限调整到 72；它们不是“额外申请 224 个”或“释放 72 个”。CUTLASS 的包装分别对应 `setmaxnreg.inc` 与 `setmaxnreg.dec`。

按三个 warpgroup 各 128 个线程计算，调整后的预算为：

| Warpgroup | 每线程目标寄存器数 | 线程数 | 32-bit 寄存器总数 | 主要用途 |
| --- | ---: | ---: | ---: | --- |
| WG0 | 224 | 128 | 28672 | 分数、指数权重、行统计、O 的分块读改写 |
| WG1 | 72 | 128 | 9216 | 描述符、坐标、循环状态、少量 scale 数据 |
| WG2 | 208 | 128 | 26624 | 展开的 FP8/BF16 转换、scale、地址与循环临时量 |
| 合计 | 平均 168 | 384 | **64512** | 合计 252 KiB |

预算不等于每个时刻都活跃的变量数量。源码给出了三个分支的目标上限；具体编译产物的实际分配、spill 和初始寄存器数，需要看对应构建的编译报告。

**WG0：分数和输出片段占用寄存器。** 每线程一开始就要读两组 32 个 FP32 分数，随后还要计算指数、维护 `mi/li/real_mi`，并以 64 个 FP32 一组读取和缩放 O。写回阶段又需要 FP32 到 BF16 的转换临时量。`CHUNK_SIZE=64` 限制了一次保留的输出片段大小，但这个角色的寄存器需求仍然明显高于纯指令发射者。

**WG1：主要持有描述符与调度状态。** warp4、5、6 各由一个线程提交 MMA/TMA，矩阵本体保存在 SMEM 和 TMEM，计算或搬运由异步硬件完成。warp7 虽然 32 个线程全部工作，每线程也只准备两个 token 的索引和 scale，因此这一组不需要按 softmax 的峰值配置资源。分配调整发生在 `warpgroup_idx == 1` 分支入口，四个 warp 一起执行，之后才进一步按 warp 分工。

**WG2：展开反量化与预读需要临时空间。** 它显式展开多个 token、多个特征段的 load、scale 转换与 BF16 写出，并提前加载下一轮 FP8。较大的寄存器预算给编译器保留这些临时量和安排指令重叠的空间。这里为 WG2 设定的每线程寄存器目标上限为 208。

`setmaxnreg` 在 CTA 的寄存器池内调配资源：WG1 降低上限，释放出的预算可供 WG0、WG2 请求；增加上限时如果资源暂时不足，发出请求的 warp 会等待。要求同一 warpgroup 的所有 warp 执行相同调整，因此不能只在 warp4 的选举线程里调用它。对应语义见 [PTX setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg)。

这份实现没有在每个 KV 块或每个请求之间来回切换寄存器上限，而是在专职分支入口调整一次，持续执行各自的工作。因为 WG0、WG1、WG2 同时存在，如果统一按 224 配置，384 个线程就要预算 `224 × 384 = 86016` 个寄存器；按角色划分后只需 64512 个。

源码对调度变量的位置也有两条具体提示：`run_main_loop` 的元数据读取不能简单提到专职分支外，WG1 内还重新取得一次 `warp_idx`，注释说明相应改动会影响 spilling。它们说明，在这种高资源占用的 kernel 中，变量的存活范围与角色分工和算术本身一样，会影响最终生成的代码。
