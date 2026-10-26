---
title: basic-train 从 MLP 手算到 Adam 和混合精度主权重
date: 2026-10-26
tags:
  - AI-Infra
  - Training
  - Megatron
  - PyTorch
  - Mixed-Precision
categories:
  - 基础
mathjax: true
---

从一个 2 → 3 → 2、隐藏层使用 ReLU 的 MLP 出发，手算前向、反向和一次参数更新。随后讨论 Adam 的历史状态、混合精度主权重及其 Megatron 实现，再用参数量 $\Psi$ 建立显存账本，最后通过 batch 与梯度累积衔接 TP/DP 和 FSDP 的取舍。

<!-- more -->

# basic-train 从 MLP 手算到 Adam 和混合精度主权重

一次训练迭代需要完成三件事：用当前参数计算预测和 loss（Forward），沿计算图求出梯度（Backward），再用梯度与优化器状态更新参数（Optimizer step）。前向留下的中间结果、反向产生的梯度和跨迭代保存的优化器状态，生命周期并不相同。后面的 TP、DP、PP 和分布式优化器，都会改变这些计算或状态在设备之间的分工。

数学部分采用行向量与右乘权重的约定；数值核对使用 CPU 上的 FP64，舍入实验使用 FP16、BF16 和 FP32。Megatron 源码对照固定在 `core_v0.12.0` 的提交 `d580efc68a9f0dbf1945f834f6f6200cd01d3343`。

## 1 基础 MLP 与单卡训练

### 1.1 基本MLP

网络有两个输入特征、三个隐藏单元和两个输出。隐藏层使用 ReLU，输出层是线性层。

```text
X [B, 2]
    │  W₁ [2, 3]，b₁ [3]
    ▼
Z [B, 3] ── ReLU ──▶ H [B, 3]
                         │  W₂ [3, 2]，b₂ [2]
                         ▼
                      Ŷ [B, 2] ── 与目标 Y 比较 ──▶ 标量 L
```

其中 $B$ 是 batch 中的样本数。计算式为：

$$
Z=XW_1+b_1,\qquad H=\operatorname{ReLU}(Z),\qquad
\hat Y=HW_2+b_2.
$$

偏置沿 batch 维广播。参数总数为 $2\times3+3+3\times2+2=17$。

为使手算简单，使用平方误差，并明确归一化方式：**对每个样本的两个输出求平方和，再对样本取平均**。

$$
L=\frac{1}{2B}\sum_{n=1}^{B}\sum_{j=1}^{2}
(\hat Y_{n,j}-Y_{n,j})^2.
$$

因此 $\partial L/\partial\hat Y=(\hat Y-Y)/B$。对应代码是：

```python
loss = 0.5 * (prediction - target).square().sum(dim=-1).mean()
```

这里不能直接替换成 `0.5 * F.mse_loss(prediction, target)` 的默认形式：默认 MSE 对所有元素平均，在这个两输出模型上还会多除以 2。两种定义都可以使用，但手算、代码和梯度累积必须采用同一个定义。

### 1.2 手算一次前向

先取一个样本，即 $B=1$：

$$
X=\begin{bmatrix}1&2\end{bmatrix},\qquad
Y=\begin{bmatrix}0&0\end{bmatrix}.
$$

初始参数为：

$$
W_1=\begin{bmatrix}1&-1&1\\1&1&-1\end{bmatrix},
\qquad b_1=\begin{bmatrix}0&0&0\end{bmatrix},
$$

$$
W_2=\begin{bmatrix}1&-1\\-2&1\\1&2\end{bmatrix},
\qquad b_2=\begin{bmatrix}0&0\end{bmatrix}.
$$

第一层线性变换：

$$
Z=\begin{bmatrix}
1\times1+2\times1&
1\times(-1)+2\times1&
1\times1+2\times(-1)
\end{bmatrix}
=\begin{bmatrix}3&1&-1\end{bmatrix}.
$$

ReLU 将第三个隐藏单元的输出截为零：

$$
H=\begin{bmatrix}3&1&0\end{bmatrix}.
$$

第二层线性变换：

$$
\hat Y=\begin{bmatrix}
3\times1+1\times(-2)+0\times1&
3\times(-1)+1\times1+0\times2
\end{bmatrix}
=\begin{bmatrix}1&-2\end{bmatrix}.
$$

于是：

$$
L=\frac12\left(1^2+(-2)^2\right)=2.5.
$$

Forward 还要保留 backward 所需的信息：第二层求权重梯度时要用 $H$，ReLU 反向要知道哪些位置处于正半轴。

### 1.3 从 loss 逐层手算梯度

下面用 $G_A=\partial L/\partial A$ 表示张量 $A$ 的梯度。因为 loss 是标量，$G_A$ 与 $A$ 的 shape 相同。

#### 1.3.1 线性层为什么有两条反向路径

先看一般线性层 $V=UW+b$。按元素展开：

$$
V_{n,j}=\sum_i U_{n,i}W_{i,j}+b_j.
$$

对同一个权重 $W_{i,j}$，不同样本都会贡献 loss，因此：

$$
\frac{\partial L}{\partial W_{i,j}}
=\sum_n\frac{\partial L}{\partial V_{n,j}}U_{n,i}.
$$

对同一个输入 $U_{n,i}$，不同输出位置都会贡献 loss，因此：

$$
\frac{\partial L}{\partial U_{n,i}}
=\sum_j\frac{\partial L}{\partial V_{n,j}}W_{i,j}.
$$

写成矩阵式就是：

$$
G_W=U^TG_V,\qquad G_U=G_VW^T,\qquad
G_b=\sum_nG_V[n,:].
$$

这给出训练系统里常见的两个名字：**WGRAD** 计算本层参数梯度，供优化器更新；**DGRAD** 计算本层输入梯度，供前面的计算继续反向传播。在拿到 $G_V$、保存的 $U$ 和本次前向使用的 $W$ 后，两者在数学上没有先后依赖；实际执行次序和并发取决于实现及资源。

#### 1.3.2 loss 产生输出梯度

本例 $B=1$，所以：

$$
G_{\hat Y}=\hat Y-Y=\begin{bmatrix}1&-2\end{bmatrix}.
$$

#### 1.3.3 第二层同时产生参数梯度和隐藏层梯度

$$
G_{W_2}=H^TG_{\hat Y}
=\begin{bmatrix}3\\1\\0\end{bmatrix}
\begin{bmatrix}1&-2\end{bmatrix}
=\begin{bmatrix}3&-6\\1&-2\\0&0\end{bmatrix}.
$$

$$
G_{b_2}=\begin{bmatrix}1&-2\end{bmatrix}.
$$

送往隐藏层的梯度为：

$$
G_H=G_{\hat Y}W_2^T
=\begin{bmatrix}
1\times1+(-2)\times(-1)&
1\times(-2)+(-2)\times1&
1\times1+(-2)\times2
\end{bmatrix}
=\begin{bmatrix}3&-4&-3\end{bmatrix}.
$$

第三个隐藏单元的输出虽然为零，**对这个输出本身的梯度仍然可以非零**。是否继续传过 ReLU，要看下一步的局部导数。

#### 1.3.4 ReLU 决定哪些梯度继续向前传

对于 $z>0$，ReLU 的导数为 1；对于 $z<0$，导数为 0。零点不可微，PyTorch 在零点采用 0；本例没有落在零点。[不可微位置的梯度约定][autograd]

$$
G_Z=G_H\odot\mathbf1[Z>0]
=\begin{bmatrix}3&-4&-3\end{bmatrix}
\odot\begin{bmatrix}1&1&0\end{bmatrix}
=\begin{bmatrix}3&-4&0\end{bmatrix}.
$$

其中 $\odot$ 表示逐元素乘。

第三个隐藏单元的两侧出现了两种不同原因造成的零梯度：第二层对应行的 $G_{W_2}$ 为零，是因为输入激活 $H_3=0$；第一层对应列的 $G_{W_1}$ 为零，是因为 ReLU 的局部导数使 $G_{Z_3}=0$。

#### 1.3.5 第一层的参数梯度与输入梯度

$$
G_{W_1}=X^TG_Z
=\begin{bmatrix}1\\2\end{bmatrix}
\begin{bmatrix}3&-4&0\end{bmatrix}
=\begin{bmatrix}3&-4&0\\6&-8&0\end{bmatrix},
$$

$$
G_{b_1}=\begin{bmatrix}3&-4&0\end{bmatrix}.
$$

若还需要输入梯度：

$$
G_X=G_ZW_1^T
=\begin{bmatrix}3&-4&0\end{bmatrix}
\begin{bmatrix}1&1\\-1&1\\1&-1\end{bmatrix}
=\begin{bmatrix}7&-1\end{bmatrix}.
$$

原始输入通常不需要更新，因此训练时可以不请求它的梯度；若这里的 $X$ 是前一层的输出，这份 DGRAD 就是前一层 backward 的输入。**某层权重被冻结，也不意味着可以跳过该层 DGRAD：前面如果还有需要训练的层，梯度仍要传过去。**

到这里 backward 已经完成，但所有参数仍是原值。特别是，计算 $G_H$ 时必须使用本次前向对应的 $W_2$，不能先更新 $W_2$ 再用新权重传播梯度。

### 1.4 用 SGD 完成第一次更新

最简单的 SGD 只使用当前参数和当前梯度：

$$
\theta_{t+1}=\theta_t-\eta g_t.
$$

取学习率 $\eta=0.01$，更新后得到：

$$
W_1'=\begin{bmatrix}0.97&-0.96&1\\0.94&1.08&-1\end{bmatrix},
\qquad b_1'=\begin{bmatrix}-0.03&0.04&0\end{bmatrix},
$$

$$
W_2'=\begin{bmatrix}0.97&-0.94\\-2.01&1.02\\1&2\end{bmatrix},
\qquad b_2'=\begin{bmatrix}-0.01&0.02\end{bmatrix}.
$$

对同一个样本重新前向：

$$
Z'=\begin{bmatrix}2.82&1.24&-1\end{bmatrix},\qquad
\hat Y'=\begin{bmatrix}0.233&-1.366\end{bmatrix},
$$

$$
L'=\frac12(0.233^2+1.366^2)=0.9601225.
$$

样本 loss 从 2.5 降到了 0.9601225。

#### 1.4.1 用 PyTorch 对照手算

PyTorch 的 `nn.Linear(in_features, out_features)` 将权重保存为 `[out_features, in_features]`，实际计算相当于 $X\cdot\texttt{weight}^T+b$。因此下面初始化时要把本文的 $W_1,W_2$ 转置，检查梯度时也要转置回来。[Linear 文档][linear]

以下代码可以在 CPU 上独立运行：

```python
import torch
from torch import nn

dtype = torch.float64

def fresh_model():
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2))
    model = model.to(dtype=dtype)
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(
            [[1., -1., 1.], [1., 1., -1.]], dtype=dtype).T)
        model[0].bias.zero_()
        model[2].weight.copy_(torch.tensor(
            [[1., -1.], [-2., 1.], [1., 2.]], dtype=dtype).T)
        model[2].bias.zero_()
    return model

model = fresh_model()
x = torch.tensor([[1., 2.]], dtype=dtype, requires_grad=True)
y = torch.tensor([[0., 0.]], dtype=dtype)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

optimizer.zero_grad(set_to_none=True)
prediction = model(x)
loss = 0.5 * (prediction - y).square().sum(dim=-1).mean()
loss.backward()

torch.testing.assert_close(loss, torch.tensor(2.5, dtype=dtype))
torch.testing.assert_close(model[0].weight.grad.T, torch.tensor(
    [[3., -4., 0.], [6., -8., 0.]], dtype=dtype))
torch.testing.assert_close(model[0].bias.grad, torch.tensor(
    [3., -4., 0.], dtype=dtype))
torch.testing.assert_close(model[2].weight.grad.T, torch.tensor(
    [[3., -6.], [1., -2.], [0., 0.]], dtype=dtype))
torch.testing.assert_close(model[2].bias.grad, torch.tensor(
    [1., -2.], dtype=dtype))
torch.testing.assert_close(x.grad, torch.tensor([[7., -1.]], dtype=dtype))

optimizer.step()
with torch.no_grad():
    new_prediction = model(x)
    new_loss = 0.5 * (new_prediction - y).square().sum(dim=-1).mean()
torch.testing.assert_close(new_prediction, torch.tensor(
    [[0.233, -1.366]], dtype=dtype))
torch.testing.assert_close(new_loss, torch.tensor(0.9601225, dtype=dtype))
print(loss.item(), new_loss.item())
```

`backward()` 根据计算图计算梯度，并将叶子张量的梯度累积到 `.grad`；`optimizer.step()` 根据已有梯度执行更新。`zero_grad()` 清理的是优化器所管理参数的梯度，不会把权重归零，也不会清空 Adam 的历史状态。上面的 `x` 不在优化器里，如果重复使用并请求 `x.grad`，要单独清理它。[Autograd 说明][autograd]

## 2 优化器与训练状态

### 2.1 Momentum 先保留方向的历史

SGD 每次只看当前梯度。Momentum 引入一个跨更新保存的缓冲：

$$
u_t=\mu u_{t-1}+g_t,\qquad
\theta_t=\theta_{t-1}-\eta u_t.
$$

这里使用 $u_0=0$ 的基础形式，不含 dampening 或 Nesterov。持续同方向的梯度会累积；反复变号的贡献会部分抵消。从系统角度看，每个参数现在需要额外保存一个同形状的历史张量。

Adam 还会跟踪梯度的幅度，为不同参数坐标调整更新尺度。

### 2.2 Adam 的平方项是二阶矩

记第 $t$ 次更新使用的梯度为 $g_t$，从 $m_0=v_0=0$ 开始：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)(g_t\odot g_t).
$$

$m_t$ 是梯度的指数移动平均，保留符号；$v_t$ 是梯度逐元素平方的指数移动平均，即非中心二阶矩估计。

因为初始值是零，需要校正初始化带来的偏小：

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},
$$

$$
\theta_t=\theta_{t-1}-\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

分母按历史梯度尺度调整各坐标的更新，$\epsilon$ 防止除零。“二阶”指统计矩，Adam 使用一阶梯度，不计算 Hessian。[Adam Algorithm 1][adam]

### 2.3 在同一个 MLP 上追踪两次 Adam 更新

将模型重置到第 1.2 节的初始参数，每次使用同一个样本。设置 $\eta=0.01$、$\beta_1=0.9$、$\beta_2=0.999$、$\epsilon=10^{-8}$。选 $W_2$ 第一行第一列这个初始值为 1 的参数观察。

第一次 backward 得到 $g_1=3$：

$$
m_1=0.3,\quad v_1=0.009,\quad
\hat m_1=3,\quad\hat v_1=9.
$$

于是更新量约为 $0.01\times3/3=0.01$，权重变为约 $0.99$。第二次梯度由**全部参数更新后的模型重新前向和反向得到**，不能继续把原来的梯度 3 当作新梯度。

| 更新次数 | 当前梯度 $g_t$ | $m_t$ | $v_t$ | $\hat m_t$ | $\hat v_t$ | 更新后的权重 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.000000 | 0.300000 | 0.009000 | 3.000000 | 9.000000 | 0.990000 |
| 2 | 2.456800 | 0.515680 | 0.015026866 | 2.714105 | 7.517192 | 0.980100828 |

表内保留有限小数，验证脚本使用完整精度与 PyTorch Adam 对照。对于第一次更新、非零梯度且忽略 $\epsilon$ 时，$\hat m_1/\sqrt{\hat v_1}=\operatorname{sign}(g_1)$；这个简化不适用于一般的后续更新。

在 PyTorch 优化器中，常见状态字段对应如下：[AdamW 状态初始化][adamw-init]

| 数学对象 | 代码状态 | 生命周期 |
| --- | --- | --- |
| $m$ | `exp_avg` | 跨多次 step 保留 |
| $v$ | `exp_avg_sq` | 跨多次 step 保留 |
| 更新计数 $t$ | `step` | 实际更新时推进，用于偏差校正 |
| 当前梯度 $g$ | 参数的 `.grad` 或框架梯度 buffer | 累积完成后用于本次更新，下轮重新准备 |

恢复训练时，只有权重 checkpoint 不足以恢复优化器轨迹，还需要 $m/v/t$；完整续训还涉及学习率调度、随机数、数据进度和适用时的 loss scaler 状态。

### 2.4 AdamW 将权重衰减与梯度自适应处理分开

对本文的基础 AdamW 形式，可以写成：

$$
\theta_t=(1-\eta\lambda)\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

$\lambda$ 是 weight decay 系数。衰减项直接作用于参数；用于计算 $m/v$ 的仍是目标函数的梯度。若把 $\lambda\theta$ 加入梯度再交给 Adam，衰减项也会进入历史统计并被自适应缩放，通常不等价于上述解耦形式。[AdamW 更新实现][adamw-update]

例如 $\theta=1$、$\eta=0.01$、$\lambda=0.1$，初始 $m=v=0$ 且显式提供零梯度，一次 AdamW step 后参数为 $0.999$。这里“零梯度张量”有别于 `.grad is None`；常见 PyTorch 实现会跳过后者对应参数的更新。

## 3 混合精度与主权重

Adam 说明了如何决定更新量，混合精度则要求进一步区分：算子用什么精度计算，梯度用什么精度累积，参数用什么精度保存并更新。FP32 主权重解决的是低精度参数写回时的信息损失，这个问题在 SGD 中同样存在。

### 3.1 动态范围与有效位数是两件事

下表中的尾数字段不包含正规数隐含的前导 1：

| 格式 | 指数字段 | 尾数字段 | 对训练的直接影响 |
| --- | ---: | ---: | --- |
| FP32 | 8 位 | 23 位 | 表达较细的参数变化，常用于更新与状态 |
| FP16 | 5 位 | 10 位 | 动态范围较窄，小梯度可能下溢 |
| BF16 | 8 位 | 7 位 | 指数范围与 FP32 相近，但同量级数之间间隔较大 |

BF16 的范围较宽，不能推出它能精细表达参数上的小增量。低精度是否需要 loss scaling，与是否需要高精度地保存权重更新，也必须分开判断。

### 3.2 一个很小的更新怎样被吞掉

设权重初始值为 1，每次更新量固定为 $\Delta=10^{-4}$。

FP16 在紧邻 1 的下方，相邻可表示数的间隔为 $2^{-11}=0.00048828125$。$1-10^{-4}=0.9999$ 舍入到 FP16 后仍是 1。如果每一步都从这个已经舍入的值继续减：

$$
w_{t+1}^{\rm low}
=\operatorname{cast}_{\rm FP16}(w_t^{\rm low}-10^{-4}),
$$

那么本例每次都写回 1，小更新不会积累。即便减法临时使用 FP32，最后只保存 FP16 结果，仍然会丢失这个差值。[Mixed Precision Training 第 3.1 节][mixed]

改为持续更新同一份 FP32 主权重，再从它生成低精度计算副本：

$$
w_{t+1}^{\rm main}=w_t^{\rm main}-10^{-4},\qquad
w_{t+1}^{\rm model}=\operatorname{cast}_{\rm FP16}(w_{t+1}^{\rm main}).
$$

| 步数 | 只保存 FP16 的权重 | 持久 FP32 主权重 | 从主权重转换出的 FP16 权重 |
| --- | ---: | ---: | ---: |
| 1 | 1.000000000 | 0.999899983 | 1.000000000 |
| 2 | 1.000000000 | 0.999799967 | 1.000000000 |
| 3 | 1.000000000 | 0.999699950 | 0.999511719 |
| 10 | 1.000000000 | 0.998999834 | 0.999023438 |

前两步模型副本仍显示为 1，但主权重已经保留变化，第三步累积的变化跨过了 FP16 的舍入边界。BF16 也有这种写回舍入问题，验证脚本包含两种格式的对照。

### 3.3 FP32 累加与 loss scaling 分别解决什么

| 机制 | 作用发生在哪里 | 主要解决的问题 |
| --- | --- | --- |
| GEMM 内使用 FP32 累加 | 一个算子的乘加过程 | 减少长累加链的低精度误差 |
| FP32 梯度 buffer | 跨 microbatch 累加参数梯度 | 提高多次梯度相加的精度 |
| FP32 main/master weight | 跨 optimizer step 保存参数 | 保留低精度参数上难以表达的小更新 |
| loss scaling | backward 前放大 loss，更新前还原梯度 | 降低小梯度在低精度反向中下溢的风险 |

这些位置不同，不能用“内部已经是 FP32 累加”推断一定不需要主权重，也不能指望在梯度已经变成零之后再转成 FP32 恢复信息。

loss scaling 使用链式法则：

$$
L_s=sL\quad\Longrightarrow\quad g_s=sg,\qquad g=g_s/s.
$$

例如 FP32 中的 $10^{-8}$ 直接转 FP16 会成为零；先乘 1024、转 FP16，再在 FP32 中除以 1024，可以保留接近原值的结果。真实训练中必须在 backward **之前**缩放 loss，才能影响反向计算中的梯度。

使用 loss scaling 时，整个梯度累积窗口保持同一个 scale；在完整梯度上进行 unscale、非有限值检查，随后按原尺度裁剪，再更新参数。检测到溢出时跳过对应更新并调整 scale。BF16 常常不需要为动态范围使用 loss scaling，但主权重仍可能有价值。[AMP 梯度累积与裁剪示例][amp]

### 3.4 为什么在优化器封装里看到主权重

优化器封装处在“拿到完整梯度”和“发布下一轮参数”的边界，适合管理两份参数之间的同步：

```text
初始化一次：model weight ──复制并转 FP32──▶ main weight

每次更新：
低精度 model weight ── forward / backward ──▶ 参数梯度
                                                │
                                  准备梯度，适用时 unscale 和 clip
                                                ▼
                                  FP32 main weight + Adam m/v
                                                │ optimizer step
                                                ▼
                                       新的 FP32 main weight
                                                │ 转成模型所需精度并复制
                                                ▼
                                       新的 model weight
```

这解释了源码上的组织方式：主权重常由混合精度优化器或其 wrapper 管理，Adam 的内部更新则使用这份高精度参数。算法状态与精度管理在同一条调用链上协作，但各自的存在原因不同。

还存在另一种常见用法：PyTorch 模型参数本来就保持 FP32，仅用 autocast 选择部分算子的执行精度。此时 FP32 参数本身就承担持久更新的职责，不应自动再画一份同大小的独立 FP32 master copy；autocast 也不等价于把整个模型永久转换成 FP16。[AMP 示例][amp]

## 4 Megatron 中的参数与梯度

本节查看 `Float16OptimizerWithFloat16Params` 路径中的 FP16/BF16 模型参数与 FP32 主权重。

### 4.1 谁持有参数以及更新谁

构造 `Float16OptimizerWithFloat16Params` 时，代码为可训练的 FP16/BF16 参数创建 FP32 副本，将底层优化器 `param_groups` 中的参数引用换成该副本，并给原参数设置 `main_param` 引用。原本就是 FP32 的参数进入另一组，直接使用原参数。[参数组构造][meg-main]

| 名称 | 含义 | 主要使用者 |
| --- | --- | --- |
| `model_param` | 模型 forward/backward 使用的参数对象 | 模型与算子 |
| `model_param.main_param` | 与该低精度参数对应的 FP32 主权重引用 | 混合精度优化器 wrapper |
| `fp32_from_float16_groups` | wrapper 保存的 FP32 主权重分组 | 梯度准备、状态保存与参数复制 |
| 底层 optimizer 的 `param_groups` | 在这条路径中已指向主权重 | Adam/SGD 等实际更新实现 |

`fp32_from_float16_groups` 保存新分配的 FP32 主权重；`model_param.main_param` 与底层 optimizer 的 `param_groups` 引用同一张量，不增加参数数据副本。

### 4.2 main_grad 不是 main weight

`main_param` 保存参数值，`main_grad` 保存梯度。Megatron 的参数与梯度 buffer 代码把 `param.main_grad` 映射到连续梯度 buffer 的一段，方便梯度累积和通信。[梯度 buffer 映射][meg-buffer]

在常规 hook 路径中，局部参数梯度加入 `param.main_grad`，之后 `param.grad` 可以被置为 `None`；在启用梯度累积融合的路径中，WGRAD kernel 可以直接累加到 `main_grad`，并通过标志避免 hook 再累加一次。[梯度 hook][meg-hook] [WGRAD 融合路径][meg-linear]

因此，看到 `.grad is None` 不能立即判断“没有算出梯度”：需要继续检查梯度是否已经进入框架维护的 buffer。同样，`main_grad` 的 dtype 要结合配置和分支判断，不能仅根据名字认定永远是 FP32。

`_copy_model_grads_to_main_grads()` 会把对应梯度提供给 `main_param.grad`；代码中的 `.float()` 在输入已经是 FP32 时可能复用同一个张量，因此函数名包含 copy 也不保证每条路径都有新的物理副本。连续 buffer 的存储生命周期还由 buffer 管理者控制。[梯度准备与参数复制][meg-copy]

### 4.3 一次 step 的实际顺序

`MixedPrecisionOptimizer.step()` 的关键依赖关系是：

1. `prepare_grads()` 把梯度准备到主参数侧；存在 scaler 时执行 unscale、非有限值检查并更新 scaler。
2. 如果检测到非有限值，提前返回，不执行本次底层参数更新。
3. 配置了梯度裁剪时，对准备好的梯度进行裁剪。
4. `step_with_ready_grads()` 调用底层 `self.optimizer.step()`。
5. `_copy_main_params_to_model_params()` 将更新后的主权重复制回模型参数。

因此，底层 Adam 看到的是主参数和已经准备好的梯度；计算模型在参数复制完成后使用新版本权重。Megatron 在此处更新 scaler 的具体位置与通用 PyTorch 示例不完全相同，读调用链时应以固定实现为准。[优化器 step][meg-step]

上层 `training.py::train_step()` 组织梯度清理、forward/backward 调度和 optimizer step。[训练循环][meg-train]

### 4.4 为什么代码中的矩阵乘看起来转置不同

本文使用 $Y=XW$，$W$ 的 shape 是 `[in, out]`。Megatron 这里的 Linear 与 PyTorch 一样按 `[out, in]` 保存权重，因此对应公式是：

$$
Y=XW_{\rm code}^T,\qquad
G_X=G_YW_{\rm code},\qquad
G_{W_{\rm code}}=G_Y^TX.
$$

代码里的 `grad_output.matmul(weight)` 和 `grad_output.t().matmul(total_input)` 因此与前面的手算一致。[Linear 反向实现][meg-linear]

## 5 参数量与显存占用

用 $\Psi$ 表示模型参数总数，包含权重和偏置；本例 $\Psi=17$。按参数保存的状态可以用“参数量 × 每参数字节数”记账。先区分各类状态的生命周期，再计算它们的存储开销。

### 5.1 训练状态的生命周期

| 对象 | 谁生产 | 谁消费 | 一般何时不再需要旧内容 |
| --- | --- | --- | --- |
| $X$ 和目标 $Y$ | 数据管线 | forward、loss，部分 backward | 本 microbatch 的相关计算完成后 |
| 激活与反向所需信息 | forward | 对应算子的 backward | 对应反向消费后；重计算会改变保存策略 |
| 激活梯度，如 $G_H,G_Z$ | 下游 backward | 上游 backward | 上游消费后，通常不跨 step 保留 |
| 参数梯度 | 各 microbatch 的 backward | 梯度同步、裁剪、优化器 | 本次更新后可清零或释放，下轮重新累积 |
| model weight | 初始化及更新后的参数复制 | forward/backward | 参数对象长期存在，值随更新改变 |
| main weight | 初始化时构造，随后持续更新 | 底层优化器、模型参数复制 | 跨 step 保留 |
| Adam 的 $m/v/t$ | 优化器 | 后续每次更新 | 跨 step 保留 |

“旧内容不再需要”与“显存已归还给设备”不是同一件事。张量可能仍有其他引用，buffer 可以长期分配而重复使用，PyTorch 分配器也可以缓存已经释放的块。

### 5.2 按参数量计算存储开销

例如采用低精度模型权重、FP32 梯度、FP32 主权重与 FP32 Adam 状态，不重复计算别名，逻辑状态账本为：

$$
\underbrace{2\Psi}_{\text{model weight}}
+\underbrace{4\Psi}_{\text{gradient}}
+\underbrace{4\Psi}_{\text{main weight}}
+\underbrace{4\Psi+4\Psi}_{m,v}
=18\Psi\ \text{bytes}.
$$

若梯度占 $2\Psi$，对应账本是 $16\Psi$；若模型参数本来就保留 FP32 且没有额外主权重，FP32 参数、梯度和 $m/v$ 也是 $16\Psi$。这些数都不含激活、临时张量、通信 buffer、对齐及分配器开销。

## 6 batch 与梯度累积

参数量决定需要维护多少训练状态，batch 的组织则影响每次计算的激活占用和更新频率。多卡训练需要将这两部分开销与通信代价一起考虑。

### 6.1 microbatch 与梯度累积

一次 optimizer step 使用的样本可以一次送入，也可以拆成多个 microbatch。拆分的意义是让每个 microbatch 的前向与反向先完成，随后释放其不再需要的激活，而参数梯度继续累积。

设第 $k$ 个 microbatch 有 $b_k$ 个样本，其平均 loss 为 $L_k$，总样本数 $B=\sum_kb_k$。完整 batch 的目标是：

$$
L=\sum_k\frac{b_k}{B}L_k,\qquad
\nabla_\theta L=\sum_k\frac{b_k}{B}\nabla_\theta L_k.
$$

因此，应在整轮累积开始时清零，给每个 microbatch 的 loss 乘以 $b_k/B$ 后 backward，最后只调用一次 step。只有所有 microbatch 等大时，这个系数才能统一写成 $1/K$。

```text
清空梯度
    microbatch 1：forward → 按样本占比缩放 loss → backward → 累加梯度
    microbatch 2：forward → 按样本占比缩放 loss → backward → 累加梯度
    ...
使用完整梯度更新参数一次
```

例如三个样本拆成 `1 + 2`，正确权重是 `1/3` 和 `2/3`。把两个 microbatch 的平均 loss 简单相加再除以 2，会使第一个样本获得过高权重。若目标按有效 token 平均，就应按有效 token 数构造权重，而不是直接套用样本数。

这个等价关系要求各 microbatch 使用同一份参数、目标函数能按样本分解，并保持随机操作与状态更新的语义一致。BatchNorm 的 batch 统计会受拆分影响；Dropout 的随机数对应关系也需要考虑。即便数学等价，浮点归约顺序不同仍可能产生小差异。

每个 microbatch 后都调用 `step()` 会使后面的 microbatch 使用更新后的参数，从而改变训练轨迹。梯度累积不需要跨 microbatch 保留已消费的计算图，因此无需设置 `retain_graph=True`。

### 6.2 从 microbatch 衔接 TP/DP 与 FSDP

固定全局 batch 和 DP 副本数时，减小 microbatch 会增加梯度累积次数。后续讨论 TP/DP 时，将把这一变化与每卡 GEMM 形状、激活显存和梯度同步放在一起分析。

FSDP 在 DP 组内分片训练状态。在 `FULL_SHARD` 路径中，每次前后向都有相应的参数聚集；使用 `no_sync()` 累积完整梯度又会提高显存占用。因此，小 microbatch 节省激活显存的收益，需要与重复参数通信和梯度驻留的代价一起权衡。[FSDP 分片与梯度累积][fsdp]

## 附录 验证与自测

配套 [验证脚本](verify_basic_train.py) 可以从笔记库根目录运行：

```bash
python assets/basic-train/verify_basic_train.py
```

脚本需要 PyTorch，可在 CPU 上运行。手算、梯度累积、Adam/AdamW 与主权重算例均已通过检查，环境为 Python 3.11.15、PyTorch 2.5.1，完整结果见 [verification.json](verification.json)。正文之外的误差核对如下：

| 检查 | 结果 |
| --- | --- |
| 中心有限差分与解析梯度 | 最大绝对误差约 $4.94\times10^{-10}$ |
| 错误地对两个 microbatch 均值等权平均 | 最大参数梯度差约 0.583333 |

读完后，可以用以下问题检查是否真正串起了闭环：

- 为什么第三个隐藏单元的 $G_H$ 非零，但对应第一层参数梯度为零？
- 为什么求完 $G_{W_2}$ 之后，不能立刻改写 $W_2$ 再计算 $G_H$？
- `1 + 2` 两个 microbatch 的平均 loss 应该各乘什么？为什么不能各除以 2？
- 当前参数和梯度一样，Adam 下一步为什么还可能不同？
- 低精度模型权重连续两步没变化，主权重是否也一定没变化？
- `.grad is None`、存在 `main_grad`、存在 `main_param`，分别说明什么？

## 参考资料

- [Adam 原论文 Algorithm 1][adam]：梯度的一阶矩、非中心二阶矩、偏差校正与更新。
- [Mixed Precision Training][mixed]：FP32 主权重、loss scaling 与算术精度。
- [PyTorch Linear][linear]、[Autograd mechanics][autograd]：权重布局与自动求导语义。
- [PyTorch AMP examples][amp]：loss scaling、梯度累积和裁剪。
- [PyTorch v2.6.0 AdamW 状态初始化][adamw-init]、[更新代码][adamw-update]：与数值实验所用 PyTorch 2.5.1 分开标注，便于复核实现。
- [Megatron 优化器参数组][meg-main]、[step][meg-step]、[梯度与参数复制][meg-copy]、[训练循环][meg-train]：均固定到同一提交。
- [Megatron 梯度 buffer][meg-buffer]、[梯度 hook][meg-hook]、[Linear 反向][meg-linear]：梯度存储与计算路径。
- [PyTorch FSDP][fsdp]：`FULL_SHARD` 的参数聚集与 `no_sync()` 的完整梯度累积。

[adam]: https://arxiv.org/abs/1412.6980
[mixed]: https://arxiv.org/abs/1710.03740
[linear]: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
[autograd]: https://docs.pytorch.org/docs/stable/notes/autograd.html
[amp]: https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/docs/source/notes/amp_examples.rst#L1-L156
[adamw-init]: https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/optim/adamw.py#L145-L185
[adamw-update]: https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/optim/adamw.py#L379-L437
[meg-main]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/optimizer/optimizer.py#L517-L581
[meg-copy]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/optimizer/optimizer.py#L631-L674
[meg-step]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/optimizer/optimizer.py#L405-L501
[meg-buffer]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/distributed/param_and_grad_buffer.py#L635-L659
[meg-hook]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/distributed/distributed_data_parallel.py#L401-L423
[meg-linear]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/core/tensor_parallel/layers.py#L445-L545
[meg-train]: https://github.com/NVIDIA/Megatron-LM/blob/d580efc68a9f0dbf1945f834f6f6200cd01d3343/megatron/training/training.py#L1213-L1267
[fsdp]: https://docs.pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.no_sync
