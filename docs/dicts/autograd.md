---
title: 自动微分
comments: true
---

## 微分方法

微分有四种方法: 手动微分, 数值微分, 符号微分和自动微分. 如图[^2]所示.

<figure markdown='1'>
  ![](https://img.ricolxwz.io/e8b99a520b4a39e9675ad00859811bf4.png){ loading=lazy width='600' }
</figure>

- **手动微分**: 一种通过手动推导和计算来求解函数导数的方法. 根据微积分中的求导法则(如链式法则, 乘法法则等), 手动推导出该函数的导数表达式. 并将这个手动推导出来的公式用计算机代码表示, 以便在给定输入值时计算出相应的导数值
- **数值微分**: 数值微分根据离散的数据点估计函数的导数, 这意味着, 它和其他方法的主要区别就是它计算出来的是导数的近似值. 常用的方法包括正向差分算法(使用某个点以及前一个点的值来估计导数), 后向差分算法(使用某个点及后一个点的值来估计导数), 中心差分算法(综合正向和后向差分, 通常提供更高的精度).
- **符号微分**: 符号微分需要对表达式进行符号解析和规则转换, 这可能涉及对复杂函数结构的模式匹配和简化, 将输入的函数转化为其对应的解析形式的导数表达式, 最终结果是一个以符号为基础的解析函数, 如, 使用符号微分对表达式进行求导, 可以直接得到类似$2x+3\sin(x)$这样明确的解析形式. 如果函数很复杂, 解析出来的导数表达式可能极其冗长甚至不便于使用. 它适用于需要明确解析导数公式的场合, 例如数学分析, 公式推导等. **它侧重于给出一个具体的解析导数表达式, 而不是在求出某个点的导数值**
- **自动微分**: 自动微分通过在数值计算的过程中对基本运算步骤的导数链式求导规则进行系统地分解和累积, 最终在一个具体点处高效地求得函数在该点的导数值. **它侧重于对给定输入点求出相应的导数值, 而不是求出导函数的解析式**. 它分为正向模式和反向模式, 上图所展示的自动微分属于正向模式, 总结起来, 正向模式是从输入开始, 往输出传播导数信息; 后向模式是从输出开始, 往输入传播导数信息

## 自动微分

???+ info "前置提醒"

    我们将$\frac{\partial a}{\partial b}$称为$a$相对于$b$的偏导数.

自动微分分为两种形式, Forward Mode和Backward Mode. 🌟用最最最浓缩的话讲就是: 链式法则需要有一个起点, 而这个起点对于正向模式来讲就是输入变量对于自身的导数是1, 对于反向模式来讲就是输出变量对于自身的导数是1. 这个导致了在使用链式法则的时候计算先后的不同. 对于一个链式求导$\frac{\partial y}{\partial x}=\frac{\partial y}{\partial u}\cdot \frac{\partial u}{\partial x}$, 正向模式和反向模式都是这个公式. 但是正向模式是先计算$\frac{\partial u}{\partial x}=\frac{\partial u}{\partial x}\cdot \frac{\partial x}{\partial x}=\frac{\partial u}{\partial x}\cdot 1$, 然后计算$\frac{\partial y}{\partial u}$, 然而反向模式是先计算$\frac{\partial y}{\partial u}=\frac{\partial y}{\partial y}\cdot\frac{\partial y}{\partial u}=1\cdot \frac{\partial y}{\partial u}$, 然后再计算$\frac{\partial u}{\partial x}$. 相当于给你了一个火柴, 也就是$\frac{\partial x}{\partial x}=1$或者$\frac{\partial y}{\partial y}=1$, 然后启动这个过程.🌟

???+ tip "另一种描述方法"

    对于链式法则$\frac{d v_{i+1}}{d v_{i-1}}=\frac{d v_{i+1}}{d v_i}\cdot \frac{d v_i}{d v_{i-1}}$来说:

    - 正向模式中$\frac{d v_i}{d v_{i-1}}$是已知数, 需要求导的是$\frac{d v_{i+1}}{d v_i}$, 缓存的是中间变量对于输入的导数
    - 反向模式中$\frac{d v_{i+1}}{d v_i}$是已知数, 需要求导的是$\frac{d v_i}{d v_{i-1}}$, 缓存的是输出对于中间变量的导数

### 正向模式

<figure markdown='1'>
  ![](https://img.ricolxwz.io/9896183d8df1e4f92cbbb37a6961e12a.png){ loading=lazy width='500' }
  <figcaption>正向模式例子. 定义输出函数为$y=f(x_1, x_2)=\ln(x_1)+x_1x_2-\sin(x_2)$, 计算$(x_1, x_2)=(2, 5)$处的偏导数$\frac{\partial y}{\partial x_1}$</figcaption>
</figure>

### 反向模式

<figure markdown='1'>
  ![](https://img.ricolxwz.io/da7f074ddd48d72e35c1ed7c7f06ebd2.png){ loading=lazy width='500' }
  <figcaption>反向模式例子. 定义输出函数为$y=f(x_1, x_2)=\ln(x_1)+x_1x_2-\sin(x_2)$, 计算$(x_1, x_2)=(2, 5)$处的偏导数$\frac{\partial y}{\partial x_1}$和偏导数$\frac{\partial y}{\partial x_2}$</figcaption>
</figure>

上述过程其实和反向传播算法是吻合的, BP算法也是正向传播求出所有神经元的权重, 然后通过反向模式求出损失函数对应于每一个神经元的权重的偏导数. 由于只有一个标量的输出, 所以大多数的中间偏导数都能被重复利用(缓存输出对于中间变量的偏导数)而不用对于每个神经元的权重都重复计算一次, 所以反向模式特别适合标量输出, 输入维度较大的场景; 而正向模式特别适合输入维度较小, 输出维度较大的场景.

### Jacobian矩阵 {#Jacobian-Matrix}

Jacobian矩阵定义为一个由多个函数的多个偏导数组成的矩阵. 假设有一个从$\mathbb{R}^n$到$\mathbb{R}^m$的函数$\mathbf{f}$, 其中的第$(i, j)$个元素的含义是第$i$个函数对第$j$个变量的偏导数.

$$\mathbf{J} =
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \cdots & \frac{\partial y_1}{\partial x_n} \\
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \cdots & \frac{\partial y_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial y_m}{\partial x_1} & \frac{\partial y_m}{\partial x_2} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{bmatrix}$$

**若设置单个种子**, 正向模式和反向模式一次分别能计算Jacobian矩阵的一列和一行, 如图所示.

<figure markdown='1'>
  ![](https://img.ricolxwz.io/b608f936c0a583e468b44541fab6d1c3.png){ loading=lazy width='500' }
  <figcaption>总共有$m$个函数, $n$个输入</figcaption>
</figure>

**若设置单个种子:**

- 对于正向模式, 一次程序计算能够求所有函数对于一个输入的偏导数(能够缓存的是中间变量对于输入的偏导数), 对应的就是Jacobian矩阵中的一列, 所以, 如果想用正向模式计算出所有函数对于所有输入的偏导数, 需要计算$n$次
- 对于反向模式, 一次程序计算能够求一个函数对于所有输入的偏导数(能够缓存的是输出对于中间变量的偏导数), 对应的就是Jacobian矩阵中的一行, 所以, 如果想用反向模式计算出所有函数对于所有输入的偏导数, 需要计算$m$次

存储整个Jacobian矩阵通常是不现实的, 因为它的规模可能会非常大, 消耗巨大的存储空间. 此外, 在很多实际问题中, 我们通常只需要用到Jacobian矩阵的特定操作, 而不需要完整存储整个矩阵, 这就是引入JVP和VJP的原因.

#### Jacobian-向量积 {#JVP}

JVP是指用Jacobian矩阵$\mathbf{J}$左乘一个向量$\mathbf{v}$:

$$
\mathbf{Jv} =
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \cdots & \frac{\partial y_1}{\partial x_n} \\
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \cdots & \frac{\partial y_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial y_m}{\partial x_1} & \frac{\partial y_m}{\partial x_2} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{bmatrix}
\begin{bmatrix}
v_1 \\ v_2 \\ \vdots \\ v_n
\end{bmatrix}
$$

假设你在训练一个大型神经网络模型, 这个模型的参数非常巨大. 在训练的过程中, 你通常只能得到全局梯度方向(即对所有参数求偏导之后的结果), 但有时你希望在当前参数$\mathbf{x}$上, 对某个特定参数方向$\mathbf{v}$(例如, 沿着某些参数子集的改变方向)来衡量函数输出的变化率. 换句话说, 你想知道如果沿着$\mathbf{v}$这个方向轻微移动参数, 那么函数输出会以怎么样的斜率变化. 该方向向量$\mathbf{v}$在对应参数的位置上为非零(表示想在这部分参数方向上做灵敏度分析), 在其他参数维度上为零.

使用自动微分框架(如JAX, PyTorch)可以直接得到$\mathbf{J}\cdot \mathbf{v}$的值, 而不用先计算Jacobian矩阵, 再进行点积, 因为在高维空间中, 完整构建Jacobian矩阵会非常昂贵(内存和计算上), JAX计算JVP的方法是使用正向模式AD.

在正向模式AD中, 你不需要逐个变量分开跑一次正向AD来合成$\mathbf{v}$的效果, 只要你一开始就为所有输入变量设置对应的导数种子的值为$\mathbf{v}$, **即设置多个种子**. 一次正向模式AD就能让你获得整组方向导数对输出的影响, 即$\mathbf{J}\mathbf{v}$.

???+ note "深入理解上面这句话"

    Jacobian矩阵$\mathbf{J}$的每一列对应“一个特定输入变量对于所有输出的偏导数组合”, 即固定输入参数求所有输出对它的偏导数, 这对应在使用正向AD时, 为这个输入变量设置其导数种子在对应的方向向量的分量为1, 其他为0. 当你有多个输入变量同时设置导数种子非零的时候, 正向AD会在一次传播中计算这些方向对于输出的综合影响.

#### 向量-Jacobian积 {#VJP}

VJP(Vector-Jacobian Product)是指用一个向量$\mathbf{u}$右乘Jacobian矩阵$\mathbf{J}$:

$$
\mathbf{u}^\top \mathbf{J}
=
\bigl[\,u_1 \quad u_2 \quad \cdots \quad u_m\,\bigr]
\begin{bmatrix}
\dfrac{\partial y_1}{\partial x_1} & \dfrac{\partial y_1}{\partial x_2} & \cdots & \dfrac{\partial y_1}{\partial x_n} \\[6pt]
\dfrac{\partial y_2}{\partial x_1} & \dfrac{\partial y_2}{\partial x_2} & \cdots & \dfrac{\partial y_2}{\partial x_n} \\[6pt]
\vdots & \vdots & \ddots & \vdots \\[6pt]
\dfrac{\partial y_m}{\partial x_1} & \dfrac{\partial y_m}{\partial x_2} & \cdots & \dfrac{\partial y_m}{\partial x_n}
\end{bmatrix}
$$

这里, $\mathbf{u}$的大小和输出维度$m$相同, 将其作为一个权重向量(类似于上面的方向向量)赋给输出的各个非良. 对那些在$\mathbf{u}$中设置为0的输出方向, 相当于“忽略”了对应的输出对输入参数的影响. 对那些在$\mathbf{u}$中为非零的输出分量, 则把该输出维度的偏导信息累积起来, 回溯到参数空间, 得到特定输出方向在输入参数中的灵敏度分布.

---

???+ note "JVP和VJP的含义总结"

    - JVP: 在输入方向上选择性赋0, 使得你“忽略”对无关参数的变化, 只关心指定参数子集对完整输出函数的影响
    - VJP: 在输出方向上选择性赋0, 使得你“忽略”某些输出维度的变化, 只关心指定输出子集在传播回输入参数时产生的敏感度

## 参考视频[^1]

<div style="position: relative; padding: 30% 45%;">
<iframe style="position: absolute; width: 100%; height: 100%; left: 0; top: 0;" src="//player.bilibili.com/player.html?isOutside=true&bvid=BV1PF411h7Ew&p=1&high_quality=1&autoplay=false&muted=false&t=5&as_wide=1" frameborder="yes" scrolling="no" allowfullscreen="true"></iframe>
</div>

[^1]: Deep_Thoughts (导演). (2021, 十一月 15). 13、详细推导自动微分Forward与Reverse模式 [Video recording]. https://www.bilibili.com/video/BV1PF411h7Ew/?spm_id_from=888.80997.embed_other.whitelist&t=5&bvid=BV1PF411h7Ew&vd_source=f86bed5e9ae170543d583b3f354fcaa9
[^2]: Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic differentiation in machine learning: A survey (No. arXiv:1502.05767). arXiv. https://doi.org/10.48550/arXiv.1502.05767
[^3]: Zomi酱. (2019, 九月 6). [DL]自动微分—向前模式和反向模式 [知乎专栏文章]. https://zhuanlan.zhihu.com/p/81507449
[^4]: Zomi酱. (2022, 七月 30). 【自动微分原理】AD的正反向模式 [知乎专栏文章]. Ai系统. https://zhuanlan.zhihu.com/p/518296942
