---
title: 前馈神经网络
comments: false
---

## 架构

前馈神经网络, Feedforward NN, 的架构如图所示.

<figure markdown='1'>
![](https://img.ricolxwz.io/182181d298a387078d650701b708d254.png){ loading=lazy width='500' }
</figure>

具体说明如下:

- 输入层: 位于网络最底层, 输入变量$x_1, x_2, ..., x_n$表示输入特征. 每个输入节点代表一个特征, 输入层只复杂将数据传递到下一层(隐藏层), 不执行任何计算
- 隐藏层: 网络中间的层. 每个隐藏层的神经元接受来自上一层的加权求和输入, 并加上一个偏置项$b_m$, 然后通过激活函数$f$进行线性变换, 公式为$o_m=f(z_m)=f(\sum_{i=1}^n w_{im}x_i+b_m)$
- 输出层: 位于网络最顶层, 输出结果$o_k$是隐藏层经过类似的加权和计算后的输出, 公式为$o_k=f(z_k)=f(\sum_{i=1}^m w_{wk}o_i+b_k)$

除此之外, FNN还有两个非常重要的特征:

- 每一个神经元只接受前一个层的输出
- 当前层的每一个神经元都与前一个层的所有神经元相连
- 每个神经元的输入在完成加权和计算后, 会通过一个激活函数, 这个激活函数不局限于阶跃函数, 可以是ReLu, Sigmoid, Tanh函数, 最常用的是Sigmoid函数. 特别注意, 这个激活函数要可微

## 反向传播算法 {#backpropagation-algorithm}

对于每一个训练样本$\{\bm{x}, t\}$, $\bm{x}=\{x_1, x_2, ..., x_n\}$, $t$为其标签. 将其传入网络, 直到输出层, 这个过程称为正向传播, 将其输出$o$与标签$t$进行比较, 计算误差, 根据误差, 从输出层到输入层逐级反向传播, 调整每个神经元的权重, 以减小误差, 这个过程就是反向传播. 权重的更新公式为$w_{pq}^{new}=w_{pq}^{old}+\Delta w_{pq}$. 这种过程会被不断重复, 即正向->反向->正向->反向, 直到输出层输出结果的误差在可以接受的范围内. "正向->反向"这样的一次更新就称为一次迭代.

那么, 我们怎么计算这个权重变化$\Delta w_{pq}$的呢? 参考线性回归, 我们可以定义一个误差损失函数然后使用梯度下降算法解决. 如均方误差函数, MSE. 每一次迭代/每一个批次对应于下山的"一步", 在山上的每一个位置对应于一组权重配置, 梯度的方向是误差增加最快的方向, 沿着负梯度的方向移动则可以使误差减小. 目标是通过调整权重, 使模型"下坡", 最终达到地形的最低点, 这样误差最小, 模型性能最佳. 用来下山的步子被称为"学习率", 这是算法的一个超参数.

<figure markdown='1'>
![](https://img.ricolxwz.io/4392978da300d80f0485a0aa396966ff.png){ loading=lazy width='500' }
</figure>

要注意的是, 梯度下降算法并不保证能找到全局最小值, 它只会找到基于起点的最近的局部最小值.

<figure markdown='1'>
![](https://img.ricolxwz.io/21ad32b9aa5ba67e5b0f6abc71d55d08.png){ loading=lazy width='500' }
</figure>

假设 $w_{pq}(t)$表示的是从神经元$p$到神经元$q$在$t$这个时间的权重, 那么下一次在$t+1$这个时间的权重为$w_{pq}(t+1)=w_{pq}(t)+\Delta w_{pq}$, 其中$\Delta w_{pq}=\eta\cdot \delta_q\cdot o_p$, 即权重变化与神经元$p$在激活函数激活后的输出$o_p$, 神经元$q$的误差$\delta_q$成正比.

神经元$q$的误差$\delta_q$要分两种情况计算:

1. 若$q$是输出层的神经元, 则$\delta_q=(t_q-o_q)f'(z_q)$, 见[图](https://img.ricolxwz.io/5f941efe6d1e0e2a24f4cc02e2b5f50c.png)
2. 若$q$是隐藏层的神经元, 则$\delta_q=f'(z_q)\sum_i w_{qi}\delta_i$, 见[图](https://img.ricolxwz.io/240c3a21ce4f4009099695a2a1c28f42.png)

注意, $i$是$q$后面的神经元, 即顺序为$p\rightarrow q\rightarrow i$; 这里的$z_q$是在激活函数激活前的输出, $o_q=f(z_q)$. 可以被证明$f'(z_q)=f(z_q)(1-f(z_q))$(前提是使用sigmoid激活函数), 所以有$f'(z_q)=o_q(1-o_q)$. 上面的误差计算公式可以写为:

1. 若$q$为输出层神经元, 则$\delta_q=(t_q-o_q)o_q(1-o_q)$
2. 若$q$为隐藏层神经元, 则$\delta_q=o_q(1-o_q)\sum_i w_{qi}\delta_i$

注意, $i$是$q$后面的神经元, 即顺序为$p\rightarrow q\rightarrow i$.

???+ tip "Tip"

    🌈🥚: 反向传播公式推导请见[这里](/algorithm/neural-network/backpropagation).

### 训练过程 {#training-procedure}

1. 初始化所有的权重和截距为较小的随机数
2. 重复循环, 直到停止条件被满足
    1. 正向传播: 计算神经网络的输出
    2. 反向传播
        1. 计算输出层神经元的误差$\delta$, 更新输出层的权重和截距
        2. 从输出层开始, 对于神经网络的每一层, 重复循环, 直到输入层
            1. 反向传播$\delta$
            2. 更新两层之间的权重
3. 检查是否满足条件, 若训练集的误差已经低于某一个值或已经达到了最大轮次, 则停止循环

???+ example "例子"

    假设学习率为$0.9$.

    === "1--->"

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/efbaeae02cd8777f9b80255112fe680c.png){ loading=lazy width='550' }
        </figure>

    === "2--->"

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/c5859d33d92cbffd51fa2b05db6f50e3.png){ loading=lazy width='650' }
        </figure>

    === "3--->"

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/4fbfa4681dde83cc2983beabf18b2b5c.png){ loading=lazy width='650' }
        </figure>

    === "4--->"
        
        <figure markdown='1'>
        ![](https://img.ricolxwz.io/ad1fbdd3eb510d21579643d0c982ea49.png){ loading=lazy width='650' }
        </figure>

    === "5--->"

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/478525aca15db7ed94df36a9839ac82d.png){ loading=lazy width='650' }
        </figure>

    === "6--->"

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/e053763d1f673ce6b687928f85addb82.png){ loading=lazy width='650' }
        </figure>

    === "7"

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/84b6a3083f59e94953f1dff056775fd7.png){ loading=lazy width='650' }
        </figure>

### 其他梯度下降算法

标准的梯度下降算法计算出$\sum_i w_{qi}\delta_i$, 所有的误差之后再更新权重$\delta_q$, 而随机梯度下降算法随机选一个$i$层神经元来计算$\delta_q$. 而小批量梯度下降是选取一小部分$i$层神经元来计算$\delta_q$.

常用的优化算法总结:

- 标准梯度下降, standard gradient descent
- 随机梯度下降, SGD, stochastic gradient descent
- [动量法](#动量), momentum
- 自适应梯度算法, adagard
- Nesterov加速算法, NAG
- RMSProp
- AdaDelta
- Adam

## 通用逼近定理

根据Cyberko和Hornik等人在1989年的研究, 任何连续函数都可以通过一个单隐藏层的神经网络, 以任意小的误差进行逼近. 这意味着即使是简单的神经网络, 也能处理复杂的连续函数. 根据Cyberko 1988年的研究, 任何函数(包括不连续函数)都可以通过一个具有两个隐藏层的神经网络, 以任意小的误差进行逼近. 这意味着即使是非连续的复杂函数, 神经网络也能够很好的逼近.

这两个定理属于存在性定理, 也就是说, 它们仅仅说明了在理论上, 这样的神经网络能够逼近任意的函数, 但是并没有告诉我们应该如何选择网络的具体架构(如隐藏层侧数量, 每层神经元的数量等)以及如何设置超参数.

## 神经元的数量 {#neuron-num}

### 输入层

- 数值属性: 每个属性对应一个神经元
- 类别属性: 如果该属性有$k$个值, 那么就需要$k$个神经元, 并使用one-hot编码方式. 这种编码方式将具有$k$个值的属性表示为$k$个二进制的属性, 只有一个位置是$1$, 其余位置是$0$, 如假设一个天气属性有三个取值, 晴天, 阴天, 雨天, 那么, 晴天可以表示为$1\ 0\ 0$, 阴天可以表示为$0\ 1\ 0$, 雨天可以表示为$0\ 0\ 1$

### 输出层

- $k$类问题: 对于一个有$k$个类别的问题, 输出层会有$k$个神经元, 也用one-hot编码
- 二分类问题: 可以使用两个神经元, 使用one-hot编码; 也可以使用一个神经元, 配合sigmoid激活函数, 若输出值接近$0$, 属于第一类, 接近$1$则属于第二类

### 隐藏层 

隐藏层神经元对误差的影响如[图](https://img.ricolxwz.io/54225f471feeb87e7664d00730b2f0cf.png).

- 隐藏层神经元过多: 导致过拟合
- 隐藏层神经元过少: 导致欠拟合

思想就是开始的时候使用较少的神经元, 然后逐步增加神经元的数量, 直到模型的误差不再显著减少.

1. 一开始使用较少的隐藏神经元来初始化网络
2. 训练网络, 直到均方误差或其他损失指标不再显著减少
3. 此时, 向隐藏层中添加一些新神经元, 并使用随机初始化的权重重新训练网络, 均方误差会减少, 如[图](https://img.ricolxwz.io/b8176513be1475a51dbd250cd4e9bcdc.png)
4. 重复上述步骤, 直到满足终止条件, 例如添加新神经元不会导致显著的误差减少, 或者隐藏层达到设定的最大大小

## 学习率 {#learning-rate}

神经网络的性能和学习率的相关性非常大. 

- 若学习率太小: 收敛地很慢
- 若学习率很大: 振荡, 超出最小值

我们无法在训练前就预知最优的学习率.

<figure markdown='1'>
![](https://img.ricolxwz.io/8a0280ce5fa12521df322cc115dfe04d.png){ loading=lazy width='500' }
</figure>

学习率可以是固定的, 也可以随时间变化. 后者一开始学习率交大, 随着时间推移, 慢慢减小. 一开始学习率较大能够造成更大的权重变化, 能够减少训练批次, 甚至还可能跳过某些局部最小值; 而后, 学习率慢慢减小, 防止振荡的发生. 公式有两种:

- $\eta_n = \frac{\eta_{n-1}}{1+d*n}$
- $\eta_n = \eta_0 e^{-dn}$

其中, $\eta_n$表示第$n$批次的学习率, $d$是一个超参数, 表示衰减率.

## 动量 {#动量}

动量, momentum, 通过在权重更新公式中引入一个额外的动量项, 使得当前的权重更新依赖于之前的更新, 从而减少振荡并允许使用更大的学习率. 计算公式为$\Delta w_{pq}(t)=\eta\delta_q o_p + \mu(w_{pq}(t)-w_{pq}(t-1))$.

## 初始化 {#weight-initialization}

神经网络模型的性能非常依赖于权重和截距的初始化.常见的做法有:

- 标准的做法: 从$-1$到$1$之间选择小的随机数
- Xavier初始化: 权重从一个正态分布中产生, $\sigma=\sqrt{\frac{2}{N_{in}+N_{out}}}$, 其中$N_{in}$是输入神经元的数量, $N_{out}$是输出神经元的数量, 注意, 这里的输入输出神经元不是整个神经网络的输入输出神经元, 是相对于当前层神经元来说的上一层/下一层神经元, 当前层神经元就是权重/截距待更新的神经元