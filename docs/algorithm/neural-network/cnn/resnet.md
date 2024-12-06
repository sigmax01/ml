---
title: ResNet
comments: false
---

## 背景[^1]

最近的研究显示网络的层数是一个比较重要的问题. 并且最近的ImageNet竞赛中, 所有方法都利用了"非常深度"的模型. 那么就会产生一个问题: "训练好的网络是否就是简单的堆叠更多的层?"

一个臭名昭著的阻碍就是梯度消失/爆炸现象. 这个问题, 已经很大程度上被归一化初始化和中间归一化层(如BN)解决了, 使得网络可以在较深的层数下通过SGD和反向传播开始收敛. 

<figure markdown='1'>
![](https://img.ricolxwz.io/bca8ca0cfe726369f49294dd19322075.png){ loading=lazy width='600' }
<figcaption>使用25层和56层的普通网络在CIFAR-10数据集上的表现, 训练误差(左), 测试误差(右)</figcaption>
</figure>

可以从上图中发现, 随着层数的增加, 训练误差和测试误差不降反增. 这说明这不是过拟合, 因为过拟合是模型在训练集上表现优秀, 但是在验证集/测试集上表现较差.

所以, 既然**不是因为过拟合以及梯度消失导致的**, 那原因是什么?

> Let us consider a shallower architecture and its deeper counterpart that adds more layers onto it. There exists a solution by construction to the deeper model: the added layers are identity mapping, and the other layers are copied from the learned shallower model. The existence of this constructed solution indicates that a deeper model should produce no higher training error than its shallower counterpart. But experiments show that our current solvers on hand are unable to find solutions that are comparably good or better than the constructed solution (or unable to do so in feasible time).

这是由于**"退化问题"**导致的, 退化问题最明显的表现就是给网络叠加更多的层之后, 性能却快速下降的情况. 按照道理, 给网络叠加更多的层, 浅层网络的解空间是包含在深层网络的解空间中的, 深层网络的解空间至少存在不差于浅层网络的解, 因为只需要将增加的层变成恒等映射, 其他层的权重原封不动复制浅层网络, 就可以获得和浅层网络同样的性能. 更好地解明明存在, 为什么找不到, 找到的反而是更差的解?

???+ note "解空间是什么"

    解空间是指网络所有可能的权重和偏执值的组合形成的集合. 比如一个简单的网络有$W_1$, $W_2$, $b$三个参数, 每个参数可以取无数种值, 所有可能的$(W_1, W_2, b)$组合就是解空间, 对于深层网络, 解空间更加复杂, 因为参数的数量随着深度和宽度的增加呈现指数级增长.

???+ note "恒等映射是什么意思"

    恒等映射(Identity Mapping)是指一个输入通过某种函数变换后, 输出仍然等于输入本身, 用数学表达就是$f(x)=x$. 在神经网络中, 恒等映射的含义是, 某一层网络的输出完全等于它的输入, 没有做任何额外的变换.

    理论上, 深度网络新增的层如果只学习恒等映射, 深层网络就不会比浅层网络更差, 这样, 深层网络至少可以"退化"为浅层网路. 恒等映射是一个假想的最低性能保障, 如果网络能学到恒等映射, 至少不会比浅层网络更差.

???+ example "退化问题比方"

    打个比方, 你有一张比别人更大的藏宝图(解空间), 理论上你能找到更多的保障, 但是如果你没有很好的工具(优化算法), 反而会因为地图太复杂, 找不到最好的路径, 结果挖了一些不值钱的东西(次优解), 甚至挖错地方.

退化问题的原因可以总结为:

- 优化难度随着深度的增加非线性上升: 深层网络拥有更多参数, 解空间的维度和复杂性呈现指数级增长, 这使得优化算法需要在一个更大的高维空间内找到最优解, 难度大大增加
-  误差积累: 深层网路中的每一层都会对后续层的输入产生影响, 如果某些层的输出有轻微偏差, 这些偏差可能随着层数的增加而累积放大, 导致最终的误差增大

由此一个想法就自然而然的产生, 对于一个56层的神经网络相比于20层的浅层网络, 如果后36层是恒等映射, 那么56层的神经网络不就和20层的一样好了吗? 更近一步, 如果这36层神经网络相比于恒等映射更好上那么一点点(跟接近最优函数), 那么不就起到正优化作用了吗, ResNet的insight由此诞生.[^2]

## 概念[^1]

深度残差神经网络, Deep Residual Network, 简称ResNet, 它由微软研究院何凯明等人在2015年首次提出, 在深度学习领域产生了深远的影响. 它通过一种创新的"残差学习"机制, 从而实现了对非常深度网络的有效训练.

<figure markdown='1'>
![](https://img.ricolxwz.io/2c69b6152dba08a21c64cb133c5020a2.png){ loading=lazy width='600' }
<figcaption>在ImageNet上的训练结果. 细线表示训练误差, 粗线表示验证误差. 左图是18和34层普通网络的表现, 右图是18和34层ResNet的表现</figcaption>
</figure>

### 怎么做

基于背景中最后一段话的考虑, 我们在所拟合的函数中加入恒等函数. 假设在某一层内, 最优的函数记为$H(x)$, 那么我们所拟合的目标函数$F(x)$定义为$F(x):=H(x)-x$, 函数$F(x)$被称为“残差函数”.

每个残差模块都有一条跳跃连接(Shortcut Connection), 这条跳跃连接不对输入做任何复杂变化, 只是简单地将输入直接传递过去(即“恒等映射”). 而经过叠加的卷积层(或者其他变换层)处理后得到的输出, 会与这条“原样传递”的输入信号相加在一起. 恒等映射不会增加额外的参数也不会增加计算复杂度, 如下图所示.

<figure markdown='1'>
![](https://img.ricolxwz.io/0ac52a866f3a39e361c7406ff4e8a214.png){ loading=lazy width='400' }
</figure>

### 为什么

由此可见, 我们所需要的函数由两部分组成: 恒等函数和残差函数. 恒等函数的存在, 避免了“负优化”问题, 而残差函数则起到了“锦上添花”的作用.

> To the extreme, if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers.

🌟如果一个网络的最优映射恰好就是恒等映射, 那么通过学习一个函数使其输出趋近于0, 即$F(x)=x+f(x)\simeq x$, 而$f(x)\simeq 0$要比让一堆非线型层直接组合出恒等函数更加容易. 换句话说, 传统的深层网络在没有残差结构的时候, 加入最优解就是恒等映射, 要让多层非线型变换层叠加后输出结果是输入本身是相对困难的, 因为这些层被来就是用来学习复杂映射的, 很难精确地"什么也不做", 而在ResNet的框架下, 由于有$F(x)=x+f(x)$这种形式, 如果最优解是恒等映射, 那么训练只需要让$f(x)$的输出逼近于0, 就能得到恒等映射的效果, 这比通过堆叠多层非线型层直接拟合恒等映射要容易得多.🌟

> If one hypothesizes that multiple nonlinear layers can asymptotically approximate complicated functions, then it is equivalent to hypothesize that they can asymptotically approximate the residual functions, i.e., H(x) − x

> Although both forms should be able to asymptotically approximate the desired functions (as hypothesized), the ease of learning might be different

> This reformulation is motivated by the counterintuitive phenomena about the degradation problem (Fig. 1, left). As we discussed in the introduction, if the added layers can be constructed as identity mappings, a deeper model should have training error no greater than its shallower counterpart. The degradation problem suggests that the solvers might have difficulties in approximating identity mappings by multiple nonlinear layers. With the residual learning reformulation, if identity mappings are optimal, the solvers may simply drive the weights of the multiple nonlinear layers toward zero to approach identity mappings.

这些讲的也是同一个意思.

> In real cases, it is unlikely that identity mappings are optimal, but our reformulation may help to precondition the problem. If the optimal function is closer to an identity mapping than to a zero mapping, it should be easier for the solver to find the perturbations with reference to an identity mapping, than to learn the function as a new one.

这其实也是同一个意思, 其中, “零映射”可以理解为一种输出完全不依赖于输入的映射, 就是将所有的输入都映射为零的函数. 假设目标函数更加接近于“恒等映射”而不是零映射, 那么优化器从恒等映射出发时需要的调整(perturbation)会更小, 优化的效率会更高, 而如果目标函数接近零映射, 那么优化器就需要从“没有特性”的状态重新开始学习函数的全部结构, 这通常会更加困难.

引入了残差结构后:

- **减少了优化难度**: 在传统的深度网络中, 每一层都需要去拟合目标函数本身. 而在ResNet的残差块中, 我们将目标函数拆解为恒等映射+残差函数的形式. 这样, 网络只需要关注学习更加简单的残差函数, 而不必重新开始拟合整个映射. 这使得优化问题更加接近于学习小的, 便宜量式的调整, 有助于减少训练过程中的优化障碍
- **避免负优化**: 没有引入残差的时候, 额外的层可能会在优化初期大幅偏离已有模型的性能. 通过提供恒等映射的捷径(Shortcut Connection), 新加入的层在最初近似为恒等映射, 即使它们的参数还未充分优化, 也不会使性能变得更差, 这保障了深度网络至少可以"退化"回一个较浅的网络性能, 不会因为层数的增加而性能下降

## 细节[^1]

### 维度统一

在实现Shortcut Connection的时候, 残差层的输出和输入的维度必须一致, 因为在公式中维度相同的矩阵才可以相加. 文章将残差块看作是$\bm{y} = \mathcal{F}(\bm{x}, \{W_i\}) + \bm{x}$, 其中$\{W_i\}$表示的是当前层的权重参数集合, 表示网络需要学习的参数. 后面这个$\bm{x}$的维度必须和残差块的输出的维度相同, 如果不相同话, 可以在Shortcut Connection那边做一个线性变换, 去把维度对应上: $\bm{y} = \mathcal{F}(\bm{x}, \{W_i\}) + W_s\bm{x}$. 注意, 这里使用$W_s$的唯一目的是保持维度的统一, 如果维度本来就统一, 就没必要使用$W_s$, 因为不使用$W_s$更加经济高效, 可以减少参数量, 简单的加上$\bm{x}$已经足够解决退化问题.

### 残差函数结构

残差函数$\mathcal{F}$的结构是可选的, 可以是两层或者多层神经网络. 但是如果$\mathcal{F}$只有一层的话, 那么就相当于$\bm{y}=W_1\bm{x}+\bm{x}$. 作者并没有发现这样做的优势.

作者还发现上述的残差函数结构中不仅仅可以含有全连接层还可以是卷积层. 对于卷积层, 残差块输入的通道数和输出的通道数通常是相等的(需要逐个通道相加). 如果不一样的话, 可能需要使用Shortcut Connection上的线型变换(如$W_s$)或者使用零填充...

### 架构

作者测试了普通网络和残差网络的表现.

<figure markdown='1'>
![](https://img.ricolxwz.io/5637f76757e67bc2888327a110a9937a.png){ loading=lazy width='350' }
<figcaption>参与ImageNet的网络架构图. VGG-19(左, 19.6 billion FLOPs), 普通网络(中, 3.6 billion FLOPs), ResNet(右, 3.6 billion FLOPs). 点状的Shortcut Connection表示会对维度进行提升(参考前面的$W_s$), 不是点状的Shortcut Connection表示不对维度进行改变</figcaption>
</figure>

#### 普通网络

普通网络借鉴了VGG网络的设计哲学.

???+ tip "卷积操作后特征图的大小计算公式"

    对于一个特征图, 卷积操作后输出特征图的大小计算公式为:

    $Output\ Size = \frac{Input\ Size - Kernel\ Size + 2 \times Padding}{Stride}+1$

    如果使用的是3*3的卷积核, padding=1, stride=1, 那么输入输出的特征图的大小是不变的, 因为这种情况下, 卷积核会在边界添加填充, 使得卷积后的输出和输入具有相同的高度和宽度.

卷积层大多数用的是3*3的卷积核, 并且遵循两条简单的设计原则: 1. 对于具有相同大小输入输出特征图的层之间的卷积核的数量应该是相同的; 2. 如果特征图的大小减半了, 那么卷积核的数量应该翻倍以保持层之间的复杂度大致相同.

???+ note "为什么特征图越小, 卷积核越多"

    在网络的前期, 输入特征图较大, 如果使用过多卷积核, 计算量会大幅度增加, 后期特征图逐渐缩小, 增加卷积核数量对计算复杂度的影响较小, 但是可以大幅提升表达能力. 

    底层特征相对简单, 种类虽然多, 但是它们的表示复杂度较低, 不需要大量的卷积核就可以捕捉到这些模式. 高层特征更加复杂, 它们需要更多的滤波器来进行区分和表示. 

与VGG不同的是, 这里直接采用stride=2, padding=0, 3*3的卷积核实现降采样.

???+ example "直接使用卷积降采样"

    比如特征图的大小从$56$到$28$, 输出的大小可以这样计算$\frac{56-3+2*0}{2}+1=28.5$. 

    :fontawesome-solid-circle-question: 注意, 大多数的深度学习框架会默认执行向下取整, 所以输出特征图的大小是$28$.

???+ note "为什么不采用池化而采用卷积层降采样"

    池化层(尤其是最大池化)通过选择池化窗口中的最大值来进行下采样, 这虽然有效减少了计算量和特征图的尺寸, 但是会丢失大量的位置信息. 卷积层不仅仅是对输入进行下采样, 它还能学习到空间中的局部特征, 通过不同卷积核来捕捉更多的空间模式. 相比之下, 池化只是简单的选取最大值或者平均值, 无法进行更加复杂的学习.

    所以, 越来越多的现代网络架构如ResNet开始减少池化层的使用, 代之以卷积层的下采样.

以此得到的普通网络比VGG网络的复杂度更低, 可以计算它们俩的FLOPS, FLOPS由以下的几个因素决定: 输入特征图高度, 输入特征图宽度, 输入通道数, 卷积核高度, 卷积核宽度, 输出通道数.

#### 残差网络

基于上面的普通网络, 作者插入了Shortcut Connections. 在输入和输出特征图维度(通道)相同的情况下, 可以直接使用恒等映射(即上图中实现的连接). 当维度(通道)升高的情况下, 它们采取了两种措施.

1. Shortcut Connection依旧采用的是恒等映射, 但是会对新增额外的零通道. 假设, 输入的通道数是64, 在一系列操作后, 通道数增加到了128, 则这种恒等映射需要进行变换, 多出的64张特征图全部填充为0. 这种不会引入额外的训练参数
2. 使用1*1的卷积核进行升维操作. 具体操作参考[这里](/algorithm/neural-network/cnn/#increase-dimension)

## 实现

作者对于ImageNet上训练的ResNet实现如下: 

在训练的时候, 将图片的最短边随机resize到[256, 480], 然后参考AlexNet中的[随机裁剪](/algorithm/neural-network/cnn/alexnet/#random-cropping)对图片进行224*224的随机裁剪(包括它的水平反转)以防止过拟合. 然后, 减去每个像素的平均值, 去掉很多不必要的背景信息, 例如全局亮度 :fontawesome-solid-circle-question:. 然后在每一个卷积层之后采用BN. 对所有的参数进行从头训练. 采用mini-batch SGD, batch的大小为256. 学习率从0.1开始下降当错误率遇到平稳的情况除以10避免震荡. 采用0.0001的权重衰减(就是[L2正则化](/algorithm/linear-regression/#l2-regularization), 防止模型过拟合), [动量](/algorithm/neural-network/fnn/#动量)设置为0.9. 并没有使用Dropout. 

在测试的时候, 采取了"10-crop"的方法, 就是说, 从输入图像中采样10个不同的区域来进行评估, 减少由于图像中不同位置的信息丢失带来的影响. 并且, 图像会被调整为多个不同的尺度, 例如, 缩放到$224, 256, 384, 480, 640$, 然后对每个尺寸进行预测, 去平均值, 增强模型的尺度不变性. 

[^1]: He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for image recognition (No. arXiv:1512.03385). arXiv. https://doi.org/10.48550/arXiv.1512.03385
[^2]: Apache. (2022, 二月 4). 深度学习之残差神经网络（ResNet） [知乎专栏文章]. https://zhuanlan.zhihu.com/p/463935188
