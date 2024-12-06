---
title: ResNet
comments: false
---

## 背景[^1]

最近的研究显示网络的层数是一个比较重要的问题. 并且最近的ImageNet竞赛中, 所有方法都利用了"非常深度"的模型. 那么就会产生一个问题: "训练好的网络是否就是简单的堆叠更多的层?"

一个臭名昭著的阻碍就是梯度消失/爆炸现象. 这个问题, 已经很大程度上被归一化初始化和中间归一化层(如BN)解决了, 使得网络可以在较深的层数下通过SGD和反向传播开始收敛. 

<figure markdown='1'>
![](https://img.ricolxwz.io/bca8ca0cfe726369f49294dd19322075.png){ loading=lazy width='400' }
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

## 概念

深度残差神经网络, Deep Residual Network, 简称ResNet, 它由微软研究院何凯明等人在2015年首次提出, 在深度学习领域产生了深远的影响. 它通过一种创新的"残差学习"机制, 从而实现了对非常深度网络的有效训练.

<figure markdown='1'>
![](https://img.ricolxwz.io/2c69b6152dba08a21c64cb133c5020a2.png){ loading=lazy width='400' }
<figcaption>在ImageNet上的训练结果. 细线表示训练误差, 粗线表示验证误差. 左图是18和34层普通网络的表现, 右图是18和34层ResNet的表现</figcaption>
</figure>

基于上面的考虑, 我们在所拟合的函数中加入恒等函数. 假设在某一层内, 最优的函数记为$H(x)$, 那么我们所拟合的目标函数$F(x)$定义为$F(x):=H(x)-x$, 函数$F(x)$被称为“残差函数”.

<figure markdown='1'>
![](https://img.ricolxwz.io/0ac52a866f3a39e361c7406ff4e8a214.png){ loading=lazy width='200' }
</figure>

由此可见, 我们所需要的函数由两部分组成: 恒等函数和残差函数. 恒等函数的存在, 避免了“负优化”问题, 而残差函数则起到了“锦上添花”的作用.

> To the extreme, if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers.

🌟如果一个网络的最优映射恰好就是恒等映射, 那么通过学习一个函数使其输出趋近于0, 即$F(x)=x+f(x)\simeq x$, 而$f(x)\simeq 0$要比让一堆非线型层直接组合出恒等函数更加容易. 换句话说, 传统的深层网络在没有残差结构的时候, 加入最优解就是恒等映射, 要让多层非线型变换层叠加后输出结果是输入本身是相对困难的, 因为这些层被来就是用来学习复杂映射的, 很难精确地"什么也不做", 而在ResNet的框架下, 由于有$F(x)=x+f(x)$这种形式, 如果最优解是恒等映射, 那么训练只需要让$f(x)$的输出逼近于0, 就能得到恒等映射的效果, 这比通过堆叠多层非线型层直接拟合恒等映射要容易得多.🌟

引入了残差结构后:

- **减少了优化难度**: 在传统的深度网络中, 每一层都需要去拟合目标函数本身. 而在ResNet的残差块中, 我们将目标函数拆解为恒等映射+残差函数的形式. 这样, 网络只需要关注学习更加简单的残差函数, 而不必重新开始拟合整个映射. 这使得优化问题更加接近于学习小的, 便宜量式的调整, 有助于减少训练过程中的优化障碍
- **避免负优化**: 没有引入残差的时候, 额外的层可能会在优化初期大幅偏离已有模型的性能. 通过提供恒等映射的捷径(Shortcut Connection), 新加入的层在最初近似为恒等映射, 即使它们的参数还未充分优化, 也不会使性能变得更差, 这保障了深度网络至少可以"退化"回一个较浅的网络性能, 不会因为层数的增加而性能下降

[^1]: He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for image recognition (No. arXiv:1512.03385). arXiv. https://doi.org/10.48550/arXiv.1512.03385
[^2]: Apache. (2022, 二月 4). 深度学习之残差神经网络（ResNet） [知乎专栏文章]. https://zhuanlan.zhihu.com/p/463935188
