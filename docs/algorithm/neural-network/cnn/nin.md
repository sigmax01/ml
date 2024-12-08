---
title: NiN
comments: false
---

# NiN[^1]

## 动机

### 卷积层是一个GLM

CNN由卷积层和池化层交替连接构成. 卷积层中线性滤波器(卷积核)和底层感受野做内积, 然后在输入的每个局部部分应用非线性激活函数, 产生的结果叫做特征图.

CNN的滤波器是一个底层数据块的广义线性模型(GLM). 下面是GLM的解释: 在某些情况下, 线性回归模型是不合适的, 如果:
1. X和y之间的关系不是线性的, 它们之间存在某种非线性的关系, 例如, y随着X的增加而呈现指数级增长
2. y中的误差方差不是常数, 会随着X的变化而变化(异方差)
3. 因变量不是连续的, 而是离散的/分类的

广义线性模型(GLM)是线性回归的扩展, 它可以处理更加复杂的情况. 当线性回归模型不适用的时候, GLM可以通过调整结果来更好地拟合数, 它主要包括三个主要组成部分:
1. 随机成分(Random Component), 指的是响应变量(因变量)的分布类型. 与传统的回归假设响应变量服从正态分布不同, GLM允许响应变量服从多种分布, 如二项分布, 泊松分布, 伽马分布等, 这使得GLM能够处理分类数据, 计数数据等不同类型的因变量
2. 系统成分(Systematic Component), 指的是多个预测变量(自变量)通过线性组合的方式影响因变量, 即模型, 即模型中的线性预测器部分, 通常表示为$\eta=\beta_0+\beta_1X_1+\beta_2X_+...+\beta_pX_p$
3. 连接函数(Link Function), 连接函数用于将系统成分(线性预测起)和因变量的期望值联系起来. 它定义了因变量的期望值和线性预测器之间的关系. 通过选择合适的连接函数, GLM能够捕捉因变量和预测变量之间的非线性关系

CNN中的卷积层能够被视为一个GLM的原因在于:

- 随机成分: 在卷积层中, 卷积操作对感受野进行处理, 得到一个特征图, 每个特征图的值可以看作是卷积操作之后的响应, 类似于GLM中的响应变量. 卷积操作本身并没有对响应变量的值的分布进行限制, 可能包含多种类型, 类似于GLM中的“随机成分”
- 系统成分: 卷积操作本质上是对输入数据进行加权求和, 即感受野和卷积核做内积(多变量线性组合), 得到一个值, 这类似于GLM中的系统成分
- 连接函数: 经过卷积操作得到的结果会通过一个非线性激活函数(例如ReLU)进行处理, 这个激活函数在CNN中起到了连接函数的作用, 将线性组合的结果和卷积后的激活结果联系起来, 映射到了非线性特征空间

### GLM的缺陷

> By abstraction we mean that the feature is invariant to the variants of the same concept [2]. Replacing the GLM with a more potent nonlinear function approximator can enhance the abstraction ability of the local model. GLM can achieve a good extent of abstraction when the samples of the latent concepts are linearly separable, i.e. the variants of the concepts all live on one side of the separation plane defined by the GLM.

在这里, “抽象”的意思是指某个特征对于同一个概念的不同变体保持不变性, 具体来说, 这意味着无论同一个概念有多少不同的表现形式或者变化, 这个特征都能够保持一致, 从而有效地代表该概念的核心特征. 如, 无论一个猫的爪子如何在不同方向, 颜色, 姿态变化, 都属于猫爪子这个概念. CNN的隐含假设是猫的爪子和猫的尾巴的样本是线性可分的, 而实际情况是, 可能猫的尾巴上有一些特征和猫的爪子上有点像, 导致无法线性可分. 如果所有的猫的爪子样本和猫的尾巴样本在特征空间可以被一个超平面在搞维度中分开, 那么GLM就可以很好的工作, 但是实际情况是, GLM无法在高维找到一个很好的线性边界, 因此需要更加强大的非线性函数逼近器来捕捉这种复杂的, 非线性的关系. 该观点参考[^2].

在传统的CNN中, 为了表达这种高维的非线性函数逼近器, 他们用一系列过于完备(over-complete)的卷积核去应对潜空间中的所有概念(concept), 以确保能够捕捉到数据中潜在的变异或者特征.但是, 太多的卷积核描述一个概念可能会导致后面层的负担变大(我们需要考虑所有前一个层的所有组合), 因此, 他们从网络复杂度, 过拟合风险的角度再次说明了在局部使用更好的抽象表示方法的必要性.

### Maxout的启发

Maxout是由Goodfellow等人[^6]提出的一种新的激活函数. 在利用了Maxout技术的网络中, 每个神经元的输出不是通过传统的激活函数, 如ReLU, sigmoid等进行非线性变换, 而是通过选择多个线性变换中的最大值来拟合一种凸函数的非线性变换. 假设我们有多个仿射变换(就是卷积操作之后没有经过激活函数处理). Maxout会选择其中最大的一项作为该神经元的输出. 具体来说, Maxout将第$i$个仿射结果$z_i$划分为具有$k$个值的组$z_{i, j}, j\in[1, k]$($j$表示组内序号), 而不是直接将ReLu等函数作用于那个仿射结果上, 如$ReLU(z_i)$, 然后这个Maxout单元的输出为$Maxout(z_i)=\max_{j\in [1, k]}z_{i, j}$, 而$z_{i, j}$表示的是这个仿射结果对应的第$j$个线性函数, 所以Maxout能够产生一个分段性的输出, 选择的是$z_{i1}$到$z_{ij}$中较大的那个线性函数, 这种分段线性能力是的Maxout在理论上能够表示任意的凸函数. 而且Maxout激活函数还有一个重要的特点就是它是可以学习的, 而不是像传统的ReLU那样是固定的, Maxout对每一个仿射结果都可以学习$k$个不同的线性变换, 并根据数据的特征选择最合适的线性变换, 该观点参考[^3].

Maxout运用在仿射特征图上的表现就是, 假设有$96$个通道, 然后设置$k=4$, 说明以$4$张仿射特征图(通道)作为一组, 那么总共有$24$组, 对于每一个组的$4$个通道中的每一个像素点, Maxout层会选出这$4$个通道中值最大的像素作为这个位置的输出值, 其他位置的像素同理. 这样, 输出特征图的数量(通道)会减少到$24$个, 这种方法又叫做仿射特征图的最大池化(maximum pooling over affine feature maps).

与普通的卷积层相比, ReLU的非线性特性确实给网络带来了很多表达能力, 但是它仍然保留了一些线性结构, 换句话说, ReLU在输入大于$0$的区域表现为线性. Maxout通过选择不同的线性函数来表达更加复杂的决策边界, 尤其是处理那些输入空间凸集可分的数据时候, 比单一的ReLU更有表达力, 这也导致Maxout在使用之前有一个前提条件, 那就是输入空间中概念的表现形式都在一个凸集里面[^4], 但这在实际情况下不一定是成立的. 所以, 有必要提出一种更加普遍的函数逼近器(激活函数). 作者由此提出了NiN的架构.

## 创新

### NiN的设计

NiN, Network in Network通过一个称为"微网络"(micro network)的结构来替代传统的GLM. 在这个工作中, 作者采用的是MLP作为这个非线性函数逼近器, 该结构又被称为mlpconv.

<figure markdown='1'>
  ![](https://img.ricolxwz.io/0ede5e8e0112ff9b4584ad3a4187afe3.png){ loading=lazy width='500' }
</figure>

这个夹在卷积层之间的MLP对于所有的感受野都是共享权重的, 它会随着卷积核一起滑动(就是接受卷积核的输出, sliding a micro network). 具体来说, 对input进行卷积, 然后将卷积得到的特征图放到MLP中. 这个和之前的SMLP(Structured Multilayer Perceptron)[^5]不太一样, 那个是不会滑动的, 也就是输入的不同区域输入一个共享的MLP.

### 全局平均池化

[^1]: Lin, M., Chen, Q., & Yan, S. (2014). Network In network (No. arXiv:1312.4400). arXiv. https://doi.org/10.48550/arXiv.1312.4400
[^2]: 月来客栈. (2021, 十一月 26). NIN一个即使放到现在也不会过时的网络 [知乎专栏文章]. 深深深-深度学习. https://zhuanlan.zhihu.com/p/337035992
[^3]: Jasmine_Feng (导演). (2020, 十月 15). 小茉的花书笔记——神经网络激活函数之maxout [Video recording]. https://www.bilibili.com/video/BV1bD4y1d7Zz/?spm_id_from=333.337.search-card.all.click&vd_source=f86bed5e9ae170543d583b3f354fcaa9
[^4]: Teng-Sun. (2017, 十月 17). 深度学习方法（十）：卷积神经网络结构变化——Maxout Networks，Network In Network，Global Average Pooling. Csdn. https://blog.csdn.net/stt12345678/article/details/78261858
[^5]: Gülçehre, Ç., & Bengio, Y. (2013). Knowledge matters: Importance of prior information for optimization (No. arXiv:1301.4083). arXiv. https://doi.org/10.48550/arXiv.1301.4083
[^6]: Goodfellow, I. J., Warde-Farley, D., Mirza, M., Courville, A., & Bengio, Y. (2013). Maxout networks (No. arXiv:1302.4389). arXiv. https://doi.org/10.48550/arXiv.1302.4389
