---
title: AlexNet
comments: false
---

## 架构[^1]

### ReLU激活函数

AlexNet采用的是非饱和非线性函数, 而不是才用传统的饱和非线性函数, 如$\tanh$或者是sigmoid. 饱和的意思就是说当$x$的值变得很大或者很小的时候, 饱和函数的值会解禁期极限, 在这些区域, 导数趋近于$0$, 导致梯度下降更新缓慢, 称为"梯度消失问题". 梯度消失会导致网络学习效率大幅降低, 尤其是在深度网络中, 靠近输入层的参数几乎无法被有效更新. 

$max(0, x)$是AlexNet采用的非饱和非线性函数. 输出为正值的时候, 导数恒为$1$, 输出为负值的时候, 导数恒为$0$. 

<figure markdown='1'>
![](https://img.ricolxwz.io/ef635b798a4cfa5fe4207d4d54967ff5.png){ loading=lazy width='300' }
</figure>

从图中可以看出, 使用一个ReLU的四层CNN达到$25\%$错误率的速度比使用$\tanh$激活函数的CNN快$6$倍. 

> Jarrett et al. [11] claim that the nonlinearity f (x) = |tanh(x)| works particularly well with their type of contrast normalization followed by local average pooling on the Caltech-101 dataset.

虽然早期的研究也用了$|\tanh(x)|$作为激活函数, 但是他们的主要目标是避免过拟合, 而不是真正的加速模型的训练.

### 分布式GPU训练

AlexNet的作者使用了两块GTX 580训练, 至于这里为什么是$2$, 是因为它的内存太小了, 导致无法把整个网络放在上面. 

得益于GPU能够直接访问和写入其他GPU的内存, 不用经过宿主的通用内存, GPU的分布式训练得以发展. 

他们将卷积核(神经元)分半分配到两个GPU上, 每个GPU只处理一部分神经元. 并且, 他们还使用了一种优化通信的策略: 在普通的卷积神经网络中, 某一层的所有卷积核(或神经元)会从上一层的所有特征图中获取输入, 比如, 层3的核会从层2的所有特征图中获取信息, 这种全连接方式有助于模型更好地获取全局信息, 但是在多GPU并行训练的时候会带来巨大的通信开销. 该文章引入了一个新的方法, 仅在部分层进行GPU间的通信, 比如, 层3的核心仍然从层2的所有特征图中获取输入(即, 跨GPU通信仍然会发生), 但是从层3到层4的时候, 每个GPU上的核心只能从同一个GPU的特征图获取输入.

不同的连接模式会影响通信量: 全连接(每层的所有神经元都连接)通信量大但是性能受限; 部分连接(限制某些层或者某些神经元的连接)减少通信但是可能会影响模型的性能. 可以使用交叉验证来测试不同的连接模式, 直到找到通信量和计算量之间的最佳平衡点, 目标是调整通信量, 使其成为总计算量中可以接受的比例.

这个结构其实和一种"柱状"的CNN较为类似, 每个柱都是一部分独立的CNN分支, 这些柱可以看作是并行的网络结构, 每个柱都会生成自己的输出, 通过后续层将这些输出融合在一起. 文中提到的是非独立的柱状CNN, 这个柱不是完全独立的, 柱之间会共享某些信息, 通过两块GPU分别负责不同的柱, 能够实现并行计算.

由于网络绝大部分的参数都集中在了全连接层的第一层上, 而全连接层的第一层接受来自卷积层的最后一层的输出作为输入, 为了保持双GPU网络和单GPU网络的参数总量相近, 设计者决定不减少最后一层卷积层和全连接层的参数数量.

### 局部归一化[^2]

局部归一化, Local Response Normalization (LRN)首次在AlexNet中提出, 通过实验证实可以提高模型的泛化能力, 但是提升的很少, 以至于后面都不再使用, 甚至有人觉得它是一个伪命题, 因此饱受争议.

[^1]: Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems, 25. https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
[^2]: LoveMIss-Y. (2019, 三月 26). 深度学习饱受争议的局部响应归一化(LRN)详解. Csdn. https://blog.csdn.net/qq_27825451/article/details/88745034