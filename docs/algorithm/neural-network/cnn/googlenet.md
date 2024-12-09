---
title: GoogLeNet
comments: false
---

# GoogLeNet[^1]

## 背景

GoogLeNet是2014年Christian Szegedy提出的一种全新的深度学习结构, 在这之前的AlexNet, VGG等结构都是通过增大网络的深度(层数)来获得更好的训练效果, 但是层数的增加会带来很多副作用, 比如过拟合, 梯度消失/爆炸等[^2]. 它是2014年ImageNet比赛的冠军, 它的主要特点是网络不仅有深度, 而且在横向上具有“宽度”, 并且能够在增加深度和宽度的情况下节约计算资源. 从名字GoogLeNet更是致敬了LeNet. GoogLeNet中最核心的部分是其内部的子网络结构Inception, 该结构灵感来源于NiN(Network in Network).

### 算力约束

最近的实验证明, 深度学习领域很大的提升是由于新算法, 新网络架构的出现, 而不是新的数据集, 新的硬件设施. 它们提交到ILSVRC 2014的GoogleNet架构实际上比两年前的由Krizhevsky等人[^6]提出的架构少12倍的参数, 但是准度显著高于后者. 另外一个值得关注的点是, 随着手机和嵌入式设备的发展, 本地的算力变得越来越重要. GoogLeNet的优化重心并不完全是准度上的提升, 而是综合考虑了算力的稀缺性等因素, 对于多数的实验, GoogLeNet将推理的算力预算控制在了15亿线性操作内.

### 使用卷积层减少特征图尺寸

尽管最大池化可能会导致空间信息的丢失, 比如在ResNet中就使用了[设置stride=2](/algorithm/neural-network/cnn/resnet/#plain-network)来减半特征图的尺寸. 但是相同的卷积层结构(指使用最大池化)也成功地被运用于定位, 问题检测等任务中.

### 多尺度滤波器

Serre等人[^5]借鉴了灵长类动物的视觉皮层的工作原理, 使用了不同尺度的Gabor滤波器来处理多尺度特征, 这些滤波器的参数(如频率, 方向, 尺度等)在训练过程中保持不变. Inception类似于它们的模型, 但是Inception模型所有的滤波器都是通过学习得到的, 此外, Inception模型中的层会被重复多次, 在GoogLeNet中, 最终的网络深度达到了22层.

### NiN

[Network-in-Network](/algorithm/neural-network/cnn/nin)是一个由Lin等人[^4]提出的用于增加神经网络表示能力的一种方法. 经典CNN中的卷积层其实是利用线性滤波器对图像进行内积运算, 在每个局部输出后面跟着一个非线性的激活函数, 最终得到的叫做特征图. 而这种卷积滤波器是一种广义线性模型(GLM, Generalized Linear Model). 所以用CNN进行特征提取的时候, 其实就隐含地假设了特征是线性可分的, 可实际问题往往是难以线性可分的. GLM的抽象能力是比较低的, 自然而然地我们想到用一种抽象能力更强的模型去替换它, 从而提升传统CNN的表达能力. NiN通过在卷积操作之后加入一个微型的MLP, 从而增加网络的非线性表示能力[^3]. 他们使用了$1\times 1$的卷积核进行跨通道池化, 在GoogLeNet中也大量使用了这种结构, 但是在GooLeNet中, 它有两重意义, 一个是用于提高网络的抽象表达能力, 另一个关键的是用于降维操作移除计算瓶颈, 这不仅仅增加了网络的深度, 而且拓宽了网络的宽度, 于此同时没有增大计算压力.

### R-CNN

当时的物体检测的SOTA是由Girshick等人[^7]提出的R-CNN(Regions with CNN), R-CNN将物体检测问题分为了两个子问题: 一个是区域提议, 另一个是特征提取. R-CNN首先通过传统的区域提议方法来生成候选框, 这些候选框是图像中可能包含物体的区域, 然后, 对每一个生成的候选区域, R-CNN使用CNN提取特征, 通常会使用一个预训练的网络, 如AlexNet来提取这些特征, 再后, R-CNN在每个候选区域提取到的特征上使用一个SVM进行分类, 为了进一步提高物体检测的精度, R-CNN还使用了边界框回归来对候选框的位置进行微调. R-CNN这种方法利用了低层次候选框的准度, 也利用了当时强大的SOTA CNN的支持. 在GoogLeNet中对于物体检测任务也使用了类似的方法, 但是对于两个阶段都有显著地增, 如使用Multi-box prediction, 意味着对于每一个物体, 会生成多个候选框, 也使用了集成学习的思想(通过多个独立训练的模型对候选框进行分类, 然后民主/加权投票).

## 动机

### 高质量训练集短缺

最简单直接的提高神经网络性能的方法是增大它的体积(包括深度和宽度). 但是这意味着参数的数量很大, 导致过拟合, 特别是带标签的样本太少的时候. 创造高质量训练集的成本是特别大的, 特别是当标注员都分不清的时候, 如下图的西伯利亚哈士奇和爱斯基摩犬.

<figure markdown='1'>
  ![](https://img.ricolxwz.io/d00c62f52d0b6d1abc779568e8a2ba7c.png){ loading=lazy width='500' }
</figure>

### 计算资源短缺

即便网络体积是均匀增长的, 也会导致计算资源需求极速增长. 例如, 在深度视觉网络中, 如果两个卷积层连接在一起, 则其滤波器数量的任何均匀增长都会导致计算量呈二次增加. 如果添加的许多滤波器是没有有效利用的, 如滤波器的大多数权重都被设置为$0$, 那么很多的计算资源是被浪费的.

### 简化稀疏网络

最基本的解决上述问题的方法是将视线从全连接转移到稀疏连接的架构, 甚至是在卷积层内部. 稀疏连接的一个原因是模仿大脑的结构和工作方式, 大脑中的神经网络也不是完全连接的, 只有一部分神经元之间有直接的连, 因此参数量大大减少. 但问题是现有的硬件大多针对密集矩阵的运算进行优化, 非均匀稀疏模型的设计更加复杂, 需要更加精密的工程和计算基础设施类实现, 也就是说, 即使使用了稀疏连接, 稀疏矩阵的计算时间也不比密集矩阵小很多[^8].

那么有没有解决办法呢, 将稀疏矩阵聚类为相对密集的矩阵, Arora等人[^9]的研究指出如果一个数据集的概率分布能够被一个非常大, 非常稀疏的网络描述的话, 那么可以通过分析激活值的统计特性和对高度相关的输出进行聚类来逐层构建出一个最有网络, 这说明臃肿的稀疏网络可以被不失性能的简化, 也就是简化后的网络比原来的全连接网络的性能更高[^8]. 虽然数学证明需要很多的条件, 但是这个现象和Hebbian Priciple很像, 简单的说, 如果两个神经元常常同时产生动作电位, 或者说同时激活(fire), 这两个神经元之间的连接就会变强, 反之变弱(neurons that fire together, wire together), 所以说, 这个结论在条件不是很全的情况下也可以用.

所以现在的问题就是找到一个架构使得能够使用稀疏网络, 同时又能够更高效地利用我们目前的硬件. 正如Hebbian所说的一样, 许多文献指出, 将稀疏矩阵聚类为多个较为密集的子矩阵, 并对每个子矩阵进行单独计算能够创造出SOTA的稀疏矩阵乘法效率. Inception架构是该文章第一作者的一个案例研究, 目的是评估一个假设的网络拓扑构建算法的效果, 这个算法的目标是近似一个稀疏结构, 即使用密集的, 现成的组件来模拟这些稀疏结构. 尽管最初的设计和构建非常具有推测性, 但是在进行了两轮的调整后, Inception架构已经超过了NiN的表现. 在进一步调整学习率, 超参数和训练方法后, Inception架构的性能得到了显著提升, 特别是在定位和目标检测等任务中.

## 架构

正如前面所说的, Inception架构的目的是找到怎么才能在CNN中将已经存在的密集组件近似为最佳稀疏结构. 由于我们的网络要具有平移不变性(即图片平移不会影响结果), 所以网络是基于CNN的, 要做的是找到一种最佳的局部构造, 并在空间上重复应用. Arora等人[^9]提出了一种逐层构建的方式, 在这种方式中, 应该分析网络的最后一层的相关统计信息, 并将具有高度相关性的神经元聚类.

???+ example "如何进行逐层构建"

    例如, 最后一层包含1000个神经元, 每个神经元对应一个不同的类别, 通过计算这些神经元的相关性, 发现200个神经元的相关性较高, 将这200个高度相关的神经元分成10个簇, 每个簇包含20个功能近似的神经元, 根据上述簇, 为倒数第二层设计10个子模块, 每个模块复杂处理对应簇内的特定类别的信息. 重复这一过程, 直到构建完整个网络的所有层. 前一层根据下一层的模块划分, 前一层的每个神经元只连接到最后一层对应模块的神经元, 而不与其他模块的神经元连接.

作者假设在CNN中, 前面层的每个神经元都对应于输入图像的某一部分区域, 这些单元被组合为滤波器组(filter banks). 靠近输入的低层, 它们的神经元通常会集中在图像的局部区域, 这意味着, 会有很多的聚焦于一个局部区域的簇, 这可以通过$1\times 1$的卷积核覆盖. 相反的, 可能会有数量较少的, 但是分布更广的簇, 这些簇可以通过卷积覆盖更大的区域. 随着区域越来越大, 簇的数量会越来越少.

???+ question "对于Inception架构以及filter bank这样设计的理解"

    个人理解: Inception架构本质上就是根据下一层的结果通过关联一些相似的滤波器, 将它们归为一类, 然后这些滤波器所对应的神经元只会下一层的对应类的神经元连接, 以模拟稀疏网络. 如果不是稀疏连接而是全连接的话, 使用$3\times 3$的滤波器就可以了, 因为所有的神经元都是连接的. 但是对于Inception来说, 如果是不同的簇之间的神经元上下层是没有连接的, 这也就是说, 我们要手动将关注局部和全局的簇所对应的神经元都连接起来, 这也就是为什么Inception会有一个filter bank的概念, $1\times 1$的卷积核所形成的簇代表的是局部特征, 对应的是代表局部特征的簇所在的神经元之间的连接, 而$5\times 5$的卷积核所形成的簇代表的是全局特征, 对应的是代表全局特征的簇所在的神经元之间的连接, 如果缺少$5\times 5$, 那么这部分的连接就消失了, 网络的可解释性就会降低.

[^1]: Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., & Rabinovich, A. (2014). Going deeper with convolutions (No. arXiv:1409.4842). arXiv. https://doi.org/10.48550/arXiv.1409.4842
[^2]: GoogLeNet. (2022). 收入 百度百科. https://baike.baidu.com/item/GoogLeNet/22689587
[^3]: Network in Network 简单理解. (2015, 十二月 26). Emanuel’s Notes. http://yoursite.com/2015/12/26/nin/index.html
[^4]: Lin, M., Chen, Q., & Yan, S. (2014). Network In network (No. arXiv:1312.4400). arXiv. https://doi.org/10.48550/arXiv.1312.4400
[^5]: Serre, T., Wolf, L., Bileschi, S., Riesenhuber, M., & Poggio, T. (2007). Robust object recognition with cortex-like mechanisms. IEEE Transactions on Pattern Analysis and Machine Intelligence, 29(3), 411–426. IEEE Transactions on Pattern Analysis and Machine Intelligence. https://doi.org/10.1109/TPAMI.2007.56
[^6]: Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems, 25. https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
[^7]: Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014). Rich feature hierarchies for accurate object detection and semantic segmentation (No. arXiv:1311.2524). arXiv. https://doi.org/10.48550/arXiv.1311.2524
[^8]: Liiiii_Regina. (2021, 十月 19). GoogLeNet 论文阅读笔记. Csdn. https://blog.csdn.net/qq_43488473/article/details/120848718
[^9]: Arora, S., Bhaskara, A., Ge, R., & Ma, T. (2013). Provable bounds for learning some deep representations (No. arXiv:1310.6343). arXiv. https://doi.org/10.48550/arXiv.1310.6343
