---
title: GoogLeNet
comments: false
---

# GoogLeNet[^1]

## 背景

GoogLeNet是2014年Christian Szegedy提出的一种全新的深度学习结构, 在这之前的AlexNet, VGG等结构都是通过增大网络的深度(层数)来获得更好的训练效果, 但是层数的增加会带来很多副作用, 比如过拟合, 梯度消失/爆炸等[^2]. 它是2014年ImageNet比赛的冠军, 它的主要特点是网络不仅有深度, 而且在横向上具有“宽度”, 并且能够在增加深度和宽度的情况下节约计算资源. 从名字GoogLeNet更是致敬了LeNet. GoogLeNet中最核心的部分是其内部的子网络结构Inception, 该结构灵感来源于NiN(Network in Network).

最近的实验证明, 深度学习领域很大的提升是由于新算法, 新网络架构的出现, 而不是新的数据集, 新的硬件设施. 它们提交到ILSVRC 2014的GoogleNet架构实际上比两年前的由Krizhevsky等人[^6]提出的架构少12倍的参数, 但是准度显著高于后者. 另外一个值得关注的电视, 随着手机和嵌入式设备的发展, 本地的算力变得越来越重要. GoogLeNet的优化重心并不完全是准度上的提升, 而是综合考虑了算力的稀缺性等因素, 对于多数的实验, GoogLeNet将推理的算力预算控制在了15亿线性操作内.

尽管最大池化可能会导致空间信息的丢失, 比如在ResNet中就使用了[设置stride=2](/algorithm/neural-network/cnn/resnet/#plain-network)来减半特征图的尺寸. 但是相同的卷积层结构(指使用最大池化)也成功地被运用于定位, 问题检测等任务中. Serre等人[^5]借鉴了灵长类动物的视觉皮层的工作原理, 使用了不同尺度的Gabor滤波器来处理多尺度特征, 这些滤波器的参数(如频率, 方向, 尺度等)在训练过程中保持不变. Inception类似于它们的模型, 但是Inception模型所有的滤波器都是通过学习得到的, 此外, Inception模型中的层会被重复多次, 在GoogLeNet中, 最终的网络深度达到了22层.

[Network-in-Network](/algorithm/neural-network/cnn/nin)是一个由Lin等人[^4]提出的用于增加神经网络表示能力的一种方法. 经典CNN中的卷积层其实是利用线性滤波器对图像进行内积运算, 在每个局部输出后面跟着一个非线性的激活函数, 最终得到的叫做特征图. 而这种卷积滤波器是一种广义线性模型(GLM, Generalized Linear Model). 所以用CNN进行特征提取的时候, 其实就隐含地假设了特征是线性可分的, 可实际问题往往是难以线性可分的. GLM的抽象能力是比较低的, 自然而然地我们想到用一种抽象能力更强的模型去替换它, 从而提升传统CNN的表达能力. NiN通过在卷积操作之后加入一个微型的MLP, 从而增加网络的非线性表示能力[^3]. 他们使用了$1\times 1$的卷积核进行跨通道池化, 在GoogLeNet中也大量使用了这种结构, 但是在GooLeNet中, 它有两重意义, 一个是用于提高网络的抽象表达能力, 另一个关键的是用于降维操作移除计算瓶颈, 这不仅仅增加了网络的深度, 而且拓宽了网络的宽度, 于此同时没有增大计算压力.

当时的物体检测的SOTA是由Girshick等人[^7]提出的R-CNN(Regions with CNN), R-CNN将物体检测问题分为了两个子问题: 一个是区域提议, 另一个是特征提取. R-CNN首先通过传统的区域提议方法来生成候选框, 这些候选框是图像中可能包含物体的区域, 然后, 对每一个生成的候选区域, R-CNN使用CNN提取特征, 通常会使用一个预训练的网络, 如AlexNet来提取这些特征, 再后, R-CNN在每个候选区域提取到的特征上使用一个SVM进行分类, 为了进一步提高物体检测的精度, R-CNN还使用了边界框回归来对候选框的位置进行微调. R-CNN这种方法利用了低层次候选框的准度, 也利用了当时强大的SOTA CNN的支持. 在GoogLeNet中对于物体检测任务也使用了类似的方法, 但是对于两个阶段都有显著地增, 如使用Multi-box prediction, 意味着对于每一个物体, 会生成多个候选框, 也使用了集成学习的思想(通过多个独立训练的模型对候选框进行分类, 然后民主/加权投票).

[^1]: Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., & Rabinovich, A. (2014). Going deeper with convolutions (No. arXiv:1409.4842). arXiv. https://doi.org/10.48550/arXiv.1409.4842
[^2]: GoogLeNet. (2022). 收入 百度百科. https://baike.baidu.com/item/GoogLeNet/22689587
[^3]: Network in Network 简单理解. (2015, 十二月 26). Emanuel’s Notes. http://yoursite.com/2015/12/26/nin/index.html
[^4]: Lin, M., Chen, Q., & Yan, S. (2014). Network In network (No. arXiv:1312.4400). arXiv. https://doi.org/10.48550/arXiv.1312.4400
[^5]: Serre, T., Wolf, L., Bileschi, S., Riesenhuber, M., & Poggio, T. (2007). Robust object recognition with cortex-like mechanisms. IEEE Transactions on Pattern Analysis and Machine Intelligence, 29(3), 411–426. IEEE Transactions on Pattern Analysis and Machine Intelligence. https://doi.org/10.1109/TPAMI.2007.56
[^6]: Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems, 25. https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
[^7]: Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014). Rich feature hierarchies for accurate object detection and semantic segmentation (No. arXiv:1311.2524). arXiv. https://doi.org/10.48550/arXiv.1311.2524
