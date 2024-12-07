---
title: GoogLeNet
comments: true
---

## 背景

GoogLeNet是2014年Christian Szegedy提出的一种全新的深度学习结构, 在这之前的AlexNet, VGG等结构都是通过增大网络的深度(层数)来获得更好的训练效果, 但是层数的增加会带来很多副作用, 比如过拟合, 梯度消失/爆炸等[^2]. 它是2014年ImageNet比赛的冠军, 它的主要特点是网络不仅有深度, 而且在横向上具有“宽度”, 并且能够在增加深度和宽度的情况下节约计算资源. 从名字GoogLeNet更是致敬了LeNet. GoogLeNet中最核心的部分是其内部的子网络结构Inception, 该结构灵感来源于NiN(Network in Network).

最近的实验证明, 深度学习领域很大的提升是由于新算法, 新网络架构的出现, 而不是新的数据集, 新的硬件设施. 它们提交到ILSVRC 2014的GoogleNet架构实际上比两年前的由Krizhevsky等人提出的架构少12倍的参数, 但是准度显著高于后者. 另外一个值得关注的电视, 随着手机和嵌入式设备的发展, 本地的算力变得越来越重要. GoogLeNet的优化重心并不完全是准度上的提升, 而是综合考虑了算力的稀缺性等因素, 对于多数的实验, GoogLeNet将推理的算力预算控制在了15亿线性操作内.

尽管最大池化可能会导致空间信息的丢失, 比如在ResNet中就使用了[设置stride=2](/algorithm/neural-network/cnn/resnet/#plain-network)来减半特征图的尺寸. 但是相同的卷积层结构(指使用最大池化)也成功地被运用于定位, 问题检测等任务中. Serre等人借鉴了灵长类动物的视觉皮层的工作原理, 使用了不同尺度的Gabor滤波器来处理多尺度特征, 这些滤波器的参数(如频率, 方向, 尺度等)在训练过程中保持不变. Inception类似于它们的模型, 但是Inception模型所有的滤波器都是通过学习得到的, 此外, Inception模型中的层会被重复多次, 在GoogLeNet中, 最终的网络深度达到了22层.

Network-in-Network是一个由Lin等人提出的用于增加神经网络表示能力的一种方法. 该网络的亮点在于,

[^1]: Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., & Rabinovich, A. (2014). Going deeper with convolutions (No. arXiv:1409.4842). arXiv. https://doi.org/10.48550/arXiv.1409.4842
[^2]: GoogLeNet. (2022). 收入 百度百科. https://baike.baidu.com/item/GoogLeNet/22689587
[^3]: 放风筝的猪. (不详). Network in network（NIN）网络结构详解，网络搭建. Csdn. 取读于 2024年12月7日, 从 https://blog.csdn.net/weixin_45897172/article/details/128736549
