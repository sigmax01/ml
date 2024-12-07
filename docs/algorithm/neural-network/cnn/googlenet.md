---
title: GoogLeNet
comments: false
---

## 背景

GoogLeNet是2014年Christian Szegedy提出的一种全新的深度学习结构, 在这之前的AlexNet, VGG等结构都是通过增大网络的深度(层数)来获得更好的训练效果, 但是层数的增加会带来很多副作用, 比如过拟合, 梯度消失/爆炸等[^1]. 它是2014年ImageNet比赛的冠军, 它的主要特点是网络不仅有深度, 而且在横向上具有“宽度”, 并且能够在增加深度和宽度的情况下节约计算资源. 从名字GoogLeNet更是致敬了LeNet. GoogLeNet中最核心的部分是其内部的子网络结构Inception, 该结构灵感来源于NiN(Network in Network).

最近的实验证明, 深度学习领域很大的提升是由于新算法, 新网络架构的出现, 而不是新的数据集, 新的硬件设施. 它们提交到ILSVRC 2014的GoogleNet架构实际上比两年前的由Krizhevsky等人提出的架构少12倍的参数, 但是准度显著高于后者. 另外一个值得关注的电视, 随着手机和嵌入式设备的发展, 本地的算力变得越来越重要. GoogLeNet的优化重心并不完全是准度上的提升, 而是综合考虑了算力的稀缺性等因素, 对于多数的实验, GoogLeNet将推理的算力预算控制在了15亿线性操作内.

[^1]: GoogLeNet. (2022). 收入 百度百科. https://baike.baidu.com/item/GoogLeNet/22689587
