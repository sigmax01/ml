---
title: ViT
comments: false
---

## 摘要

当年, Transformer已经成为了NLP的de-facto标准(当然现在也是), 它在计算机视觉上的引用可以说是寥寥无几.在视觉领域, 注意力机制要不和卷积网络结合使用, 要不是替代其中的某一些组件, 但整体结构保持不变. 作者展示了这种对于卷积的依赖关系是多余的, 纯Transformer架构就可以直接应用于图像的块序列而得到非常好的分类效果. 作者先在大量数据上得到一个预训练模型, 然后将该模型在较小规模或者中等规模的数据集(例如ImageNet, CIFAR-10等)进行有监督的微调, 与当时的SOTA相比展现了出色的结果并且需要更少的资源训练.

???+ note "什么是image patches"

    "image patches"是指将输入的原始图像分割为一系列较小的矩形块(通常是大小相等的正方形块). 这些小块每一个都可以视为一个独立的图像片段(patch). 然后每个patch的像素会被展开为一维序列, 然后...

## 背景

基于注意力机制的架构, 特别是Transformer已经成为NLP领域的model of choice. 目前最主流的方法是先在大型语料库上进行预训练然后在较小的下游任务数据集上微调. 正是由于Transformer的计算效率和可扩展性, 训练庞大的前所未有的网络变得可能, 甚至有超过100B参数的. 随着模型和数据集的增长, 似乎还没有看到性能的饱和.

在计算机视觉领域, 然而, 卷积架构仍然是主流. 收到Transformer在NLP领域成功的启发, 有一些工作试图将类似于CNN的架构和自注意力机制结合在一起[^1][^2], 另一些工作试图将整个卷积层替换掉[^3][^4]. 随后的模型, 虽然在理论上是高效率的, 但是由于使用了特殊的注意力结构, 导致并没有很好的扩展到现代的硬件加速器上. 因此, 对于大尺度的图片识别, 传统的类ResNet目前还是SOTA的状态.

受到Transformer在NLP领域的启发, 作者测试了在最少的可能改动下将一个标准的Transformer直接用于图片. 为了实现这一点, 他们将一张图片分为了很多的patches并且对这些patches进行编码, 得到的序列被输入到Transformer中, 这些patches和Transformer中的Token是等价的. 与GPT等模型不同的是, 在预训练阶段, 他们使用的是有监督学习, 而不是自监督学习.

当我们在中等规模的数据集如ImageNet上进行训练, 且不实用高强度的正则化方法时, 这些模型的准确率比同等规模的ResNets低几个百分点. 这种令人失望的结果或许是预期的, 因为Transformer缺乏CNN所具有的一些归纳偏置, 如平移不变性, 局部特征, 因此对于数据量不足的情况下无法很好地泛化.

???+ note "什么是归纳偏置"

    归纳偏置, Inductive Biases, 或者叫做“归纳偏好”, 是学习算法自身在一个庞大的假设空间中对假设进行选择的启发或者“价值观”[^5]. 如CNN通过卷积核在输入数据的局部区域内进行操作, 强调了局部特征的重要性(locality); 并且, 图片平移后识别结果应该具有不变性(translation equivariance), CNN通过卷积核在空间上滑动实现这一点. 每个神经元仅仅与局部区域内的一小部分神经元相连接, 这种稀疏连接能够大幅减少计算量并提高性能, 所以GooLeNet通过密集部件模拟稀疏网络. 这里的局部特征, 平移不变性, 稀疏网络就是归纳偏置.

然而, 如果模型在大规模的数据集上(14M-300M)训练, 他们发现这种大规模的训练范式强于CNN的归纳偏置. 他们的ViT模型在大规模与训练并迁移到较少的任务时, 取得了优异的成果. 他们使用公共的ImageNet-21K或者内部的JFT-300M数据集预训练后, ViT在多项图片识别benchmark上接近或者打败了SOTA. 特别的, 最优的ViT模型在ImageNet上达到了88.55%, 在ImageNet-ReaL上达到了90.72%的准确率, 在CIFAR-100上达到了94.55%的准确率, 在VTAB19个任务套件上达到了77.63%的准确率.

[^1]: Wang, X., Girshick, R., Gupta, A., & He, K. (2018). Non-local neural networks (No. arXiv:1711.07971). arXiv. https://doi.org/10.48550/arXiv.1711.07971
[^2]: Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-end object detection with transformers (No. arXiv:2005.12872). arXiv. https://doi.org/10.48550/arXiv.2005.12872
[^3]: Ramachandran, P., Parmar, N., Vaswani, A., Bello, I., Levskaya, A., & Shlens, J. (2019). Stand-alone self-attention in vision models (No. arXiv:1906.05909). arXiv. https://doi.org/10.48550/arXiv.1906.05909
[^4]: Wang, H., Zhu, Y., Green, B., Adam, H., Yuille, A., & Chen, L.-C. (2020). Axial-DeepLab: Stand-alone axial-attention for panoptic segmentation (No. arXiv:2003.07853). arXiv. https://doi.org/10.48550/arXiv.2003.07853
[^5]: Young. (2019, 三月 20). 深度学习的归纳偏置是什么？ [知乎回答]. 知乎. https://www.zhihu.com/question/41404496/answer/627673667
