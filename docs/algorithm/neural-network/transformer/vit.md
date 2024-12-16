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

## 相关工作

Transformer由Vaswani等人提出[^6], 起初用于机器翻译, 现在已经在很多NLP任务中成为了SOTA方法. 基于Transformer的大模型常常是在大规模语料库上训练然后在特定任务上进行微调. BERT[^7]在预训练阶段采用了一种类似“去噪”的自监督学习任务(denoising self-supervised pre-training task), 其核心是MLM, 即在输入句子中随机mask掉一部分次, 然后让模型在不看到这些遮盖词原本信息的情况下, 基于上下文语义来预测被遮盖位置应填入的词汇, 这一过程可以理解为一种“去噪”任务, 输入文本相当于被“噪声”污染过的句子, 而BERT需要学会利用上下文信息将“噪声”清除, 恢复出原本的正确内容. GPT[^8]在预训练阶段使用的是语言建模的优化目标.

由于自注意力机制的全局特性, 直接将自注意力应用于图像的话, 每个像素都要关注其他的所有像素. 这所产生的复杂度是输入像素点数量的平方, 显然无法扩展到现实中的输入尺寸. 因此, 为了在图像处理领域运用Transformer, 在过去的工作中, 研究者尝试了多种不同的近似方法. Parmar等人[^9]将自注意力机制仅仅应用于每个查询像素的局部领域, 而不是全局, 这种局部的多头点积自注意力块可以完全取代卷积计算. 在不同的战线上, 由Child等人[^10]提出的Sparse Transformer将输入特征划分为多个区域或块, 然后在这些较小的块内部使用全局或者局部自注意力, 使模型在每个块内建立特征的全局关联. 同时, 在块与块之间采用某种稀疏的连接规则(如跳跃式的远程连接), 而非对所有块进行完全的全局交互, 这样, 每个块内部可以有比较密集的自注意力计算(相当于“局部全局”注意力), 而块与块之间的自注意力交互则被限制在某些特定特征上, 从而降低了整体计算量. 另一种由Weissenborn等人[^11]提出的方法还是对输入特征进行分块的, 不同的是, 他们更强调利用分层结构来减少全局交互, 而不同于Child等人提出的固定块之间的跳跃式稀疏连接. 在计算情况下, 还可以只对行维度或列维度上进行全局的块交互[^4][^12]. 许多这些特殊设计的注意力架构在计算机视觉任务中展现了具有前景的结果, 但是在硬件加速器上需要复杂的工程设计以实现高效运算.

和作者提出的模型关系最大的是由Cordonnier等人[^13]提出的模型. 他们从输入图像中提取大小为2\*2的patches, 然后在此之上应用完整的自我注意力. 这个模型和ViT非常像, 但是作者进一步证明在大规模数据集上预训练的ViT甚至能够比当时的CNN领域的SOTA更好(还是输给了数据集大小, orz). 并且, Cordonnier等人使用的是一个小的2\*2的patch, 这使得他们的模型仅仅能够被应用到小分辨率图像上, 而作者提出的ViT也能够处理中等分辨率的图像.

[^1]: Wang, X., Girshick, R., Gupta, A., & He, K. (2018). Non-local neural networks (No. arXiv:1711.07971). arXiv. https://doi.org/10.48550/arXiv.1711.07971
[^2]: Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-end object detection with transformers (No. arXiv:2005.12872). arXiv. https://doi.org/10.48550/arXiv.2005.12872
[^3]: Ramachandran, P., Parmar, N., Vaswani, A., Bello, I., Levskaya, A., & Shlens, J. (2019). Stand-alone self-attention in vision models (No. arXiv:1906.05909). arXiv. https://doi.org/10.48550/arXiv.1906.05909
[^4]: Wang, H., Zhu, Y., Green, B., Adam, H., Yuille, A., & Chen, L.-C. (2020). Axial-DeepLab: Stand-alone axial-attention for panoptic segmentation (No. arXiv:2003.07853). arXiv. https://doi.org/10.48550/arXiv.2003.07853
[^5]: Young. (2019, 三月 20). 深度学习的归纳偏置是什么？ [知乎回答]. 知乎. https://www.zhihu.com/question/41404496/answer/627673667
[^6]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2023). Attention is all you need (No. arXiv:1706.03762). arXiv. https://doi.org/10.48550/arXiv.1706.03762
[^7]: Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding (No. arXiv:1810.04805). arXiv. https://doi.org/10.48550/arXiv.1810.04805
[^8]: Radford, A. (2018). Improving language understanding by generative pre-training. https://www.mikecaptain.com/resources/pdf/GPT-1.pdf
[^9]: Parmar, N., Vaswani, A., Uszkoreit, J., Kaiser, Ł., Shazeer, N., Ku, A., & Tran, D. (2018). Image transformer (No. arXiv:1802.05751). arXiv. https://doi.org/10.48550/arXiv.1802.05751
[^10]: Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). Generating long sequences with sparse transformers (No. arXiv:1904.10509). arXiv. https://doi.org/10.48550/arXiv.1904.10509
[^11]: Weissenborn, D., Täckström, O., & Uszkoreit, J. (2020). Scaling autoregressive video models (No. arXiv:1906.02634). arXiv. https://doi.org/10.48550/arXiv.1906.02634
[^12]: Ho, J., Kalchbrenner, N., Weissenborn, D., & Salimans, T. (2019). Axial attention in multidimensional transformers (No. arXiv:1912.12180). arXiv. https://doi.org/10.48550/arXiv.1912.12180
[^13]: Cordonnier, J.-B., Loukas, A., & Jaggi, M. (2020). On the relationship between self-attention and convolutional layers (No. arXiv:1911.03584). arXiv. https://doi.org/10.48550/arXiv.1911.03584
