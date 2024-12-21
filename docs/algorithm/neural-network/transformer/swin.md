---
title: Swin
comments: false
---

# Swin[^1]

## 概要

该研究展现了一种新的视觉Transformer, 叫做Swin Transformer. 它是一种能够胜任计算机视觉的一般性[骨干网络](/dicts/backbone). 将Transformer从语言领域应用到视觉领域面临着诸多的挑战, 这些挑战源于两者之间的差异. 例如[视觉实体在尺度上的差异](/dicts/large-variation-in-scale-visual-entities/), 图像的高像素分辨率. 为了解决这些差异产生的问题, 作者提出了一种**层次化**的Transformer, 使用**滑动窗口**计算表示. 这个滑动窗口的方案通过**将自我注意力限制在不重叠的局部窗口中, 同时允许不同窗口间的连接**, 从而提高效率. 这种层次化的结构具有在不同尺度上建模的灵活性, 并且其计算复杂度和图像的大小呈线性关系. 这些特性使得Swin Transformer能够兼容广泛的视觉任务, 包括图像分类, 在ImageNet-1K上达到87.3%的top-1精度; 密集预测任务, 如目标检测, 在COCO测试集上, 58.7的框平均精度和51.1的掩码平均精度; 语义分割, 在ADE20K上达到53.5mIoU. 它的性能已经超过了之前的SOTA, 在[COCO](/dicts/coco)上, Box AP +2.7, Mask AP +2.6. 在[ADE20K](/dicts/ade20k)上, +3.2 mIoU, 展现了基于Transformer的模型作为视觉主干网络的潜力. 这种**层次化**的设计和**滑动窗口**的方法被证明对于所有的MLP架构都是有益的. 代码可以在[https://github.com/microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)找到.

## 背景

计算机视觉领域的建模长期以来一直由卷积神经网络主导. 从AlexNet及其在ImageNet图像分类挑战赛上的革命性表现开始, CNN架构通过更大的规模, 更广泛的连接和更加复杂的卷积形式变得越来越强大. CNNs作为各种视觉任务的主干网络, 它们的架构的进步带来了性能的提升, 从而显著促进了整个领域的发展. 

在另一个赛道上, 自然语言处理中网络架构的演变则走了一条不同的道路, 如今盛行的架构是Transformer. Transformer模型专为序列建模和[转导任务](/dicts/inductive-transductive-learning)而设计, 其显著的特点在于利用注意力机制来模拟数据中的长程连接关系. 其在语言领域的巨大成功促使研究人员探究其在计算机视觉领域的应用, 最近在特定的任务上, 例如图像分类和联合视觉-语言建模上, 取得了令人鼓舞的成果.

在这个研究中, 他们旨在找到扩展Transformer的功能, 使其成为一种通用的计算机视觉领域的[骨干网络](/dicts/backbone), 正如它在NLP领域和CNN在视觉领域那样. 作者观察到, 将其在语言领域的出色表现转移到视觉领域面临的重大挑战, 可以用两种模态之间的差异来解释. 其中的一个差异是[尺度](/dicts/large-variation-in-scale-visual-entities/), 于作为语言Transformer处理基本单元的词元不同, 视觉元素的尺度差异很大, 这个问题在目标检测任务中受到了极大关注. 现在的基于Transformer的模型, 所有的token都是固定尺度的, 这对于在视觉领域的应用是不合适的. 另外一个差异是图像中像素的分辨率远远高于文本段落中的文字. 许多视觉任务, 如语义分割, 需要像素级密集预测, 如果图像的分辨率一大, 对于Transformer模型来说是难以处理的, 假设我们对于整张图片的所有像素运用全局注意力, 其复杂度$O(B\cdot N^2\cdot D)$和图像的大小是呈现二次方关系的. 

为了解决这个问题, 作者提出了一种Transformer[通用主干网络](/dicts/backbone), 叫做Swin Transformer. **它会构建分层特征图, 并且其计算复杂度和图像的大小呈现线性关系. 它通过从小patches开始, 并在更深的Transformer层中逐渐合并相邻补丁, 构建层次表示(如下图所示)**. 利用这些分层特征图, Swin Transformer模型可以方便地利用高级稠密预测技术, 例如特征金字塔网络(FPN)或者U-Net. 他们将图像划分为非重叠的窗口(红色框), 并在窗口内局部计算自注意力, 且每个窗口中patch的数量固定, 因此复杂度是图像大小的线性函数(见下方复杂度计算).  

<figure markdown='1'>
![](https://img.ricolxwz.io/77e84ae173ab3e1ff94dd4d5a678ac96.webp#only-light){ loading=lazy width='400' }
![](https://img.ricolxwz.io/77e84ae173ab3e1ff94dd4d5a678ac96_inverted.webp#only-dark){ loading=lazy width='400' }
<figcaption>红色的边框区域表示一个窗口, 被红色的边框围起来的是数个patch. (a) Swin Transformer通过在深层合并patches构造层次结构并且由于只在窗口内部有注意力, 窗口内patch的数量是固定的, 所以复杂度和图像大小线性相关. (b) </figcaption>
</figure>

???+ note "Swin和ViT复杂度简单计算"

    假设图像的大小为$H\times W$, 大小的单位都是像素.

    - ViT: 假设图像在每一层都会被分为$N$个patches. 每一层都要在这些固定的patches之间计算注意力, 所以每一层的复杂度都为$O(N^2\cdot D)$, $D$是每个patch的嵌入维度. 假设patches的大小为$Q\times Q$, 那么$N=\frac{H\times W}{Q^2}$, 代入到公式里面, 就是$O(\frac{(H\times W)^2\cdot D}{Q^4})$. 这表明, ViT的时间复杂度和图像分辨率$H\times W$的平方呈正比
    - Swin: 假设第$i$层每个窗口的大小为$M_i\times M_i$, 其窗口的数量为$\frac{H\times W}{M_i^2}$, 每个窗口内, 会被划分为固定数量的patches, 假设为$P$, 那么在每个窗口内的自注意力复杂度为$O(P^2\cdot C_i)$, $C_i$是这些patches在第$i$层的嵌入维度, 所以第$i$层的复杂度是$O(\frac{H\times W\times P^2\times C_i}{M_i^2})$, 由于在最底层$M_i\gg P$, 且随着$i$的变大, $P$是不变的, $M_i$是2倍扩大的, 所以$\frac{P^2}{M_i^2}$是单调递减的, 所以复杂度可以简化为$O(H\cdot W\cdot C_i)$. 这表明, Swin的时间复杂度和图像分辨率$H\times W$呈正比

这些优点使得Swin Transformer成为各种视觉任务的通用主干网络, 这和之前的基于Transformer架构的[ViT](/algorithm/neural-network/transformer/vit)形成对比, 后者所有的patch嵌入都只有单一尺度的特征, 无法有效学习到不同尺度的信息, 且只会产生单一分辨率的特征图(即那个窗口的大小是固定的), 所以有二次方的复杂度. 

[^1]: Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows (No. arXiv:2103.14030). arXiv. https://doi.org/10.48550/arXiv.2103.14030
