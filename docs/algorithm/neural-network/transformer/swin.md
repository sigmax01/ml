---
title: Swin
comments: false
---

# Swin[^1]

## 概要

该研究展现了一种新的视觉Transformer, 叫做Swin Transformer. 它是一种能够胜任计算机视觉的一般性[骨干网络](/dicts/backbone). 将Transformer从语言领域应用到视觉领域面临着诸多的挑战, 这些挑战源于两者之间的差异. 例如[视觉实体在尺度上的差异](/dicts/large-variation-in-scale-visual-entities/), 图像的高像素分辨率. 为了解决这些差异产生的问题, 作者提出了一种**层次化**的Transformer, 使用**移位窗口**计算表示. 这个移位窗口的方案通过**将自我注意力限制在不重叠的局部窗口中, 同时允许不同窗口间的连接**, 从而提高效率. 这种层次化的结构具有在不同尺度上建模的灵活性, 并且其计算复杂度和图像的大小呈线性关系. 这些特性使得Swin Transformer能够兼容广泛的视觉任务, 包括图像分类, 在ImageNet-1K上达到87.3%的top-1精度; 密集预测任务, 如目标检测, 在COCO测试集上, 58.7的框平均精度和51.1的掩码平均精度; 语义分割, 在ADE20K上达到53.5mIoU. 它的性能已经超过了之前的SOTA, 在[COCO](/dicts/coco)上, Box AP +2.7, Mask AP +2.6. 在[ADE20K](/dicts/ade20k)上, +3.2 mIoU, 展现了基于Transformer的模型作为视觉主干网络的潜力. 这种**层次化**的设计和**移位窗口**的方法被证明对于所有的MLP架构都是有益的. 代码可以在[https://github.com/microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)找到.

## 背景

### CV领域的骨干网络

计算机视觉领域的建模长期以来一直由卷积神经网络主导. 从AlexNet及其在ImageNet图像分类挑战赛上的革命性表现开始, CNN架构通过更大的规模, 更广泛的连接和更加复杂的卷积形式变得越来越强大. CNNs作为各种视觉任务的主干网络, 它们的架构的进步带来了性能的提升, 从而显著促进了整个领域的发展. 

### NLP领域的骨干网络

在另一个赛道上, 自然语言处理中网络架构的演变则走了一条不同的道路, 如今盛行的架构是Transformer. Transformer模型专为序列建模和[转导任务](/dicts/inductive-transductive-learning)而设计, 其显著的特点在于利用注意力机制来模拟数据中的长程连接关系. 其在语言领域的巨大成功促使研究人员探究其在计算机视觉领域的应用, 最近在特定的任务上, 例如图像分类和联合视觉-语言建模上, 取得了令人鼓舞的成果.

### 图像和文本模态的差异

在这个研究中, 他们旨在找到扩展Transformer的功能, 使其成为一种通用的计算机视觉领域的[骨干网络](/dicts/backbone), 正如它在NLP领域和CNN在视觉领域那样. 作者观察到, 将其在语言领域的出色表现转移到视觉领域面临的重大挑战, 可以用两种模态之间的差异来解释. 其中的一个差异是[尺度](/dicts/large-variation-in-scale-visual-entities/), 于作为语言Transformer处理基本单元的词元不同, 视觉元素的尺度差异很大, 这个问题在目标检测任务中受到了极大关注. 现在的基于Transformer的模型, 所有的token都是固定尺度的, 这对于在视觉领域的应用是不合适的. 另外一个差异是图像中像素的分辨率远远高于文本段落中的文字. 许多视觉任务, 如语义分割, 需要像素级密集预测, 如果图像的分辨率一大, 对于Transformer模型来说是难以处理的, 假设我们对于整张图片的所有像素运用全局注意力, 其复杂度$O(B\cdot N^2\cdot D)$和图像的大小是呈现二次方关系的. 

### Swin的提出

#### 层次化架构 {#hierarchical-structure}

为了解决这些问题, 作者提出了一种Transformer[通用主干网络](/dicts/backbone), 叫做Swin Transformer. **它会构建分层特征图, 并且其计算复杂度和图像的大小呈现线性关系. 它通过从小patches开始, 并在更深的Transformer层中(更深的stage)逐渐合并相邻补丁, 构建层次表示(如下图所示)**. 利用这些分层特征图, Swin Transformer模型可以方便地利用高级稠密预测技术, 例如特征金字塔网络(FPN)或者U-Net. 他们将图像划分为非重叠的窗口(红色框), 并在窗口内局部计算自注意力, 且每个窗口中patch的数量固定, 因此复杂度是图像大小的线性函数(见下方复杂度计算).  

<figure markdown='1'>
<!-- ![](https://img.ricolxwz.io/77e84ae173ab3e1ff94dd4d5a678ac96.webp#only-light){ loading=lazy width='400' }
![](https://img.ricolxwz.io/77e84ae173ab3e1ff94dd4d5a678ac96_inverted.webp#only-dark){ loading=lazy width='400' } -->
![](https://img.ricolxwz.io/c78aa09e2a4a017c112084c2eca559a8.webp#only-light){ loading=lazy width='600' }
![](https://img.ricolxwz.io/c78aa09e2a4a017c112084c2eca559a8_inverted.webp#only-dark){ loading=lazy width='600' }
<figcaption>(a) Swin Transformer通过在深层合并patches构造层次的特征图, 由于只在窗口内部有注意力, 窗口内patch的数量是固定的, 所以复杂度和图像大小线性相关. (b) ViT只能产生一个低分辨率的特征图, 而且其复杂度和图像大小的平方呈正比</figcaption>
</figure>

???+ warning "层次化结构是指不同stage之间的结构具有层次化"

    🌟请注意, 这种patches合并, 窗口变大, 特征图分辨率减半的行为只会发生在不同的stage之间, 不会发生在同一个stage内部, 每个stage内部的窗口大小是不会变的, 每个stage内部会有多个transformer块, 这些transformer块的区别是窗口的位置会交替变化([移位窗口](#shifted-window)). 上图表示的是不同的stage之间的区别, 而不是同一个stage内部的区别.(看下面那张完整的Swin Transformer架构图).🌟

???+ note "Swin和ViT复杂度简单计算"

    假设图像的大小为$H\times W$, 大小的单位都是像素.

    - ViT: 假设图像在每一层都会被分为$N$个patches. 每一层都要在这些固定的patches之间计算注意力, 所以每一层的复杂度都为$O(N^2\cdot D)$, $D$是每个patch的嵌入维度. 假设patches的大小为$Q\times Q$, 那么$N=\frac{H\times W}{Q^2}$, 代入到公式里面, 就是$O(\frac{(H\times W)^2\cdot D}{Q^4})$. 这表明, ViT的时间复杂度和图像分辨率$H\times W$的平方呈正比
    - Swin: 假设第$i$个stage每个窗口的大小为$M_i\times M_i$, 其窗口的数量为$\frac{H\times W}{M_i^2}$, 每个窗口内, 会被划分为固定数量的patches, 假设为$P$, 那么在每个窗口内的自注意力复杂度为$O(P^2\cdot C_i)$, $C_i$是这些patches在第$i$个stage的嵌入维度, 所以第$i$个stage的复杂度是$O(\frac{H\times W\times P^2\times C_i}{M_i^2})$, 由于在最底部的stage$M_i\gg P$, 且随着$i$的变大, $P$是不变的, $M_i$是2倍扩大的, 所以$\frac{P^2}{M_i^2}$是单调递减的, 所以复杂度可以简化为$O(H\cdot W\cdot C_i)$. 这表明, Swin的时间复杂度和图像分辨率$H\times W$呈正比

这些优点使得Swin Transformer成为各种视觉任务的通用主干网络, 这和之前的基于Transformer架构的[ViT](/algorithm/neural-network/transformer/vit)形成对比, 后者所有的patch嵌入都只有单一尺度的特征, 无法有效学习到不同尺度的信息, ^^且只会产生单一分辨率的特征图^^, 所以有二次方的复杂度, ^^而Swin Transformer随着网络的加深, 其特征图的分辨率会逐渐降低, 而通道数逐渐增加^^.

#### 移位窗口 {#shifted-window}

Swin Transformer中的另一个比较重要的设计就是移位窗口, Shifted Window. 这种移位窗口发生在同一个stage的不同transformer块之间, 使得不同block的窗口之间产生了连接/重叠/交叉, 让原本独立的窗口之间产生了联系, 使得模型能够学习到更大范围的信息, 提升了在不同窗口之间的理解能力. 如下图所示.

<figure markdown='1'>
![](https://img.ricolxwz.io/c1b8074569e7a51ce2c3008dd1caf2dc.webp#only-light){ loading=lazy width='500' }
![](https://img.ricolxwz.io/c1b8074569e7a51ce2c3008dd1caf2dc_inverted.webp#only-dark){ loading=lazy width='500' }
<figcaption>移位窗口. 在左边, 使用的是一种常规的分割策略. 在右边, 是将前一个block的窗口进行移动.</figcaption>
</figure>

这个策略同样在真实世界降低延迟方面也很有效, 所有在同一个窗口内不同的patch的$Q$向量共享的都是同一个键集合, 而不是像传统的滑动窗口(sliding window)方法那样不同的patch的$Q$向量使用的是不同的键集合, 这样可以极大降低延迟和内存占用. 作者的实验显示在消耗的算力差不多的情况下移位窗口策略的延迟相比滑动窗口显著降低. 这种移位窗口的approach被证明对所有的MLP架构是有益的.

<figure markdown='1'>
![](https://img.ricolxwz.io/8b85d471089af1b92eb97119e1bc71ea.webp#only-light){ loading=lazy width='500' }
![](https://img.ricolxwz.io/8b85d471089af1b92eb97119e1bc71ea_inverted.webp#only-dark){ loading=lazy width='500' }
<figcaption>传统滑动窗口计算某一个patch对其他patches注意力的时候, 必须要用到它的周围一圈的patches组成的键集合; 但是移位窗口计算某一个patch对其他patches注意力的时候, 只用同一个窗口下的patches组成的键集合. 如b2, b3, b4, b5, b6, b7计算注意力的时候用到的都是同一个键集合. 但是b2, b3计算注意力的时候用到的不是同一个键集合</figcaption>
</figure>

???+ note "理解共用一个键集合"

    参考: https://github.com/microsoft/Swin-Transformer/issues/118#issuecomment-993124909

    > In traditional sliding window method. Different querys, q and q', will have different neighboring windows, which indicates different key sets. The computation is unfriendly to memory access and the real speed is thus not good.
    > 
    > In non-overlapping window method used in Swin Transformer. Different queries within a same red window, for example, q and q', will share a same key set.

### 评估

作者所提出的Swin Transformer在图像分类, 目标检测和予以分割等识别任务上取得了强大的性能. 它在性能上超越了ViT/DeiT, ResNe(X)t模型, 并且在延迟上所差无几. 它在[COCO](/dicts/coco)上的成绩为58.7 box AP, 51.1 mask AP, 超越了之前的SOTA, +2.7 box AP, +2.6 mask AP. 在[ADE20K](/dicts/ade20k)语义分割上, 在验证集上达到了53.5 mIoU, 较之前的SOTA增加了3.2 mIoU. 它也在ImageNet-1K图像分类的top-1的准度上达到了87.3%. 

---

作者深信, 一个在计算机视觉领域和自然语言处理领域统一的架构会对两个领域都有极大的帮助, 因为它能促进视觉和文本信号的联合建模, 并且来自这两个领域的建模只是可以更加深入地共享. 他们相信通过Swin Transformer在各种视觉任务上的出色表现, 能够让社区更加深入地推动这种信念, 并鼓舞视觉和语言信号的统一建模.

[^1]: Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows (No. arXiv:2103.14030). arXiv. https://doi.org/10.48550/arXiv.2103.14030
[^2]: 木盏. (2021, 十月 18). Swin transformer全方位解读【ICCV2021马尔奖】. Csdn. https://blog.csdn.net/leviopku/article/details/120826980
[^3]: 最容易理解的Swin transformer模型(通俗易懂版)—海_纳百川—博客园. (不详). 取读于 2024年12月21日, 从 https://www.cnblogs.com/chentiao/p/18379629