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

当他们在中等规模的数据集如ImageNet上进行训练, 且不实用高强度的正则化方法时, 这些模型的准确率比同等规模的ResNets低几个百分点. 这种令人失望的结果或许是预期的, 因为Transformer缺乏CNN所具有的一些归纳偏置, 如平移不变性, 局部特征, 因此对于数据量不足的情况下无法很好地泛化.

???+ note "什么是归纳偏置"

    归纳偏置, Inductive Biases, 或者叫做“归纳偏好”, 是学习算法自身在一个庞大的假设空间中对假设进行选择的启发或者“价值观”[^5]. 如CNN通过卷积核在输入数据的局部区域内进行操作, 强调了局部特征的重要性(locality); 并且, 图片平移后识别结果应该具有不变性(translation equivariance), CNN通过卷积核在空间上滑动实现这一点. 每个神经元仅仅与局部区域内的一小部分神经元相连接, 这种稀疏连接能够大幅减少计算量并提高性能, 所以GooLeNet通过密集部件模拟稀疏网络. 这里的局部特征, 平移不变性, 稀疏网络就是归纳偏置.

然而, 如果模型在大规模的数据集上(14M-300M)训练, 他们发现这种大规模的训练范式强于CNN的归纳偏置. 他们的ViT模型在大规模与训练并迁移到较少的任务时, 取得了优异的成果. 他们使用公共的ImageNet-21K或者内部的JFT-300M数据集预训练后, ViT在多项图片识别benchmark上接近或者打败了SOTA. 特别的, 最优的ViT模型在ImageNet上达到了88.55%, 在ImageNet-ReaL上达到了90.72%的准确率, 在CIFAR-100上达到了94.55%的准确率, 在VTAB19个任务套件上达到了77.63%的准确率.

## 相关工作

Transformer由Vaswani等人提出[^6], 起初用于机器翻译, 现在已经在很多NLP任务中成为了SOTA方法. 基于Transformer的大模型常常是在大规模语料库上训练然后在特定任务上进行微调. BERT[^7]在预训练阶段采用了一种类似“去噪”的自监督学习任务(denoising self-supervised pre-training task). GPT[^8]在预训练阶段使用的是语言建模的优化目标.

???+ note "什么是“去噪”自监督学习任务"

    其核心是MLM, 即在输入句子中随机mask掉一部分词, 然后让模型在不看到这些遮盖词原本信息的情况下, 基于上下文语义来预测被遮盖位置应填入的词汇, 这一过程可以理解为一种“去噪”任务, 输入文本相当于被“噪声”污染过的句子, 而BERT需要学会利用上下文信息将“噪声”清除, 恢复出原本的正确内容.

由于自注意力机制的全局特性, 直接将自注意力应用于图像的话, 每个像素都要关注其他的所有像素. 这所产生的复杂度是输入像素点数量的平方, 显然无法扩展到现实中的输入尺寸. 因此, 为了在图像处理领域运用Transformer, 在过去的工作中, 研究者尝试了多种不同的近似方法. Parmar等人[^9]将自注意力机制仅仅应用于每个查询像素的局部领域, 而不是全局, 这种局部的多头点积自注意力块可以完全取代卷积计算. 在不同的战线上, 由Child等人[^10]提出的Sparse Transformer将输入特征划分为多个区域或块, 然后在这些较小的块内部使用全局或者局部自注意力, 使模型在每个块内建立特征的全局关联. 同时, 在块与块之间采用某种稀疏的连接规则(如跳跃式的远程连接), 而非对所有块进行完全的全局交互, 这样, 每个块内部可以有比较密集的自注意力计算(相当于“局部全局”注意力), 而块与块之间的自注意力交互则被限制在某些特定特征上, 从而降低了整体计算量. 另一种由Weissenborn等人[^11]提出的方法还是对输入特征进行分块的, 不同的是, 他们更强调利用分层结构来减少全局交互, 而不同于Child等人提出的固定块之间的跳跃式稀疏连接. 在计算情况下, 还可以只对行维度或列维度上进行全局的块交互[^4][^12]. 许多这些特殊设计的注意力架构在计算机视觉任务中展现了具有前景的结果, 但是在硬件加速器上需要复杂的工程设计以实现高效运算.

和作者提出的模型关系最大的是由Cordonnier等人[^13]提出的模型. 他们从输入图像中提取大小为2\*2的patches, 然后在此之上应用完整的自我注意力. 这个模型和ViT非常像, 但是作者进一步证明在大规模数据集上预训练的ViT甚至能够比当时的CNN领域的SOTA更好(还是输给了数据集大小, orz). 并且, Cordonnier等人使用的是一个小的2\*2的patch, 这使得他们的模型仅仅能够被应用到小分辨率图像上, 而作者提出的ViT也能够处理中等分辨率的图像.

对于将CNNs和自注意力结合起来也有许多有趣的研究. Bello等人[^14]在CNN提取的特征图上添加自注意力, 增加模型对全局信息的感知, 提升图像分类效果. Hu等人[^15], Carion等人[^2]在CNN输出的基础上应用Transformer架构, 使模型更好地定位和识别图像中的目标对象. Wang等人[^1], Sun等人[^16]在时序序列的视觉特征中加入自注意力, 可以同时关注视频帧之间的全局关联和局部特征, 从而改善动作识别, 视频分类等任务, Wu等人[^17]通过在CNN中整合自注意力, 进一步提升图像分类精度, Locatello等人[^18]在没有明确标注的条件下, 将CNN和自注意力结合, 有助于物体识别. Chen等人[^19], Lu等人[^20], Li等人[^21]通过在CNN视觉特征和文本特征之间引入自注意力, 对图像和文本特征进行统一编码和融合, 从而实现如图文匹配, 图文问答, 多模态理解等复杂任务.

另外一个比较相关的模型是image GPT(iGPT)[^22], 他们将Transformer应用于减少分辨率/颜色空间的图像上, 这个模型被当作一个生成式模型进行无监督预训练, 然后得到的表示被微调或进行“线性探针”以获得分类性能, 在ImageNet上达到了72%的准度.

???+ note "什么是“线性探针”"

    线性探针通常是指在模型的某一层(通常是中间层或者是预训练模型的特定层)上提取特征表示, 然后在这些表示上训练一个简单的线性分类器(例如线性回归或者SVM)来预测特定的下游任务标签, 以此来衡量该层表示中所蕴含的信息和任务可分性.

作者的研究属于不断增多的研究工作之一: 这些研究都试图在比标准的ImageNet数据集更大规模的范围内开展图像识别的探索和实践. 换句话说, 他们的工作与其他的一些论文一起, 正推动图像识别从仅局限于ImageNet数据集, 扩展到处理更大, 更加广泛的图像数据. 此前的一些工作研究了CNN在数据集规模增大时的性能变化, 以及从大规模数据集(如ImageNet-21k和JFT-300M)进行迁移学习的效果. 本研究和这些工作相似, 也使用了上述的大型数据集, 但是和以往的ResNet模型不同的是, 本研究的重点在于使用Transformer模型进行训练和研究.

## 方法论

<figure markdown='1'>
  ![](https://img.ricolxwz.io/68495fc7236b9721a4b529966ca65c5e.png){ loading=lazy width='500' }
  <figcaption>模型overview. 将图片分为固定大小的patches, 然后对它们进行线性编码, 加入位置嵌入, 将结果喂到一个标准的Transformer编码器中. 为了能够执行分类任务, 作者使用了向序列中加入"classification token"的标准方式(类似于BERT). Transformer的编码器来源于Vvaswani等人.</figcaption>
</figure>

### ViT

对于模型的overview如上图所示. 这个标准的Transformer会收到1D的token嵌入序列. 为了处理2D的图像, 他们将图像$\mathbf{x} \in \mathbb{R}^{H \times W \times C}$首先转化为一个展平的2D patches $\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, $(H, W)$表示的是原图的分辨率, $C$表示的是通道数, $(P, P)$表示的是patch的分辨率, $N=HW/P^2$表示的是产生的patches的数量, 同时也是输入Transformer的序列长度. Transformer对于所有的层潜向量大小是固定的$D$, 所以他们使用可训练的线性变换对这些patches进一步展平到$D$维, 他们将这个线性变换的结果叫做patch嵌入(如下面的第一个公式所示).

$$
\begin{aligned}
\mathbf{z}_0 &= [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1 \mathbf{E}; \mathbf{x}_p^2 \mathbf{E}; \cdots; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{\text{pos}},
\quad \mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}, \quad \mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}\\[6pt]
\mathbf{z}'_\ell &= MSA(LN(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}, \quad \ell = 1 \dots L\\[6pt]
\mathbf{z}_\ell &= MLP(LN(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell, \quad \ell = 1 \dots L\\[6pt]
\mathbf{y} &= LN(\mathbf{z}_L^0)
\end{aligned}
$$

???+ note "上面公式中变量的含义"

    $\mathbf{z}$的下标$\ell$表示的是第$\ell$层, $\mathbf{z}_{\ell}$表示的是第$\ell$层Transformer的输出. $\mathbf{z}$的上标表示的是该层第几个输入, 如$\mathbf{z}^{0}_{0}$表示的是第0层的第0个输入. $\mathbf{E}$表示的是那个可学习的线性变换, 用于将patch转换为$D$维的嵌入. $L$是层的总数量. Transformer编码器由多头自注意力层(MSA)和多层感知机(MLP)层交替堆叠而成. 在每一个block之前会进行LN, 并在block之后有一个残差连接(MSA和MLP都算一个block).

和BERT的`[class]`token类似, 我们会在patches的嵌入组成的序列前面挂一个可学习的嵌入, 即$\mathbf{x}_{\text{class}}$, 来汇总整个序列的信息, 即在ViT中$\mathbf{z}^0_0=\mathbf{x}_{\text{class}}$. 经过多层的Transformer编码器之后, 这个token的状态会被更新, 最终在最后一层($L$层)的输出中, 它的含义就是整个图像的信息. 这个汇总的全局信息会被送入一个小型网络中, 又叫做“分类头”(classfication head), 这个分类头的输出维度的大小和类别数量相等. 在训练的时候, 为了让模型学到更加通用, 更加丰富的特征表示, 这个分类头由含有一个隐藏层的MLP构成, 中间可能会有非线性激活(如GELU), 这样可以对特征进行进一步的非线性映射, 分类头的输出会和真实标签比较, 计算交叉熵损失然后反向传播. 在微调的时候, 分类头通常被设置为一个单一的线性层(没有隐藏层), 因为在微调的时候, 模型已经有比较好的表示能力, 只需要一个简单的分类器, 对$\mathbf{z}^0_L$进行线性映射, 然后通过softmax输出最终的类别概率.

位置嵌入会被添加到patch嵌入以保持位置信息. 作者使用的是标准的可学习1D位置嵌入, 因为他们并没有发现使用2D嵌入有非常明显的性能提升. 生成的嵌入向量序列被作为编码器的输入.

#### 归纳偏置

作者注意到ViT相比于CNNs来说, 对于图片特定的偏置归纳明显更少. 在CNN中, 局部性, 二维邻域结构和平移等效性这些归纳偏置存在于模型的每一层中. 在ViT中, 只有Transformer块内部的MLP层(不是classification head的那个MLP)在某种程度上具有局部性和平移等变性(我认为只有局部性). ViT对图像二维邻域结构信息的使用非常有限, 它只在模型的开端阶段有所体现, 即通过将图像切分为patch来引入某种局部结构, 在微调阶段, 可以对positional embedding进行相应的调整(在下面一小节会讲), 这是唯二使用2D位置信息的时机. 除此之外, 在模型初始化的时候, 位置嵌入并没有编码任何2D位置信息(前面说了, 使用2D的性能没有提升多少, 所以没有用2D). 换句话说, 初始的positional embeddings并不像卷积或者网格编码那样直接携带强烈的空间结构信息, 需要在训练过程中“从零开始”去学习patch之间的空间关系.

???+ note "什么是二维邻域结构"

    “二维邻域结构”指的是模型在处理数据时体现出来的二维拓扑关联性, 即把输入数据(如图像)看作是有明确行列关系的二维网格. 卷积核在二维图像平面上滑动, 访问“邻近的”像素块, 从而保留了数据的空间排列方式和二维结构. 例如, 我们有一张灰度图像, 可以表示为一个二维像素矩阵:

    $$
    \begin{matrix}
    p_{11} & p_{12} & p_{13} & p_{14}\\
    p_{21} & p_{22} & p_{23} & p_{24}\\
    p_{31} & p_{32} & p_{33} & p_{34} \\
    p_{41} & p_{42} & p_{43} & p_{44} \\
    \end{matrix}
    $$

    当我们把卷积核应用到图像的左上角的时候, 它会作用在下列二维邻域上:

    $$
    \begin{matrix}
    p_{11} & p_{12} & p_{13} \\
    p_{21} & p_{22} & p_{23} \\
    p_{31} & p_{32} & p_{33}
    \end{matrix}
    $$

    这里的$p_{11}, p_{12}, p_{13}, p_{21}, p_{22}, p_{23}, p_{31}, p_{32}, p_{33}$组成了一个“二维邻域”, 这个邻域是有空间结构的.

???+ note "为什么MLP被认为是Local的"

    “局部性”强调的是模型倾向于在小范围的局部区域中发现特征. 当CNN用小卷积核在输入图像上滑动的时候, 每个卷积核一次只能“看”一个有限大小的局部感受野, 而不会看到其他地方的东西.

    对于MLP, local不是指图像空间上的locality, 是指MLP对每个输入的patch embedding独立, 相同的进行处理(共享权重), 并不考虑其他patch的信息(因为它没有像自注意力那样的全局关联计算), 换句话说, 对于序列中的每个token(patch embedding), MLP执行的变换是相同的. 从另一个角度理解, MLP是point-wise的操作, 它不会根据全局结构或者位置差异对不同的token赋予不同的处理方式, 而是对每个token单独且同样地变换.

    对于MSA, 自注意力机制会计算序列中任意两个token之间的注意力权重, 也就是说, 输出不仅仅取决于每个token自身的特征, 还高度依赖于其他token的关系, 能够“看到”整个序列的全局结构和上下文信息, 因此具有非局部性.

???+ question "为什么MLP被认为是Translationally Equivariant的(我认为MLP和MSA都没有这个性质)"

    “平移不变形”指的是, **无论特征出现在图像的哪个位置, 网络都能在那个相应的位置产出类似的响应, 并且模型无需为不同位置的同类特征单独学习不同的参数**.

    对于MSA, 由于ViT会引入位置编码, 这使得当图像进行平移之后, 其token的编码会发生变化, 这会使得自注意力分数产生显著变化, 即特征在不同位置会导致自注意力的重新分布, 如此一来, 网络就无法在保证自注意力层参数$W_q, W_k, W_v$不变的情况下, 对平移后的特征图产生与平移前的特征产生完全相同的响应.

    对于MLP而言, 由于ViT会引入位置编码, 那么相同的特征在不同的位置也会呈现出不同的输入信号(因为位置编码的注入改变了输入特征), 从而导致MLP在其参数不改变的情况下, 特征在不同位置的输出不再相同, 所以MLP其实也是没有Translationally Equivariant的.

???+ note "为什么切patch会引入局部的二维结构"

    切patch保留了在patch内部的二维结构信息, 而patches之间的这些二维结构信息是完全被摧毁的, 即patch是不知道自己在图像的哪个位置, 以及相邻的patch有哪些这些二维位置信息的. 模型需要在训练中学习patches之间的相对关系和空间结构.

#### 混合架构

除了这种在原始图像上切patches的做法, 作者还提出了一种混合模型架构, 在Transformer之前先使用CNN对图像进行特征提取. 即首先使用CNN对原图进行处理, 得到一个中间层的特征图. 这个特征图会被再切分成patches, 然后通过线性投影层$\mathbf{E}$对每个patch进行展平. 特别的, 如果patch大小是1*1的时候, 每个patch相当于是特征图的单个像素, 并将每个patch线性映射到$D$维, 然后添加那个`[class]`和位置嵌入.

### 微调和高分辨率



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
[^14]: Bello, I., Zoph, B., Vaswani, A., Shlens, J., & Le, Q. V. (2020). Attention augmented convolutional networks (No. arXiv:1904.09925). arXiv. https://doi.org/10.48550/arXiv.1904.09925
[^15]: Hu, H., Gu, J., Zhang, Z., Dai, J., & Wei, Y. (2018). Relation networks for object detection (No. arXiv:1711.11575). arXiv. https://doi.org/10.48550/arXiv.1711.11575
[^16]: Sun, C., Myers, A., Vondrick, C., Murphy, K., & Schmid, C. (2019). VideoBERT: A joint model for video and language representation learning (No. arXiv:1904.01766). arXiv. https://doi.org/10.48550/arXiv.1904.01766
[^17]: Wu, B., Xu, C., Dai, X., Wan, A., Zhang, P., Yan, Z., Tomizuka, M., Gonzalez, J., Keutzer, K., & Vajda, P. (2020). Visual transformers: Token-based image representation and processing for computer vision (No. arXiv:2006.03677). arXiv. https://doi.org/10.48550/arXiv.2006.03677
[^18]: Locatello, F., Weissenborn, D., Unterthiner, T., Mahendran, A., Heigold, G., Uszkoreit, J., Dosovitskiy, A., & Kipf, T. (2020). Object-centric learning with slot attention (No. arXiv:2006.15055). arXiv. https://doi.org/10.48550/arXiv.2006.15055
[^19]: Chen, Y.-C., Li, L., Yu, L., Kholy, A. E., Ahmed, F., Gan, Z., Cheng, Y., & Liu, J. (2020). UNITER: UNiversal image-TExt representation learning (No. arXiv:1909.11740). arXiv. https://doi.org/10.48550/arXiv.1909.11740
[^20]: Lu, J., Batra, D., Parikh, D., & Lee, S. (2019). ViLBERT: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks (No. arXiv:1908.02265). arXiv. https://doi.org/10.48550/arXiv.1908.02265
[^21]: Li, L. H., Yatskar, M., Yin, D., Hsieh, C.-J., & Chang, K.-W. (2019). VisualBERT: A simple and performant baseline for vision and language (No. arXiv:1908.03557). arXiv. https://doi.org/10.48550/arXiv.1908.03557
[^22]: Chen, M., Radford, A., Wu, J., Jun, H., Dhariwal, P., Luan, D., & Sutskever, I. (2020, 七月 12). Generative pretraining from pixels. International Conference on Machine Learning. https://www.semanticscholar.org/paper/Generative-Pretraining-From-Pixels-Chen-Radford/bc022dbb37b1bbf3905a7404d19c03ccbf6b81a8
