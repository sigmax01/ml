---
title: Adapter
comments: true
---

# Adapter[^1]

## 概要

对预训练的模型进行微调是NLP领域有效的迁移学习方法. 然而, 在很多下游任务中, 参数微调效率低下. 对于每一个全新的任务都需要一个新的模型. 为了解决这个问题, 作者提出了一种利用adapter模块进行迁移的方法. Adapter模块提供了模型的整体**紧凑性和可扩展性**(这两个形容词的意思见下小节), 每当需要为一个新任务进行适配的时候, 只需要增加相应的adapter模块, 这些模块通常包含较少的参数, 相比于完全训练一个新的模型或者对整个模型进行微调, 节省了大量的计算资源和存储空间. 并且, 在添加新的任务的时候, 不需要调整之前为其他任务添加的adapter, 只需要新增相应的adapter模块, 需要注意的是, 每个任务都有其独立的adapter, 彼此之间互不干扰, 确保模型能够同时处理多个任务, 避免了“灾难性遗忘”(catastrophic forgetting)的问题. 预训练模型的参数将保持固定, 或者说“冻结”, 使得在不同任务之间有一个高程度的参数共享. 为了证明adapter模块的有效性, 作者最近将刚刚提出的BERT模型应用到26个不同的文本分类任务中, 包括GLUE基准测试. Adapters实现了接近SOTA的性能, 同时每个任务仅仅只需要增加几个参数. 在GLUE基准测试中, 作者将模型的性能控制在和全量微调的0.4%左右, 但是对于每个任务仅增加了3.6%的参数, 同比之下, 微调对每个任务会增加100%的参数.

???+ note "什么是灾难性遗忘"

    灾难性遗忘, Catastrophic Forgetting, 是指在连续学习多个任务的过程中, 学习新知识的过程会迅速破坏之前获得的信息, 而导致模型性能在旧任务中急剧下降[^3]. 这种现象在顺序学习或者增量学习中尤为常见. 灾难性遗忘主要由于以下几个原因: a) 共享参数, 大多数神经网络通过共享大量参数来学习和表示不同的任务, 当模型在训练新任务的时候, 导致新任务的学习过程覆盖了旧任务的知识; b) 缺乏记忆机制, 传统的神经网络缺乏有效的机制来区分和保护不同任务的重要参数, 导致新任务的学习过程覆盖了旧任务的知识; c) 优化冲突: 不同任务之间的优化目标可能存在冲突, 训练新任务时优化的方向可能和保留旧任务性能的方向相反, 从而导致旧任务的性能下降.

???+ tip "GLUE基准测试"

    GLUE(General Language Understanding Evaluation)基准测试是纽约大学和华盛顿大学联合开发的一套用于评估自然语言处理(NLP)模型在多项语言理解任务上的综合性能的基准. 它包含了多个字任务, 包括但是不限于单句理解任务, 句子对理解任务, 问答和推理任务.

## 背景

预训练模型能够通过迁移学习在许多NLP任务上取得很强的性能. BERT, 一个基于Transformer的网络, 在大规模语料库上进行无监督预训练, 在文本分类和抽取式问答上达到了SOTA的表现.

### 紧凑型和可扩展性

在该研究中, 任务以流(stream)的形式连续到达, 系统需要逐一进行处理, 而不是一次性获取所有任务. 目标是在不为每一个任务都训练一个新模型的情况下, 找到一种机制使得在每个任务上都表现良好. 高程度的任务间共享(意味着大部分模型参数或者结构在不同任务间是共享的)对于云服务等应用尤其有用, 因为在这些应用中, 模型需要被训练以依次处理来自客户的众多任务. 作者提出了一种迁移学习的方法, 能够产生**紧凑性和可扩展性**的下游模型. 紧凑性意味着模型对每个任务增加一小部分参数就能解决很多不同的任务. 可扩展性意味着模型能够在不出现灾难性遗忘的前提下, 被增量训练以处理新任务. 作者提出的方法就能产生这样一个模型并不牺牲性能.

### 基于特征的迁移和微调

目前在NLP中最流行的两种迁移学习的方法是feature-based迁移和微调. 相反, 作者展现的是一种基于adapter模块[^2]的替代方案. 基于特征的迁移学习其核心思想是利用预训练模型生成的特征表示, 例如, 实值的词嵌入, 句子级嵌入, 段落级嵌入, 将这些特征作为输入提供给为下游特定任务设计的独立模型(如分类器, 回归模型, 序列标注模型等等). 微调包括调整预训练模型的参数已使其适应下游任务. 最近的工作证明微调常常比基于特征的迁移性能更高.

### 参数量

基于特征的迁移学习和微调对于每个人物都需要一堆新的权重参数. 如果网络的较底层能够共享权重的话, 微调更加节约参数. 但是, 作者提出的adapter方法更加节约参数:laughing:. 下图展示了这种trade-off.

<figure markdown='1'>
![](https://img.ricolxwz.io/eb18642fbd8a8d7ab88bc333671b123d.webp#only-light){ loading=lazy width='400' }
![](https://img.ricolxwz.io/eb18642fbd8a8d7ab88bc333671b123d_inverted.webp#only-dark){ loading=lazy width='400' }
<figcaption>对于adapter和微调准度和任务相关参数数量的trade-off. y轴已经针对全量微调的准度进行归一化. 下/中/上三条曲线分别展示了20分位, 50分位, 80分位的GLUE基准测试中9个任务的性能</figcaption>
</figure>

x轴展示的是对于每个任务训练的参数. 这个对应于解决每个额外任务所需要的边际增加. 基于adapter的调整在达到相似性能的情况下可以少训练两个数量级的参数.

### 数学表示

**Adapters是在预训练模型的层之间添加的新模块**. 基于adapter和基于特征的迁移学习, 微调之间的差异如下: 考虑一个参数为$\bm{w}$函数(神经网络)$\phi_{\bm{w}}(\bm{x})$. 基于特征的迁移学习结合了$\phi_{\bm{w}}(\bm{x})$和一个全新的函数$\mathcal{X}_{\bm{v}}$, 产生$\mathcal{X}_{\bm{v}}(\phi_{\bm{w}}(\bm{x}))$. 只有新的基于任务的参数$\bm{v}$, 会被额外训练. 微调对于每个任务来说都包含了对原始参数$\bm{w}$的修改, 限制了产生模型的紧凑型. 对于adapter来说, 也定义了一个新的函数, $\psi_{\bm{w}, \bm{v}}(\bm{x})$, 但是这里的$\bm{w}$是从预训练那里搬过来的. ^^作者还设计了$\bm{v}_0$, 保证在训练开始的时候, adapter模块不会显著改变模型的输出, 即$\psi_{\bm{w}, \bm{v}}(\bm{x})\simeq \phi_{\bm{w}}(\bm{x})$, 这对于保证模型的初始性能和稳定性非常重要, 有助于加快收敛.^^ 在训练的过程中, 只有$\bm{v}$被调整. 对于深层网络, 定义$\psi_{\bm{w}, \bm{v}}(\bm{x})$通常包括在原网络$\phi_{\bm{w}}$中添加新的层. 如果选择$|\bm{v}|<<|\bm{w}|$, 那么产生的模型对于很多任务来说需要的参数$\sim |\bm{w}|$. 因为$\bm{w}$是固定的, 所以模型可以被扩展到其他任务而不影响之前的任务(没有灾难性遗忘).

### 多任务和持续学习

基于adapter的调优与多任务和持续学习相关. 多任务学习可以带来紧凑的模型, 这意味着, 通过同时训练一个模型来处理多个相关任务, 模型的参数量可以相对较少, 但是, 多任务学习有一个缺点, 每个任务需要不同类型的数据, 如SA需要标注有情感标签的文本, NER需要标注实体的文本, 问答系统需要上下文与问题及答案的配对, 而且不同任务的数据量差异可能很大, 确保每个任务的数据质量和数量平衡是一个挑战, 还有不同任务的数据格式差异, 需要设计复杂的数据处理流水线来转换这些数据, 使其适应共享模型的输入格式, 不同的任务有不同的重要性和难度, 需要为每个任务设计合适的损失函数和权重, what's more, 不同任务的梯度更新方向可能不一致, 导致训练过程中梯度冲突, 影响模型的收敛... 持续学习系统目标是从无穷无尽的流式任务中学习, 这种范式具有挑战性因为网络在重新训练后会忘记之前的任务(灾难性遗忘). **Adapters的不同之处在于不同的任务之间不会发生交互, 而且共享的权重是被冻结的**, 这意味着该模型在少量任务特定参数的代价下能够完美地记住先前的任务.

### 效果

作者证明在一个大量的, 多样化的文本分类任务上, adapters能够实现NLP参数的高效率微调. 最重要的创新是发明一个高效的adapter模块能够和base模型产生很好的集成. 作者提出了一种简单但是高效, 的瓶颈结构. 在GLUE基准测试中, 作者的方法几乎达到了和全量微调的BERT一样的效果, 但是作者只使用了3%的任务相关的参数, 但是微调用了100%的参数. 作者观察到在另外17个公共文本数据集达到了相似的效果. 总结来说, 基于adapter的调优能够产生一个单一的, 可扩展的在文本识别领域能够接近SOTA的模型.

???+ note "什么是瓶颈结构"

    瓶颈结构, Bottleneck Structure, 是深度学习中常用的一种架构设计方法. 它的核心思想是通过在网络中引入一个或者多个较小的, 维度受限的层, 从而迫使模型在有限的容量下提取和表示最关键的特征, 这种设计不仅能够减少模型的参数量, 还能提高模型的训练效率和泛化能力. 例如, 使用1*1卷积核跨通道池化减小特征图的维度.

## 方法

作者提出了一个在多个下游任务上调优大模型的策略. 他们的方法有三个重要的特征: (i) 它能够达到很好的性能; (ii) 禁止序列化任务训练, 也就是说, 它不需要同时访问所有的数据集; (iii) 对于每个任务它只加一小部分参数. 这些特性对于云服务特别有用, 因为有许多的下游任务都需要有一个模型, 所以这种高强度的参数共享是必要的.

为了达到这些特性, 他们提出了一种瓶颈结构, adapter模块. 调优adapter模块包含对于每个下游任务增加一小部分的新的参数. 当对深度网络执行普通的微调的时候, 会修改网络的顶层, 这是必要的, 因为上游任务(即预训练时使用的任务)和下游任务(即你希望微调后网络执行的新任务)在标签空间和损失函数上存在差异. Adapter模块执行的是一种更加通用的架构修改, 以将与训练网络重新用于下游任务. 特别的, 这个adapter调优的策略包含向预训练的网络注入新的层(\^***\^, 相同的概念反复讲来讲去...). 原始网络的参数是被冻结的, 然而新的adapter层以随机的方式初始化(之前明明说在训练开始的时候, 要使得新函数和原函数相似... 所以怎么可能是随机的?). 在标准的调优中, 会移除预训练模型的顶层, 然后根据下游任务的需求, 设计一个新的输出层, 在新的任务数据上, 同时更新预训练模型的所有权重(包括底层和新添加的顶层). 与此相反, 在adapter调优中, 原始网络的参数是冻结的, 它们在下游任务是共享的. 

Adapter模块有两个核心的特性: 参数数量少而且近乎相同的初始化(near-identity initialization). 这个adapter模块相比于原始网络的层必须要小, 使得增加下游任务的时候总体模型的大小增长相对缓慢. 近乎相同的初始化, 这个特性是需要的, 能使得训练更加稳定, 通过新函数和原函数近似恒等, 保证在训练开始的时候原始网络的输出不受影响. ^^在训练的过程中, adapters可能会对网络中信息的流动和激活值的分布产生影响, 通过这种方式, adapter能够调整模型的表现, 使其更好地适应特定任务和数据的分布. 如果某些adapter模块在特定情况下不需要, 可以选择忽略它们, 这有助于减少计算开销, 并避免不必要的参数更新. 在作者后续的研究中, 他们观察到某些adapter对网络的影响力更大, 这意味着并非所有的adapter模块在调整模型的时候都同样重要.^^ 此外, 如果adapter的初始设置导致偏离了原始函数太多, 模型在训练的过程中可能无法顺利进行(难以收敛).

### Transformer网络的实例化

作者的实例化基于adapter的文本Transformer调优, 这些模型在许多的NLP任务中达到了SOTA, 包括翻译, 文本分类问题. 作者考虑的是由Vaswani等人[^4]提出的标准的Transformer架构.

由于adapter模块的设计空间极大, 研究人员可以在其架构上做出很多选择, 在面对多种可能的设计选择的时候, 作者决定采用一种较为简单的adapter架构, 这种简单设计不仅易于实现和理解, 还能在多个数据集上取得很好的性能. 虽然作者最终选择了简单的设计, 但是他们并没有忽视其他更加复杂的架构可能带来的潜在优势, 因此, 他们进行了多种更加复杂的设计实验, 以评估其性能.

<figure markdown='1'>
![](https://img.ricolxwz.io/188d8d2ffcf8b5a1b8be33ca52db1d9d.webp#only-light){ loading=lazy width='400' }
![](https://img.ricolxwz.io/188d8d2ffcf8b5a1b8be33ca52db1d9d_inverted.webp#only-dark){ loading=lazy width='400' }
<figcaption>Adapter模块的架构以及它和Transformer的集成</figcaption>
</figure>

在Transformer层中添加两个adapter模块, 一个是在多头注意力的投影层之后, 另一个是在2层FNN之后(左) Adapter模块采用瓶颈设计, 这意味着它们具有较少的参数量, adapter内部使用了残差连接, 有助于减少“退化问题”, **在adapter调优的时候, adapter模块/LN/分类头(没有显示在图中)的参数会参与训练, 其余参数全部冻结.** 

???+ note "Adapter的位置"

    注意到左边Transformer层里面有两个残差连接, 每个残差连接前都会有一个投影, 下面的那个投影对应的是把所有的头的输出拼接起来, 然后通过一个线性变换将拼接后的矩阵投影会原始维度, 和输入的维度match. 上面的那个投影是因为这个2xFNN通常会对维度进行扩展, 所以需要通过线性变换降为输入的维度. 残差连接之后的结果会被喂到LN里面.

    Adapter模块所在的位置是**在LN之前, 在投影之后**. 

为了限制参数的数量, 作者提出了一种“瓶颈结构”. Adapters首先将原来$d$维的特征映射到一个较小的维度, $m$, 应用非线性变换, 然后在映射回$d$维, 以进行残差连接. 每个adapter的参数(包括截距)为$2md+d+m$. 通过设置$m\ll d$, 可以限制每个任务新增的参数量. 在实际中, $2md+d+m$约为$0.5-8\%$的原网络参数量. 这个“瓶颈维度”, $m$, 提供了一种简单的平衡性能和参数效率的方法. Adapter内部使用了残差连接, ^^如果投影层(右图最上面的那个层)的参数初始化为零, 则该adapter模块就被初始化一个近似恒等函数, 这就实现了在训练开始的时候, adapter模块不会显著改变模型的输出, 即$\psi_{\bm{w}, \bm{v}}(\bm{x})\simeq \phi_{\bm{w}}(\bm{x})$^^.

除了训练adapter模块中的层, 对于每个下游任务, 他们还训练了Transformer层中的两个LN层. 这种方法类似于条件BN和自调节, 他们的共同特点是通过减少少量的参数来实现网络的快速适应, 这些方法每层只需要大概$2d$的参数来对任务进行有效的适应. 

???+ note "什么是条件批归一化"

    条件批归一化(Conditional Batch Normalization, CBN)是一种在标准批归一化的基础上进行扩展的方法, 目的是使得模型能够根据不同的条件动态地调整归一化过程. 在传统的BN中, 网络的每一层都会在训练过程中对其输入进行归一化, 具体做法是减去均值, 除以标准差, 并且学习缩放和偏移参数. CBN时归一化的一个扩展, 它根据不同的条件来调整归一化的参数(如均值, 方差, 缩放参数和偏移参数), 这些条件可以时输入相关的(例如输入的某些特征), 也可以是任务相关的(例如在多任务学习中, 不同任务可能有不同的归一化策略).

[^1]: Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., Laroussilhe, Q. de, Gesmundo, A., Attariyan, M., & Gelly, S. (2019). Parameter-efficient transfer learning for NLP (No. arXiv:1902.00751). arXiv. https://doi.org/10.48550/arXiv.1902.00751
[^2]: Rebuffi, S.-A., Bilen, H., & Vedaldi, A. (2017). Learning multiple visual domains with residual adapters (No. arXiv:1705.08045). arXiv. https://doi.org/10.48550/arXiv.1705.08045
[^3]: 大规模语言模型—灾难性遗忘-行麦科技. (不详). 取读于 2024年12月17日, 从 https://www.aihomecaring.com/?jishu/89.html
[^4]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2023). Attention is all you need (No. arXiv:1706.03762). arXiv. https://doi.org/10.48550/arXiv.1706.03762
