---
title: LoRA
comments: false
---

# LoRA[^1]

## 概要

NLP邻域最重要的一个范式是先使用通用领域的大规模数据进行预训练然后将其搬到特定的任务或者领域上. 重新训练模型所有参数的完整微调方法是不太可行的, 拿GPT-3 175B举个例子, 部署每一个任务特定的微调模型, 每个都要以训练175B参数为代价. 作者提出了低秩适配(Low-Rank Adaptation, LoRA), 它会冻结预训练模型的权重, 并将可训练的秩分解矩阵(trainable rank decomposition matrices)注入到Transformer模型架构的每一层中「怎么感觉和[Adapter](/algorithm/neural-network/transfer-learning/adapter)有点像」, 对于下游任务显著减少了可训练的参数量. 对比使用Adam优化器微调的GPT-3 175B, LoRA可以将可训练的参数量减少了10000倍, 同时, 对于GPU内存的要求减少了3倍. 使用LoRA调优后的模型性能和使用微调的模型性能相当或甚至更好, 但是前者有更少的可训练参数, 更高的训练吞吐量, **并且, 和adapters不一样的是, 没有额外的推理延迟**. ^^作者通过实验发现了语言模型适配中的秩缺失现象, 解释和确认了LoRA方法的有效性.^^ 相关的代码包括在PyTorch上使用LoRA可以在[这里](https://github.com/microsoft/LoRA)找到. 

## 背景

### 微调的问题

很多在NLP领域的应用依赖于将一个大型的预训练语言模型搬到许多下游任务中. 这样的adaptation经常是通过微调实现的, 微调会更新预训练模型的所有参数, 然而, 这样的操作不是紧凑的. 随着大模型每隔几个月就被训练一次, 这从对GPT-2或者RoBERTa大模型而言仅仅是“不太方便”, 转变为对拥有1750亿可训练参数的GPT-3而言的部署挑战.

### Adapter/Prompt的问题

许多研究者试图通过仅调整部分参数或者对于每个新的任务都学习一个外部模块. 这样一来, 对于每个任务只需要在预训练模型之外存储和加载一小部分task-specific的参数, 极大提高了在部署阶段的操作效率. 🌟然而, 现有的方法通常存在一些问题, 一方面, 这些方法通过增加模型的深度来实现某些功能(如使用adapter模块[^2]), 另一方面, 这些方法可能会减少模型能够处理的序列长度(如使用Prompt调优, 加入了一些额外的指令), 即模型在处理长文本或长时间序列数据时的能力受到限制, 这两个因素导致推理时间的显著增加. 更重要的是, 这些方法常常无法匹敌微调的基线, 导致需要在效率和性能之间做一个trade-off.🌟

### LoRA的提出

作者受到Li等人[^6]和Aghajanyan等人[^7]研究工作的启发. 🌟他们的研究表明, 学习到的过参数化模型(Over-Parameterized Model)实际上存在于一个低内在维度空间(Low Intrinsic Dimension)中. 基于这一发现, 作者假设在模型的**调优/适应**过程中, 权重的变化**也**具有低内在秩(Intrinsic Rank). 这促使作者提出了低秩适应(LoRA)的方法.🌟 ^^LoRA允许我们通过在调优过程中优化某些特定密集层(Dense Layer)的权重变化的秩分解矩阵(Rank Decomposition Matrices)来间接"训练"这些密集层, 同时保持预训练的权重冻结.^^ 如下图所示.

<figure markdown='1'>
![](https://img.ricolxwz.io/1ed1d2f0549dd53c0a7684355d9f56a4.webp#only-light){ loading=lazy width='200' }
![](https://img.ricolxwz.io/1ed1d2f0549dd53c0a7684355d9f56a4_inverted.webp#only-dark){ loading=lazy width='200' }
<figcaption>LoRA. 会对原始权重和调整权重进行相加操作, $d$表示密集层的理论最大秩, $r$表示在调优中的权重变换矩阵的秩</figcaption>
</figure>

拿GPT-3 175B举个例子, 某些密集层的理论最大秩为12288(即图中的$d=12288$), 但是在实际调优的过程中其**权重变化受限于低维度**(根据Li等人的研究), 如图中$r$甚至可以是$1$或者$2$已经足够, 而远远不需要达到理论上的最大秩, 这使得LoRA在存储和计算上非常高效.

### LoRA优点

LoRA有以下的优点:

- 一个预训练模型可以被共享, 并构建用于不同任务的许多小型LoRA模块. 我们可以冻结共享模型, 并通过替换上图的$\bm{A}$和$\bm{B}$来有效切换任务, 从而显著降低存储需求和任务切换开销
- LoRA通过优化注入较小的低秩矩阵, 使得训练过程更加高效, 并降低了硬件门槛, 尤其是在使用自适应优化器时候的效果显著, 提升可达到3倍. 传统的训练方法需要计算大部分参数的梯度并维护优化器状态
- 🌟LoRA在调优的时候对密集层的权重是一个**间接的, 线性的**调整, 在部署到特定任务前, LoRA对于权重的调整会被预先整合到原始密集层的权重中, 即在推理过程中这个LoRA模块就是不存在的. 而Adapter模块在推理的时候仍然是存在的, 因为adapter是在transformer层之间插入独立的神经网络层, 这些层包含独立的权重和偏置参数, 不能被简单地整合到某个层的权重中. 因此, LoRA的延迟和原始网络是相等的, 而Adapter新增了adapter模块内部的延迟🌟
- LoRA和其他方法在功能和实现上相互独立, 互不干扰, 所以可以和其他方法协同工作, 如Prefix-Tuning, 进一步提升模型的性能和效率

[^1]: Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-rank adaptation of large language models (No. arXiv:2106.09685). arXiv. https://doi.org/10.48550/arXiv.2106.09685
[^2]: Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., Laroussilhe, Q. de, Gesmundo, A., Attariyan, M., & Gelly, S. (2019). Parameter-efficient transfer learning for NLP (No. arXiv:1902.00751). arXiv. https://doi.org/10.48550/arXiv.1902.00751
[^3]: Rebuffi, S.-A., Bilen, H., & Vedaldi, A. (2017). Learning multiple visual domains with residual adapters (No. arXiv:1705.08045). arXiv. https://doi.org/10.48550/arXiv.1705.08045
[^4]: Li, X. L., & Liang, P. (2021). Prefix-tuning: Optimizing continuous prompts for generation (No. arXiv:2101.00190). arXiv. https://doi.org/10.48550/arXiv.2101.00190
[^5]: Lester, B., Al-Rfou, R., & Constant, N. (2021). The power of scale for parameter-efficient prompt tuning (No. arXiv:2104.08691). arXiv. https://doi.org/10.48550/arXiv.2104.08691
[^6]: Li, Y., & Liang, Y. (2019). Learning overparameterized neural networks via stochastic gradient descent on structured data (No. arXiv:1808.01204). arXiv. https://doi.org/10.48550/arXiv.1808.01204
[^7]: Aghajanyan, A., Zettlemoyer, L., & Gupta, S. (2020). Intrinsic dimensionality explains the effectiveness of language model fine-tuning (No. arXiv:2012.13255). arXiv. https://doi.org/10.48550/arXiv.2012.13255

