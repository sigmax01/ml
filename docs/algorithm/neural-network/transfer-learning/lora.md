---
title: Adapter
comments: false
---

## 概要

NLP邻域最重要的一个范式是先使用通用领域的大规模数据进行预训练然后将其搬到特定的任务或者领域上. 重新训练模型所有参数的完整微调方法是不太可行的, 拿GPT-3 175B举个例子, 部署每一个任务特定的微调模型, 每个都要以训练175B参数为代价. 作者提出了低秩适配(Low-Rank Adaptation, LoRA), 它会冻结与训练模型的权重, 并将可训练的秩分解矩阵(trainable rank decomposition matrices)注入到Transformer模型架构的每一层中「怎么感觉和[Adapter](/algorithm/neural-network/transfer-learning/adapter)有点像」, 对于下游任务显著减少了可训练的参数量. 对比使用Adam优化器微调的GPT-3 175B, LoRA可以将可训练的参数量减少了10000倍, 同时, 对于GPU内存的要求减少了3倍. 使用LoRA调优后的模型性能和使用微调的模型性能相当或甚至更好, 但是前者有更少的可训练参数, 更高的训练吞吐量, **并且, 和adapters不一样的是, 没有额外的推理延迟**. ^^作者通过实验发现了语言模型适配中的秩缺失现象, 解释和确认了LoRA方法的有效性.^^ 相关的代码包括在PyTorch上使用LoRA可以在[这里](https://github.com/microsoft/LoRA)找到.