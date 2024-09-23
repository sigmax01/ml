---
title: Transformer
comments: true
---

## 起源和发展

2017年, Google在*Attention is All You Need*中提出了Transformer结构用于序列标注, 在翻译任务中超过了之前最优秀的[递归神经网络RNN](/算法/神经网络/递归神经网络); 与此同时, Fast AI在*Universal Language Model Fint-tuning for Text Classification*中提出了一种名为ULMFiT的迁移训练方法, 将在大规模数据上预训练好的[LSTM模型](/算法/神经网络/递归神经网络#LSTM)迁移用于文本分类, 只用很少的标注数据就达到了最佳性能.

这些开创性的工作促成了两个著名的Transformer模型的出现:

- [GPT](https://ai.com) (the Generative Pretrained Transformer)
- [BERT](https://github.com/google-research/bert) (Bidirectional Encoder Representations from Transformers)

通过将Transformer结构和无监督学习相结合, 我们不再需要对每一个任务都从头开始训练模型, 并且几乎在所有NLP任务上都远远超过先前的最强基准. 

GPT和BERT被提出后, NLP领域出现了越来越多基于Transformer结构的模型, 其中比较有名的有:

<figure markdown='1'>
![](https://img.ricolxwz.io/aeac87fbc55d405f507b73b96ac912e4.png){ loading=lazy width='600' }
</figure>

虽然新的Transformer模型层出不穷, 它们采用不同的预训练目标, 在不同的数据集上进行训练, 但是依然可以按照模型结构将它们大致分为三类:

- 纯Encoder模型, 例如BERT, 又称为自编码(auto-encoding)Transformer模型
- 纯Decoder模型, 例如GPT, 又称为自回归(auto-regressive)Transformer模型
- Encoder-Decoder模型, 例如BART, T5, 又称Seq2Seq(sequence-to-sequence)Transformer模型

[^1]: 第二章：Transformer 模型 · Transformers快速入门. (不详). 取读于 2024年9月23日, 从 https://transformers.run/c1/transformer/#%E6%B3%A8%E6%84%8F%E5%8A%9B%E5%B1%82
