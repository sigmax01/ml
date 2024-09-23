---
title: Transformer
comments: true
---

## 起源和发展

2017年, Google在*[Attention is All You Need](https://arxiv.org/abs/1706.03762)*中提出了Transformer结构用于序列标注, 在翻译任务中超过了之前最优秀的[递归神经网络RNN](/算法/神经网络/递归神经网络); 与此同时, Fast AI在*Universal Language Model Fint-tuning for Text Classification*中提出了一种名为ULMFiT的迁移训练方法, 将在大规模数据上预训练好的[LSTM模型](/算法/神经网络/递归神经网络#LSTM)迁移用于文本分类, 只用很少的标注数据就达到了最佳性能.

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

## 架构

我们以Encoder-Decoder模型为例, 来看Transformer的架构.

将模型视为一个黑盒子, 它会接受一段一门语言的输入, 然后转换为另一门语言的输出.

<figure markdown='1'>
![](https://img.ricolxwz.io/02a726e225f7010356c7d65bfb3ee8d5.png){ loading=lazy width='700' }
</figure>

打开这个黑箱子, 我们会发现有一个编码组件, 一个解码组件, 以及它们之间的连接.

<figure markdown='1'>
![](https://img.ricolxwz.io/c262cf206d9060f0b436de9590852701.png){ loading=lazy width='500' }
</figure>

编码组件由编码器堆叠形成, 论文中使用了$6$层编码器, 注意这里的层数不是固定的, 可以尝试编程其他的数字. 解码组件也由同样多的编码器堆叠形成.

- 编码器的主要任务是接受输入序列, 并将其转换为一个包含语义和结构信息的向量, 这表示了输入序列中的重要信息和上下文关系
- 解码器的主要任务是根据编码器的输出, 逐步生成目标输出序列, 解码器会根据已经生成的部分, 以及编码器提供的输入信息, 决定下一个词的生成

<figure markdown='1'>
![](https://img.ricolxwz.io/576e37e4cd6ed2d8923b3e274417e5e2.png){ loading=lazy width='500' }
</figure>

所有的编码器的结构都是相同的, 虽然它们的权重不一样. 每个编码器都可以分为两层.

- 第一层是自注意力层, self-attention. 当编码器处理输入句子中的一个特定单词的时候, 不仅仅单独编码这个单词, 还会去关注输入句子中的其他单词, 这样就可以理解这个单词在整个句子的上下文关系
- 第二层是[前馈神经网络](/算法/神经网络/前馈神经网络)

<figure markdown='1'>
![](https://img.ricolxwz.io/1da193b987cd1d6838e4665b4c19d548.png){ loading=lazy width='500' }
</figure>

解码器也会有这两层, 除了这两层之外, 还有一个夹在中间的编码-解码注意力层. 自注意力层用于关注解码器已经生成的部分的重要信息和上下文关系, 编码-解码注意力层用于参考/关注编码器输入序列的编码表示. 

<figure markdown='1'>
![](https://img.ricolxwz.io/1dcad850e25c516fee17a32ed76452e1.png){ loading=lazy width='600' }
</figure>

[^1]: 第二章：Transformer 模型 · Transformers快速入门. (不详). 取读于 2024年9月23日, 从 https://transformers.run/c1/transformer/#%E6%B3%A8%E6%84%8F%E5%8A%9B%E5%B1%82
[^2]: Alammar, J. (不详). The Illustrated Transformer. 取读于 2024年9月23日, 从 https://jalammar.github.io/illustrated-transformer/
[^3]: 细节拉满，全网最详细的Transformer介绍（含大量插图）！. (不详). 知乎专栏. 取读于 2024年9月23日, 从 https://zhuanlan.zhihu.com/p/681532180