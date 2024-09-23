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

## 张量

上面, 我们讲了Transformer的大致轮廓, 下面, 我们来看一下向量/张量在组件之间的流动. 

就如其他NLP模型一样, 我们最开始会使用[词嵌入算法, embedding algorithm](https://aitutor.liduos.com/02-langchain/02-3.html)将类别数据(如单词或者符号)转换为连续的数值向量. 实际中向量一般是$256$维或者$512$维, 这里为了简化起见, 将每个词表示为一个$4$维向量. 

<figure markdown='1'>
![](https://img.ricolxwz.io/226c51fe49f5d580c0554d4820df362e.png){ loading=lazy width='600' }
</figure>

这个词嵌入算法只会发生在最底部的编码器中, 相同的是所有的编码器都会收到一个由$4$维向量组成的列表. 这个列表的大小是一个超参数, 如果一个句子达不到这个长度, 那么就填充全为$0$的$4$维词向量; 如果句子超出了这个长度, 则做截断. 第一个编码器输入的是词向量的列表, 后面的编码器的输入是上一个编码器的输出, 输入的词向量列表大小和输出的词向量列表大小是相同的.

<figure markdown='1'>
![](https://img.ricolxwz.io/dbeb1331cff42a9f74fa2ff22148327f.png){ loading=lazy width='500' }
</figure>

## 编码器

前面我们提到, 编码器会接受一个词向量的列表作为输入, 它会把向量列表输入到自注意力层, 然后经过前馈神经网络层, 最后得到输出, 传入下一个编码器. 每个位置的词向量都会经过自注意力层, 得到的每个输出向量都会单独经过前馈神经网络层, 每个向量经过的前馈神经网络都是一样的.

<figure markdown='1'>
![](https://img.ricolxwz.io/eb79b0cfd8d61a555d7f654cb4022e11.png){ loading=lazy width='500' }
</figure>

[^1]: 第二章：Transformer 模型 · Transformers快速入门. (不详). 取读于 2024年9月23日, 从 https://transformers.run/c1/transformer/#%E6%B3%A8%E6%84%8F%E5%8A%9B%E5%B1%82
[^2]: Alammar, J. (不详). The Illustrated Transformer. 取读于 2024年9月23日, 从 https://jalammar.github.io/illustrated-transformer/
[^3]: 细节拉满，全网最详细的Transformer介绍（含大量插图）！. (不详). 知乎专栏. 取读于 2024年9月23日, 从 https://zhuanlan.zhihu.com/p/681532180