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

### 张量

上面, 我们讲了Transformer的大致轮廓, 下面, 我们来看一下向量/张量在组件之间的流动. 

就如其他NLP模型一样, 我们最开始会使用[词嵌入算法, embedding algorithm](https://aitutor.liduos.com/02-langchain/02-3.html)将类别数据(如单词或者符号)转换为连续的数值向量. 实际中向量一般是$256$维或者$512$维, 这里为了简化起见, 将每个词表示为一个$4$维向量. 

<figure markdown='1'>
![](https://img.ricolxwz.io/226c51fe49f5d580c0554d4820df362e.png){ loading=lazy width='600' }
</figure>

这个词嵌入算法只会发生在最底部的编码器中, 相同的是所有的编码器都会收到一个由$4$维向量组成的列表. 这个列表的大小是一个超参数, 如果一个句子达不到这个长度, 那么就填充全为$0$的$4$维向量; 如果句子超出了这个长度, 则做截断. 第一个编码器输入的向量叫作词向量, 它的输入是词向量的一个列表, 后面的编码器的输入是上一个编码器的输出, 又叫作上下文向量, 所有向量的列表大小都是相同的. 词向量和上下文向量广义统称嵌入向量.

<figure markdown='1'>
![](https://img.ricolxwz.io/dbeb1331cff42a9f74fa2ff22148327f.png){ loading=lazy width='500' }
</figure>

### 编码器

前面我们提到, 编码器会接受一个向量的列表作为输入, 它会把向量列表输入到自注意力层, 然后经过前馈神经网络层, 最后得到输出, 传入下一个编码器. 每个位置的向量都会经过自注意力层, 得到的每个输出向量都会单独经过前馈神经网络层, 每个向量经过的前馈神经网络都是一样的.

<figure markdown='1'>
![](https://img.ricolxwz.io/eb79b0cfd8d61a555d7f654cb4022e11.png){ loading=lazy width='500' }
</figure>

### 自注意力层

别被自注意力, self attention这么高大上的词给唬住了, 但是作者在读论文*Attension is All You Need*之前就没有听说过这个词, 下面来分析一下自注意力的机制.

假设我们需要翻译的句子是: The animal didn't cross the street because it was too tired. 

这个句子中的it是一个代词, 那么it指的是什么呢? 是animal还是street? 这个问题对人来说是简单的, 但是对机器来说不是那么容易, 当处理it的时候, 自注意力机制能够让it和animal关联起来. 即当处理每一个词的时候, 自注意力机制能够查找在输入序列中其他的能够让当前词编码更优的词.

在RNN中, 处理每一个输入的时候, 会考虑前面传过来的隐藏状态. Transfommer使用的是自注意力机制, 把其他单词的理解融入处理当前的单词.

<figure markdown='1'>
![](https://img.ricolxwz.io/a103df16bceed84e7dd0dac59042db48.png){ loading=lazy width='400' }
</figure>

如上图, 当我们在第五层编码器(即最后一层编码器)编码it的时候, 有相当一部分的注意力集中在The animal上, 把这两个单词的信息融合到了it这个单词中.

下面我们来看如何使用向量来计算自注意力, 然后再看如何使用矩阵来实现自注意力.

#### 计算Query, Key, Value向量

计算自注意力的第一步是, 对输入编码器的每个向量, 都创建三个向量, 分别是Query向量, Key向量, Value向量. 这三个向量是向量分别和三个矩阵相乘得到的, 这三个矩阵就是我们要学习的参数. 注意到这些新的向量比向量的维度更小, 如, 若编码器的输入/输出的向量的维度是$512$, 则新的三个向量的维度是$64$. 虽然在这里选用$64$, 但是这并不是必须的, 选择较小维度主要是为了优化计算效率, 尤其是在多头注意力(后面会讲)的计算过程中.

<figure markdown='1'>
![](https://img.ricolxwz.io/45dbc2a47b2cd6d2ef8ba28ef2fac164.png){ loading=lazy width='500' }
</figure>

上图中, 有两个嵌入向量$x_1$和$x_2$, $x_1$和$W^Q$权重矩阵做乘法得到Query向量$q_1$, ... 那么什么是Query, Key, Value呢? 它们本质上都是向量, 为了帮助我们更好的理解自注意力被抽象为三个名字, 往下面读, 你就会知道它们扮演什么角色.

#### 计算注意力分数

第二步是计算注意力分数, Attention Score. 假设我们现在计算第一个词Thinking的注意力分数, 即需要根据Thinking这个词, 对于句子中的其他位置的每个词放置多少的注意力. 

这些分数, 是通过计算Thinking对应的Query向量和其他位置每个词的Key向量的点积得到的. 如果我们计算句子中第一个位置单词的Attension Score, 那么第一个分数就是$q_1$和$k_1$的点积, 第二个分数就是$q_1$和$k_2$的点积.

<figure markdown='1'>
![](https://img.ricolxwz.io/f64cbdcf1d883ede36b26067e34f4e3e.png){ loading=lazy width='500' }
</figure>

第三步是把每个分数除以$8$(论文中Key向量的维度$64$开方得到的), 这一步是为了得到更稳定的梯度.

第四步是把这些分数送到Softmax函数, 可以将这些分数归一化, 使得所有的分数加起来等于$1$.

<figure markdown='1'>
![](https://img.ricolxwz.io/03d0a60b60a0a28f52ed903c76bb9a22.png){ loading=lazy width='500' }
<!-- <figcaption>第四步</figcaption> -->
</figure>

这些分数决定了编码当前位置的词, 即Thinking的时候, 对所有位置的词分别有多少的注意力. 很明显, 在上图的例子中, 当前位置的词Thinking对自己有最高的注意力$0.88$, 但有时, 关注其他位置上的词也很有用.

第五步是得到每个位置的分数后, 将每个分数和每个Value向量相乘, 这种做法背后的直觉理解是: 对于分数高的位置, 相乘后的值就越大, 我们把更多的注意力放到了它们的身上; 对于分数低的位置, 相乘后的值就越小, 这些位置的词可能相关性是不大的, 这样我们就忽略了这些位置的词.

第六步是把上一步得到的向量相加, 就得到了自注意力层在这个位置即Thinking的输出. 

<figure markdown='1'>
![](https://img.ricolxwz.io/087b831f622f83e4529c1bbf646530f0.png){ loading=lazy width='500' }
</figure>

上面的这张图囊括了计算自注意力计算的全过程, 最终得到的向量, 当前例子是Thinking最后的向量会输入到前馈神经网络. 但是, 这样每次只能计算一个位置的输出, 在实际的代码实现中, 自注意的计算是通过矩阵实现的, 这样可以加速计算, 一次就得到了所有位置的输出向量.

[^1]: 第二章：Transformer 模型 · Transformers快速入门. (不详). 取读于 2024年9月23日, 从 https://transformers.run/c1/transformer/#%E6%B3%A8%E6%84%8F%E5%8A%9B%E5%B1%82
[^2]: Alammar, J. (不详). The Illustrated Transformer. 取读于 2024年9月23日, 从 https://jalammar.github.io/illustrated-transformer/
[^3]: 细节拉满，全网最详细的Transformer介绍（含大量插图）！. (不详). 知乎专栏. 取读于 2024年9月23日, 从 https://zhuanlan.zhihu.com/p/681532180