---
title: Transformer
comments: false
---

## 起源和发展

2017年, Google在*[Attention is All You Need](https://arxiv.org/abs/1706.03762)*中提出了Transformer结构用于序列标注, 在翻译任务中超过了之前最优秀的[递归神经网络](/algorithm/neural-network/rnn); 与此同时, Fast AI在*Universal Language Model Fint-tuning for Text Classification*中提出了一种名为ULMFiT的迁移训练方法, 将在大规模数据上预训练好的[LSTM模型](/algorithm/neural-network/递归神经网络#LSTM)迁移用于文本分类, 只用很少的标注数据就达到了最佳性能.

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
- 第二层是[前馈神经网络](/algorithm/neural-network/fnn)

<figure markdown='1'>
![](https://img.ricolxwz.io/1da193b987cd1d6838e4665b4c19d548.png){ loading=lazy width='500' }
</figure>

解码器也会有这两层, 除了这两层之外, 还有一个夹在中间的编码器-解码器注意力层. 自注意力层用于关注解码器已经生成的部分的重要信息和上下文关系, 编码器-解码器注意力层用于参考/关注编码器输入序列的编码表示. 

<figure markdown='1'>
![](https://img.ricolxwz.io/1dcad850e25c516fee17a32ed76452e1.png){ loading=lazy width='600' }
</figure>

## 张量

上面, 我们讲了Transformer的大致轮廓, 下面, 我们来看一下向量/张量在组件之间的流动. 

就如其他NLP模型一样, 我们最开始会使用[词嵌入算法, embedding algorithm](https://aitutor.liduos.com/02-langchain/02-3.html)将类别数据(如单词或者符号)转换为连续的数值向量. 实际中向量一般是$256$维或者$512$维, 这里为了简化起见, 将每个词表示为一个$4$维向量. 

<figure markdown='1'>
![](https://img.ricolxwz.io/226c51fe49f5d580c0554d4820df362e.png){ loading=lazy width='600' }
</figure>

这个词嵌入算法只会发生在最底部的编码器中, 相同的是所有的编码器都会收到一个由$4$维向量组成的列表. 这个列表的大小是一个超参数, 如果一个句子达不到这个长度, 那么就填充全为$0$的$4$维向量; 如果句子超出了这个长度, 则做截断. 第一个编码器输入的向量叫作词向量, 它的输入是词向量的一个列表, 后面的编码器的输入是上一个编码器的输出, 又叫作上下文向量, 所有向量的列表大小都是相同的. 词向量和上下文向量广义统称嵌入向量.

<figure markdown='1'>
![](https://img.ricolxwz.io/dbeb1331cff42a9f74fa2ff22148327f.png){ loading=lazy width='500' }
</figure>

## 编码器

前面我们提到, 编码器会接受一个向量的列表作为输入, 它会把向量列表输入到自注意力层, 然后经过前馈神经网络层, 最后得到输出, 传入下一个编码器. 每个位置的向量都会经过自注意力层, 得到的每个输出向量都会单独经过前馈神经网络层, 每个向量经过的前馈神经网络都是一样的.

<figure markdown='1'>
![](https://img.ricolxwz.io/eb79b0cfd8d61a555d7f654cb4022e11.png){ loading=lazy width='500' }
</figure>

## 自注意力层

别被自注意力, self attention这么高大上的词给唬住了, 但是作者在读论文*Attension is All You Need*之前就没有听说过这个词, 下面来分析一下自注意力的机制.

假设我们需要翻译的句子是: The animal didn't cross the street because it was too tired. 

这个句子中的it是一个代词, 那么it指的是什么呢? 是animal还是street? 这个问题对人来说是简单的, 但是对机器来说不是那么容易, 当处理it的时候, 自注意力机制能够让it和animal关联起来. 即当处理每一个词的时候, 自注意力机制能够查找在输入序列中其他的能够让当前词编码更优的词.

在RNN中, 处理每一个输入的时候, 会考虑前面传过来的隐藏状态. Transfommer使用的是自注意力机制, 把其他单词的理解融入处理当前的单词.

<figure markdown='1'>
![](https://img.ricolxwz.io/a103df16bceed84e7dd0dac59042db48.png){ loading=lazy width='400' }
</figure>

如上图, 当我们在第五层编码器(即最后一层编码器)编码it的时候, 有相当一部分的注意力集中在The animal上, 把这两个单词的信息融合到了it这个单词中.

下面我们来看如何使用向量来计算自注意力, 然后再看如何使用矩阵来实现自注意力.

### 计算Query, Key, Value向量 {#计算三个向量}

计算自注意力的第一步是, 对输入编码器的每个向量, 都创建三个向量, 分别是Query向量, Key向量, Value向量. 这三个向量是向量分别和三个矩阵相乘得到的, 这三个矩阵就是我们要学习的参数. 注意到这些新的向量比向量的维度更小, 如, 若编码器的输入/输出的向量的维度是$512$, 则新的三个向量的维度是$64$. 虽然在这里选用$64$, 但是这并不是必须的, 选择较小维度主要是为了优化计算效率, 尤其是在多头注意力(后面会讲)的计算过程中.

<figure markdown='1'>
![](https://img.ricolxwz.io/45dbc2a47b2cd6d2ef8ba28ef2fac164.png){ loading=lazy width='500' }
</figure>

上图中, 有两个嵌入向量$x_1$和$x_2$, $x_1$和$W^Q$权重矩阵做乘法得到Query向量$q_1$, ... 那么什么是Query, Key, Value呢? 它们本质上都是向量, 为了帮助我们更好的理解自注意力被抽象为三个名字, 往下面读, 你就会知道它们扮演什么角色.

### 计算注意力分数 {#计算注意力分数}

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

### 使用矩阵计算自注意力

第一步是计算Query, Key, Value矩阵, 首先, 我们把两个嵌入向量即Thinking和Machines放到一个矩阵$X$中, 然后分别和$3$个权重相乘, 得到Query, Key, Value矩阵, $W^Q, W^K, W^V$是我们通过训练得到的.

<figure markdown='1'>
![](https://img.ricolxwz.io/eea2dcbfa49df9fb799ef8e6997260bf.png){ loading=lazy width='300' }
</figure>

接着, 由于我们使用了矩阵计算, 我们可以把上面的第二步和第六步压缩为一步, 直接得到输出.

<figure markdown='1'>
![](https://img.ricolxwz.io/752c1c91e1b4dbca1b64f59a7e026b9b.png){ loading=lazy width='500' }
</figure>

## 多头注意力机制 {#多头注意力机制}

论文还通过增加多头注意力机制, 进一步完善了自注意力层. 注意力头是多头注意力机制中的一个子组件, 它是一个独立的注意力计算单元. 每个注意力头都有自己的Query, Key, Value矩阵. 多头注意力机制从下面的两个方面扩展了自注意力层的能力:

- 它扩展了模型关注不同位置的能力. 就如在[计算注意力分数](#计算注意力分数)中的第四步讲到的那样, 最后生成的$z_1$向量被自己Thinking主导($0.88$), 而只包含了Machines的很小一部分信息($0.12$). 多头注意力则允许模型同时学习多个不同的注意力分布. 例如, 某些注意力头可能专注于局部信息, 而其他头可能捕捉更远距离的依赖
- 多头注意力机制赋予自注意力层多个"子表示空间". 每个注意力头都有自己独立的矩阵, 会将输入映射到不同的向量空间. 这些空间可以理解为不同的"子表示空间". 不同的注意力头就像不同的"视角", 捕捉输入中不同的信息. 可以将每个注意力头理解为一个"专家", 每个专家会根据不同的规则和视角来分析输入数据, 并生成对一个输入的独特理解

<figure markdown='1'>
![](https://img.ricolxwz.io/ebef9242633eaeaa58c7ae3429b33d13.png){ loading=lazy width='600' }
</figure>

我们为每组注意力维护单独的$W^Q, W^K, W^V$权重矩阵. 将输入$X$和每组注意力$W^Q, W^K, W^V$相乘, 得到$8$组$Q, K, V$矩阵. 接着, 我们用$Q, K, V$计算每组的$Z$矩阵, 就得到$8$个$Z$矩阵$Z_0, Z_1, ..., Z_7$.

<figure markdown='1'>
![](https://img.ricolxwz.io/9a245789280ff24b8637f0ffe7f2f8a0.png){ loading=lazy width='500' }
</figure>

接下来就有点麻烦了, 因为前馈神经网络接受的是$1$个矩阵(每个词的一个向量), 所以我们需要有一种方法把$8$个矩阵整合为一个矩阵. 怎么才能做到呢? 

1. 把$8$个矩阵拼接起来
2. 把拼接后得到的矩阵和$W^O$权重矩阵相乘, 这个$W^O$是随着模型一起训练的
3. 得到最终的矩阵$Z$, 这个矩阵包含了所有注意力头的信息, 输入到前馈神经网络

<figure markdown='1'>
![](https://img.ricolxwz.io/9a721b7e3b77140f0a51e6cb38117209.png){ loading=lazy width='600' }
</figure>

这就是多头注意力机制的全部内容, 下面是一张汇总图.

<figure markdown='1'>
![](https://img.ricolxwz.io/3cd76d3e0d8a20d87dfa586b56cc1ad3.png){ loading=lazy width='600' }
</figure>

既然我们已经谈到了多头注意力, 现在让我们重新回顾一下之前的翻译例子, 看下当我们编码单词it的时候, 不同的注意力头关注的是什么部分. 例如, 下图中含有$2$个注意力头, 其中的一个注意力头最关注的是"The animal", 另外一个注意力头最关注的是$tired$, 因此"it"在最后的输出中融合了"animal"和"tired".

<figure markdown='1'>
![](https://img.ricolxwz.io/6cfe032799b48017bbb21103a0cc4892.png){ loading=lazy width='400' }
</figure>

如果我们把所有的注意力头都在图上画出来, 会变成这个样子.

<figure markdown='1'>
![](https://img.ricolxwz.io/9cd4154bc491304fb8b0518cff1b872c.png){ loading=lazy width='400' }
</figure>

## 词嵌入[^4]

在真正把数据输入到第一层编码层之前, 需要对文本做一定的处理. 其中第一步是将非结构化的信息转化为结构化的信息, 这一个步骤叫做"文本表示", 主要有3种方式: 独热编码, 整数编码, 词嵌入.

 其中, 独热编码无法表示词语之间的联系, 而且这种过于稀疏的向量, 会导致计算和存储的效率不高. 整数编码是用一种数字来表示一个词, 如猫用1表示, 狗用2表示, 牛用3表示, 这种方法也无法表示词语之间的关系, 而且, 不利于模型的解释. 

词嵌入, word embedding跟独热编码和整数编码的目的一样, 不过他有更多的优点. 词嵌入并不是指某个具体的算法, 跟以上2种方式相比, 这种方式有几个明显的优势, 它可以降文本通过一个低微向量表达, 但是不像独热编码那么长. 寓意相似的词在向量空间上有比较相近; 通用性也比较强, 可以用在不同的任务中.                       

<figure markdown='1'>
![](https://img.ricolxwz.io/67d533bc3180e699d01468936c6acd7c.webp){ loading=lazy width='500' }
</figure>

2种主流的词嵌入算法有Word2vec和GloVe. 

### Word2vec

这是一种基于统计方法来获得词向量的方法, 他是2013年由谷歌的Mikolov提出的一种新的词嵌入方法. 在2018年之前比较主流, 但是随着BERT, GPT2.0的出现, 这种方式已经不算效果最好的方法了.

Word2vec的2种训练模式有两种, CBOW和Skip-gram. CBOW通过上下文来预测当前值, 相当于一句话中扣掉了一个词, 让你猜这个词是什么. Skip-gram用当前此来预测上下文. 相当于给你一个词, 让你猜前面和后面可能出现什么词.

CBOW的模型通常包括以下的步骤:

1. 输入表示, 输入是一组上下文单词, 通常是独热编码之后的结果. 假设词汇表的大小是$V$, 上下文窗口大小为$c$, 即每个上下文的独热向量大小为$V$, 输入矩阵的形状是$c\times V$
2. 嵌入层, 使用嵌入矩阵$W$将高维稀疏的独热向量映射为低维密集向量. 嵌入矩阵的大小为$V\times d$, 其中$d$是嵌入维度, 对于每个上下文单词, 它的嵌入结果为$\bm{e_i}=\bm{W}^T\cdot x_i$, 其中, $\bm{x_i}$是第$i$个上下文单词的独热向量
3. 平均上下文向量: 将所有上下文单词的嵌入向量取平均, $\bm{h}=\frac{1}{c}\sum_{i=1}^c\bm{e_i}$, 这里的$\bm{h}$是一个$d$维的向量, 作为中心单词的语义表示
4. 输出层: 使用一个线性变换和Softmax激活函数将$\bm{h}$转换为概率分布$p(w_{target}|w_{context})=Softmax(\bm{U}\cdot \bm{h})$, 其中, $\bm{U}$是权重矩阵, 大小为$d\times V$, 输出是对词汇表中每个单词的概率预测

## 位置编码

位置编码是为了让模型理解输入序列中元素的顺序信息而设计的. 在Transformer结构中, 自注意力机制本身是无序的, 为了解决这个问题, 位置编码被引入, 将位置信息注入到自注意力层的输入数据.

为了让模型理解单词的顺序, 论文引入了一个带有位置编码的向量, 这种做法背后的直觉是: 将这些表示位置的向量添加到嵌入向量中, 会在它们运算$Q, K, V$向量和点击的时候提供有用的距离信息.

<figure markdown='1'>
![](https://img.ricolxwz.io/32993fd5f1b712dc93db93830d5900ec.png){ loading=lazy width='600' }
</figure>

如果我们假设嵌入向量的维度是$4$, 那么位置编码向量可能如下所示.

<figure markdown='1'>
![](https://img.ricolxwz.io/d1ad72abf15e81966a448248e7d4c8b7.png){ loading=lazy width='600' }
</figure>

那么这些位置编码向量到底遵循什么模式呢?

在下图中, 每一行代表一个位置编码向量. 第一行对应于序列中第一个单词的位置编码向量. 每一行都包含$512$个值, 每个值的范围在$-1$到$1$之间. 对这些向量进行可视化处理, 可以看到向量遵循的模式.

<figure markdown='1'>
![](https://img.ricolxwz.io/afad13e06cc0454f7d4a3ddafb6ccf32.png){ loading=lazy width='500' }
</figure>

这是一个真实的例子, 包含了$20$个单词, 每个嵌入向量的维度是$512$, 你可以看到, 它看起来像是从中间一分为二. 这是因为左半部分的值是由正弦函数产生的, 右半部分的值是由余弦函数产生的, 然后将它们拼接起来, 得到每个位置编码向量. 论文中的位置编码模式和上述略有不同, 它不是直接拼接两个向量, 而是将两个向量交织在一起, 如下图所示.

<figure markdown='1'>
![](https://img.ricolxwz.io/c35ed6b9a6405146557671f2819881d9.png){ loading=lazy width='500' }
</figure>

## 残差连接

残差连接, Residual Connection的本质是将输入直接跳过某一层操作, 并与该层的输出相加, 再进行后续处理. 残差最初是ResNet引入的, 主要目的是解决深层神经网络中的梯度消失和梯度爆炸问题. 

在编码器的每一个子层周围, 都会有一个围绕它的残差连接还有一个层标准化.

<figure markdown='1'>
![](https://img.ricolxwz.io/01883e4cf179997b19a95c3826c83215.png){ loading=lazy width='400' }
</figure>

将Add&Normalize层可视化, 可以得到.

<figure markdown='1'>
![](https://img.ricolxwz.io/b831c7d0981cae1f9e0127c27d1e5391.png){ loading=lazy width='400' }
</figure>

可以看到, 在自注意力中, 输入的嵌入向量$x_1$和$x_2$经过自注意力机制处理后, 生成新的表示$z_1$和$z_2$, 残差连接会将输入的$x_1$和$x_2$直接与自注意力的输出$z_1$和$z_2$相加. 然后, 经过层归一化, 这一过程的输出将进一步被传递给前馈神经网络.

解码器的子层里面也有层标准化, 假设Transformer是由$2$层编码器和$2$层解码器组成的, 如下图所示.

<figure markdown='1'>
![](https://img.ricolxwz.io/406921881ee31e9f56f9d7300f41f57e.png){ loading=lazy width='600' }
</figure>

## 解码器

我们已经了解了解码器中的大部分概念, 现在我们来看一下, 编码器和解码器是如何协同工作的.

编码器的输出会进一步被处理为Key矩阵和Value注意力矩阵, 这些向量将在解码器的编码器-解码器注意力层中使用, 帮助解码器在生成输出的时候关注输入序列的相关部分.

<figure markdown='1'>
![](https://img.ricolxwz.io/0973bef4fa0892557b1049e436f097e7.gif){ loading=lazy width='600' }
</figure>

解码阶段的每个时间步都会输出一个翻译后的单词. 重复这个过程, 直到输出一个结束符, 就完成了所有输出. 解码器每一步的输出都会在下一个时间步输入到第一个解码器. 正如对编码器的输入所做的处理, 我们把解码器的输入向量, 也加上位置编码向量, 来表示每一个词的位置.

<figure markdown='1'>
![](https://img.ricolxwz.io/dce969fd736d3fc3535cc1222bceab2d.gif){ loading=lazy width='600' }
</figure>

解码器中的自注意力层和编码器中的自注意力层不太一样, 在解码器里面, 自注意力层只允许关注输出序列中早于当前位置的单词(后面的单词还没有生成...), 具体的做法是, 在自注意力层的分数经过Softmax层(注意这里的Softmax不是指最终的解码器经过的Softmax层, 是在计算自注意力的时候的Softmax层, 详情见[这里](#计算三个向量))之前, 屏蔽当前位置之后的那些位置.

编码器-解码器注意力层的原理和[多头注意力机制](#多头注意力机制)类似, 不同之处是, 编码器-解码器注意力层使用的是前一层解码器的输出来构造Query矩阵, 而Key矩阵和Value矩阵来自于解码器的最终输出.

## 线性层和Softmax层

解码器在每个时间步的输出是一个向量, 其中的每个元素都是浮点数, 那么我们怎么把这个向量转化为单词呢? 这就是由线性层和后面的Softmax层来实现的.

线性层是一个普通的全连接神经网络, 可以把解码器输出的向量映射到一个更长的的向量, 这个向量被称为logits向量. 假设我们的模型认识$10000$个唯一的英文单词, 那么logits向量的维度就是$10000$, 每个数表示一个单词的分数. 

然后, Softmax层会把这些分数转换为概率, 把所的分数转换为正数, 并且加起来等于$1$. 然后选择最高概率的那个数字对应的词, 就是这个时间步的输出单词.

<figure markdown='1'>
![](https://img.ricolxwz.io/97652550f350209b238757b1f9660497.png){ loading=lazy width='500' }
</figure>

## 训练过程

在上面, 我们了解的是一个已经训练好的Transformer的前向传播过程. 下面会讲讲是怎么训练的. 

在训练的过程中, 模型会经过上面讲的所有前向传播的步骤. 不同的是, 因为我们是在有标签的数据集上训练, 所以可以比较模型的输出和真实的标签.

为了可视化, 我们假定输出词汇表只包含$6$个单词: "a", "am", "i", "thanks", "student"和"<eos\>". "<eos\>"表示句子末尾. 注意, 这个输出词汇表是在训练之前的数据预处理阶段就构造好的. 

<figure markdown='1'>
![](https://img.ricolxwz.io/16f981983e1247ef0eec0459e97b737e.png){ loading=lazy width='500' }
</figure>

构造好输出词汇表后, 我们就可以使用One-Hot编码使用相同长度的向量来表示词汇表中的一个词. 例如, 我们可以把单词"am"用下面的向量来表示.

<figure markdown='1'>
![](https://img.ricolxwz.io/085330e629dc9f7d6d5e49d5f9acec9b.png){ loading=lazy width='500' }
</figure>

## 损失函数

假设我们正在训练模型, 并且是训练周期的第一步, 目标是把"merci"翻译为"thanks". 这意味着我们希望模型的最终输出的概率分布, 会指向"thanks", 表示这个词的可能性最高, 但是鉴于我们的模型还没有训练好, 它输出的概率分布可能和我们希望的概率分布相差甚远.

<figure markdown='1'>
![](https://img.ricolxwz.io/74bf8db4543d6187960ee3cea18e1703.png){ loading=lazy width='500' }
</figure>

由于模型的参数都是随机初始化的. 第一步模型在每个词输出的概率都是随机的. 我们可以把这个概率和正确的概率做对比, 然后使用反向传播来调整模型的权重, 使得输出的概率分布更加接近真实输出. 

那么要怎么比较两个概率分布, 我们可以借助信息论的工具, 详情见[交叉熵](https://gk.ricolxwz.de/information-theory/what-is-information/#交叉熵)和[KL散度](https://gk.ricolxwz.de/information-theory/what-is-information/#KL散度).

但是注意我们这是一个过度简化的例子. 现实来说, 我们会翻译一个句子, 而不是一个词. 例如, 输入是"je suis étudiant", 期望输出是"i am a student". 这意味着, 我们模型需要输出多个概率分布, 满足如下条件:

- 每个概率分布都是一个向量, 长度是词汇表的大小(在我们的例子中是$6$, 实际中是$30000$, ...)
- 第一个概率分布中, 最高概率对应的单词是"i"
- 第二个概率分布中, 最高概率对应的单词是"am"
- 依此类推, 直到第$5$个概率分布中, 最高概率对应的单词是"<eos\>"表示没有单词了

<figure markdown='1'>
![](https://img.ricolxwz.io/cb2581e2f1b3f673f0e1b53e2c100a26.png){ loading=lazy width='500' }
</figure>

我们使用例子中的句子训练模型, 希望产生如上图所示的概率分布(最佳, 理想状态, 实际上很难达到), 我们的模型在一个足够大的数据集上, 经过长时间的训练后, 可能产生的概率分布如下图所示.

<figure markdown='1'>
![](https://img.ricolxwz.io/3c8ce3f741d432fcadf75c93d13a20a5.png){ loading=lazy width='500' }
</figure>

在测试时, 如果你要翻译的句子是训练集中的一部分, 那输出的结果不能说明什么. 我们希望的是模型在没见过的句子上也能给出准确地翻译. 注意, 概率分布向量中, 每个位置都会有一点概率, 即使这个位置不是输出对应的单词, 这是Softmax中一个很有用的特性, 有助于训练过程. 

这种方法叫作贪心解码, greedy decoding, 在每个时间步, 模型会选择当前概率最高的单词作为输出, 这种优点是速度快, 缺点是只选择当前看起来最优的单词, 可能会错误全局更好的解决方案.

还有一种方法是集束搜索, 与贪心解码不同, 集束搜索会保留多个单词. 假设bean size是$2$, 这意味着模型在每一步中保留两个概率最高的单词作为候选项. 例如, 生成第一个单词的时候, 它可能会选择"I"和"a", 然后在下一个时间步中, 分别基于"I"和"a"生成后续的单词, 并继续计算这些路径的总得分, 模型会根据总和得分决定哪条路径最优. 这种方法可以避免贪心解码的局部最优解问题, 因为它在每一步都保留了多条候选路径, 允许模型在更大范围内搜索可能的最优解.

[^1]: 第二章：Transformer 模型 · Transformers快速入门. (不详). 取读于 2024年9月23日, 从 https://transformers.run/c1/transformer/#%E6%B3%A8%E6%84%8F%E5%8A%9B%E5%B1%82
[^2]: Alammar, J. (不详). The Illustrated Transformer. 取读于 2024年9月23日, 从 https://jalammar.github.io/illustrated-transformer/
[^3]: 细节拉满，全网最详细的Transformer介绍（含大量插图）！. (不详). 知乎专栏. 取读于 2024年9月23日, 从 https://zhuanlan.zhihu.com/p/681532180
[^4]: easyAI-人工智能知识库. (2020, 二月 18). 一文看懂词嵌入word embedding（2种算法+其他文本表示比较）. Medium. https://easyaitech.medium.com/%E4%B8%80%E6%96%87%E7%9C%8B%E6%87%82%E8%AF%8D%E5%B5%8C%E5%85%A5word-embedding-2%E7%A7%8D%E7%AE%97%E6%B3%95-%E5%85%B6%E4%BB%96%E6%96%87%E6%9C%AC%E8%A1%A8%E7%A4%BA%E6%AF%94%E8%BE%83-c7dd8e4524db