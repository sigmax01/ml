---
title: Transformer
comments: true
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

## 动机

### RNN等模型的缺陷

RNN模型无法并行执行, 在计算第t的词的时候, 前面的t-1个词必须全部运算完成. 如果时序比较长的话, 可能会导致前面的信息到后面就丢失掉了. 如果不想丢掉的话, 就需要做一个比较大的ht, 这会导致很高昂的内存开销.

一开始的时候, 这些注意力机制可能都是和RNN结合起来使用的, 而没有成为一个独立的体系.

???+ tip "端到端记忆网络"

    “端到端”指的是一种系统设计方法, 意味着从输入到输出的整个过程由一个单一的系统或者模型直接完成, 通常不需要人工干预或者多个独立的模块.

    端到端记忆网络是主要用于处理需要长期依赖记忆的任务. 它在功能上和LSTM类似, 但是在结构或者说工作方式上有显著不同. 前者引入了一个独立的外部记忆模块, 该模块是一个可供模型读取和写入的内存池, 模型通过注意力机制与这个外部记忆池的交互来保存和获取信息, 从而支持长期依赖. 而LSTM本身是通过门机制来管理和控制记忆的, 它的记忆是隐式的, 即记忆是通过递归传递的内部状态(cell state)来存储的, 而不是通过外部记忆池进行显式存储.

### CNN等模型的缺陷

使用CNN的时候, 每一次它去看的是一个比较小的窗口, 如3\*3的卷积核. 如果两个像素隔的比较远的时候, 需要较多的层才能把这两个像素融合起来.

但是Transformer模型通过注意力机制每一次能够看到所有的像素, 在一层中就能够看到. CNN比较好的地方是它有很多个输出通道, 每个通道可以去识别不同的模式. 所以说它提出了一个多头注意力机制, 模拟CNN的多输出通道的效果.

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

在RNN等模型的缺陷中, 处理每一个输入的时候, 会考虑前面传过来的隐藏状态. Transfommer使用的是自注意力机制, 把其他单词的理解融入处理当前的单词.

<figure markdown='1'>
![](https://img.ricolxwz.io/a103df16bceed84e7dd0dac59042db48.png){ loading=lazy width='400' }
</figure>

如上图, 当我们在第五层编码器(即最后一层编码器)编码it的时候, 有相当一部分的注意力集中在The animal上, 把这两个单词的信息融合到了it这个单词中.

下面我们来看如何使用向量来计算自注意力, 然后再看如何使用矩阵来实现自注意力.

### 计算Query, Key, Value向量 {#计算三个向量}

计算自注意力的第一步是, 对输入编码器的每个向量, 都创建三个向量, 分别是Query向量, Key向量, Value向量. 这三个向量是向量分别和三个矩阵相乘得到的, 这三个矩阵就是我们要学习的参数. 注意到这些新的向量比向量的维度更小, 如, 若编码器的输入/输出的向量的维度是$512$, 则新的三个向量的维度是$64$. 虽然在这里选用$64$, 但是这并不是必须的, 选择较小维度主要是为了优化计算效率以及提取多种特征, 尤其是在多头注意力(后面会讲)的计算过程中.

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

???+ tip "累加/点积注意力计算方法"

    最常用的两种注意力函数是累加和点积, 累加是将Query和Key组合起来, 通过一个小型的神经网络生成注意力分数. 它和点积的区别主要在于计算方式的非线性化. 但是由于点积在计算上的性能远远大于累加, 而且在实际中往往更加节省内存空间, 所以这篇文章用的是点积.

???+ note "为什么要除以sqrt(dk)"

    当dk比较大的时候, QK^T点积过程中需要累加的项就特别多, 导致最终的attention分数的数值过大. 累积dk次后, 结果的均值和方差都会成比例增大, 由于点积结果的方差和dk是成正比的, 所以通过除以sqrt(dk), 可以使得结果的方差保持常数, 这样, 无论dk的大小如何, 点积结果的数值范围都会被规范到一个固定的范围内.

### 使用矩阵计算自注意力

第一步是计算Query, Key, Value矩阵, 首先, 我们把两个嵌入向量即Thinking和Machines放到一个矩阵$X$中, 然后分别和$3$个权重相乘, 得到Query, Key, Value矩阵, $W^Q, W^K, W^V$是我们通过训练得到的.

<figure markdown='1'>
![](https://img.ricolxwz.io/eea2dcbfa49df9fb799ef8e6997260bf.png){ loading=lazy width='300' }
</figure>

接着, 由于我们使用了矩阵计算, 我们可以把上面的第二步和第六步压缩为一步, 直接得到输出.

<figure markdown='1'>
![](https://img.ricolxwz.io/752c1c91e1b4dbca1b64f59a7e026b9b.png){ loading=lazy width='500' }
</figure>

???+ note "GPU并行计算"

    在实际训练中, 通常是`batch_size`个序列直接输入, 每个序列都对应自己的`Q, K, V`矩阵, 所以可以利用GPU等并行设备, 同时对batch中的所有序列进行计算, 从而显著提高计算速度. 如果`batch_size`为1, 就无法利用GPU的序列间并行计算能力, 导致处理速度较慢(但是内部计算`QK^T`矩阵乘法还是并行的).  

    所以, GPU的并行计算体现在两个方面:

    1. **Batch内序列间的并行计算**
    2. **Q, K, V序列内进行矩阵乘法操作的并行计算**

???+ note "注意力的范围"

    在标准的Transformer模型中, 不同的序列(即同一batch中的不同样本)之间是没有注意力的, 注意力仅仅局限在单个序列内部, 模型不会跨序列进行注意力汇聚.

???+ note "如何计算attention层的复杂度"

    设$N$是序列长度, $D$是特征维度, $B$是batch_size.

    Q和K的维度都是$N\cdot D$. 两个矩阵$m\cdot n$和$n\cdot p$相乘的复杂度是$O(m\cdot p\cdot n)$. 所以QK^T的复杂度是$O(N^2\cdot D)$. Softmax处理QK^T结果的复杂度是$O(N^2)$, 将softmax结果和V进行矩阵乘法, V的维度是$N\cdot D$, 继续运用矩阵乘法的复杂度计算公式, 复杂度是$O(N^2\cdot D)$. 总共有$B$个序列, 所以结果为$O(B\cdot N^2\cdot D)$.

???+ note "各个层的输入/输出神经元数量"

    每个层的输入/输出神经元数量都是$d_{model}$, 一般为512. 对于GPT-3, 是12888. 这是为了确保skip connection中输入和输出维度相等. 

## 多头注意力机制 {#多头注意力机制}

???+ note "深入理解MSA"

    在这条补丁下面的可能是比较粗浅的理解(而且可能和PyTorch的实现有点出入), 这条补丁是较深入的理解:

    1. **输入:** 

        输入`x`的形状为`(batch_size, seq_len, hidden_size)`

    2. **线性投影生成`Q`, `K`, `V`:** 

        通过三个独立的线性层, 进行线性变换, 即通过`W_K`, `W_Q`, `W_V`, 分别生成`Q, K, V`, 形状均为`(batch_size, seq_len, hidden_size)`

    3. **拆分为多个头:** 

        1. 使用`view`或者`reshape`将`Q`, `K`, `V`的形状从`(batch_size, seq_len, hidden_size)`转换为`(batch_size, seq_len, num_heads, hidden_size/num_heads)`
        2. 使用`transpose`将维度顺序调整为`(batch_size, num_heads, seq_len, hidden_size/num_heads)`

    4. **计算注意力得分:**

        1. 对每个头独立计算注意力得分, 得分的形状为`(batch_size, num_heads, seq_len, seq_len)`

        2. 通过`softmax`函数对得分进行归一化, 得到注意力权重

    5. **计算注意力输出:**
    
        使用注意力权重和每个头独立的`V`进行乘法, 得到注意力输出, 形状为`(batch_size, num_heads, seq_len, hidden_size/num_heads)`

    6. **拼接所有头的输出:**
    
        1. 使用`transpose`将注意力输出的维度顺序调整回`(batch_size, seq_len, num_heads, hidden_size/num_heads)`
        2. 使用`view`或者`reshape`将tensor拼接回原始的嵌入维度, 形状为`(batch_size, seq_len, hidden_size)`

    7. **最终线性层:**
    
        然后通过一个线性层对拼接后的注意力输出进行最终转换, 得到模型的输出. 这个线性层的作用是对输出进行整合, 增强模型的表达能力.

    虽然在原文中写了每个头有自己独立的权重矩阵`W^i_Q`, `W^i_K`, `W^i_V`, 但是在实际的实现中, 通常采用上述这种共享大的`W_Q`, `W_K`, `W_V`, 切分`Q`, `K`, `V`的做法. 

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

???+ note "拼接之后的维度需要满足残差连接的条件"

    由于有残差连接的存在, 输入的维度必须是和输出的维度是一样的, 所以, 选择的dk和dv是原始维度/h. 然后拼接的时候就可以回到原来的维度.

<figure markdown='1'>
![](https://img.ricolxwz.io/9a721b7e3b77140f0a51e6cb38117209.png){ loading=lazy width='600' }
</figure>

这就是多头注意力机制的全部内容, 下面是一张汇总图.

<figure markdown='1'>
![](https://img.ricolxwz.io/3cd76d3e0d8a20d87dfa586b56cc1ad3.png){ loading=lazy width='600' }
</figure>

总结来说, 我们将高维的Q, K和V投影到多个低维子空间(每个子空间对应一个头), 在这些低维子空间中分别计算注意力, 最终将各个头的结果拼接起来, 再进行一次线性变换, 得到输出.

既然我们已经谈到了多头注意力, 现在让我们重新回顾一下之前的翻译例子, 看下当我们编码单词it的时候, 不同的注意力头关注的是什么部分. 例如, 下图中含有$2$个注意力头, 其中的一个注意力头最关注的是"The animal", 另外一个注意力头最关注的是$tired$, 因此"it"在最后的输出中融合了"animal"和"tired".

<figure markdown='1'>
![](https://img.ricolxwz.io/6cfe032799b48017bbb21103a0cc4892.png){ loading=lazy width='400' }
</figure>

如果我们把所有的注意力头都在图上画出来, 会变成这个样子.

<figure markdown='1'>
![](https://img.ricolxwz.io/9cd4154bc491304fb8b0518cff1b872c.png){ loading=lazy width='400' }
</figure>

## 词嵌入

请见[这里](/algorithm/neural-network/word-embedding).

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

关于残差连接, 可以参考[ResNet](/algorithm/neural-network/cnn/resnet).

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

## 掩码

为什么需要掩码呢? 这个可以从两个方面来说, 训练阶段和推理阶段.

- **训练阶段:** 在训练的时候, decoder输入的是完整的目标序列(包括未来词), 而不是逐步生成的部分序列. 目标序列被送入自注意力层之后, 如果不加mask, 模型在计算第t个词的注意力的时候, 会看到整个目标序列(包括t+1, t+2, …), 导致模型可以利用未来的词生成当前的词, 这种信息泄露会导致训练过程中模型无法学习到正确的因果关系, 从而在推理阶段表现不加

- **推理阶段:** 在推理的过程中, decoder的输入是逐步生成的序列, 在t时刻, decoder的输入是从第1到第t-1时刻生成的词, 理论上, 此时未来的词(第t+1, t+2, …)根本不存在, 似乎不需要mask. 但是由于自注意力机制的实现是对于整个序列(包括还未填充的位置)计算注意力分布, 如果不加mask, decoder的自注意力层仍会尝试对后续未生成的位置(这些位置可能被初始化为零向量或者其他占位符)来计算注意力分布, 即使这些未生成(未填充)的位置没有真实的信息, 注意力分布的结果可能会受到干扰

具体的做法是将后面的值替换成一个非常大的负数. 注意, 不能替换为0, 不然的话经过softmax之后其他地方的值会受到影响(变小). 负数经过softmax之后就会变成0.

## 归一化

归一化的方式主要分为两种, BN和LN:

- **批归一化(BatchNorm):** 是对每个batch内的样本的每个特征进行归一化, 具体来说, 它会计算整个batch中的每个特征的均值和方差, 然后基于这些统计量对所有的样本进行归一化. 因此, BN的计算是跨样本的, 即竖着计算. 注意, 学习和预测时候的BN是不一样的, 学习的时候是这个batch里面的特征取均值和方差; 预测的时候是整个样本集的特征取均值和方差

- **层归一化(LayerNorm):** 是对每个样本的所有特征进行归一化. 它计算每个样本的所有特征的均值和方差, 因此是横着归一化的

所以, BN抹平了不同特征之间的大小关系, 而保留了不同样本之间的大小关系. 这样, 如果具体任务依赖于不同样本之间的联系, BN更有效, 尤其是在CV领域, 不同图片样本之间的大小关系得以保留. LN抹平了不同样本之间的大小关系, 而保留了不同特征之间的大小关系. 所以, LN更加适合NLP任务, 一个样本实际上就是不同的词向量, 通过LN可以保留特征之间的关系.

对于NLP来说, 归一化是三维的, 因为每一个序列样本由多个单词组成. 它的坐标分别为batch(表示某一个样本也就是某个序列), seq(表示某个序列中的单词), feature(表示这个单词的词向量).

## 局部, 受限注意力机制

在Transformer模型中, 传统过的注意力机制允许序列中的每个元素都能关注到序列中的所有其他元素. 这种全局注意力机制虽然强大, 但是当处理图像, 音频和视频等大型输入和输出的时候, 计算成本非常高.

**局部, 受限注意力机制**是为了解决这个问题而提出的方法. 它通过限制每个元素的注意力范围, 只允许它关注到序列中的局部区域, 从而降低了计算复杂度.

它常见的实现方式有:

- **窗口注意力:** 将输入序列分成固定大小的窗口, 每个元素只能关注到自己所在窗口内的元素, 这种方法显著减少了注意力计算量, 尤其对于长序列
- **膨胀注意力:** 以指数方式扩大注意力窗口, 使模型能够捕捉长距离依赖关系, 同时保持较高的效率
- **稀疏注意力:** 根据预定义的模式或学习到的注意力分布, 选择一小部分元素进行关注

[^1]: 第二章：Transformer 模型 · Transformers快速入门. (不详). 取读于 2024年9月23日, 从 https://transformers.run/c1/transformer/#%E6%B3%A8%E6%84%8F%E5%8A%9B%E5%B1%82
[^2]: Alammar, J. (不详). The Illustrated Transformer. 取读于 2024年9月23日, 从 https://jalammar.github.io/illustrated-transformer/
[^3]: 细节拉满，全网最详细的Transformer介绍（含大量插图）！. (不详). 知乎专栏. 取读于 2024年9月23日, 从 https://zhuanlan.zhihu.com/p/681532180
