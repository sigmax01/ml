# (2023-08-02) Attention is all you need (关注就是你所需要的一切)

<table><tbody><tr><td style="background-color: rgb(219, 238, 221);"><p><strong><span style="color: rgb(25, 60, 71)"><span style="background-color: rgb(219, 238, 221)">Author:</span></span></strong><span style="color: rgb(25, 60, 71)"><span style="background-color: rgb(219, 238, 221)"> Ashish Vaswani; Noam Shazeer; Niki Parmar; Jakob Uszkoreit; Llion Jones; Aidan N. Gomez; Lukasz Kaiser; Illia Polosukhin;</span></span></p></td></tr><tr><td style="background-color: rgb(243, 250, 244);"><p><strong><span style="color: rgb(25, 60, 71)"><span style="background-color: rgb(243, 250, 244)">Journal: (Publication Date: 2023-08-02)</span></span></strong></p></td></tr><tr><td style="background-color: rgb(219, 238, 221);"><p><strong><span style="color: rgb(25, 60, 71)"><span style="background-color: rgb(219, 238, 221)">Journal Tags:</span></span></strong></p></td></tr><tr><td style="background-color: rgb(243, 250, 244);"><p><strong><span style="color: rgb(25, 60, 71)"><span style="background-color: rgb(243, 250, 244)">Local Link: </span></span></strong><span style="color: rgb(25, 60, 71)"><span style="background-color: rgb(243, 250, 244)"><a href="zotero://open-pdf/0_K3RI73ET" rel="noopener noreferrer nofollow">Vaswani 等 - 2023 - Attention is all you need.pdf</a></span></span></p></td></tr><tr><td style="background-color: rgb(219, 238, 221);"><p><strong><span style="color: rgb(25, 60, 71)"><span style="background-color: rgb(219, 238, 221)">DOI: </span></span></strong><span style="color: rgb(25, 60, 71)"><span style="background-color: rgb(219, 238, 221)"><a href="https://doi.org/10.48550/arXiv.1706.03762" rel="noopener noreferrer nofollow">10.48550/arXiv.1706.03762</a></span></span></p></td></tr><tr><td style="background-color: rgb(243, 250, 244);"><p><strong><span style="color: rgb(25, 60, 71)"><span style="background-color: rgb(243, 250, 244)">Abstract Translation: </span></span></strong><em><span style="color: rgb(25, 60, 71)"><span style="background-color: rgb(243, 250, 244)">主要的序列转导模型基于编码器-解码器配置中的复杂递归或卷积神经网络。性能最佳的模型还通过注意力机制连接编码器和解码器。我们提出了一种新的简单网络架构，即 Transformer，它完全基于注意力机制，完全省去了递归和卷积。对两项机器翻译任务的实验表明，这些模型在质量上上乘，同时并行化程度更高，并且需要的训练时间明显减少。我们的模型在 WMT 2014 英语到德语的翻译任务中实现了 28.4 BLEU，比现有的最佳结果有所改进，包括超过 2 BLEU 的集成。在 WMT 2014 英语到法语翻译任务中，我们的模型在 8 个 GPU 上训练 3.5 天后，建立了一个新的单模型最先进的 BLEU 分数，即 41.8，这只是文献中最佳模型的训练成本的一小部分。我们表明，Transformer 成功地将其应用于具有大量和有限训练数据的英语选区解析，从而很好地推广到其他任务。</span></span></em></p></td></tr><tr><td style="background-color: rgb(219, 238, 221);"><p><strong><span style="color: rgb(25, 60, 71)"><span style="background-color: rgb(219, 238, 221)">Tags:</span></span></strong></p></td></tr><tr><td style="background-color: rgb(243, 250, 244);"><p><strong><span style="color: rgb(25, 60, 71)"><span style="background-color: rgb(243, 250, 244)">Note Date: </span></span></strong><span style="color: rgb(25, 60, 71)"><span style="background-color: rgb(243, 250, 244)">2024/11/30 19:56:28</span></span></p></td></tr></tbody></table>

## 导言背景

### RNN模型的缺陷

“This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.” ([Vaswani 等, 2023, p. 2](zotero://select/library/items/AWW2Z4WB)) ([pdf](zotero://open-pdf/library/items/K3RI73ET?page=2))

RNN模型无法并行执行, 在计算第t的词的时候, 前面的t-1个词必须全部运算完成. 如果时序比较长的话, 可能会导致前面的信息到后面就丢失掉了. 如果不想丢掉的话, 就需要做一个比较大的ht, 这会导致很高昂的内存开销.

“In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.” ([Vaswani 等, 2023, p. 2](zotero://select/library/items/AWW2Z4WB)) ([pdf](zotero://open-pdf/library/items/K3RI73ET?page=2))

一开始的时候, 这些注意力机制可能都是和RNN结合起来使用的, 而没有成为一个独立的体系.

### CNN模型的缺陷

“In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12].” ([Vaswani 等, 2023, p. 2](zotero://select/library/items/AWW2Z4WB)) ([pdf](zotero://open-pdf/library/items/K3RI73ET?page=2))

使用CNN的时候, 每一次它去看的是一个比较小的窗口, 如3\*3的卷积核. 如果两个像素隔的比较远的时候, 需要较多的层才能把这两个像素融合起来.

但是Transformer模型通过注意力机制每一次能够看到所有的像素, 在一层中就能够看到. CNN比较好的地方是它有很多个输出通道, 每个通道可以去识别不同的模式. 所以说它提出了一个多头注意力机制, 模拟CNN的多输出通道的效果.

### 端到端记忆网络

“End-to-end memory networks are based on a recurrent attention mechanism instead of sequencealigned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].” ([Vaswani 等, 2023, p. 2](zotero://select/library/items/AWW2Z4WB)) ([pdf](zotero://open-pdf/library/items/K3RI73ET?page=2))

“端到端”指的是一种系统设计方法, 意味着从输入到输出的整个过程由一个单一的系统或者模型直接完成, 通常不需要人工干预或者多个独立的模块.

端到端记忆网络是主要用于处理需要长期依赖记忆的任务. 它在功能上和LSTM类似, 但是在结构或者说工作方式上有显著不同. 前者引入了一个独立的外部记忆模块, 该模块是一个可供模型读取和写入的内存池, 模型通过注意力机制与这个外部记忆池的交互来保存和获取信息, 从而支持长期依赖. 而LSTM本身是通过门机制来管理和控制记忆的, 它的记忆是隐式的, 即记忆是通过递归传递的内部状态(cell state)来存储的, 而不是通过外部记忆池进行显式存储.

## 模型架构

### 简易结构

“Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn)” ([Vaswani 等, 2023, p. 2](zotero://select/library/items/AWW2Z4WB)) ([pdf](zotero://open-pdf/library/items/K3RI73ET?page=2))

编码器会将输入的字符映射到一个连续的表示空间(潜空间)中.

“Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time.” ([Vaswani 等, 2023, p. 2](zotero://select/library/items/AWW2Z4WB)) ([pdf](zotero://open-pdf/library/items/K3RI73ET?page=2))

解码器得到编码器的输出之后会根据它一个一个的生成字符, 注意这里的输出序列的长度和输入序列的长度是不一样的, 这是因为seq2seq中得到的文本长度和原始文本长度大概率是有差异的(比如英文转中文).

“At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.” ([Vaswani 等, 2023, p. 2](zotero://select/library/items/AWW2Z4WB)) ([pdf](zotero://open-pdf/library/items/K3RI73ET?page=2))

这里用到了自回归(auto-regressive), 也就是说, 解码器在生成yt的时候已经有y1-yt-1这些输出了, 这些输出继续当作它的输入.

### 批归一化和层归一化

“LayerNorm” ([Vaswani 等, 2023, p. 3](zotero://select/library/items/AWW2Z4WB)) ([pdf](zotero://open-pdf/library/items/K3RI73ET?page=3))

- **批归一化(BatchNorm):** 是对每个batch内的样本的每个特征进行归一化, 具体来说, 它会计算整个batch中的每个特征的均值和方差, 然后基于这些统计量对所有的样本进行归一化. 因此, BN的计算是跨样本的, 即竖着计算. 注意, 学习和预测时候的BN是不一样的, 学习的时候是这个batch里面的特征取均值和方差; 预测的时候是整个样本集的特征取均值和方差
    
- **层归一化(LayerNorm):** 是对每个样本的所有特征进行归一化. 它计算每个样本的所有特征的均值和方差, 因此是横着归一化的
    

所以, BN抹平了不同特征之间的大小关系, 而保留了不同样本之间的大小关系. 这样, 如果具体任务依赖于不同样本之间的联系, BN更有效, 尤其是在CV领域, 不同图片样本之间的大小关系得以保留. LN抹平了不同样本之间的大小关系, 而保留了不同特征之间的大小关系. 所以, LN更加适合NLP任务, 一个样本实际上就是不同的词向量, 通过LN可以保留特征之间的关系.

对于NLP来说, 归一化是三维的, 因为每一个序列样本由多个单词组成. 它的坐标分别为batch(表示某一个样本也就是某个序列), seq(表示某个序列中的单词), feature(表示这个单词的词向量).

### 注意力机制

“The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention.” ([Vaswani 等, 2023, p. 4](zotero://select/library/items/AWW2Z4WB)) ([pdf](zotero://open-pdf/library/items/K3RI73ET?page=4))

“Additive attention computes the compatibility function using a feed-forward network with a single hidden layer.” ([Vaswani 等, 2023, p. 4](zotero://select/library/items/AWW2Z4WB)) ([pdf](zotero://open-pdf/library/items/K3RI73ET?page=4))

最常用的两种注意力函数是累加和点积, 累加是将Query和Key组合起来, 通过一个小型的神经网络生成注意力分数. 它和点积的区别主要在于计算方式的非线性化. 但是由于点积在计算上的性能远远大于累加, 而且在实际中往往更加节省内存空间, 所以这篇文章用的是点积.

“We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4. To counteract this effect, we scale the dot products by √1dk .” ([Vaswani 等, 2023, p. 4](zotero://select/library/items/AWW2Z4WB)) ([pdf](zotero://open-pdf/library/items/K3RI73ET?page=4))

当dk比较大的时候, QK^T点积过程中需要累加的项就特别多, 导致最终的attention分数的数值过大. 累积dk次后, 结果的均值和方差都会成比例增大, 由于点积结果的方差和dk是成正比的, 所以通过除以sqrt(dk), 可以使得结果的方差保持常数, 这样, 无论dk的大小如何, 点积结果的数值范围都会被规范到一个固定的范围内.

### 为什么需要mask

这个可以从两个方面来说, 训练阶段和推理阶段.

- **训练阶段:** 在训练的时候, decoder输入的是完整的目标序列(包括未来词), 而不是逐步生成的部分序列. 目标序列被送入自注意力层之后, 如果不加mask, 模型在计算第t个词的注意力的时候, 会看到整个目标序列(包括t+1, t+2, …), 导致模型可以利用未来的词生成当前的词, 这种信息泄露会导致训练过程中模型无法学习到正确的因果关系, 从而在推理阶段表现不加
    
- **推理阶段:** 在推理的过程中, decoder的输入是逐步生成的序列, 在t时刻, decoder的输入是从第1到第t-1时刻生成的词, 理论上, 此时未来的词(第t+1, t+2, …)根本不存在, 似乎不需要mask. 但是由于自注意力机制的实现是对于整个序列(包括还未填充的位置)计算注意力分布, 如果不加mask, decoder的自注意力层仍会尝试对后续未生成的位置(这些位置可能被初始化为零向量或者其他占位符)来计算注意力分布, 即使这些未生成(未填充)的位置没有真实的信息, 注意力分布的结果可能会受到干扰
    

具体的做法是将后面的值替换成一个非常大的负数. 注意, 不能替换为0, 不然的话经过softmax之后其他地方的值会受到影响(变小). 负数经过softmax之后就会变成0.

### 多头注意力机制

“we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively.” ([Vaswani 等, 2023, p. 4](zotero://select/library/items/AWW2Z4WB)) ([pdf](zotero://open-pdf/library/items/K3RI73ET?page=4))

将高维的Q, K和V投影到多个低维子空间(每个子空间对应一个头), 在这些低维子空间中分别计算注意力, 最终将各个头的结果拼接起来, 再进行一次线性变换, 得到输出.

“In this work we employ h = 8 parallel attention layers, or heads. For each of these we use dk = dv = dmodel/h = 64.” ([Vaswani 等, 2023, p. 5](zotero://select/library/items/AWW2Z4WB)) ([pdf](zotero://open-pdf/library/items/K3RI73ET?page=5))

由于有残差连接的存在, 输入的维度必须是和输出的维度是一样的, 所以, 选择的dk和dv是原始维度/h. 然后拼接的时候就可以回到原来的维度.

## 结论

### 局部, 受限注意力机制

“We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video.” ([Vaswani 等, 2023, p. 10](zotero://select/library/items/AWW2Z4WB)) ([pdf](zotero://open-pdf/library/items/K3RI73ET?page=10))

在Transformer模型中, 传统过的注意力机制允许序列中的每个元素都能关注到序列中的所有其他元素. 这种全局注意力机制虽然强大, 但是当处理图像, 音频和视频等大型输入和输出的时候, 计算成本非常高.

**局部, 受限注意力机制**是为了解决这个问题而提出的方法. 它通过限制每个元素的注意力范围, 只允许它关注到序列中的局部区域, 从而降低了计算复杂度.

它常见的实现方式有:

- **窗口注意力:** 将输入序列分成固定大小的窗口, 每个元素只能关注到自己所在窗口内的元素, 这种方法显著减少了注意力计算量, 尤其对于长序列
    
- **膨胀注意力:** 以指数方式扩大注意力窗口, 使模型能够捕捉长距离依赖关系, 同时保持较高的效率
    
- **稀疏注意力:** 根据预定义的模式或学习到的注意力分布, 选择一小部分元素进行关注