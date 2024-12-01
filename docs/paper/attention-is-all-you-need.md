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

## 结论

### 局部, 受限注意力机制

“We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video.” ([Vaswani 等, 2023, p. 10](zotero://select/library/items/AWW2Z4WB)) ([pdf](zotero://open-pdf/library/items/K3RI73ET?page=10))

在Transformer模型中, 传统过的注意力机制允许序列中的每个元素都能关注到序列中的所有其他元素. 这种全局注意力机制虽然强大, 但是当处理图像, 音频和视频等大型输入和输出的时候, 计算成本非常高.

**局部, 受限注意力机制**是为了解决这个问题而提出的方法. 它通过限制每个元素的注意力范围, 只允许它关注到序列中的局部区域, 从而降低了计算复杂度.

它常见的实现方式有:

- **窗口注意力:** 将输入序列分成固定大小的窗口, 每个元素只能关注到自己所在窗口内的元素, 这种方法显著减少了注意力计算量, 尤其对于长序列
    
- **膨胀注意力:** 以指数方式扩大注意力窗口, 使模型能够捕捉长距离依赖关系, 同时保持较高的效率
    
- **稀疏注意力:** 根据预定义的模式或学习到的注意力分布, 选择一小部分元素进行关注