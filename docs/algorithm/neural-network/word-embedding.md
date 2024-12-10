---
title: 词嵌入
comments: false
---

# 词嵌入[^1]

## 词汇表征

### One-Hot表示

词汇表是一种用one-hot编码来表示词的方法, 例如, 如果man在字典里是第5391个, 那么就可以表示为一个向量, 只在5391这个位置上是1, 其他地方为0, 使用$O_{5391}来代表这个量$; 如果woman的编号是9853, 那么这个向量在9853处为1, 其他地方为0, 使用$O_{9853}$来代表这个量. 其他的词如king, queen, apple, orange都可以这样表示出来. 这种表示方法最大的缺点就是将每个词孤立了起来, 这样使得算法对相关词的泛化能力不强.

<figure markdown='1'>
  ![](https://img.ricolxwz.io/c2ca6351fdf76fe5b747211d159b8472.png){ loading=lazy width='500' }
</figure>

假如你已经学习到了一个语言模型, 但你看到"I want a glass of orange \_\_"的时候, 那么下一个词的时候会是什么? 很可能是juice. 即使你的学习算法已经学习到了"I want a glass of orage juice"这样的一个很可能的橘子, 但是如果看到"I want a glass of apple __", 因为算法不知道apple和orange的关系很接近, 就像man和woman, king和queen一样. 所以算法很难从已经知道的orange juice是一个很常见的东西, 而明白apple juice也是很常见的东西或者说是常见的句子. 因为任何两个one-hot向量的内积都是0, 说明apple和orange是不相关的, 还无法表示相关的程度, 如无法知道apple, orange的相关程度和apple, peach的相关程度.

### 词嵌入表示

换一种表示方式, 如果不用one-hot表示, 而是用特征化的表示来表示每个词, man, woman, king, queen, apple, orange或者字典里面的任何一个单词, 我们学习这些词的特征或者数值. 举一个例子, 对于这些词, 比如我们想知道这些词在Gender上的表示是怎么样的, 假定男性的性别是+1, 女性的性别是-1, 那么man的这个Gender的属性可以用+1表示, woman的Gender属性可以用-1表示. 最终根据经验king就是-0.95(剩下的0.05人妖是吧), queen是+0.97. 另外一个特征是这些词有多高贵(Royal), man, woman和高贵没多大关系, 所以它们的特征值接近于0, 而king和queen很高贵, apple, orange跟高贵也没多大关系. 那么年龄呢? man和woman一般没有年龄的意思, 也许man和woman隐藏着成年人的意思, 但也可能接近于young和old之间, 所以它们的值也接近于0, 而通常king和queen都是成年人, apple和orange就跟年龄没什么关系了. 还有一个特征, 这个词是否是食物, man不是食物, woman不是食物, king和queen也不是, 但是apple和orange是食物. 当然还可以有很多其他特征, 如Size, Cost, Alive, Action, Noun, Verb等等.

<figure markdown='1'>
  ![](https://img.ricolxwz.io/c679a2cc7a679aaa676627f906fc80b4.png){ loading=lazy width='500' }
</figure>

所以你可以想很多的特征, 为了说明, 假设有300个特征, 这样的话对于每个词就有了一列数字, 如用300维的向量来表示man这个词, 使用$e_{5391}$来表示这个量. 现在, 如果我们用这种方法来表示apple和orange, 那么apple和orange的表示肯定会非常相似, 可能有一些特征不太一样, 比如颜色, 口味.. 但是总的来说apple和orange的大部分特征都是差不多的, 或者说有相似的值, 这样它们两个词的内积就会比较大.

后面的几个小节里面, 我们会找到一种学习词嵌入的方式, 这里只是希望你能够理解这种高维度特征的表示能够比one-host更好的表示不同的单词. 而我们最终学习的特征不会像这里一样那么好理解, 没有像第一个特征是性别, 第二个特征是高贵, 第三个特征是年龄等等这些, 新的特征表示的东西肯定会更难搞清楚, 或者说根本没必要搞清楚. 尽管如此, 接下来小节要学的特征表示方法能够高效地发现apple和orange比king和orange, queen和orange更加相似.

#### 词嵌入可视化

如果我们能够学习到一个300维的特征向量, 或者说是300维的词嵌入, 通常我们可以做一件事情, 就是把这300维度的数据嵌入到一个二维空间里面, 这样就可以可视化了. 常用的可视化算法是t-SNE算法[^2]. 如果观察这种词嵌入的表示方法, 你会发现man和woman这些词聚集在一块, 如下图, king和queen聚集在一块, 动物聚集在一起, 水果聚集在一起.

<figure markdown='1'>
  ![](https://img.ricolxwz.io/0455d8f6163ce2dc94ea360c32a9012d.png){ loading=lazy width='500' }
</figure>

这种表示方式用的是在300维的空间里的特征表示, 这叫做嵌入(embeddings), 如orange会被嵌入到300维空间的一个点上, apple这个词会被嵌入到300维空间的另一个点上, 在上图中由于画不出300维, 所以使用一个3维的点代替. t-SNE算法就是把这个空间映射到低维空间, 可以画出一个2维图像然后观察.

## 使用词嵌入

在上小节中, 了解了不同单词的特征化表示, 这节会看到我们如何把这种表示方法应用到NLP中.

### NER任务

在命名实体识别任务(Named Entity Recognition, NER)任务中, 我们会依赖上下文判断一个词的实体类别. 假如有一个句子"Saly Johnson is an orange farmer", 如果要找出人名, 你会发现Sally Johnson是一个人名, 之所以能确定Sally Johnson是一个人名而不是一个公司名, 是因为这种判断依赖于上下文, 你知道种橙子的农名一定是一个人, 而不是一个公司.

但是如果你用特征化的表示方法, 即词嵌入. 那么用词嵌入作为输入, 如果你看到一个新的输入"Robert Lin is an apple farmer". 因为你知道apple和orange很相似, 即apple和orange的词向量很相似, 那么相相当于有了类似的上下文, 算法可以很容易就知道Robert Lin也是一个人的名字. 再举一个例子, 如果这个时候变成了"Robert Lin is a durian cultivator", 训练集里面甚至没有durian和cultivator这两个词, 但是如果有一个已经学习好的词嵌入, 它会告诉你durian是水果, 就像orange一样, cultivator和farmer差不多, 那么也可以推断出Robert Lin是一个人.

<figure markdown='1'>
  ![](https://img.ricolxwz.io/4927908c6964b1d8d55d9008a7e543ac.png){ loading=lazy width='500' }
</figure>

### 迁移学习

词嵌入能够达到这样的效果, 其中的一个原因是学习词嵌入的算法会考察非常大的文本集, 也许是从网上找到的, 这样你可以考察很大的数据集甚至是1亿个单词, 甚至达到100亿都是合理的. 通过考察大量的**无标签**的文本, 你可以发现orange和durian相近, farmer和cultivator相近. 尽管你只有一个很小的训练集, 也许训练集里只有100000个单词, 你也可以通过迁移学习将从互联网上免费获得的大量的词嵌入表示运用到你的任务中.

<figure markdown='1'>
  ![](https://img.ricolxwz.io/d89785032c4d1b77f250e06a473e0d70.png){ loading=lazy width='500' }
</figure>

如上图所示, 迁移学习的步骤可以表示为:

1. 先从大量的文本集中学习词嵌入, 或者说下载网上预训练好的词嵌入模型
2. 用这些词嵌入模型迁移到你的任务中, 比如说用这个300维的嵌入表示你的单词
3. 在你的任务上, 可以选择要不要微调, 用新的数据稍微调整一下某些词的词嵌入, 实际情况下, 如果你有很大的训练集你才可以这么做, 如果不是很大, 通常不会在微调词嵌入上面花费力气

词嵌入在语言模型, 机器翻译用的少一点, 尤其是做语言模型或者机器翻译任务的时候, 这些任务你有大量的数据. 在其他的迁移学习情形中也一样, 如果你从某一任务A迁移到某一任务B, 只有A中有大量数据, 而B中的数据较少的时候, 迁移的过程才有用. 所以对于很多的NLP任务这些都是对的, 而对于一些语言模型和机器翻译则不然.

### 人脸编码

词嵌入和人脸编码之间有其妙的关系, 在人脸识别中, 我们训练了一个Siamese网络结构[^5], 这个网络会学习不同人脸的128维表示, 然后通过比较编码结果来判断两个图片是否是一个人脸, 词嵌入的意思和这个差不多, 在人脸识别领域大家更喜欢用编码这个词来指代这些词向量. 但是有一个显著的不同就是, 在人脸识别中, 我们训练一个网络, 任意给出一个人脸照片, 甚至是没有见过的照片, 神经网络都会计算出一个相应的编码结果. 而学习词嵌入是有一个固定的词汇表的, 比如10000个单词, 我们学习每个词的一个固定的编码, 而像一些没有出现过的单词就被标记为未知单词. 现代语言模型已经通过方法如WordPiece[^3]或者Byte Pair Encoding(BPE)[^4]解决了未知单词的限制, 但是不变的是, 它们都有一个词汇表.

### 类比推理

之前我们讲到, 词嵌入的作用之一是捕捉相似词之间的关系. 词嵌入还有一个特性就是它能实现类比推理, 它更关注捕捉不同实体之间的关系模式. 假如提出一个问题, man如果对应woman, 那么king应该对应什么, 都知道king应该对应queen, 这里强调的是性别差异模式, 能否有一种算法能够自动推导出这种关系.

<figure markdown='1'>
  ![](https://img.ricolxwz.io/bedd9aba9a35a07b6bb3d6d499077f07.png){ loading=lazy width='500' }
</figure>

我们使用一个四维向量来表示man, 标识为$e_{man}$, 而旁边这个表示woman的嵌入向量, 称其为$e_{woman}$, 对king和queen也是同样的表示方法. 这里为了简化表示假设使用的是$4$维的嵌入向量, 对向量$e_{man}$和$e_{woman}$进行减法运算, 即:

$$\mathbf{e}_{\text{man}} - \mathbf{e}_{\text{woman}} =
\begin{bmatrix}
-1 \\
0.01 \\
0.03 \\
0.09
\end{bmatrix}
-
\begin{bmatrix}
1 \\
0.02 \\
0.02 \\
0.01
\end{bmatrix}
=
\begin{bmatrix}
-2 \\
-0.01 \\
0.01 \\
0.08
\end{bmatrix}
\approx
\begin{bmatrix}
-2 \\
0 \\
0 \\
0
\end{bmatrix}$$

类似的, 假如你用$e_{king}$减去$e_{queen}$, 最后也会得到类似的结果:

$$\mathbf{e}_{\text{king}} - \mathbf{e}_{\text{queen}} =
\begin{bmatrix}
-0.95 \\
0.93 \\
0.70 \\
0.02
\end{bmatrix}
-
\begin{bmatrix}
0.97 \\
0.95 \\
0.69 \\
0.01
\end{bmatrix}
=
\begin{bmatrix}
-1.92 \\
-0.02 \\
0.01 \\
0.01
\end{bmatrix}
\approx
\begin{bmatrix}
-2 \\
0 \\
0 \\
0
\end{bmatrix}$$

这个结果表示, man和woman主要的差异是gender上的差异, 而king和queen之间的主要差异, 根据向量的表示, 也是gender上的差异, 这就是为什么$e_{man}-e_{woman}$和$e_{king}-e_{queen}$的结果是相同的. 所以的出这种方法对应的就是当算法被问及man对woman相当于king对什么的时候, 算法所做的就是计算$e_{man}-e_{woman}$, 然后找到一个向量也就是找出一个词, 使得$e_{man}-e_{woman}\simeq e_{king}-e_{?}$, 也就是说, 当这个新词是queen的时候, 式子的左边会近似地等于右边. 这种思想源于Tomas Mikolov等人的研究[^6].

<figure markdown='1'>
  ![](https://img.ricolxwz.io/d999f07762bbe97c0eb0c884eb2220f3.png){ loading=lazy width='500' }
</figure>

那么, 如何将这种思想写成算法呢? 在上图中, 词嵌入向量在一300维的空间里面, 每个单词对应的是300维空间上的一个点. 所示的箭头代表的就是向量在gender这一特征的差值, 可以看到man, woman的差值非常接近于king, woman之间的差值. 为了得出上述类比推理, 你能做的就是找到单词$w$使得$e_{man}-e_{woman}\simeq e_{king}-e_{w}$这个等式成立. 我们要做的是把$e_w$放到等式的一边, 于是等式的另一边就是$e_{king}-e_{man}+e_{woman}$, 这个式子的意思就是找到单词$w$最大化$e_w$和$e_{king}-e_{man}+e_{woman}$的相似度. 如果理想的话, 应该会得到单词queen. 但是, 如果查看一些研究文献不难发现, 通过这种方法做类比推理的准确率大概只有30%-75%.

在继续之前, 需要明确一下上图中左边的这个平行四边形. 之前我们谈到过可以使用t-SNE算法将词嵌入可视化, t-SNE是一种非线性降维算法, 它将高维数据(如300维的词向量)映射到低维空间(如2维)以便于可视化, 这种映射方式复杂且非线性, 因此在降维后, 数据之间的几何关系可能会被扭曲. 在原始的高维空间中, 类比关系如man\:woman::king:queen可以通过集合关系(如平行四边形)表示, 而在t-SNE降维后的2维空间中, 由于映射的非线性特性, 这种几何关系通常无法保持, 平行四边形的形状和方向可能会完全改变甚至丢失.

余弦函数是一种常用的相似度测量函数. 在余弦相似度中, 向量$u$和$v$之间的相似度可以被定义为$sim(u, v)=\frac{u^Tv}{||u||_2||v||_2}$. 如果$u$和$v$非常相似, 那么它们的内积会非常大, 当夹角是$90$度的话, 那么余弦相似度是0, 所以说, 这种相似性取决于向量$u$和$v$之间的角度.

<figure markdown='1'>
  ![](https://img.ricolxwz.io/fc0d12a75b44f0ad50c7817db6b35c6d.png){ loading=lazy width='700' }
</figure>

用到上述的例子中, 就是计算$e_w$和$e_{king}-e_{man}+e_{woman}$的余弦相似度.

## 嵌入矩阵

当你应用算法来学习词嵌入的时候, 实际上是在学习一个嵌入矩阵.

和之前一样, 假设我们的词汇表含有$10000$个单词, 词汇表里有a, aaron, orange, zulu, 等等词汇. 我们要做的就是学习一个嵌入矩阵$E$, 它是一个$300\times 10000$的矩阵. 假设orange的单词编号是$6257$, 使用$O_{6257}$来表示这个one-hot向量, 显然它的形状是$10000\times 1$, 它不像下图(右侧)中的那么短, 它的高度应该和左边的那个嵌入矩阵的宽度($10000$)相等.

<figure markdown='1'>
  ![](https://img.ricolxwz.io/1d1a0c6e49eaa13069990aa098f06e5c.png){ loading=lazy width='500' }
</figure>

假设这个嵌入矩阵叫做$E$. 如果用$E$去乘右边的$O_{6257}$, 那么就会得到一个$300$维的向量, 产生的矩阵的形状是$300\times 1$的, 也就是一个列向量. 这个列向量的第一个元素(图中编号6)对应的就是嵌入矩阵中orange列的第一个元素(图中编号5), 以此类推, **得到的列向量与orange列构成的向量是相等的. 所以说每个单词的词向量其实就存储在嵌入矩阵的对应列中**.

在下个小节中会随机初始化矩阵$E$, 然后使用梯度下降法来学习这$300\times 10000$的矩阵的各个参数. $E$乘以one-hot向量就会得到它的词向量. 在我们手动计算这个词向量的时候, 是很方便的. 在实际中, 由于所有词的one-hot编码会组成一个非常大又非常稀疏的矩阵, 嵌入矩阵和这个矩阵相乘的计算效率非常低下. 由于刚才我们提到“某个单词的词向量其实就是嵌入矩阵中的对应列”, 所以在实际中会使用一个专门的函数来单独查找嵌入矩阵的对应列, 而不是用矩阵乘法去取出那个列.

## 学习嵌入矩阵

在本小节中, 要学习一些具体的算法来学习词嵌入. 在深度学习应用于学习词嵌入的历史上, 人们一开始使用的算法比较复杂, 但是随着时间的推移, 研究者们不断发现他们能用更加简单的算法来实现一样好的效果, 特别是在数据集比较大的情况下. 但是有一件事情就是, 现在很多最流行的算法都非常简单, 如果我们一开始就介绍这种简单的算法, 可能会觉得有点摸不着头发, 稍微从简单一点的算法开始, 可以更容易对算法的运作方式有一个更加直观的了解, 之后我们会对算法进行简化, 使得我们能够明白即使一些简单的算法也能得到非常好的效果.

假如你在构建一个语言模型, 并且用神经网络来实现这个模型. 在训练过程中, 你可能想要你的神经网络能够做到比如输入"I want a glass of orange __", 然后预测这句话的下一个词. 在每个单词下面, 我们都写上了它们在词汇表中对应的序号. 从第一个词"I"开始, 建立一个one-hot向量表示这个单词, 用为经过初始化的$E$乘以这个$O_{4343}$, 得到嵌入向量$e_{4343}$, 然后对于其他的词都做同样的操作... 于是现在你有6个300维的向量, 把他们全部放到神经网络里面, 经过神经网络之后再通过softmax层, 这个softmax产生的输出代表10000个单词作为下一个词的可能性, 然后根据真实的下一个词计算误差反向传播, 更新嵌入矩阵$E$.

<figure markdown='1'>
  ![](https://img.ricolxwz.io/a0fdd6fae7201e9347ff9a8a2199313c.png){ loading=lazy width='500' }
</figure>

实际上更加常见的做法是使用一个固定的历史窗口, 举个例子, 你总是想预测四个单词后的下一个单词, 注意这里的4是超参数. 这就意味着你的神经网络输入的不是一个6个300维的向量, 而是4个300维的向量.

<figure markdown='1'>
  ![](https://img.ricolxwz.io/c20828ff376bed49b8ce6268ffad46c0.png){ loading=lazy width='500' }
</figure>

这个是早期比较成功的学习词嵌入的方法之一. 我们先来概括一下这个算法, 看看我们怎样来推导出更加简单的算法. 假设你的训练集中有这样一个更长的句子"I want a glass of orange juice to go aloing with my cereal.". 我们将需要预测的词称为目标词, 即juice为目标词. 在上述方法中, 目标词是通过前面的4个词推导出来的.

[^1]: 深度学习笔记. (不详). 取读于 2024年12月10日, 从 http://www.ai-start.com/dl2017/html/lesson5-week2.html#header-n169
[^2]: Maaten, L. van der, & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9(86), 2579–2605.
[^3]: Song, X., Salcianu, A., Song, Y., Dopson, D., & Zhou, D. (2021). Fast WordPiece tokenization (No. arXiv:2012.15524). arXiv. https://doi.org/10.48550/arXiv.2012.15524
[^4]: Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units (No. arXiv:1508.07909). arXiv. https://doi.org/10.48550/arXiv.1508.07909
[^5]: Koch, G. R. (2015). Siamese neural networks for one-shot image recognition. https://www.semanticscholar.org/paper/Siamese-Neural-Networks-for-One-Shot-Image-Koch/f216444d4f2959b4520c61d20003fa30a199670a
[^6]: Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space (No. arXiv:1301.3781). arXiv. https://doi.org/10.48550/arXiv.1301.3781
