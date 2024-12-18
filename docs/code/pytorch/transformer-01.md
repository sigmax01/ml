---
title: Transformer-01
comments: false
---

# [Transformer-01](https://colab.research.google.com/drive/1MoxcUOQKAhBoSpJNsB-8NifN48C5VmAz?usp=sharing)[^3]

## 词嵌入

```py title='输入'
class TokenEmbedding(nn.Module):
  def __init__(self, vocab_size, hidden_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, hidden_size)

  def forward(self, x):
    embedded = self.embedding(x)
    return embedded

def test_token_embedding():
  vocab_size = 100
  hidden_size = 512
  batch_size = 32
  seq_len = 10

  # 随机生成输入数据, 形状为(batch_size, seq_len)
  x = torch.randint(0, vocab_size, (batch_size, seq_len))

  # 创建TokenEmbedding模块
  token_embedding = TokenEmbedding(vocab_size, hidden_size)

  # 计算嵌入输出
  output = token_embedding(x)

  print("输入形状: ", x.shape)
  print("输出形状", output.shape)

test_token_embedding()
```

``` title='输出'
输入形状:  torch.Size([32, 10])
输出形状 torch.Size([32, 10, 512])
```

???+ note "`nn.Embedding()`的用法"

    `nn.Embedding`的第一个参数表示的是词表的大小, 第二个参数表示的是词向量的维度, 其实还有第三个变量`padding_idx`表示的是如果词汇表产生的编码是`padding_idx`的话, 那么产生的词向量为全0. 例如, `padding_idx`被设置为0, 在词表中有一个词汇`<PAD>`的编码是0, 则句子中的所有`<PAD>`的词向量都对应为一个全0的词向量. 注意, 这里`<PAD>`的作用是用于补齐句子的长度[^1].

    `nn.Embedding`这个类有一个属性`weight`, 它是`nn.parameter.Parameter`类型的, 作用就是存储真正的词嵌入矩阵. 如果不给`weight`赋值, 那么会自动对其进行初始化, 先会定义一个`nn.parameter.Parameter`对象, 然后对该对象调用类内部的`self.reset_parameters`方法, 这个方法会将其调整为均值为0, 方差为1的正态分布[^2]. 

???+ note "`nn.Embedding.from_pretrained()`用法"

    `nn.Embedding.from_pretrained()`这个方法允许你使用已经训练好的嵌入向量来初始化`nn.Embedding`, 可以来自CBOW, Skip-gram, GloVe等等, 而不是从随机初始化开始. 它的第一个参数是`pretrained_embeddings`, 这是一个包含预训练嵌入矩阵的2D FloatTensor, 第一个维度是词汇表的大小, 第二个维度是嵌入向量的维度. 第二个参数是`freeze`, 这是可选的, 若`freeze=True`, 则嵌入权重在训练过程中不会被更新, 如果是`freeze=False`, 则嵌入权重 ^^可以^^ 在训练中进行微调, `padding_idx`同`nn.Embedding`.

## 位置嵌入

在原始论文中他们使用的位置嵌入是通过两个函数生成的.

$$
\begin{aligned}
PE(pos, 2i) &= \sin\left(\frac{pos}{10000^{\frac{2i}{d}}}\right) \\
PE(pos, 2i+1) &= \cos\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)
\end{aligned}
$$

其中, $i$表示的是维度索引, $pos$表示的是token在sequence中的位置. 

```py title='输入'
class PositionalEmbedding(nn.Module):
  def __init__(self, max_len, hidden_size):
    super().__init__()
    self.hidden_size = hidden_size

    # 创建一个位置编码表, 形状为(max_len, 1)
    # 提前进行unsqueeze在末尾添加一个维度以适应后面的广播, 原先是(max_len, ), 经过unsqueeze之后, 是(max_len, 1)
    position = torch.arange(0, max_seq_len).unsqueeze(1).float()

    # 计算三角函数内分母的倒数, 形状为(hidden_size/2, )
    div_term = torch.exp(torch.arange(0, embed_model_dim, 2).float() * -(np.log(10000.0) / embed_model_dim))

    # 初始化位置嵌入矩阵为零矩阵, 形状为(max_len, hidden_size)
    pe = torch.zeros(max_seq_len, embed_model_dim)

    # 广播机制会将div_term扩展到(1, hidden_size/2)
    # 相乘产生的结果大小为(max_len, hidden_size/2), pe的大小为(max_len, hidden_size)
    # 偶数列使用正弦函数
    pe[:, 0::2] = torch.sin(position * div_term)
    # 奇数列使用余弦函数
    pe[:, 1::2] = torch.cos(position * div_term)

    # 将位置编码信息注册为buffer, 模型训练的时候不会更新
    self.register_buffer("pe", pe)
  
  def forward(self, x):
    # x的形状为(batch_size, seq_len, hidden_size)
    # 由seq_len可能发生改变, 有可能seq_len=4, 而我们的self.pe已经预先生成了长度为5000的序列的所有token的嵌入向量了, 所以需要对self.pe进行截取, 只取前1-seq_len个token的位置嵌入向量
    seq_len = x.size(1)
    
    # 将位置编码加入到向量上
    # self.pe[:seq_len, :]的形状为(seq_len, hidden_size), 意思是前面的seq_len个token的位置向量
    # unsqueeze(0)使其形状变为(1, seq_len, hidden_size), 便于与输入tensor相加, 这个过程中有广播
    x = x + self.pe[:seq_len, :].unsqueeze(0)

    return x
```

???+ note "`unsqueeze()`的作用"

    `unsqueeze()`函数的作用是在指定位置上给tensor增加一个维度, 具体来说, 它可以将一个形状为`[N, ...]`的tensor扩展为一个形状为`[N, 1, ...]`的tensor.

    在上述代码中, 我们需要对位置进行`unsqueeze`, 这是为后面的广播做准备, 也就是`position * div_term`. 由于`position`经过`unsqueeze`之后, 形状为`(max_len, 1)`, 而`div_term`的形状为`(hidden_size/2, )`. 
    
    进行相乘操作的同时会进行广播, `div_term`在广播时, 其初始形状被视为`(1, hidden_size/2)`(从右到左匹配, 见[这里](https://py.ricolxwz.de/numpy/broadcast/#%E5%B9%BF%E6%92%AD%E8%A7%84%E5%88%99)), `(max_len, 1)`和`(1, hidden_size/2)`相乘后最后得到的`pe[:, 0::2]`/`pe[:, 1::2]`的大小为`(max_len, hidden_size/2)`.

???+ note "`register_buffer()`的作用"

    `register_buffer()`用于注册持久缓冲区. 主要用于将一个tensor作为模型的一部分进行注册, 但该tensor不会被视为可学习的参数. 使用这种方式注册的tensor会在模型保存和加载的时候一同保存, 这意味着即使该tensor不参与梯度更新, 所以它们通常用于存储一些固定的数据, 如位置编码. 当模型从一个设备移动到另一个设备的时候, 注册的缓冲区也会跟着移动, 这简化了在不同硬件上的部署.
    
???+ note "`self.pe[:seq_len, :]`的动机"

    由于位置嵌入矩阵一般是不会改变的, 所以一般的做法是一次性生成一个很长序列的位置嵌入矩阵, 比如长度为5000序列的位置嵌入矩阵, 我们可以通过截取前`seq_len`的方式获取到第`1`个到第`seq_len`个token的位置嵌入向量, 而不用再去计算一遍位置嵌入矩阵. 也就是说, `PositionalEmbedding`构造器的输入`max_len`可以搞一个很长的值, 可以远大于需求. 当然, 也要考虑实际内存容量. 这个和实际序列的最长长度没有关系, 比如说, 实际序列长度是4, 6, 2, 7, 最长长度是7, 但是这里的`max_len`可以是6000, 一下子生成所有token的位置嵌入向量.

[^1]: 苦行僧. (2021, 七月 28). 基于Pytorch的torch.nn.embedding()实现词嵌入层. Csdn. https://blog.csdn.net/weixin_43646592/article/details/119180298
[^2]: 不当菜鸡的程序媛. (不详). Pytorch的默认初始化分布 nn.Embedding.weight初始化分布. Csdn. 取读于 2024年12月18日, 从 https://blog.csdn.net/vivi_cin/article/details/135564011
[^3]: He W. (2024, 七月 8). 手撕经典算法 #3 Transformer篇. Hwcoder - Life Oriented Programming. https://hwcoder.top/Manual-Coding-3