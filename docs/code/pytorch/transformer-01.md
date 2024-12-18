---
title: Transformer-01
comments: false
---

# [Transformer-01](https://colab.research.google.com/drive/1MoxcUOQKAhBoSpJNsB-8NifN48C5VmAz?usp=sharing)

## 词嵌入

首先, 我们需要将在输入序列中的每个词转换为词向量. 假设每个嵌入向量的大小是512维, 假设词典的大小是100, 那么嵌入矩阵的大小就是512\*100. 这个矩阵能够通过线性变换将one-hot编码的词汇转化为对应的512维词嵌入. 假设我们批次的大小是32, 也就是说一个批次有32个序列, 单个序列中token的数量是10, 那么一个批次的维度是32\*10\*512.

```py
class EmbeddingLayer(nn.Module):
  def __init__(self, vocab_size, embed_dim):
    super().__init__()
    self.embed = nn.Embedding(vocab_size, embed_dim)
  def forward(self, x):
    out = self.embed(x)
    return out
```

???+ tip "词嵌入模型的训练"

    在上述代码中, 我们并没有对这个模型展开训练, 要使嵌入层可以训练, 需要创建一个完整的模型, 该模型需要具有明确的优化目标和损失函数. 常见的方法是CBOW, Skip-gram, GloVe.

???+ note "`nn.Embedding()`的用法"

    `nn.Embedding`的第一个参数表示的是词表的大小, 第二个参数表示的是词向量的维度, 其实还有第三个变量`padding_idx`表示的是如果词汇表产生的编码是`padding_idx`的话, 那么产生的词向量为全0. 例如, `padding_idx`被设置为0, 在词表中有一个词汇`<PAD>`的编码是0, 则句子中的所有`<PAD>`的词向量都对应为一个全0的词向量. 注意, 这里`<PAD>`的作用是用于补齐句子的长度[^1].

    `nn.Embedding`这个类有一个属性`weight`, 它是`nn.parameter.Parameter`类型的, 作用就是存储真正的词嵌入矩阵. 如果不给`weight`赋值, 那么会自动对其进行初始化, 先会定义一个`nn.parameter.Parameter`对象, 然后对该对象调用类内部的`self.reset_parameters`方法, 这个方法会将其调整为均值为0, 方差为1的正态分布[^2]. 

???+ note "`nn.Embedding.from_pretrained()`用法"

    `nn.Embedding.from_pretrained()`这个方法允许你使用已经训练好的嵌入向量来初始化`nn.Embedding`, 可以来自CBOW, Skip-gram, GloVe等等, 而不是从随机初始化开始. 它的第一个参数是`pretrained_embeddings`, 这是一个包含预训练嵌入矩阵的2D FloatTensor, 第一个维度是词汇表的大小, 第二个维度是嵌入向量的维度. 第二个参数是`freeze`, 这是可选的, 若`freeze=True`, 则嵌入权重在训练过程中不会被更新, 如果是`freeze=False`, 则嵌入权重 ^^可以^^ 在训练中进行微调(需要自己定义损失函数训练), `padding_idx`同`nn.Embedding`.

[^1]: 苦行僧. (2021, 七月 28). 基于Pytorch的torch.nn.embedding()实现词嵌入层. Csdn. https://blog.csdn.net/weixin_43646592/article/details/119180298
[^2]: 不当菜鸡的程序媛. (不详). Pytorch的默认初始化分布 nn.Embedding.weight初始化分布. Csdn. 取读于 2024年12月18日, 从 https://blog.csdn.net/vivi_cin/article/details/135564011