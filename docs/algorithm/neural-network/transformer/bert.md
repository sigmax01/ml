---
title: BERT
comments: true
---

## 背景

### NLP领域中的两大类任务

NLP领域中的两大类任务包括: 句子级任务和词级任务.

- **句子级任务:** 这类任务的重点是从整体上分析句子的含义, 处理句子之间的关系, 研究它们的逻辑或者语义关联. 如**自然语言推理**(Natural Language Inference, NLI), 目标是判断两个句子之间的关系, 这种关系通常包含三种类型: 蕴含(Entailment, 一个句子的含义可以从另一个句子中推导出来); 矛盾(Contradiction, 两个句子互相矛盾); 中立(Neutral, 两个句子没有明显的逻辑关联)
    
- **词级任务:** 这类任务的重点是对文本中的具体单词或者短句进行精细化处理, 输出粒度较细, 关注单个词或者词组的特定意义或者功能. 如**命名实体识别**(Named Entity Recognition, NER), 目标是识别文本中的特定实体, 并对其进行分类. 再如**问答抽取**, 目标是从一段文本中抽取具体的问题和答案
    
### 预训练模型用于下游任务的方法

主要有两种将预训练模型用于下游任务的方法: 一种是feature-based, 另一种是fine-tuning based.

- **模型架构:** feature-based方法通过生成的预训练特征和下游任务的特定架构结合, 因此需要针对每个下游任务设计和调整任务模型. fine-tuning方法通常不需要重新设计任务模型, 只需要在预训练模型的顶部添加一层简单的输出层(如分类层), 训练的时候直接调整整个模型的参数
    
- **训练方式:** feature-based的预训练模型是”冻结”的, 不能更新参数; 而fine-tuning方法会对所有参数进行更新, 因此需要更多的计算资源和训练数据
    
- **性能表现:** fine-tuning通常在性能上更优, 因为它可以通过微调参数更好地适应下游任务的需求; feature-based在资源有限或者任务数量较小的情况下表现较好, 因为它不需要大规模调整预训练模型
    
- **适用场景:** feature-based更适合小型数据集或者对计算资源要求较低的场景, 特别是在需要快速实验多个任务的时候. fine-tuning更加适合需要高性能的引用, 特别是在有足够数据和计算资源的情况下
    
总的来说, feature-based中预训练模型作为”特征提取器”, 固定不动, 适合资源受限的场景, 任务架构设计灵活但是工程复杂. fine-tuning通过微调让模型适应任务需求, 性能更好, 但是需要更多的资源支持, 任务架构改动少, 训练流程简单.  

### GPT等模型的问题

GPT等模型的问题就是它是单方向的, 只能看到前文, 但是无法利用后文. 在句子级的任务中, 由于某些任务可能对全局的信息依赖较小, 但是仍可能会比双向模型如BERT表现略逊一筹. 在词级任务中, 这种单向信息会对任务造成严重的负面影响, 远比句子级任务中更加明显, 因为这些任务通常需要综合考虑目标词前后的信息, 而不仅仅是词的前文.

## 架构

### MLM

MLM, Masked Language Model是BERT中采用的核心预训练技术之一, 即”掩码语言模型”. 它的原理是在输入序列中, 随机选择一部分词(通常是15%)进行替换, 模型的目标是预测被掩码词汇的原始词. MLM利用句子的双向上下文信息进行预测, 包括前文信息和后文信息. 优化目标是使用交叉熵损失, 最大化模型预测掩码词的概率.

### NSP

NSP, Next Sentence Prediction描述了BERT预训练过程使用的一项额外任务, 称为”下一句预测任务”, 它是BERT的预训练目标之一, 用于增强模型对句子间关系的理解能力, 特别是针对涉及两段文本的任务.

### 计算参数量

在Attention is all you need论文中, 可以找到上面的这一段话, 那么对于BERT Large来说, 其每个头中可学习矩阵的参数数量是3\*H\*H/A, 所有头的可学习的矩阵的参数数量为3\*H\*H/A\*A=3\*H^2, 那么, 还有一个将所有头concateneate在一起的矩阵的参数量是H^2, 所以attention层的总参数量为4\*H^2.

对于全连接层, 第一层的输入是H, 但是输出是4H. 第二层的输入是4H, 但是输出是H. 所以全连接层的可学习的参数量为8\*H^2.

对于嵌入层, 可以学习的参数是30000\*H(因为它的字典大小是30000).

所以一个transformer块中的可学习参数是30000\*H+12\*H^2L, 那么总共有L个transformer块, 所以总共的可学习参数为

### 输入表示方法

为了让BERT能够处理多种不同下游任务(如情感分析, 问答系统, 文本分类等), BERT的输入表示方式被设计成能够清楚地表示单个句子或者两个句子的组合. 注意, 这个”句子”并不局限于语法上的句子, 而是可以是任意连续的文本片段, 输入可以是一个完整句子, 一个段落, 甚至是一个短语. 序列是指输入到BERT模型中的一串标记, 这串标记可以代表单个句子, 也可以是由两个句子组合成的一段连续文本.

### WordPiece

BERT使用的是WordPiece作为词嵌入方法. 模型输入的是子序列, 如对于单词unbelievable, 自注意力层输入的是##believ, ##able的词向量, 而不是整个单词的词向量.

它的主要步骤有:

1. 分词阶段: 每个单词被分割为若干个子词(子序列), 这些子词通过词表映射为唯一的索引(token IDs), 供模型使用
    
2. 嵌入阶段: 每个子词都有对饮过的词向量, 这是一种低维稠密向量表示. 如, 对于例子un, ##believ, ##able, 子词un的词向量为E_un. 子词##believ的词向量为E_##believ, 子词##able的词向量为E_##able. 这些词向量会加上位置嵌入和分段嵌入, 共同作为模型的输入
    
3. 子词输入到transformer层: 子词级别的词向量分别被输入到transformer的编码器层中进行处理, transformer层会基于上下文关系, 对每个子词表示进行动态调整

所以, 只要使用30000个子序列的词典就能得到比较好的效果, 不用特别大的词典(因为很多片段都是在词语中重复出现的). 并且它还能解决稀有词问题, 如果一个单词在词表中不存在, wordpiece会将其分解为更小的, 常见的子词, 直到找到匹配的单位.

### [cls]的设计

普通的token的最终表示依然服务于它自己的语义, [cls]的最终表示被训练为全局语义的浓缩. [cls]和普通token本质上是一样的, 但是这个无明显语义信息的符号会更加公平地融合文本中各个词的语义信息, 模型最终学会让[cls]成为全局信息的浓缩表示, 而普通token则更关注自身的语义信息.

为什么需要[cls]? 其一是它提供了统一的全局序列表示, 在自然语言处理中, 许多任务(如文本分类, 下一句预测)需要用到整个输入序列的全局语义. 如果没有[cls], 需要通过其他方式提取全局信息, 例如, 使用平均池化(对所有的token的向量取平均), 使用最大池化(对每个维度取最大值), 或者自行选择一个关键token, 但是上述的方法在实践中存在问题: 池化方式可能丢失序列的细粒度信息, 手动选择token容易引入偏差或者限制模型的能力.

### [sep]的设计

在BERT中, 许多任务涉及两个序列(句子对)的输入, 例如, 下一句预测任务需要判断两个句子之间是否有逻辑链接, 它还能有助于模型避免误将输入后的填充部分视为实际内容.

其他的方法是我们向句子A和句子B中加入不同的一个额外的嵌入向量来表示token是属于句子A还是句子B.

在BERT中, 这两种方法都用了.

### 嵌入向量

在BERT中, 需要对三种嵌入向量进行求和, 得到最终的词向量表示, 这三种嵌入向量分别是通过WordPiece得到的嵌入向量, 分句嵌入向量(Segment Embeddings)和位置嵌入向量.

### [mask]的设计

在微调的时候, 是没有[mask]是这个东西的, 这就会导致在预训练的时候和微调的时候产生了一些不同, 解决的方法是有一定的几率不替换为[mask], 保持原有的词.