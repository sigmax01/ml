---
title: GPT
comments: false
---

# GPT[^1]

## 摘要

NLP包括很多任务, 如[文本蕴含](https://en.wikipedia.org/wiki/Textual_entailment, "判断一个文本片段是否能够逻辑上推导出另一个文本片段, <br>可用于信息检索, 问答系统, 自动摘要等任务"), [问答系统](https://en.wikipedia.org/wiki/Question_answering "针对用户提出的问题, 系统能够理解问题并给出准确的答案, <br>可用于搜索引擎, 智能客服, 知识库问答等任务"), [语义相似度评估](https://en.wikipedia.org/wiki/Semantic_similarity "评估两个文本在语义上的相似度, <br> 可用于信息检索, 文本聚类, 抄袭检测等任务"), [文档分类](https://en.wikipedia.org/wiki/Document_classification "将文档归类到预定义的类别中 <br>可用于垃圾邮件过滤, 新闻分类, 情感分析等任务"). 虽然大型未标注的语料库非常充足, 但是用于特定任务的已标注的文本确非常少, 导致[判别式模型](/dicts/discriminative-and-generative-model)很难在这些NLP任务中取得很好的性能. 作者展示了一种生成式预训练模型, Generative Pre-trained Model, 它通过在大量未标注的语料上训练, 学习通用的语言表示, 然后, 针对特定任务通过少量标注数据进行微调, 可以很好的完成任务. 这种预训练模型在性能上甚至优于那些专门为特定任务而设计的判别式模型, 在12项任务中的9项都打到了SOTA的表现. 在Stories Cloze Test上获得了8.9%的绝对提升, 在RACE上为5.7%, 在MultiNLI上为1.5%.

## 背景

从raw文本中直接学习的能力对减轻NLP对监督学习依赖至关重要. 许多深度学习的方法需要大量的手工标注的数据, 导致它们在很多领域应用的潜力受到限制. 在这种情况下, 对于未标记文本中的语言信息进行建模, 相比于收集更多的高质量的标注, 不失为一种可行的方案. 而且, 即使在有足够多的带标注的数据的情况下, 以无监督方式学习良好的表示也能够显著地提高性能. 迄今为止, 最令人信服的证据是, [词嵌入预训练](/algorithm/neural-network/word-embedding/#transfer-learning)的广泛应用.

[^1]: Radford, A. (2018). Improving language understanding by generative pre-training. https://www.mikecaptain.com/resources/pdf/GPT-1.pdf
