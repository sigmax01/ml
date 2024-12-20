---
title: 归纳式/直推式学习
comments: true
---

考虑普通学习问题, 训练集为$\mathcal{D}=\{\mathbf{X}_{tr}, \mathbf{y}_{tr}\}$, 测试(未标记)$\mathbf{X}_{te}$, 众所周知, $\mathbf{X}_{te}$不会出现在训练集中, 这种情况就是inductive learning. 半监督学习的情况, 训练集为$\mathcal{D}=\{\mathbf{X}_{tr}, \mathbf{y}_{tr}, \mathbf{X}_{un}\}$, 测试$\mathbf{X}_{te}$, 此时, $\mathbf{X}_{un}$和$\mathbf{X}_{te}$都是未标记的, 但是测试的$\mathbf{X}_{te}$在训练的时候没有见过, 这种情况是transductive semi-supervised learning. 简单来说, transductive和inductive的区别在于我们想要预测的样本, 是不是我们在训练的时候已经见(用)过的. 通常transductive比inductive的效果要好, 因为inductive需要从训练generalize到测试[^1].

相当于 课后作业里留了期中考试原题的是transductive learning, 不留的是inductive learning, 而且两个都不给答案, 所以有原题的学生成绩更好[^2].

[^1]: Charles. (2018, 十一月 11). 如何理解 inductive learning 与 transductive learning? [知乎回答]. 知乎. https://www.zhihu.com/question/68275921/answer/529156908
[^2]: 牛大宝. (2020, 四月 27). 如何理解 inductive learning 与 transductive learning? [知乎回答]. 知乎. https://www.zhihu.com/question/68275921/answer/1183048048