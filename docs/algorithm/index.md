---
title: 绪论
comments: true
---

## 配套课程

该库的配套课程为[COMP5318](https://www.sydney.edu.au/units/COMP5318/2024-S2C-NE-CC).

### 友情连接

- Canvas: [https://canvas.sydney.edu.au/courses/59516](https://canvas.sydney.edu.au/courses/59516)
- Ed Discussion: [https://edstem.org/au/courses/18163](https://edstem.org/au/courses/18163)
- Modules: [https://canvas.sydney.edu.au/courses/59516/modules](https://canvas.sydney.edu.au/courses/59516/modules)
- Zoom: [https://uni-sydney.zoom.us/j/81160566160](https://uni-sydney.zoom.us/j/81160566160)

### 上课地点

- 讲座: 线上
- 实践: [J12.01.114.The School of Information Technologies.SIT Computer Lab 114](https://maps.sydney.edu.au/?room=J12.01.114)

### 上课时间

- 讲座: 2024学年第二学期星期一18:00-19:00
- 时间: 2024学年第二学期星期二20:00-21:00

### 联系方式

Nguyen Tran, nguyen.tran@sydney.edu.au

### 分数分布

- 期末考试: 60%, 纸笔考试, 2小时
- 大作业1: 15%, 写一个程序解决特定问题并汇报结果, n/a
- 大作业2: 25%, 写一个程序解决特定问题并汇报结果, n/a

### 截止日期

|作业|截止日期|完成情况|完成日期|备注|
|-|-|-|-|-|
|大作业1|第七周(9月9日)|✅|-|2人组队, 可以是不同的补习课的队友. 通过Canvas提交. 用Python写一个电脑程序解决特定的问题|
|大作业2|第十一周(10月14日)|✅||多人组队, 可以是不同的补习课的队友. 通过Canvas提交, 运用机器学习算法解决一个问题. 需要提交一个报告来讨论结果|
|期末考试|考试周||||

### 惩罚措施

大作业1和大作业2允许迟交3天, 每天会有5%的分数损失.

### 课程内容

| 周数   | 主题                                                                               |
|------|----------------------------------------------------------------------------------|
| 第一周  | 绪论, [预处理](/algorithm/preprocessing)                                              |
| 第二周  | [最邻近](/algorithm/knn)                                                            |
| 第三周  | [线性回归](/algorithm/linear-regression)                                             |
| 第四周  | [朴素贝叶斯](/algorithm/naive-bayes), [评估](/algorithm/evaluation)                     |
| 第五周  | [决策树](/algorithm/decision-tree), [集成学习](/algorithm/ensemble-learning)            |
| 第六周  | [支持向量机](/algorithm/svm), [降维](/algorithm/dimensional-reduction)                  |
| 第七周  | [前馈神经网络](/algorithm/neural-network/fnn)                                          |
| 第八周  | [卷积神经网络](/algorithm/neural-network/cnn), [递归神经网络](/algorithm/neural-network/rnn) |
| 第九周  | [Transformer](/algorithm/neural-network/transformer)                             |
| 第十周  | [聚类](/algorithm/clustering)                                                      |
| 第十一周 | [马尔可夫链](/algorithm/markov-chain)                                                 |
| 第十二周 | [强化学习](/algorithm/reinforcement-learning)                                        |

### 考试

- 时间: 2024-11-11, 2h+10min
- 形式: 闭卷, 可带经过认证的计算器
- 分数: 100, 占比50%, Hurdle
- 问题: 19个
    - 简短回答
        - 给出True/False和解释
        - 给出解释
    - 问题解决/计算题

#### 小题考点

个人认为的小题考点:

- 预处理
    - 为什么要进行预处理: not perfect, noise(distortion(human voice), spurious(outlier or mixed with non-noisy data), inconsistent(negative weight, non-existing zip code), duplicate), missing
    - 噪音如何进行处理: signal/image processing&outlier detection; use robust ml algorithm; easy to deal with inconsistent&duplicate
    - 缺失数据如何处理: ignore all examples with missing values; estimate the missing values by remaining values(nominal: replace most common in A, replace most common in A with same class; numerial: average value of nearest neighbors)
    - 为什么要进行数据聚合: data reduction(same memory&computation time); change scale; stabilize data(less variable)
    - 如何选取特征子集: brute force(try all possible pairs and see the results); embedded(e.g. decision tree, use entropy or gini); filter(based on statistical measures, e.g. mutual information, information gain; or based on correlation, e.g. relief); wrapper(use ML algorithm as the black box)
    - 如何为特征添加权重: based on domain knowledge; some algorithm, e.g. boosting can automatically add weight to features
    - 如何对连续数据进行离散化(discretization): equal width; equal frequency; clustering
    - 归一化的作用: avoid the dominance attributes with large values
    - 标准化: assume data follows Gaussian distribution, convert it to standard Gaussian distribution(average 0, standard deviation 1) 
	- 余弦相似度和皮尔逊相关系数结果的含义
- KNN
	- 

#### 大题考点

个人认为的大题考点:

- 预处理
    - [二进制化](/algorithm/preprocessing/#bit-transform)
    - [归一化](/algorithm/preprocessing/#normalization)
    - [距离计算](/algorithm/preprocessing/#euclidean-distance), 特别注意Hamming distance, counts the number of different bits
    - [相似系数计算](/algorithm/preprocessing/#similarity-score), 得到了相似系数之后, 可以计算简单匹配系数, 雅卡尔指数
    - [余弦相似度计算](/algorithm/preprocessing/#cosine-similarity)
    - [皮尔逊相关系数计算](/algorithm/preprocessing/#pearson-correlation-coefficient)
- 最邻近
	- [使用k-邻近算法进行预测](/algorithm/knn/#knn), 例如, 使用3-邻近算法, Euclidean Distance
- 朴素贝叶斯
    - [使用朴素贝叶斯算法进行预测](/algorithm/naive-bayes/#nb-algorithm)
    - [数值属性朴素贝叶斯进行预测](/algorithm/naive-bayes/#numeric-nb)
- 评估
    - [混淆矩阵计算](/algorithm/evaluation/#confusion-matrix)
- 决策树:
    - [信息熵, 信息增益的计算](/algorithm/decision-tree/#information-gain)
    - [如何选择最优属性](/algorithm/decision-tree/#how-to-choose-best-feature)
- 集成学习
    - [Bagging如何进行抽样](/algorithm/ensemble-learning/#bagging)
    - [Adaboost进行预测](/algorithm/ensemble-learning/#adaboost)
- 支持向量机
    - [核方法如何简化点积计算](/algorithm/svm/#kernel-trick)
- 降维
    - [压缩率计算](/algorithm/dimensional-reduction/#compression-rate)
- 神经网络
    - [感知机学习过程](/algorithm/neural-network/#learning-algorithm)
    - [前馈神经网络学习过程](/algorithm/neural-network/fnn/#training-procedure)
    - [反向传播算法](/algorithm/neural-network/fnn/#backpropagation-algorithm)
    - [反向传播公式推导](/algorithm/neural-network/backpropagation)
    - [卷积计算](/algorithm/neural-network/cnn/#convolutional-layer)
- 聚类
    - [K-means聚类如何分簇](/algorithm/clustering/#k-means)
    - [GMM算法如何进行分簇](/algorithm/clustering/#gmm)
    - [聚合式算法如何进行分簇](/algorithm/clustering/#agglomerative-algorithm)
    - [DBSCAN算法如何分簇](/algorithm/clustering/#dbscan)
    - [计算凝聚度/分离度](/algorithm/clustering/#conhesion-separration)
- 马尔可夫链
    - [利用马尔科夫假设进行预测](/algorithm/markov-chain/#markov-assumption)
    - [前向算法](/algorithm/markov-chain/#forward-algorithm)
    - [Viterbi算法](/algorithm/markov-chain/#viterbi)
- 强化学习
    - [Q学习算法](/algorithm/reinforcement-learning/#q-algo)
    - [深度Q学习算法](/algorithm/reinforcement-learning/#dql)

### 备注

- 讲座和补习资料会在星期六9:00发布
- 补习答案会在星期五21:00发布
- 小测验答案会在星期五20:00发布
- 在补习课上, 主要用到的是两种格式, 一种是ipynb, 另一种是pdf
- 在特定的周也可能会是理论训练
- 第一周没有补习课

## 机器学习定义

- 机器学习是不显式地编程赋予计算机能力的研究领域
- 机器学习研究的是从数据中产生模型的算法

通俗的理解就是: 根据已知的数据, 学习一个决策函数, 使其可以对未来的数据作出预测或判断.

## 人工智能与机器学习[^1]

人工智能是一个广义的领域, 是指利用技术构建出能够模仿与人类智能相关的认知功能的机器和计算机. 虽然人工智能本身通常被认为是一个系统, 但是实际上它是系统中实现的一组技术, 为的是使系统能够推理, 学习和采取行动来解决实际问题.

机器学习是人工智能的一个子集, 可让机器或者系统自动从经验中学习和改进. 机器学习不是使用显式编程, 而是使用算法来分析大量数据, 从数据洞见中学习, 然后做出明智的决策. 机器学习算法会随着训练的进行(接触越来越多的数据)不断改进以提升性能. 机器学习模型是输出,即程序通过对训练数据运行算法而学到的内容. 使用的数据越多, 获得的模型就越好.

所以说:

- 人工智能是一个更宽泛的概念,可以让机器或系统像人类一样感知, 推理, 行动或适应
- 机器学习是人工智能的一种应用, 可以让机器从数据中提取知识并自主学习

## 机器学习分类

### 理论分类

机器学习的理论重要分为三个方面:

1. 传统的机器学习: 包括线性回归, 逻辑回归, 决策树, SVM, 贝叶斯模型, 神经网络等
2. 深度学习: 是一种基于神经网络的技术, 强调通过数据的层次化表征来学习特征. 主要特点是:
    - 非/半监督式学习: 可以在没有明确标签的数据上学习特征, 减少对大量标注数据的依赖
    - 分层特征提取: 通过多层网格结构, 自底向上提取数据的高级特征, 逐层优化特征表示
    - 高校算法: 使用卷积神经网络(CNN), 递归神经网络(RNN)等结构来处理图像, 序列数据等
3. 强化学习: 强调通过与环境交互来学习最优行为策略的技术, 主要特点是:
    - 基于奖励的学习: 通过试错法, 从环境的奖励/惩罚中学习, 目标是最大化累计奖励
    - 行为主义理论: 灵感来自心理学中的行为主义理论, 模拟有机体的行为形成和调整过程
    - 无监督输入/输出对: 不需要提供像监督学习那样提供明确的输入和对应的输出对, 不需要对每一步的行为进行精确的矫正
    - 在线规划: 实时调整策略, 适应不断变化的环境

### 应用分类

- 数据挖掘: 发现数据间的关系
- 计算机视觉: 让计算机看懂世界
- 自然语言处理: 让计算机读懂文字
- 语音识别: 让机器听懂
- 机器决策: 让机器做决策

## 学习问题分类

- 监督学习: 通过输入和已知的输出来训练模型, 以便模型能够预测新的输入和输出
    - 回归问题: 输出值是一个或多个连续变量
    - 分类问题: 输出值属于有限多个类别, 是离散的
- 无监督学习: 在训练数据中没有已知的输出值, 要从数据中自行发现数据的结构, 共性等特征
    - 数据聚类: 将数据分成若干簇, 使得每个簇中的数据彼此相似
    - 异常检测: 寻找异常的数据
    - 降纬: 讲大数据集压缩成小数据集合同时丢失尽可能少的信息
- 半监督学习: 训练数据中只有一部分的数据有已知的输出值, 其余数据没有已知的输出值
- 弱监督学习: 已知的输出存在质量问题, 可能是不准确的, 不完全的或者可能存在噪音的 

???+ tip "Tip"

    - 若对于一个输入, 已经提供了正确的输出, 则称该数据带有"标签".
    - 输入又被称为"特征"
    - 输出又被称为"目标"

## 学习路径

``` mermaid
graph LR
  A[传统机器学习算法] --> B[深度学习];
  B --> C[LLM, 语言大模型];
  C --> D[VLM, 视觉和语言多模态];
```

[^1]: 人工智能与机器学习：它们有何不同？. (n.d.). Google Cloud. Retrieved June 24, 2024, from https://cloud.google.com/learn/artificial-intelligence-vs-machine-learning?hl=zh-cn
[^2]: Machine-learning-deep-learning-notes/machine-learning/machine-learning-intro.md at master · loveunk/machine-learning-deep-learning-notes. (n.d.). Retrieved June 24, 2024, from https://github.com/loveunk/machine-learning-deep-learning-notes/blob/master/machine-learning/machine-learning-intro.md