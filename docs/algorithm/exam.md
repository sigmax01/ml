## 大题考点

个人认为的大题考点(持续更新中...):

重要程度: 

- 超级重要: ☢️
- 重要: ⚠️
- 一般: ♻️
- 不重要: 🗑️
- 超级不重要: 🏴‍☠️

- 预处理
    - ♻️[二进制化](/algorithm/preprocessing/#bit-transform)
    - 🗑️[归一化](/algorithm/preprocessing/#normalization)
    - ♻️[距离计算](/algorithm/preprocessing/#euclidean-distance), 特别注意Hamming distance, counts the number of different bits
    - ⚠️[相似系数计算](/algorithm/preprocessing/#similarity-score), 得到了相似系数之后, 可以计算简单匹配系数, 雅卡尔指数
    - ⚠️[余弦相似度计算](/algorithm/preprocessing/#cosine-similarity)
    - ⚠️[皮尔逊相关系数计算](/algorithm/preprocessing/#pearson-correlation-coefficient)
- 最邻近
	- ☢️[使用k-邻近算法进行预测](/algorithm/knn/#knn), 例如, 使用2-邻近算法, Euclidean Distance
- 朴素贝叶斯
    - ☢️[使用朴素贝叶斯算法进行预测](/algorithm/naive-bayes/#nb-algorithm)
    - ⚠️[数值属性朴素贝叶斯进行预测](/algorithm/naive-bayes/#numeric-nb)
- 评估
    - ♻️[混淆矩阵计算](/algorithm/evaluation/#confusion-matrix)
    - ⚠️[计算准度的方法](/algorithm/evaluation), 包括stratification, repeated hold out, cross validation, grid search, leave out这些有啥含义, 为啥要用
- 决策树:
    - ☢️[信息熵, 信息增益的计算](/algorithm/decision-tree/#information-gain)
    - ☢️[如何选择最优属性](/algorithm/decision-tree/#how-to-choose-best-feature)
- 集成学习
    - ♻️[如何计算集成学习模型的错误率](/algorithm/ensemble-learning/#why-ensemble-learning)
    - ⚠️[Bagging如何进行抽样](/algorithm/ensemble-learning/#bagging)
    - ♻️[随机森林思想](/algorithm/ensemble-learning/#random-forest), 和Bagging差不多, 多了一个特征, 换汤不换药
    - ☢️[Adaboost进行预测](/algorithm/ensemble-learning/#adaboost), 能够计算错误率, 基分类器的权重, 归一化后/前的样本集权重
- 支持向量机
    - ♻️[给定决策边界, 计算边际距离](/algorithm/svm/#maximize-lagrange-function), 套公式
    - ⚠️[给出拉格朗日乘数和支持向量计算决策边界](/algorithm/svm/#maximize-lagrange-function), 简单的套一下公式
    - ♻️[核方法如何简化点积计算](/algorithm/svm/#kernel-trick)
- 降维
    - ⚠️[计算最佳主成分数量](/algorithm/dimensional-reduction/#确定PC的数量), 两种方法, 一种minimum percentage, 另一种elbow method(看图说话)
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