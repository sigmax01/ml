---
title: 考点
comments: true
---

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
    - ⚠️[如何通过奇异值分解确定主成分](/algorithm/dimensional-reduction/#get-pc), 看里面的例子, 取奇异值矩阵对角线的左上角部分对应的主成分
    - ☢️[压缩率计算](/algorithm/dimensional-reduction/#compression-rate)
- 神经网络
    - [感知机学习过程](/algorithm/neural-network/#learning-algorithm)
    - [前馈神经网络学习过程](/algorithm/neural-network/fnn/#training-procedure)
    - [反向传播算法](/algorithm/neural-network/fnn/#backpropagation-algorithm)
    - [反向传播公式推导](/algorithm/neural-network/backpropagation)
    - [卷积计算](/algorithm/neural-network/cnn/#convolutional-layer)
- 聚类
    - ♻️[给出两个簇中所有点的坐标, 计算簇的距离](/algorithm/clustering/#簇的距离), 考虑single link, complete link, averge link
    - ☢️[K-means聚类如何分簇](/algorithm/clustering/#k-means)
    - ⚠️[选取初始质心的方法](/algorithm/clustering/#质心选取), 无非三种方法, 选择离当前质心最远的点, 随机选取点但有最小SSE, 使用K-means++算法, 同时也可以解决空簇问题
    - ♻️[GMM算法如何进行分簇](/algorithm/clustering/#gmm), 要知道正态分布的概率密度函数$f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$, 然后才能计算每个点属于特定正态分布的概率
    - ♻️[聚合式算法如何进行分簇](/algorithm/clustering/#agglomerative-algorithm), 首先每个点当作一个簇, 然后慢慢合并距离最近的两个簇, 直到变成一整个簇, 看一下例子就行
    - ⚠️[DBSCAN算法如何分簇](/algorithm/clustering/#dbscan), 把core point, border point, noise point, MinPts, Eps的理念搞搞懂.
    - ⚠️[如何选择Eps和MinPts](/algorithm/clustering/#select-eps-minpts), 和elbow method很像, 画出一个点到$k$个最近邻居的距离图
    - ☢️[计算凝聚度/分离度](/algorithm/clustering/#conhesion-separration), 很简单, 但是很重要, 看清楚是不是平方距离 
    - ⚠️[计算轮廓系数](/algorithm/clustering/#sihouette-coefficient), 对于一个点, 一个簇, 整个聚类, 计算轮廓系数有不同, 越大越好
    - ♻️[相似度矩阵是啥](/algorithm/clustering/#correlation-similarity-matrix)
- 马尔可夫链
    - ☢️[利用马尔科夫假设进行预测](/algorithm/markov-chain/#markov-assumption), 搞清楚三种概率, initial probability, transition probability, emission probability, 然后计算状态序列的概率
    - ♻️[HMM的两个假设](/algorithm/markov-chain/#hmm-assumptions): 齐次假设和观测独立性假设
    - ♻️[HMM的三个问题](/algorithm/markov-chain/#hmm-problems), 重点关注预测问题和概率计算问题
    - ⚠️[没有前向算法的时候如何预测观测序列的概率](/algorithm/markov-chain/#evaluation-problem)
    - ☢️[前向算法](/algorithm/markov-chain/#forward-algorithm), 初始值是$A_0(k)e_k(x_1)$, 前向概率是$e_k(x_i)\sum_j f_j(i-1)a_{jk}$, 最终对所有状态前向概率求和
    - ☢️[Viterbi算法](/algorithm/markov-chain/#viterbi), 初始值是$A_0(k)e_k(x_1)$, 前向概率是$e_k(x_i)max_j V_j(i-1)a_{jk}$, 最大的Viterbi得分是最终状态, 然后通过回溯指针找到前面所有的状态
- 强化学习
    - [Q学习算法](/algorithm/reinforcement-learning/#q-algo)
    - [深度Q学习算法](/algorithm/reinforcement-learning/#dql)

## 其他

如何评估一个模型, 从以下几点出发:

- 能否有效可视化?
- 是否具有良好的可解释性?
- 能否很好的处理过拟合?
- 训练时间是否长? 消耗的计算资源是否多?
- 是否需要很大的数据集?
- 是否能够/需要降维?
- 能够处理不同大小/形状/密度的簇?