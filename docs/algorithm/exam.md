## 小题考点

个人认为的小题考点(持续更新中...):

- 预处理
    - 为什么要进行预处理: not perfect, noise(distortion(human voice), spurious(outlier or mixed with non-noisy data), inconsistent(negative weight, non-existing zip code), duplicate), missing
    - 噪音如何进行处理: signal/image processing&outlier detection; use robust ml algorithm; easy to deal with inconsistent&duplicate
    - 缺失数据如何处理: ignore all examples with missing values; estimate the missing values by remaining values(nominal: replace most common in A, replace most common in A with same class; numerial: average value of nearest neighbors)
	- 什么是数据聚合: combining two or more attributes into one
    - 为什么要进行数据聚合: data reduction(same memory&computation time); change scale; stabilize data(less variable)
	- 什么是选择特征子集: the process of removing irrelevant and redundant features
	- 为什么要选取特征子集: improves accuracy; faster building; easier to interpret
    - 如何选取特征子集: brute force(try all possible pairs and see the results); embedded(e.g. decision tree, use entropy or gini); filter(based on statistical measures, e.g. mutual information, information gain; or based on correlation, e.g. relief); wrapper(use ML algorithm as the black box)
    - 如何为特征添加权重: based on domain knowledge; some algorithm, e.g. boosting can automatically add weight to features
    - 如何对连续数据进行离散化(discretization): equal width; equal frequency; clustering
    - 归一化的作用: avoid the dominance attributes with large values
    - 标准化: assume data follows Gaussian distribution, convert it to standard Gaussian distribution(average -1, standard deviation 1) 
	- 余弦相似度和皮尔逊相关系数结果的含义: consine similarity = 0, 0; corr = -1, +1, 0
- KNN
	- 复杂度分析: m training examples with n attibutes, o(mn)
	- 加权最邻近算法: closer? bigger weight; further? smaller weight
	- 特点: require normalization; not effective for high dimensional data; sensitive to k; very accurate; slow for big datasets; 

## 大题考点

个人认为的大题考点(持续更新中...):

- 预处理
    - [二进制化](/algorithm/preprocessing/#bit-transform)
    - [归一化](/algorithm/preprocessing/#normalization)
    - [距离计算](/algorithm/preprocessing/#euclidean-distance), 特别注意Hamming distance, counts the number of different bits
    - [相似系数计算](/algorithm/preprocessing/#similarity-score), 得到了相似系数之后, 可以计算简单匹配系数, 雅卡尔指数
    - [余弦相似度计算](/algorithm/preprocessing/#cosine-similarity)
    - [皮尔逊相关系数计算](/algorithm/preprocessing/#pearson-correlation-coefficient)
- 最邻近
	- [使用k-邻近算法进行预测](/algorithm/knn/#knn), 例如, 使用2-邻近算法, Euclidean Distance
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

