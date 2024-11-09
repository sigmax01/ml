---
title: 考点
comments: true
---

重要程度: 

- 超级重要: ☢️
- 重要: ⚠️
- 一般: ♻️
- 不重要: 🗑️
- 超级不重要: 🏴‍☠️

---

- [预处理](/algorithm/preprocessing)
    - ♻️[二进制化](/algorithm/preprocessing/#bit-transform)
    - 🗑️[归一化](/algorithm/preprocessing/#normalization)
    - ♻️[距离计算](/algorithm/preprocessing/#euclidean-distance), 特别注意Hamming distance, counts the number of different bits
    - ⚠️[相似系数计算](/algorithm/preprocessing/#similarity-score), 得到了相似系数之后, 可以计算简单匹配系数, 雅卡尔指数
    - ⚠️[余弦相似度计算](/algorithm/preprocessing/#cosine-similarity), 正交表示没有关系
    - ⚠️[皮尔逊相关系数计算](/algorithm/preprocessing/#pearson-correlation-coefficient), 主要掌握正态分布公式, 均值, 标准差, 协方差的计算
- [线性回归](/algorithm/linear-regression)
    - ⚠️梯度下降权重公式: $w_{i+1}=w_i-a_i\frac{df}{dw}(w_i)$, 对于普通梯度下降, 小批量梯度下降, 随机梯度下降, 计算梯度的方法有所不同, 用于计算梯度的数据点分别为: 全部, 随机选一小部分, 随机选一个, 梯度计算公式分别为$\sum_{j=1}^n(w_i^Tx^{(j)}-y^{(j)})x^{(j)}$, $\sum_{j\in B_j}^n(w_i^Tx^{(j)}-y^{(j)})x^{(j)}$, $(w_i^Tx^{(j)}-y^{(j)})x^{(j)}$, 在计算资源需求和更快收敛之间权衡
    - ⚠️批量梯度下降复杂度分析: 其闭式解为$w=(x^Tx)^{-1}x^Ty$, $x^Tx$的复杂度是$nk^2$, 转置矩阵的复杂度为$k^3$, 所以时间复杂度是$O(nk^2+k^3)$, 空间复杂度是$O(nk+k^2)$
    - ♻️选择学习率大小: $\alpha_i=\frac{\alpha}{n\sqrt{i}}$. 注意这个学习率是每一步都会变化的, 越到后面越小. 学习率太小converge too slowly; 太大can diverge
    - ⚠️[正则化](/algorithm/linear-regression/#regularization), 重点关注L2, 其误差函数中的$\alpha$控制的是模型的复杂度, 防止过拟合, L1和L2的差别是L1会让某些特征直接消失; 还有注意一下岭回归的的损失函数, 第二项用于控制模型的复杂度, 较大的$\lambda$倾向于让参数减小, 降低复杂度
    - ♻️️[似然函数](/algorithm/linear-regression/#likelihood-function)表示的是属于1或者0的概率, $p(y_i|x_i; w)=\sigma(w^Tx_i)^{y_i}[1-\sigma(w^Tx)]^{1-y_i}$, 对于整个训练集的点都计算似然函数, 然后取对数, 得到对数似然函数, 越大越好; 对对数似然函数取反得到交叉熵误差函数, 越小越好.
    - ⚠️[给出逻辑函数, 样本点, 权重, 计算它属于哪一个类](/algorithm/linear-regression/#线性分类), 如给出分水岭是0.5, 给出权重向量, 和一些样本点, 计算这些样本点属于哪一个类别
	- ♻️[对数损失似然损失函数](/algorithm/linear-regression/#对数似然损失函数), 注意那里的$p$是是否等于$1$的概率
- [最邻近](/algorithm/knn)
	- ☢️[使用k-邻近算法进行预测](/algorithm/knn/#knn), 例如, 使用2-邻近算法, Euclidean Distance
- [朴素贝叶斯](/algorithm/naive-bayes)
    - ☢️[使用朴素贝叶斯算法进行预测](/algorithm/naive-bayes/#nb-algorithm)
    - ⚠️[数值属性朴素贝叶斯进行预测](/algorithm/naive-bayes/#numeric-nb), 注意, 计算条件概率的时候, 要计算该条件下的均值和标准差
    - ⚠️[处理缺失值问题](/algorithm/naive-bayes/#missing-values), 新样本中某些属性缺失, 不要在计算p(E|yes)**和**计算p(E|no)的时候包括那个缺失值的属性, 如没有outlook则不要包含$p(outlook|yes)$和$p(outlook|no)$; 表中的某些属性值缺失, 则不要将缺失值纳入计数, 如在yes下, outlook列中有一个缺失值, 则直接跳过, 不用管.
    - ♻️[处理零频问题](/algorithm/naive-bayes/#zero-frequency), 使用拉普拉斯, $P(E_i|yes)=(count(E_i)+1)/(count(yes)+m)$, 零频的时候$count(E_i)$应该是等于$0$的, $m$是属性可能取值的数量
- [评估](/algorithm/evaluation)
    - ♻️[混淆矩阵计算](/algorithm/evaluation/#confusion-matrix), T字开头代表正确预测, P字打底代表结果是1, 准度是(TP+TN)/(TP+TN+FP+FN), 注意它本身不是性能衡量指标, 只是一个工具
    - ⚠️[计算准度的方法](/algorithm/evaluation), 包括stratification, repeated hold out, cross validation, grid search, leave one out这些有啥含义, 为啥要用
- [决策树](/algorithm/decision-tree)
    - ☢️[信息熵, 信息增益的计算](/algorithm/decision-tree/#information-gain), 特别注意, fx-82系列计算器上没有$log_2$, 需要使用换底公式$log_2(x)=\frac{\log(x)}{\log(2)}$, 主要就是一个熵和一个条件熵的计算
    - ☢️[如何选择最优属性](/algorithm/decision-tree/#how-to-choose-best-feature), 熵越小, 纯度越高, 选择的应该是信息增益最大的属性
- [集成学习](/algorithm/ensemble-learning)
    - ♻️[如何计算集成学习模型的错误率](/algorithm/ensemble-learning/#why-ensemble-learning), 大概会有$1-(1-1/n)^m$的样本会被抽样到新的训练集中
    - ⚠️[Bagging如何进行抽样](/algorithm/ensemble-learning/#bagging)
    - ♻️[随机森林思想](/algorithm/ensemble-learning/#random-forest), 和Bagging差不多, 多了一个特征, 换汤不换药
    - ☢️[Adaboost进行预测](/algorithm/ensemble-learning/#adaboost), 能够计算错误率(初始错误率为$1/n$), 基分类器的权重, 归一化后/前的样本集权重

        ???+ danger "特别注意"

            课件上的例子中计算基分类器权重的时候使用的是以$10$为底的对数, 这是**错误**的, 应该是自然对数, 因为对于$\beta_t$求导的时候产生的结果中是自然对数. 相关[链接](https://www.digitalocean.com/community/tutorials/adaboost-optimizer).

- [支持向量机](/algorithm/svm)
    - ♻️[给定决策边界, 计算边际距离](/algorithm/svm/#maximize-lagrange-function), 套公式
    - ⚠️理解拉格朗日函数: 拉格朗日函数结合了约束条件$y_i(\boldsymbol{w}\cdot \boldsymbol{x_i}+b)\geq 1$和目标$\frac{1}{2}||\boldsymbol{w}||^2$, 最小化目标就是最大化拉格朗日函数, $max \{L(\boldsymbol{w}, b, \boldsymbol{\lambda})\}=max \sum_{i=1}^N\lambda_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_j\boldsymbol{x_i}\cdot\boldsymbol{x_j}$, 其中, 最后有两个点积, 第一个是两个训练向量的类的点击, 后面是两个训练向量特征的点积
    - ☢️[给出拉格朗日乘数和支持向量计算决策边界](/algorithm/svm/#maximize-lagrange-function), 系数矩阵的解为$\boldsymbol{w}=\sum_{i=1}^N\lambda_iy_i\boldsymbol{x_i}$
    - ⚠️软边界和硬边界: 约束条件变为$y_i(\boldsymbol{w}\cdot \boldsymbol{x_i}+b)\geq 1-\xi_i$, 目标函数变为$\frac{1}{2}||\bm{w}||^2+C\sum \xi_i$, $C$越大, 说明对于点分类越严格
    - ♻️[核方法如何简化点积计算](/algorithm/svm/#kernel-trick), 相当于上面拉格朗日方程中的$\bm{x_i}\cdot\bm{x_j}$的计算会大大简化
- [降维](/algorithm/dimensional-reduction)
    - ⚠️[计算最佳主成分数量](/algorithm/dimensional-reduction/#确定PC的数量), 两种方法, 一种minimum percentage, 另一种elbow method(看图说话)
    - ⚠️[如何通过奇异值分解确定主成分](/algorithm/dimensional-reduction/#get-pc), 看里面的例子, 取奇异值矩阵对角线的左上角部分对应的主成分
    - ☢️[压缩率计算](/algorithm/dimensional-reduction/#compression-rate), 原来$n\times m$的矩阵经过奇异值分解变成三个矩阵, 大小分别是$n\times k$, $k\times k$, $m\times k$, 所以新新占有空间为$k(1+m+n)$, 压缩率为$k(1+m+n)/(n\times m)$, $k$就是主成分的数量, 对于样本数量非常大的情况下, $m+1$可以忽略不计, 变为$k/m$
- 神经网络
    - [感知机学习过程](/algorithm/neural-network/#learning-algorithm)
    - [前馈神经网络学习过程](/algorithm/neural-network/fnn/#training-procedure)
    - [反向传播算法](/algorithm/neural-network/fnn/#backpropagation-algorithm)
    - [反向传播公式推导](/algorithm/neural-network/backpropagation)
    - [卷积计算](/algorithm/neural-network/cnn/#convolutional-layer)
- [聚类](/algorithm/clustering)
    - ♻️[给出两个簇中所有点的坐标, 计算簇的距离](/algorithm/clustering/#簇的距离), 考虑single link, complete link, averge link, 分别是距离最小, 最大, 平均
    - ☢️[K-means聚类如何分簇](/algorithm/clustering/#k-means)
    - ⚠️[选取初始质心的方法](/algorithm/clustering/#质心选取), 无非三种方法, 选择离当前质心最远的点, 随机选取点但有最小SSE, 使用K-means++算法
    - ⚠️解决空簇问题: 使用上面选择初始质心的方法, 除了SSE. 选择SEE较高的簇, 从中选择一个起始点
    - ⚠️解决离群问题: a. 在聚类开始之前移除outliers; b. 在聚类之后移除对SSE贡献异常大的点(优先)
    - ♻️[GMM算法如何进行分簇](/algorithm/clustering/#gmm), 要知道正态分布的概率密度函数$f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$, 然后才能计算每个点属于特定正态分布的概率, 注意每个分布的权重, 点属于某个分布的概率为$p(distribution\ j|x_i, \theta)=\frac{w_jp(x_i|\theta_j)}{w_1p(x_i|\theta_1)+w_2p(x_i|\theta_2)}$, 然后新的权重为$\mu_1=\sum_{i=1}^n x_i\frac{p(distribution\ 1|x_i, \theta)}{\sum_{i=1}^n p(distribution\ 1|x_i, \theta)}$
    - ♻️[聚合式算法如何进行分簇](/algorithm/clustering/#agglomerative-algorithm), 首先每个点当作一个簇, 然后慢慢合并距离最近的两个簇, 直到变成一整个簇, 看一下例子就行, 注意一下那个distance matrix
    - ☢️[DBSCAN算法如何分簇](/algorithm/clustering/#dbscan), 把core point, border point, noise point, MinPts, Eps的理念搞搞懂, 特别注意, MinPts是包括自己的; a. 任何两个核心点, 若在各自对方的Eps内, 属于同一个簇; b. 任何边界点放在与其相关联的核心点所属的簇中; c. 噪声点抛弃
    - ⚠️[如何选择Eps和MinPts](/algorithm/clustering/#select-eps-minpts), 和elbow method很像, 画出一个点到$k$个最近邻居的距离图
    - ☢️[计算凝聚度/分离度](/algorithm/clustering/#conhesion-separration), 很简单, 但是很重要, 看清楚是不是平方距离 
    - ♻️[计算轮廓系数](/algorithm/clustering/#sihouette-coefficient), 对于一个点, 一个簇, 整个聚类, 计算轮廓系数有不同, 越接近1越好, ai表示的是凝聚度, bi表示的分离度, ai越小越好, bi越大越好, $s_i=\frac{b_i-a_i}{max(a_i, b_i)}$
    - ♻️[相似度矩阵是啥](/algorithm/clustering/#correlation-similarity-matrix)
    - ⚠️如何选择簇的数量: 选择SSE的拐点对应的簇的数量, 选择轮廓系数的最大值对应的簇数量 
- [马尔可夫链](/algorithm/markov-chain)
    - ☢️[利用马尔科夫假设进行预测](/algorithm/markov-chain/#markov-assumption), 搞清楚三种概率, initial probability, transition probability, emission probability, 然后计算状态序列的概率
    - ♻️[HMM的两个假设](/algorithm/markov-chain/#hmm-assumptions): 齐次假设和观测独立性假设
    - ♻️[HMM的三个问题](/algorithm/markov-chain/#hmm-problems), 重点关注预测问题和概率计算问题
    - ⚠️[没有前向算法的时候如何预测观测序列的概率](/algorithm/markov-chain/#evaluation-problem)
    - ☢️[前向算法](/algorithm/markov-chain/#forward-algorithm), 初始值是$A_0(k)e_k(x_1)$, 前向概率是$e_k(x_i)\sum_j f_j(i-1)a_{jk}$, 最终对所有状态前向概率求和
    - ☢️[Viterbi算法](/algorithm/markov-chain/#viterbi), 初始值是$A_0(k)e_k(x_1)$, 前向概率是$e_k(x_i)max_j V_j(i-1)a_{jk}$, 最大的Viterbi得分是最终状态, 然后通过回溯指针找到前面所有的状态
- [强化学习](/algorithm/reinforcement-learning)
    - ♻️智能体: 在每一个时间步, 智能体接受状态和奖励, 执行动作. 环境接受动作, 更新状态, 发出新的奖励
    - ♻️[折扣因子是啥](/algorithm/reinforcement-learning/#mdp), 表示对未来的重视程度, 较高的$\gamma$对应更关注长期回报
    - ⚠️[两种价值函数, Bellman方程, 最优Q值函数](/algorithm/reinforcement-learning/#value-function), state-value函数衡量的是当前的状态在遵循Policy的预期累积回报; action-value函数是在当前状态下采取行动后遵循Policy的预期累积回报
    - ⚠️最优Q函数和Bellman方程: 最优Q函数就是在所有可能的Policy中, 最大的预期累积回报. 最优Q函数遵循Bellman方程, Q*等于即时回报和将来回报乘以折扣因子, $Q^*(s, a) = \mathbb{E}_{s' \sim \mathcal{E}} \left[ r + \gamma \max_{a'} Q^*(s', a') \,|\, s, a \right]$
    - ⚠️[Q学习算法](/algorithm/reinforcement-learning/#q-algo), 核心思想是维护一张Q表, 每次迭代都会对某个Q值进行增量更新(使用Bellman方程), 最终得到的Q表中的所有值近似收敛于最佳Q值
    - ⚠️[深度Q学习算法](/algorithm/reinforcement-learning/#dql), 这里只考虑状态是连续的, 但是动作不是连续的(因为输出的动作概率预测是离散的), 核心思想是维护一个Q网络, 使用和目标Q值之间的差作为损失函数$L = \left( r + \gamma \max_{a'} Q_w(s', a') - Q_w(s, a) \right)^2$, 输入状态, 给出所有动作的可能性. 在每轮迭代中, 利用$\epsilon$调控探索/利用选择动作, 并维护一个记忆池打破和时间的相关性, 从记忆池中均匀采样然后反向传播更新Q网络的权重. 为了防止目标Q值计算的不稳定, 我们引入一个目标Q网络, 使用目标Q网络来评估选择的动作, 而使用原始Q网络选择动作, 这个目标Q网络和原始Q网络的差异是目标Q网络在一段时间内会保持不变, 但是原始Q网络是每次迭代结束都会改变的, 损失函数变为$L=(r+\gamma Q_{\hat{w}}(s', argmax_{a'}Q_w(s', a')) - Q_w(s, a))^2$

## 其他

如何评估一个模型, 从以下几点出发:

- 能否有效可视化?
- 是否具有良好的可解释性?
- 能否很好的处理过拟合?
- 训练时间是否长? 消耗的计算资源是否多?
- 是否需要很大的数据集?
- 是否能够/需要降维?
- 能够处理不同大小/形状/密度的簇?
