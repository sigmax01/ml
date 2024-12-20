---
title: 聚类
comments: true
---

## 定义

聚类, Clustering, 是将数据对象分割为组(也叫作簇)的过程. 同一个簇内的对象特征相似, 不同簇内的对象特征不相似. 最简单的相似性测量方法是测量距离. 对象之间距离小说明相似度高, 对象之间距离大说明相似度小. 它是一种无监督学习, 也就是说输入一个未标注的训练集和想要的簇的数量$k$, 输出$k$个簇. 一个好的聚类应该具备: 高内聚性(即簇内的高度相似性); 高分离度(即簇之间的低相似性).

## 应用

聚类可以用于:

- 作为独立的工具对数据进行分簇
- 作为其他算法的构建模块, 例如, 作为[降维](/algorithm/dimensional-reduction)的预处理工具
- ...

举几个例子把...

1. 根据购买历史, 浏览历史将顾客分为不同特征的群体, 利用这些信息开发针对性的营销活动 
2. 分析客户的行为, 找到可能丢失的客户群体, 例如转移到其他医疗保险, 电力或者电话公司
3. 找到具有相似结构和功能的基因, 使用微阵列从数千个基因🧬中同时分析
4. 基于文件的内容找到相似的其他文件, 如专利查询, 个性化新闻推荐
5. 了解特定群体(如年轻的亚洲人)的饮食习惯和饮食模式
6. 聚类像素点, 然后用几何中心的颜色替代达到图像压缩的效果
7. 基于不同的颜色将图像分割为不同的部分

## 测量

### 点的距离

数据点A和B之间的相似性是通过距离测量的. 距离测量的方式有很多种. 请参考[相似性测量](/algorithm/preprocessing/#相似性测量).

### 簇的距离 {#簇的距离}

### 质心和中心

考虑一个含有$N$个点的簇$\{p_1, ..., p_N\}$. 质心, Centroid, 是簇的几何中心, 通常不是簇中的一个数据点. 而中心点, Medoid, 是簇中一个具有代表性的数据点, 其选取方式是: 找到簇中所有点到该点的距离和最小的那个点, 如[图](https://img.ricolxwz.io/e6cedeff0f7b2b51aa22fd01709d7a34.png)所示.

---

簇的距离可以通过多种方式衡量.

- 质心: 根据质心点之间的距离
- 中心: 根据中心点之间的距离
- 单链接: 根据簇A任意一个点和簇B任意一个点之间的最短距离
- 全链接: 根据簇A任意一个点和簇B任意一个点之间的最大距离
- 平均链接: 根据簇A所有点和簇B所有点之间的平均距离

## 分类

聚类算法主要分为四种.

- 划分式: Partitional, 代表算法为K-menas, K-medoids. 通过划分数据集生成一个簇的集合, 每个簇都对应数据中的一个子集
- 模型式: Model-based, 代表算法为高斯混合模型(GMM). 假设数据式由不同的概率分布生成的, 使用该模型来估计这些分布并分配数据点
- 层次式: Hierarchical, 代表算法为聚合式(Agglomerative), 分裂式(Divisive). 构建嵌套的簇结构, 可以通过层次图展示, 层次聚类逐步合并或分裂数据, 创建不同层次的簇
- 密度式: Density-based, 代表算法为DBSCAN. 基于数据点的密度进行聚类, 能够识别出形状不规则的簇, 并能够检测出噪声点

## 划分式

### K-means {#k-means}

K均值聚类, K-means式一种非常流行且广泛使用的划分式聚类算法. 它通过将数据集划分为$k$个簇来进行聚类. K-means因为其简单性和计算效率被广泛应用于各种领域的数据聚类任务. K-means算法要求用户预先定义簇的数量$k$, 这是它的一个主要限制.

算法的步骤为:

1. 选择$k$个初始质心: 选择$k$个样本作为初始质心
2. 迭代步骤
    1. 将每个样本分配到最近质心: 将每个样本分配到距离最近的质心所属的簇中, 形成$k$个簇, 这个步骤通过最小化样本和质心的距离来完成, 通常使用欧几里得距离
    2. 重新计算质心: 每次迭代后, 重新计算每个簇的质心, 注意, 不一定是实际的数据点
3. 检查停止条件: 当质心不再发生变化, 则算法终止; 否则, 重复第2步, 用新的质心重新进行样本分配和质心计算

<figure markdown='1'>
![](https://img.ricolxwz.io/fd9096289dd73f230d6b032b260d8949.png){ loading=lazy width='600' }
</figure>

???+ note "细节"

    - 最初的质心通常是随机选取的, 而且选的是实际存在的点
    - 质心的选取会对结果产生严重影响, 详情请见[这里](#质心选取)
    - 绝大多数的收敛发生在前几个轮次
    - 通常情况下, 停止的条件是"直到较少的质心发生变化"而不是"没有质心发生变化"
    - 复杂度为$O(n*k*i*d)$, 其中$n$为数据点的数量, $k$是簇的数量, $i$是迭代的次数, $d$是属性的数量

???+ example "例子"

    下表是五个数据点之间的距离.

    |   | A | B | C | D | E |
    |---|---|---|---|---|---|
    | **A** | 0 | 2 | 7 | 10| 1 |
    | **B** | 2 | 0 | 3 | 4 | 6 |
    | **C** | 7 | 3 | 0 | 5 | 9 |
    | **D** | 10| 4 | 5 | 0 | 1 |
    | **E** | 1 | 6 | 9 | 1 | 0 |

    现在, 使用K-means算法将其聚类为$2$个簇. 初始的质心为A和B. 展示第一轮之后的簇.

    - C和A之间的距离是$7$, C和B之间的距离是$3$, 所以C被分配到簇$2$
    - D和A之间的距离是$10$, D和B之间的距离是$4$, 所以D被分配到簇$2$
    - E和A之间的距离是$1$, E和B之间的距离是$6$, 所以E被分配到簇$1$

#### 质心选取 {#质心选取}

算法对于初始的质心很敏感, 不同的初始质心可能会产生不同的簇.


=== "好的初始质心"

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/713548f4d06b34edda13ffe7aaaddd66.png){ loading=lazy width='500' }
    </figure>

=== "差的初始质心"

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/c74b14d48e7e9e679ee96535450ff907.png){ loading=lazy width='500' }
    </figure>

选取初始质心的方法大致有两种:

1. 运行数次初始质心不同的K-means算法, 然后找到Sum of Squared Error(SSE)最小的初始质心. 什么是SSE? 对于每一个点来说, 它的误差是到最近质心的距离, SSE就是这些距离的总和. 也就是$SSE=\sum_{i=1}^k\sum_{x\in k_i}d(c_i, x)^2$, 其中$c_i$是质心, $k$是簇的数量, $x$是数据点.
2. 使用K-means++算法

##### K-means++ {#kmeans++}

K-means++是一种K-means算法的变种, 它使用不同的方法选取初始质心. 剩余的和标准的K-means算法是相同的. 它的初始质心选取方法为: 

质心是逐个选择的, 直到选出$k$个质心为止. 每一步中, 每个数据点都有一定的概率被选为质心, 该概率与该点到当前与其最近的质心的距离平方成正比, 这意味着离现有质心最远的点更有可能被选中, 这样能够确保质心分布良好, 互相分隔. 

该方法在实际使用中能够显著改进初始质心的选择, 从而提高K-means算法的整体效果. BTW, 在[COMP5045(Computational Geometry)](https://www.sydney.edu.au/units/COMP5045/2024-S1C-ND-CC)中, 其中第11周讲到的Approximation Algorithms使用的也是这个思想, 那里叫"K-Clustering"算法, 过程是一样的.

#### 问题

##### 空簇 {#empty-cluster}

K-means会产生空簇, 也就是没有点会被分配到一个簇中, 只包含一个初始质心. 解决方法有:

- 选取离当前质心最远的点 
- 使用[K-means++算法](#kmeans++)
- 从具有最高SSE的簇中选取一个新的质心来分裂该簇

如果有多个空簇出现, 可以重复上述步骤多次, 直至所有的簇都有点被分配.

##### 离群 {#outliers}

离群点, Outliers, 是指在数据集中明显偏离其他数据点的样本点. 它们与大多数数据点的分布有很大的差异, 可能在空间中距离其他数据点较远. 当数据集中存在离群点的时候, 计算得到的质心往往不够代表性, 并且会导致SSE增加. 尤其是在多次聚类运行过程中, 离群点的影响会比较明显.

一种常见的解决方法是在进行聚类之前移除掉离群点. 但是对于某些应用场景, 离群点非常重要, 移除它们可能导致信息丢失. 例如, 数据压缩, 金融分析. 另一种方法是在聚类完成之后再移除这些离群点: 通过跟踪每个点对SSE的贡献, 特别是对贡献异常高的数据点, 可以移除. 

#### 其他扩展算法

##### 二分K-means

二分K-means, Bisecting K-means, 是对标准K-means算法的一种扩展. 它会讲数据集分成两个簇, 然后选择其中的一个簇进一步拆分, 如此重复, 直到获得$k$个簇为止. 

1. 开始时所有的点都放在一个簇中
2. 重复以下过程
    1. 从当前的簇列表中, 选择一个簇用于拆分
    2. 对于指定的迭代次数, 使用K-means对选中的簇进行二分
    3. 将二分后的簇中SSE最低的两个簇添加到簇列表中, 注意这里的SSE最低的两个簇是因为要使用K-means算法进行多次二分尝试
3. 终止条件: 当簇的数量达到$k$个时

有多种方式可以选择要拆分的簇:

- 选择最大的簇
- 选择SSE最大的簇
- 基于大小和SSE的综合指标


如图所示, 是一个使用二分K-means算法聚类的过程.

<figure markdown='1'>
![](https://img.ricolxwz.io/2acfb8b14da20e49c79e2c88bfc5fe28.png){ loading=lazy width='500' }
</figure>

#### 缺陷

K-means算法在以球形分类, 同等大小, 分裂明显的原始数据上的表现非常好. 但是在非球形分类的, 复杂的, 大小不一致, 密度不一致的原始数据上表现不佳.

=== "非球形分类的原始数据"

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/678f1837a8e2d8c0a18d30fd615ce1a6.png){ loading=lazy width='500' }
    </figure>

=== "大小不一致的原始数据"

    K-means算法只会讲最大的原始簇分开.

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/813c5603544934b740c39a96095e238c.png){ loading=lazy width='500' }
    </figure>

=== "密度不一致的原始数据"

    K-means算法只会分裂最大的原始簇.

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/8cc3ff2b958c8b9fac726eeed41c80fa.png){ loading=lazy width='500' }
    </figure>

## 模型式

### GMM {#gmm}

高斯混合模型, Gaussian Mixture Model, 可以看作是对单一高斯分布的扩展, 它假设数据是由多个高斯分布混合而成的, 每个分布在模型中具有一定的权重. 

对于一个单一的高斯分布, 它的概率密度函数为$f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$, 其中$\mu$是均值, $\sigma$是标准差. 我们假设数据是由$k$个高斯分布混合而成的, 每一个分布都有自己的均值和标准差, 一个高斯分布对应的是一个簇. 我们并不知道这些高斯分布的均值和标准差, 我们的目标是通过观察到的数据来估计这些参数. 在每一轮的估计结束后, 我们会重新计算每个数据点属于每个簇的概率, 然后重新估计参数, 如此循环直至收敛. 算法可以表示为:

1. 初始化参数: 随机初始化每个高斯分布的均值和标准差, 以及每个高斯分布的权重
2. 重复以下步骤:
    1. E步: 计算每个数据点属于每个簇的概率, 也就是每个高斯分布的概率
    2. M步: 重新估计每个高斯分布的均值, 标准差和权重
3. 检查停止条件: 当参数不再发生变化, 或者达到最大迭代次数时, 算法终止

其中, E步指的是Expectation Step, M步指的是Maximization Step. E步相当于K-means算法中的分配数据点到簇的过程, M步相当于重新计算质心的过程.

???+ example "例子"

    假设有$20000$个数据点, 从两个高斯分布构成, 分布1和分布2.
   
    <figure markdown='1'>
    ![](https://img.ricolxwz.io/44e79b3181b5e130d898eac1dea29026.png){ loading=lazy width='500' }
    </figure>

    为了简化起见, 假设我们已经知道了它们的标准差$\sigma_1$和$\sigma_2$, $\sigma_1=\sigma_2=2$.

    1. 第一步, 初始化每个高斯分布的均值和标准差, 由于我们已经知道标准差了, 所以只需要初始化均值. 随机初始化均值为$\mu_1=-2, \mu_2=3$. 那么初始状态下两个分布的参数为$\theta_1=(-2, 2), \theta_2=(3, 2)$. 整体参数$\theta=(\theta_1, \theta_2)$
    2. E步, 计算每一个数据点属于分布$j(j=1, 2)$的概率. $p(distribution\ j|x_i, \theta)=\frac{w_jp(x_i|\theta_j)}{w_1p(x_i|\theta_1)+w_2p(x_i|\theta_2)}$. 其中, $w_j$是每一个分布的权重(也就是分布$j$产生数据点的概率), 所有的权重加起来应该等于$1$, 即$w_1+w_2=1$. 假设$w_1=w_2=0.5$, 那么$p(distribution\ j|x_i, \theta)=\frac{0.5p(x_i|\theta_j)}{0.5p(x_i|\theta_1)+0.5p(x_i|\theta_2)}$, 对于数据点$x_i=0$, 根据概率密度分布函数有$p(x_i|\theta_1)=0.12, p(x_i|\theta_2)=0.06$, 那么$p(distribution\ 1|x=0, \theta)=\frac{0.12}{0.12+0.06}=0.66, p(distribution\ 2|x=0, \theta)=\frac{0.06}{0.12+0.06}=0.33$, $20000$个数据点都要算一遍
    3. M步, 重新估计每一个高斯分布的均值. $\mu_1=\sum_{i=1}^n x_i\frac{p(distribution\ 1|x_i, \theta)}{\sum_{i=1}^n p(distribution\ 1|x_i, \theta)}$, $\mu_2=\sum_{i=1}^n x_i\frac{p(distribution\ 2|x_i, \theta)}{\sum_{i=1}^n p(distribution\ 2|x_i, \theta)}$. 可以看到, 新的均值是基于数据点的加权平均值计算的. 权重由每个数据点属于特定分布的概率决定
    4. 迭代和收敛, 重复步骤2和步骤3, 直到$\mu_1$和$\mu_2$的不再产生变化或变化非常小, 数据点最终分配给概率更高的分布

#### GMM vs. K-means

GMM其实是K-means算法的推广. K-means算法假设每个簇是圆形的, 而GMM则放宽了这个假设, 允许簇的形状为椭圆形, 因此更加灵活. 

<figure markdown='1'>
![](https://img.ricolxwz.io/9927e4cc93c42ff8a27e38e94dee29f7.png){ loading=lazy width='500' }
</figure>

## 层次式

层次式会产生一个嵌套的簇状结构, 这些簇会组成一棵具有层次的树, 这可以用树状图表示.

<figure markdown='1'>
![](https://img.ricolxwz.io/0eee368bd98d0ffe70ddd197c3261992.png){ loading=lazy width='400' }
</figure>

???+ example "例子"

    可以通过层次聚类对多个时间序列进行分组, 每条线代表一个时间序列, 图中右侧的柱状图展示了这些时间序列之间的相似性和聚类过程.

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/95e3ffe4afc2c2a24487a4a81847ea99.png){ loading=lazy width='300' }
    </figure>

### 优势

使用层次式聚类无需在开始之前就明确簇的数量, 可以通过在树状图的不同层级进行"切割"得到所需的簇数量. 

树状图提供了一种可视化过程, 帮助理解不同的数据点如何逐步合并为簇. 通过观察树状图, 有时可以发现数据的有意义的层次结构或分类规则.

它特别适合处理具有层次结构或者分类方法(如生物分类)的数据, 因为它能够反应数据中的嵌套关系.

### 缺陷

层次式聚类的计算复杂度较大, 限制了其在大规模数据集上的应用. 若$n$为样本数量, 其空间复杂度为$O(n^2)$(存储邻近矩阵和树状图); 时间复杂度是$O(n^3)$(每次迭代都要搜索一遍邻近矩阵).

其二是它不是增量的, incremental, 不像某些增量算法一样动态地处理新数据.

噪声和离群点会对聚类结果产生较大影响, 特别是对于Ward法. 但是对单链接, 全链接, 平均链接的影响较小. 离群点通常会形成单独的小簇, 在聚类过程中不会和其他簇合并, 可以通过去除小簇来解决这个问题.

### 分类

层次式聚合主要分为两种: 

- 聚合式, Agglomerative, bottom-up. 是一个自下往上构建树状图的过程. 即从各自的簇开始, 慢慢合并簇直到所有的数据点都属于一个簇
- 分裂式, Divisive, top-down. 是一个自上而下构建树状图的过程. 即将所有的数据点都放到一个簇里面, 然后将这个簇慢慢分类直到所有的数据点都在自己的簇里面

<figure markdown='1'>
![](https://img.ricolxwz.io/abdba99a1719ad8e2525efdb0673aa4a.png){ loading=lazy width='400' }
</figure>

更直观的图:

<figure markdown='1'>
![](https://img.ricolxwz.io/4accd298e44978a6fb715d72c1693c60.png){ loading=lazy width='500' }
</figure>

由于分裂式比聚合式的热度更低, 所以我们会注重讲解聚合式.

### 聚合式 {#agglomerative-algorithm}

聚合式是最热门的一种层次式聚合方案. 最关键的步骤就是计算两个簇之间的距离, 或者说是邻近矩阵, proximity matrix. 如何合并簇有不同的方法, 我们采用的方法是合并两个距离最近的簇.

1. 计算邻近矩阵
2. 将每个数据点都视为一个簇
3. 重复过程:
    1. 合并两个最近的簇
    2. 更新邻近矩阵 
4. 结束条件: 直到只有一个剩余的簇

如何测量簇的距离? 我们在[这里](#簇的距离)讲过.

???+ example "例子"

    === "0. 原始数据--->"

        给出$6$个数据点, 利用聚合式聚类, 距离计算方法是单链接, 曼哈顿距离.

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/b4d217b54c6aa11263371424fda25cd7.png){ loading=lazy width='200' }
        </figure>

    === "1. 利用曼哈顿距离计算邻近矩阵--->"

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/2a4f4cec8727f5f7bfca4a77f6a72a2e.png){ loading=lazy width='400' }
        </figure>

    === "2. 让每一个点都成为一个簇--->"

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/ac7f825898f7380755ffea8d0a303829.png){ loading=lazy width='400' }
        </figure>
    
    === "3. 合并两个最近的簇--->"

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/67073e7d64f9208896455f2d9e103819.png){ loading=lazy width='400' }
        </figure>
        
        这里, 我们合并的是B-C, E-F, 因为它们的距离是最近的, 都是$1$, 所以两个都合并.

    === "4. 更新邻近矩阵--->"

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/655d35594bf061310efc089b7b4ab50f.png){ loading=lazy width='400' }
        </figure>

        注意, 邻近矩阵是根据单链接距离更新的.

    === "5. 重复上述过程--->"

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/7ebde1191b78b3e72ea3cdaf3ed73711.png){ loading=lazy width='400' }
        </figure>

        由于A和B, C距离最小, 所以合并. 

    === "6. 重复上述过程--->"

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/c34411061761479a207ee0f8c70503c2.png){ loading=lazy width='400' }
        </figure>

        注意, ⚠️, 这个时候出现了两组距离都是最短距离的簇, 分别是A, B, C和E, F; D和E, F. 都有E, F这个簇, 那么我们到底先合并哪一组好呢? 我们可以随机选取一组, 譬如这里合并的是D和E, F这两个簇.

    === "7. 合并为一个簇"

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/6d339a579b337368bc35ed9c9275a591.png){ loading=lazy width='400' }
        </figure>

        最终, 合并为一个簇. 树状图可以表示为:

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/0f5a766f61a7e1416f00f7ccec1a3335.png){ loading=lazy width='250' }
        </figure>

### 分裂式

分裂式聚合相比于聚合是聚合更少见. 它是基于最小生成树, MST的一种算法.

1. 首先, 将数据点的距离关系表示为一个邻近图, 生成MST
2. 重复过程:
    1. 通过切断MST中的最长边来创建新簇
3. 结束条件: 直到所有的点都被分割为单独的簇

<figure markdown='1'>
![](https://img.ricolxwz.io/7853562a1ce61712453679f9547630c3.png){ loading=lazy width='500' }
</figure>

## 密度式

### DBSCAN {#dbscan}

带有噪声处理的基于密度的空间聚类算法, Density-Based Spatial Clustering of Applications with Noise, DBSCAN, 是一种基于密度的聚类方法, 特别适合处理带有噪声的复杂数据集. DBSCAN将高密度区域识别为一个簇, 并把低密度区域视为簇和簇之间的分割. 噪声点通常位于低密度区域, 被排除在簇之外.

<figure markdown='1'>
![](https://img.ricolxwz.io/63eda41b125787211ac30a0e3a6836a5.png){ loading=lazy width='300' }
</figure>

不同于K-means只能找圆形的簇, DBSCAN能找任意复杂形状的簇, 如S形, 半圆形...

<figure markdown='1'>
![](https://img.ricolxwz.io/c5306dba12bcc732151af1e8be634d85.png){ loading=lazy width='500' }
</figure>

DBSCAN的主要思想是, 如果A是一个簇, 那么它的密度应该高于一个阈值. 点A的领域是指在一个给定半径(成为Eps)的范围内, 围绕点A的区域. 密度是指在A的领域内点的数量, 包括A本身. 所以, A的密度就是在这个Eps半径内点的总数. 密度的阈值MinPts是密度的最低要求, 如果A的领域中的点数达到或者超过这个阈值, 即A的密度超过这个阈值, 则点A被认为是高密度点; 如果低于这个阈值, A可能被视为边界点或者噪声点.

一个点的密度取决于半径Eps. 如果:

- Eps太大: 所有的点都会有一个较大的密度$m$, $m$是数据集中所有的点的数量
- Eps太小: 所有的点的密度都等于$1$, 即只有一个自身

#### 三种点

DBSCAN中有三种点:

- 核心点: core point, 如果一个点的Eps领域内包含至少MinPts个点, 那么这个点被认为是核心点, 核心点通常位于密度聚类的内部
- 边界点: border point, 它虽然其Eps领域内的点数不足以达到MinPts, 但是它位于某个核心点的领域内, 边界点可能位于多个核心点的领域中
- 噪声点: noise point, 既不是核心点, 又不是边界点

若MinPts = $7$:

<figure markdown='1'>
![](https://img.ricolxwz.io/6135b5f3f955d0c60f60a27007a57275.png){ loading=lazy width='300' }
</figure>

若MinPts = $4$:

<figure markdown='1'>
![](https://img.ricolxwz.io/cae80e834b70837364d7afdbc8d07362.png){ loading=lazy width='300' }
</figure>

---

DBSCAN的算法步骤为: 

1. 将数据点标注为核心点, 边界点, 噪声点
2. 抛弃噪声点
3. 将剩余的点根据如下方式聚类:
    1. 任何两个核心点, 若各自在对方的Eps内, 则属于同一个簇
    2. 任何的边界点都放在与其相关联的核心点所属的簇中. 若边界点同时和多个核心点相关联, 需要解决冲突

???+ example "例子"

    === "例子1"

        给出原始数据(邻接矩阵).

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/714fabd7f1a6ae4d9577a2b69f32aff0.png){ loading=lazy width='300' }
        </figure>

        - Eps = $2$
        - MinPts = $3$

        1. 对于每一个点, 找到其邻居(在半径为Eps的圆内的点)

            - A ---> A, B
            - B ---> B, A, C
            - C ---> C, B
            - D ---> D, E
            - E ---> E, D

        2. 根据MinPts将所有的点做标注

            - 核心点: B
            - 边界点: A, C
            - 噪声点: D, E

        3. 找到簇: A, B, C

    === "例子2"

        给出原始数据(邻接矩阵).

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/5a288f9877c7a1f5771995e176046ca2.png){ loading=lazy width='300' }
        </figure>

        - Eps = $1$
        - MinPts = $2$

        1. 对于每一个点, 找到其邻居(在半径为Eps的圆内的点)

            - A ---> A, B
            - B ---> B, A
            - C ---> C
            - D ---> D, E
            - E ---> E, D

        2. 根据MinPts将所有的点做标注

            - 核心点: A, B, D, E
            - 边界点: 无
            - 噪声点: C

        3. 找到簇

            - A, B
            - D, E
 
#### 选取Eps和MinPts {#select-eps-minpts}

不同的Eps和MinPts可能会对结果产生很大影响.

<figure markdown='1'>
![](https://img.ricolxwz.io/baa3b7f497ca64173309b9cc5ad73a1f.png){ loading=lazy width='600' }
</figure>

- 如果Eps太大的话, 那么所有的点都会变成核心点
- 如果Eps太小的话, 那么所有的点都会变成噪声点

我们可以使用k-距离, k-dist来选取适当的Eps和MinPts.

k-距离是指从一个点到其第$k$个最近邻居的距离. 对于属于某个簇的点, k-距离会比较小; 对于不属于任何簇的点, 如噪声点, k-距离会比较大. 我们可以为数据集中的所有点计算其k-距离, 然后将这些距离按照升序排列并绘制成图表. 在图中, k-距离的突然上升表示在这个位置对应的Eps和MinPts是合适的.

<figure markdown='1'>
![](https://img.ricolxwz.io/f3724bfa6246c2d1b2b3697b0f990f9f.png){ loading=lazy width='500' }
</figure>

#### 变化密度

DBSCAN无法很好的处理密度不同的簇, 结果会取决于不同的Eps和MinPts.

<figure markdown='1'>
![](https://img.ricolxwz.io/8da85a2173051ffa91b70729aa2cc5ce.png){ loading=lazy width='600' }
</figure>

???+ example "例子"

    有四个簇, A, B, C, D. 都被噪声包围. 颜色越深, 表示密度越高. A和B周围噪音的密度和C, D的密度是一样的, 那么DBSCAN会找到多少个簇?

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/dc19153a09c1e9f59362d0d6266345be.png){ loading=lazy width='500' }
    </figure>

    === "若Eps较大"

        若Eps大到恰好能让C和D能够被分为不同的簇且其周围的点被视为噪声. 则A和B及它们周围的点会被视为一个簇.

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/454cc151e9067d5fe46759af30f8ed7b.png){ loading=lazy width='500' }
        </figure>

        你可以仔细品品, 这是怎么回事.

    === "若Eps较小"

        若Eps小到A和B被分为不同的簇, 并且它们周围的点被视为噪声. 则C和D及其周围的点会被视为噪音.

        <figure markdown='1'>
        ![](https://img.ricolxwz.io/d37cf7124df9a51c46410026df0caca0.png){ loading=lazy width='500' }
        </figure>

        你可以仔细品品, 这是怎么回事.

#### 复杂度

- 时间复杂度: $O(n^2)$, 其中, $n$是点的数量. 可以使用kd树降为$O(n\log n)$, 什么是kd树? 可以参考[COMP5045 Computational Geometry](https://www.sydney.edu.au/units/COMP5045)
- 空间复杂度: $O(n)$, 对于每个点, 都要保存其很小一部分的信息, 如它所属的簇和类型

#### 优缺点

优点:

- 可以形成任意形状和大小的簇
- 不需要实现指定簇的数量
- 对噪声具有鲁棒性

缺点:

- 不适合密度差异较大的数据
- 不适合高维数据
- 对输入参数Eps和MinPts敏感
- Eps和MinPts选择通常不是直观的, 需要通过一些启发方法

## 评估

给出$1000$个随机生成的数据点, 这些点没有任何明显的自然聚类结构, 只是纯粹的噪声数据. K-means算法和聚合式算法被设置为寻找$3$个簇, 即使数据是随机的, 没有明显的聚类结构. 

???+ tip "Tip"

    理论上聚合式聚类不需要指定簇的数量, 但是有时候, 为了方便或者对比结果, 人们会提前指定提取的簇的数量. 

这提示我们, 即使数据是完全随机的, 聚类算法仍然能够划分出簇. 因此, 在使用聚类算法的时候, 需要对结果进行评估, 以确保我们没有在噪声中"发现"不存在的模式的可能性.

<figure markdown='1'>
![](https://img.ricolxwz.io/dd3e986b080ac523658d640c12f53635.png){ loading=lazy width='400' }
</figure>

有两种方式评估聚类的质量:

- 无监督评估: 又叫做internal evaluation. 不依赖任何正确答案(标签)来评价聚类结果的质量. 理想的聚类应该是簇内相似度高, 簇间相似度低. 常见的内部评估指标包括轮廓系数(sihouette score), 簇内误差平方和(SSE)等
- 有监督评估: 又叫做external evaluation. 即某些领域已有的专家或已有的数据为每个点提供了正确的簇标签. 在这种情况下, 聚类结果会和已知的正确簇标签进行比较, 以评估聚类的准确性. 常见的外部评估指标包括调整兰德系数(Adjusted Rand Index, ARI), 互信息(MI)等

我们在这里侧重于讲解无监督评估.

### 无监督

#### 凝聚度和分离度 {#conhesion-separration}

一个好的聚类会产生高凝聚度和高分离度的簇.

<figure markdown='1'>
![](https://img.ricolxwz.io/fae31b619ceba5cd790a22a12106e818.png){ loading=lazy width='500' }
</figure>

一个簇$k_i$的凝聚度计算公式为$cohesion(k_i)=\sum_{x\in k_i} dist(x, c_i)$. 其中, $c_i$是簇的质心, $x$是簇中所有的数据点. 总的凝聚度为所有簇的凝聚度之和.

???+ example "例子"

    如簇1和簇2的凝聚度.

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/453d31029f49420825a92fe6edca5ccd.png){ loading=lazy width='500' }
    </figure>

    这里的距离采用的是[曼哈顿距离](/algorithm/preprocessing/#曼哈顿距离). 得到$cohesion(k_1)=1$, $cohesion(k_2)=1$. 总的凝聚度为两者之和$cohesion = 2$.

一个簇$k_i$和其他簇的分离度计算公式为$separation(k_i)=dist(c_i, c)$. 其中, $c$是所有点的质心, $c_i$是簇$k_i$的质心. 总的分离度是所有簇的分离度的加权和$separation=\sum_{i=1}^k|k_i|dist(c_i, c)$.

???+ example "例子"

    如簇1和簇2的分离度.

    <figure markdown='œ1'>
    ![](https://img.ricolxwz.io/60703076c2e2e08db9a3662858f251db.png){ loading=lazy width='500' }
    </figure>

    所有点的质心坐标为$3$, 计算得到$separation(k_1)=1.5$, $separation(k_2)=1.5$. 总的分离度是所有簇的加权和, 权重是簇内点的数量, $sparation=2*1.5+2*1.5=6$.

##### 平方距离

我们可以使用平方距离来表示凝聚度和分离度.

- $SSE=\sum_{i=1}^k\sum_{x\in k_i}(c_i, x)^2$
- $BSE=\sum_{i=1}^k|k_i|(c_i, c)^2$

???+ example "例子"

    计算簇1和簇2的凝聚度和分离度.

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/596183cd721dec4cf07deaeaf71709bb.png){ loading=lazy width='400' }
    </figure>

    - $SSE=(1-1.5)^2+(2-1.5)^2+(4-4.5)^2+(5-4.5)^2=1$
    - $BSE=2*(3-1.5)^2+2*(4.5-3)^2=9$

##### 轮廓系数 {#sihouette-coefficient}

轮廓系数, Sihouette Coefficient. 可以为单个点, 簇或者整个聚类结果进行计算, 它结合了凝聚度和分离度, 是一个综合的衡量指标.

对于某个点$i$:

- $a_i$: 点$i$到簇内所有其他点的平均距离, 代表凝聚度
- $b_i$: 首先找到点$i$到另一个簇中所有点的平均距离, 然后取这些平均距离的最小值

轮廓系数的计算公式为$s_i=\frac{b_i-a_i}{max(a_i, b_i)}$. $s_i$的范围是$[-1, 1]$. 当$s_i$接近$-1$的时候, 表示该点可能被错误聚类, 它与其他簇的点更接近; 当$s_i$接近$1$的时候, 表示该点聚类质量高, 具有高凝聚度和高分离度.

对于某个簇:

轮廓系数是簇内所有点的轮廓系数的平均值.

对于某个聚类:

轮廓系数是所有簇的轮廓系数的平均值.

#### 相似度矩阵的相关性 {#correlation-similarity-matrix}

使用相似度矩阵的相关性来评估聚类质量的原理是找到相似的数据点(即彼此距离较近的数据点)是否被分配到相同的簇中. 

- 第一个相似度矩阵从距离得出
- 第二个相似度矩阵从聚类结果得出

计算这两个相似度矩阵的相关性.

<figure markdown='1'>
![](https://img.ricolxwz.io/95d386981583dd623f16f7c177b8f877.png){ loading=lazy width='500' }
</figure>

???+ example "例子"

    有$4$个数据点被分到了两个簇, p1, p2和p3, p4.

    其相似度矩阵为:

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/4ab78bb8e74fec45c35227a40a4876fb.png){ loading=lazy width='200' }
    </figure>

    该相似度矩阵是基于点之间的距离生成的. 

    基于聚类结果, 可以得到另一个相似度矩阵, 叫做聚类相似度矩阵:

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/44b8c7c4856be6985fa0f5dd4540388e.png){ loading=lazy width='200' }
    </figure>

    这个相似度矩阵表示哪些数据点被分到了同一个簇中. 值为$1$表示两个数据点在同一个簇中, 值为$0$表示两个数据点不在同一个簇中.

    然后, 计算这两个相似度矩阵的相关性. 如果相关性高, 说明相似的数据点被分配到了同一个簇中, 聚类效果好. 如果相关性低, 说明相似的数据点没有被分配到同一个簇中, 聚类效果差.

#### 相似度矩阵的可视化

可视化一个相似度矩阵. 

<figure markdown='1'>
![](https://img.ricolxwz.io/6559e44882de330d90545c95a2358f3f.png){ loading=lazy width='500' }
</figure>

右侧的可视化的相似度矩阵用颜色表示不同点的相似性. 颜色越红, 表示相似度越高, 点之间的距离越近; 颜色越蓝, 表示相似度越低, 点之间的距离越远. 如果主对角线有清晰的块状结构, 则表示同一簇中的点彼此相似, 且不同簇之间的点不太相似, 表示聚类的效果较好.

下图是一个随机的数据的可视化相似度矩阵.

<figure markdown='1'>
![](https://img.ricolxwz.io/cb4cdc1bb438804956337090b97c20b9.png){ loading=lazy width='500' }
</figure>

可以看到, 主对角线的块状结构是不太清晰的.

### 选取簇的数量 {#choose-number-cluster}

可以参考PCA中的[肘方法](/algorithm/dimensional-reduction/#确定PC的数量). 对于不同的$k$运行聚类算法, 使用无监督评估评估聚类结果, 如SSE, 轮廓系数. 然后, 找到SSE, 轮廓系数骤然下降对应的簇的数量.

<figure markdown='1'>
![](https://img.ricolxwz.io/eda3b8fa93926f79476591ff1adb01b1.png){ loading=lazy width='500' }
</figure>