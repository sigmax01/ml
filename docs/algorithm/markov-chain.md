---
title: 马尔可夫链
comments: false
---

马尔可夫链因俄国数学家Andrey Andreyevich Markov得名, 为状态空间中从一个状态到另一个状态转换的随机过程, 该过程要求具备"无记忆"的性质, 下一状态的概率分布只能由当前状态决定, 在时间序列中和它前面的事件无关, 这种特定类型的"无记忆性"称作马尔可夫性质.

## 马尔可夫假设

马尔可夫假设是马尔可夫链的基础.公式可以表示为$p(X)=\prod_{i=1}^n p(S_t|S_{t-1})$. 它说明, 当前状态$S_{t}$只依赖于上一个状态$S_{t-1}$, 而与之前的状态$S_{1}, ..., S_{t-2}$无关. 对于其余状态也是同理的.

上述只是一阶马尔可夫假设, 即假定当前的状态仅依赖于前面一个状态. 由此衍生出$k$阶马尔可夫假设, 即假设当前状态依赖于最近的$k$个状态, 即$p(X)=\prod_{i=1}^n p(S_t|S_{t-1}, ..., S_{t-k})$. 这个概率又叫作状态转换概率.

<figure markdown='1'>
![](https://img.ricolxwz.io/7b64c88a1a92c1805aecf02b76d5c679.png){ loading=lazy width='400' }
</figure>

???+ example "例子"

    通过今天的天气预测明天的天气. 假设今天是雨天☔️, 预测明天的天气, 符合(一阶)马尔可夫假设. 下面是形象的概率图.

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/06853a90f88f172bc0e710a6f551656d.png){ loading=lazy width='300' }
    </figure>

    我们可以看到, 从雨天到晴天的概率是$0.3$, 从雨天到阴天的概率是$0.3$, 从雨天到雨天的概率是$0.4$, 所以明天大概率还是雨天. 我们可以将上图用一个矩阵来表示.

    $$
    S = \begin{bmatrix}
    S_{11} & S_{12} & \cdots & S_{1N} \\
    S_{21} & S_{22} & \cdots & S_{2N} \\
    S_{31} & S_{32} & \cdots & S_{3N} \\
    \vdots & \vdots & \ddots & \vdots \\
    S_{N1} & S_{N2} & \cdots & S_{NN} \\
    \end{bmatrix}
    $$

    其中$S_{ij}=p(S_t=j|S_{t-1}=i)$, 表示从$i$到$j$的转换概率. 那么, 我们可不可以从任意的初始状态开始, 推导出后面的所有状态呢? 假设起始概率为$\pi_i$, 表示马尔可夫链从状态$i$开始. 

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/89db496511dfb7d4cedd80c50aad8a05.png){ loading=lazy width='300' }
    </figure>

    给你一个小小的练习, 计算下列天气变化的可能性:

    - 晴天 -> 晴天 -> 多云 -> 多云
    - 多云 -> 晴天 -> 多云 -> 雨天

## 隐马尔可夫模型

在普通的马尔可夫模型中, 系统的状态是完全可见的. 也就是说, 每个时刻系统处于哪个状态是已知的, 可以直接观测到. 而在隐马尔可夫模型, HMM中, 系统的状态是隐藏的, 无法直接观测到, 但是受状态影像的某些变量是可见的. 每一个状态在可能输出的序号上都有一概率分布, 因此输出符号的序列能够透露出状态序列的一些信息.

???+ example "例子"

    假设现在我们漂到了一个岛上, 这里没有天气预报, 只有一片片的海藻, 而这些海藻的状态如干燥, 潮湿等和天气的变换有一定的关系, 既然海藻是能看到的, 那它就是*观察状态*, 天气信息看不到就是*隐藏状态*.

    再举一个例子, 如下图所示是一个普通马尔可夫模型.

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/2e166902b66dc31881b927e274c403a4.png){ loading=lazy width='400' }
    </figure>

    HMM就是在这个基础上, 加入了一个隐藏状态和观察状态的概念.

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/015f83e68047ff2b374f6a36781a7bd6.png){ loading=lazy width='400' }
    </figure>

    图中, X的状态是不可见的, 而Y的状态是可见的. 我们可以将X看成是天气情况, 而Y看成是某个人穿的衣物类型, 如下图所示.

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/5ad9c697155c00ebf6ee8e4f8fd611b4.png){ loading=lazy width='400' }
    </figure>

    我们的任务就是从这个人穿的衣物类型预测天气变化. 在这里, 有两种类型的概率:

    - 转换概率: transition probabilities, 从一个隐藏状态到另一个隐藏状态的概率
    - 观察概率: emission probabilities, 从一个隐藏状态到一个观察变量的过程

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/3faef5ee59ce156c08236dbc928ce456.png){ loading=lazy width='300' }
    </figure>

    注意⚠️

    1. 当前的隐藏状态之和前一个隐藏状态有关
    2. **某个观测状态之和生成它的隐藏状态有关, 和别的观测状态无关**

    下图给出了一个可能的观测状态和隐藏状态之间的关系. 

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/1150eb0fe6c7bfe1390438827e567784.png){ loading=lazy width='400' }
    </figure>

    可视化表达: 

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/e4556d9676b6bd8bb2ee73554008d8d1.png){ loading=lazy width='400' }
    </figure>