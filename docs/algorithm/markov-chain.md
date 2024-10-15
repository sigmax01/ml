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

