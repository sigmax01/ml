---
title: 马尔可夫链
comments: true
---

马尔可夫链因俄国数学家Andrey Andreyevich Markov得名, 为状态空间中从一个状态到另一个状态转换的随机过程, 该过程要求具备"无记忆"的性质, 下一状态的概率分布只能由当前状态决定, 在时间序列中和它前面的事件无关, 这种特定类型的"无记忆性"称作马尔可夫性质.

## 马尔可夫假设

马尔可夫假设是马尔可夫链的基础.公式可以表示为$p(X)=\prod_{i=1}^n p(S_t|S_{t-1})$. 它说明, 当前状态$S_{t}$只依赖于上一个状态$S_{t-1}$, 而与之前的状态$S_{1}, ..., S_{t-2}$无关. 对于其余状态也是同理的.

上述只是一阶马尔可夫假设, 即假定当前的状态仅依赖于前面一个状态. 由此衍生出$k$阶马尔可夫假设, 即假设当前状态依赖于最近的$k$个状态, 即$p(X)=\prod_{i=1}^n p(S_t|S_{t-1}, ..., S_{t-k})$. 这个概率又叫作状态转移概率.

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

    其中$S_{ij}=p(S_t=j|S_{t-1}=i)$, 表示从$i$到$j$的转移概率. 那么, 我们可不可以从任意的初始状态开始, 推导出后面的所有状态呢? 假设起始概率为$\pi_i$, 表示马尔可夫链从状态$i$开始. 

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/89db496511dfb7d4cedd80c50aad8a05.png){ loading=lazy width='300' }
    </figure>

    给你一个小小的练习, 计算下列天气变化的可能性:

    - 晴天 -> 晴天 -> 多云 -> 多云
    - 多云 -> 晴天 -> 多云 -> 雨天

## 隐马尔可夫模型

在普通的马尔可夫模型中, 系统的状态是完全可见的. 也就是说, 每个时刻系统处于哪个状态是已知的, 可以直接观测到. 而在隐马尔可夫模型, HMM中, 系统的状态是隐藏的, 无法直接观测到, 但是受状态影像的某些变量是可见的. 每一个状态在可能输出的序号上都有一概率分布, 因此输出符号的序列能够透露出状态序列的一些信息.

???+ example "例子"

    假设现在我们漂到了一个岛上, 这里没有天气预报, 只有一片片的海藻, 而这些海藻的状态如干燥, 潮湿等和天气的变换有一定的关系, 既然海藻是能看到的, 那它就是*观测状态*, 天气信息看不到就是*隐藏状态*.

    再举一个例子, 如下图所示是一个普通马尔可夫模型.

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/2e166902b66dc31881b927e274c403a4.png){ loading=lazy width='400' }
    </figure>

    HMM就是在这个基础上, 加入了一个隐藏状态和观测状态的概念.

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/015f83e68047ff2b374f6a36781a7bd6.png){ loading=lazy width='400' }
    </figure>

    图中, X的状态是不可见的, 而Y的状态是可见的. 我们可以将X看成是天气情况, 而Y看成是某个人穿的衣物类型, 如下图所示.

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/5ad9c697155c00ebf6ee8e4f8fd611b4.png){ loading=lazy width='400' }
    </figure>

    我们的任务就是从这个人穿的衣物类型预测天气变化. 在这里, 有两种类型的概率:

    - 转移概率: transition probabilities, 从一个隐藏状态到另一个隐藏状态的概率
    - 观测概率: emission probabilities, 从一个隐藏状态到一个观测变量的过程

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/3faef5ee59ce156c08236dbc928ce456.png){ loading=lazy width='300' }
    </figure>

    注意⚠️, HMM模型做了两个很重要的假设:

    1. 齐次马尔可夫链假设: 当前的隐藏状态之和前一个隐藏状态有关
    2. 观测独立性假设: **某个观测状态只和生成它的隐藏状态有关, 和别的观测状态无关**

    下图给出了一个可能的观测状态和隐藏状态之间的关系, 这个就是HMM所要达到的最终效果.

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/1150eb0fe6c7bfe1390438827e567784.png){ loading=lazy width='400' }
    </figure>

    可视化表达: 

    <figure markdown='1'>
    ![](https://img.ricolxwz.io/e4556d9676b6bd8bb2ee73554008d8d1.png){ loading=lazy width='400' }
    </figure>

### 参数

HMM的参数可以表示为$\lambda = (\bm{A}, \bm{B}, \bm{\pi})$, 定义隐状态的可能的取值的数量为$N$, 如雨天, 阴天, 晴天, $N=3$. 观测变量的可能的取值的数量为$M$, 如穿夹克, 穿棉袄, $M=2$. 

- 初始状态概率向量$\bm{\pi}$. 它是一个长度为$N$的向量, 其中$\pi_i$表示在初始时刻$t=1$时处于隐状态$i$的概率, 所有的初始状态满足$\sum_{i=1}^N \pi_i=1$
- 状态转移概率矩阵$\bm{A}$, $\bm{A}=[a_{ij}]$, 它是一个$N\times M$的矩阵. $a_{ij}$表示在时刻$t$处于隐状态$i$时, 下一时刻$t+1$转移到隐状态$j$的概率, 所有的转移概率满足$\sum_{j=1}^N a_{ij}=1$
- 观测概率矩阵$\bm{B}$, $\bm{B}=[b_j(o_k)]$, 它是一个$N\times M$的矩阵. $b_j(o_k)$表示在隐状态$j$下生成观测值$o_k$的概率. 对于连续的观测值, 可以使用概率密度函数来表示概率

### 假设

HMM做了两个基本假设, 在上面的例子中也提到了.

- 齐次性假设: 即假设隐藏的马尔可夫链在任意时刻$t$的状态只依赖于它在前一时刻的状态, 与其他时刻的状态和观测无关, 也与时刻$t$本身无关
- 观测独立性假设: 即假设任意时刻的观测值只依赖于该时刻的马尔可夫链的状态, 与其他观测及状态无关

### 问题

HMM的三个基本问题:

- 概率计算问题(也叫做Evaluation Problem): 给定模型$\lambda = (\bm{A}, \bm{B}, \bm{\pi})$和观测序列$\bm{O}=(o_1, o_2, ..., o_M)$, 计算观测序列$O$出现的概率$p(\bm{O}; \lambda)$. 即评估模型$\lambda$与观测序列$\bm{O}$之间的匹配程度.
- 学习问题(也叫做Learning Problem): 已知观测序列$\bm{O}=(o_1, o_2, ..., o_M)$, 估计模型$\lambda=(\bm{A}, \bm{B}, \bm{\pi})$的参数, 使得在该模型下的观测序列概率$p(\bm{O}; \lambda)$最大, 即用极大似然估计的方法估计参数
- 预测问题(也叫做Decoding Problem): 已知模型$\lambda = (\bm{A}, \bm{B}, \bm{\pi})$和观测序列$\bm{O}=(o_1, o_2, ..., o_M)$, 求对给定观测序列的条件概率$P(\bm{I}|\bm{O})$最大的状态序列$\bm{I}=(i_1, i_2, ..., i_r)$, 即给定观测序列, 求最可能的对应的状态序列

[^1]: https://blog.csdn.net/HUSTHY/article/details/104840693

#### 概率计算问题

概率计算问题, Evaluation Problem. 指的是给定一个HMM模型$\lambda = (\bm{\pi}, \bm{A}, \bm{A_0})$和一个观测序列$X=x_1, x_2, ..., x_m$, 计算该观测序列出现的概率.

???+ warning "注意"

	这边的$\bm{\pi}$很鸡肋, 其实不用写的, 直接写一个状态转移矩阵$A$就够了. 然后这边的$\pi$也不是我上面说的意思, 他这里的$\pi_i$是指时间步$i$的状态, 而我上面说的$\pi_i$是初始时刻处于隐状态$i$的概率. 最后应该加上一个观测概率矩阵$\bm{E}$, 这边漏了. 但是为了保持和课件的统一, 下面都用它的写法.

???+ example "例子"

	给定一个HMM模型如下(包含初始状态向量, 状态转移概率矩阵, 观测概率矩阵):

	<figure markdown='1'>
	![](https://img.ricolxwz.io/73629b7b37cb523a56c45c42c1a30fc4.png){ loading=lazy width='400' }
	</figure>

	计算观测序列$X=$ Shirt, Hoodie出现的概率. 

	我们可以使用枚举法: 首先, 列举出所有可能的状态序列, 由于我们的观测序列长度是$2$, 所以长度为$2$的状态序列有$3^2=9$种组合, 如, Rainy, Rainy; Rainy, Cloudy; Rainy, Sunny; ... 对于每一种状态序列, 计算其对应的观察序列$X=$ Shirt, Hoodie的条件概率, 例如, 对于状态序列Rainy, Cloudy, 计算观测序列条件概率$p(X, \{Rainy, Cloudy\})$的步骤为:

	- 初始状态: 状态从Rainy开始, 所以初始概率参见初始状态向量, 是$0.6$
	- 状态转移: 从Rainy转移到Cloudy的概率参见状态转移概率矩阵, 是$0.3$
	- 观测概率:
		- 在第一个时刻Rainy观察到Shirt的概率参考观测概率矩阵, 是$0.8$
		- 在第二个时刻Cloudy观测到Hoodie的概率参考观测概率矩阵, 是$0.1$

	所以, 结果为$0.6\times 0.3\times 0.8\times 0.0144$. 对于所有的状态序列, 如上所示计算观测序列的条件概率. 相加这$9$个条件概率, 得到最终的观测序列概率. 

	可以看到, 计算一个简单的观测序列Short, Hoodie的过程就进行了$4\times 9=36$次乘法. 令$N$为可能的状态的数量, 在这里有三个可能的状态, Rainy, Cloudy, Sunny. 令$T$为观测序列的长度, 在这里是$2$. 那么复杂度就是$2TN^T$. 在实际中, 观测序列$T$往往很大, 而状态数$N$相对来说较小, 导致该算法的复杂度异常高. 解决这种问题的方法是使用前向算法.

##### 前向算法

前向算法, Forward Algorithm, 是一种动态规划算法, 用于高效计算给定观察序列的概率. 就像之前我们干的一样, 对于所有可能的隐藏状态序列, 前向算法都会求观测序列的条件概率, 但是在计算过程中会存储中间值来加速计算.

前向概率$f_k(i)$表示在时间$i$时处于状态$k$的条件下, 观察到*部分*观测序列的概率. $f_k(i)=e_k(x_i)\sum_j f_j(i-1)a_{jk}$, 其中$e_k(x_i)=p(x_i|\pi_i=k)$是状态$k$下观察到观测序列$x_i$的观测概率, $a_{jk}=p(\pi_i=k|\pi_{i-1}=j)$是从状态$j$转移到状态$k$的转移概率. 通过上述公式, 可以递归地计算$f_k(i)$, 因为$f_k(i)$依赖于上一个时刻$i-1$地前向概率$f_j(i-1)$. 这种递归计算不需要重新计算已经求得的中间结果, 大大减少了重复计算的次数.

前向算法的时间复杂度为$N^2T$, 其中$T$是观测序列长度, $N$是状态数量, 相比枚举法的$2TN^T$的复杂度, 前向算法的效率显著提高.

前向算法主要由$3$步计算过程组成:

1. 初始化

	计算在时间步$t=1$时每个状态$k$的前向概率$f_k(1)$, 公式为$f_k(1)=A_0(k)e_k(x_1)$, 其中$A_0(k)$是初始时刻状态$k$的概率, 参考初始状态向量; $e_k(x_1)=p(x_1|\pi_1=k)$是在状态$k$下观察到第一个观测值$x_1$的概率.

2. 迭代

	对于接下来的时间步$t=2, ..., m$, 计算每个状态$k$的前向概率$f_k(i)$. $f_k(i)=e_k(x_i)\sum_j f_j(i-1)a_{jk}$. 各部分的含义参见上方.

3. 终止

	在最后一个时间步$m$结束后, 对所有状态的前向概率求和. $p(X)=\sum_{k}f_k(m)$.

???+ example "例子"

	继续上面的例子.

	$1$的初始状态向量 + $2$个矩阵:

	<figure markdown='1'>
	![](https://img.ricolxwz.io/73629b7b37cb523a56c45c42c1a30fc4.png){ loading=lazy width='400' }
	</figure>	

	1. 初始化

		- $f_{Rainy}(1)=A_0(Rainy)e_{Rainy}(Shirt)=0.6\times 0.8=0.48$
		- $f_{Cloudy}(1)=A_0(Cloudy)e_{Cloudy}(Shirt)=0.3\times 0.5=0.15$
		- $f_{Sunny}(1)=A_0(Sunny)e_{Sunny}(Shirt)=0.1\times 0.01=0.001$

	2. 迭代

		迭代计算第$i$个时间步的前向概率.

		- $f_{Rainy}(2)=e_{Rainy}(Hoodie)(f_{Rainy}(1)a_{Rainy, Rainy}+f_{Cloudy}(1)a_{Cloudy, Rainy}+f_{Sunny}(1)a_{Sunny, Rainy})=0.01\times(0.48\times 0.6+0.15\times 0.4 + 0.001\times 0.1)=0.0035$
		- $f_{Cloudy}(2)=e_{Cloudy}(Hoodie)(f_{Rainy}(1)a_{Rainy, Cloudy}+f_{Cloudy}(1)a_{Cloudy, Cloudy}+f_{Sunny}(1)a_{Sunny, Cloudy})=0.1\times(0.48\times 0.3+0.15\times 0.3 + 0.001\times 0.4)=0.0189$
		- $f_{Sunny}(2)=e_{Sunny}(Hoodie)(f_{Rainy}(1)a_{Rainy, Sunny}+f_{Cloudy}(1)a_{Cloudy, Sunny}+f_{Sunny}(1)a_{Sunny, Sunny})=0.79\times(0.48\times 0.1+0.15\times 0.3 + 0.001\times 0.5)=0.0739$

	3. 终止

		当前最后一个时间步$m=2$, $p(X)=p(Shirt, Hoodie)=f_{Rainy}(2)+f_{Cloudy}(2)+f_{Sunny}(2)=0.0035+0.0189+0.0739=0.0963$.

#### 预测问题

预测问题, Decoding Problem. 指的是给定一个HMM模型$\lambda=(\bm{\pi}, \bm{A}, \bm{A_0})$和一个观测序列$X=x_1, x_2, ..., x_m$, 计算最可能对应的隐藏序列.

##### Viterbi算法

Viterbi算法是一种动态规划算法, 用于计算每个状态的最优路径的概率, 称之为Viterbi得分. 由于它在每个时间点, 只需要维护每个状态的最高得分路径, 所以和前向算法类似, 能提高计算效率.

给定时间$i$和状态$k$. 状态的Viterbi得分$V_k(i)$表示到达该状态的最优路径的概率, 计算公式为$V_k(i)=e_k(x_i)max_j V_j(i-1)a_{jk}$. 其中, $e_k(x_i)=p(x_i|\pi_i=k)$表示在当前状态$k$下, 观测到$x_i$的观测概率. $a_{jk}=p(\pi_i=k|\pi_{i-1}=j)$表示从前一状态$j$转移到当前状态$k$的转移概率. $max_j V_j(i-1)a_{jk}$用于选择上一个时间步中到达当前状态$k$的最优路径.

Viterbi得分$V_k(i)$可以递归地基于上一时间步的得分$V_j(i-1)$计算, 因此, 不需要重复计算所有路径, 而是逐步优化路径选择.

Viterbi得分可以给出最终状态结束的最佳路径的概率, 但是仅仅只靠得分本身, 我们无法确定从起始状态到最终状态的整个路径. 为了确定完整的路径, 需要从最终状态回溯到其实状态, 为了实现回溯, 在计算Viterbi得分的过程中, 需要为每个状态保存一个指针, 这个指针记录了每一步使得Viterbi得分最大的前一状态. 数学表达为$Ptr_k(i)=argmax_j V_j(i-1)a_{jk}$, 指向时间$i-1$时能提供最高得分的状态$j$. 

与前向算法类似, Viterbi算法也分为三步:

1. 初始化

	在初始时间步$t=1$的时候, 计算每个状态的Viterbi得分, 公式为$V_k(1)=A_0(k)e_k(x_1)$. 其中$A_0(k)$是初始时刻状态$k$的概率, 参考初始状态向量; $e_k(x_1)=p(x_1|\pi_1=k)$是在状态$k$下观察到第一个观测值$x_1$的概率. 

2. 迭代

	对于时间步$t=2, ..., m$:

	1. Viterbi得分: 计算状态$k$在时间$i$的得分$V_k(i)=e_k(x_i)max_j V_j(i-1)a_{jk}$

	2. 回溯指针: 记录状态$k$在时间$i$的回溯路径, 用于后续的路径回溯$Ptr_k(i)=argmax_j V_j(i-1)a_{jk}$  

3. 终止

	1. 确定最终状态: 在最后一个时间步$m$, 找到具有最高Viterbi得分的状态$\pi_m=argmax_k V_k(m)$

	2. 回溯路径: 从最后的状态开始, 通过回溯指针逐步确定前一个时间步的最佳状态: $\pi_{i-1}=Ptr_{\pi_i}(i)$

???+ example "例子"

	还是上面的例子. 给定一个模型:

	<figure markdown='1'>
	![](https://img.ricolxwz.io/73629b7b37cb523a56c45c42c1a30fc4.png){ loading=lazy width='400' }
	</figure>	

	和观测序列$X=$ Shirt, Hoodie.

	1. 初始化

		- $V_{Rainy}(1)=A_0(Rainy)e_{Rainy}(Shirt)=0.6\times 0.8=0.48$
		- $V_{Cloudy}(1)=A_0(Cloudy)e_{Cloudy}(Shirt)=0.3\times 0.5=0.15$
		- $V_{Sunny}(1)=A_0(Sunny)e_{Sunny}(Shirt)=0.1\times 0.01=0.001$

	2. 迭代

		这里要计算Viterbi得分和获取回溯指针.

		- Rainy: 
			- $V_{Rainy}(2)=e_{Rainy}(Hoodie)\times max(V_{Rainy}(1)a_{Rainy, Rainy}, V_{Cloudy}(1)a_{Cloudy, Rainy}, V_{Sunny}(1)a_{Sunny, Rainy})=0.01\times max(0.48\times 0.6, 0.15\times 0.4 , 0.001\times 0.1)=0.01\times 0.48\times0.6=0.0029$
			- $Ptr_{Rainy}(2)=argmax(0.48\times 0.6, 0.15\times 0.4, 0.001\times 0.1)=1$, 如$1$是Rainy
		- Cloudy: 
			- $V_{Cloudy}(2)=e_{Cloudy}(Hoodie)\times max(V_{Rainy}(1)a_{Rainy, Cloudy}, V_{Cloudy}(1)a_{Cloudy, Cloudy}, V_{Sunny}(1)a_{Sunny, Cloudy})=0.1\times max(0.48\times 0.3, 0.15\times 0.3 , 0.001\times 0.4)=0.1\times 0.48\times0.3=0.0144$
			- $Ptr_{Cloudy}(2)=argmax(0.48\times 0.3, 0.15\times 0.3, 0.001\times 0.4)=1$, 如$1$是Rainy
		- Sunny: 
			- $V_{Sunny}(2)=e_{Sunny}(Hoodie)\times max(V_{Rainy}(1)a_{Rainy, Sunny}, V_{Cloudy}(1)a_{Cloudy, Sunny}, V_{Sunny}(1)a_{Sunny, Sunny})=0.01\times max(0.48\times 0.1, 0.15\times 0.3 , 0.001\times 0.5)=0.79\times 0.48\times0.1=0.0379$
			- $Ptr_{Sunny}(2)=argmax(0.48\times 0.1, 0.15\times 0.3, 0.001\times 0.5)=1$, 如$1$是Rainy

	3. 终止

		时间步$2$的最终状态可由下列公式计算$argmax(V_{Rainy}(2), V_{Cloudy}(2), V_{Sunny}(2))=argmax(0.0029, 0.0144, 0.0379)=3$, 如$3$是Sunny. 由于$Ptr_{Sunny}=Rainy$, 所以最有可能的状态序列为Rainy, Sunny.

#### 学习问题

学习问题, Learning Problem. 指的是给定一个观测序列$X=x_1, x_2, ..., x_m$. 找出HMM模型$\lambda=(\bm{\pi}, \bm{A}, \bm{A_0})$.

##### 期望最大化算法

期望最大发算法, Expectation Maximization, 用于计算HMM模型. 它分为4步:

1. 初始化

	将模型参数$\lambda = (\bm{\pi}, \bm{A}, \bm{A_0})$随机初始化. 

2. 期望步骤

	基于当前的参数$\lambda$, 计算各隐藏状态的概率分布.

3. 最大化步骤

	利用前一步计算的概率, 更新模型参数$\lambda$, 使得给定观测数据的似然函数最大化. 这涉及预测最可能的观测序列并与实际观测序列进行比较.

4. 收敛判断

	如果模型参数更新后, $p(X\lambda)$增加, 则返回第二步继续迭代, 否则停止迭代.