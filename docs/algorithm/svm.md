---
title: 支持向量机
comments: true
---

???+ info "信息"

    - 已省略例子边框
    - 使用粗体表示向量
    - 在推导公式的过程中大量引用了🍉书的内容, 资源下载: [https://jingyuexing.github.io/Ebook/Machine_Learning/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E5%91%A8%E5%BF%97%E5%8D%8E.pdf](https://jingyuexing.github.io/Ebook/Machine_Learning/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E5%91%A8%E5%BF%97%E5%8D%8E.pdf)

## 边际最大超平面

超平面, Hyperplane是数学和机器学习中的一个概念. 它是一个$n$维空间中将空间划分为两个部分的几何对象. 具体来说, 在$n$维空间中, 超平面是一个$n-1$维子空间. 例如, 在二维空间中, 超平面是一条一维的直线, 直线将平面分成两个部分; 在三维空间中, 超平面是一个二维的平面, 平面将空间分为两个部分, 依此类推...

在左图中, 找到一个线性的决策边界(直线, 超平面)使得数据能够分开, 这道题的答案有很多, 如右图.

<figure markdown='1'>
![](https://img.ricolxwz.io/cd68b93ab0d32d042943c2abf2e34230.png){ loading=lazy width="600" }
</figure>

支持向量, support vector是离决策边界最近的样本(数据点). 边际, margin是指正负超平面(什么是正负超平面下面有讲到)之间的距离, 如左图. 当然, 也可能同时出现多个支持向量, 因为它们都是最靠近决策边界的样本点, 如右图.

<figure markdown='1'>
![](https://img.ricolxwz.io/99a4ec08371a005d310d24a6805747b0.png){ loading=lazy width="625" }
</figure>

支持向量之所以被称为向量是因为给的样本就是一个向量, 试想, 样本有很多的特征, 这些特征构成了向量... 支持向量是所有样本点(向量)的子集.

下面来看, 哪一个超平面(决策边界)更优呢? B1还是B2? 哪个超平面能够更准确地分类新数据?

<figure markdown='1'>
![](https://img.ricolxwz.io/6209988905535c0d38422ae82d63dbed.png){ loading=lazy width="300" }
</figure>

答案是B1, 因为它的边际更大. 拥有最大边际的超平面我们称之为边际最大超平面. 按照这个超平面(决策边界)分类的数据准度会更高. 如果边际过小, 意味着决策边界和支持向量非常接近, 在这种情况下, 即使是很小的变化都可能导致分类结果发生显著变化, 这意味着模型对数据扰动很敏感, 容易出现过拟合现象; 如果边际较大, 意味着模型对于数据的微小变化更具有鲁棒性, 有较好的泛化性能. 这种选择大边际的策略在统计理论中也得到了支持, 称为"结构风险最小化原理".

## 线性支持向量机

现在我们有一个二分类问题, 一共有$N$个训练样本(输入向量). 我们定义$\boldsymbol{x}$为输入向量, $y$为分类值, 即$\boldsymbol{x_i}=(x_{i1}, x_{i2}, ..., x_{im})^T, y_i=\{-1, 1\}$. 参考[线性分类](/algorithm/linear-regression/#线性分类). 我们有一个符号函数$sign$, 如果$\boldsymbol{w}\cdot \boldsymbol{x}+b$的结果$>0$, 说明样本点在决策边界的上方; 如果$\boldsymbol{w}\cdot \boldsymbol{x}+b$的结果$<0$, 说明样本点在决策边界的下方. $y=\boldsymbol{w}\cdot \boldsymbol{x} + b$就是决策边界. 

### 超平面方程式

假设输入向量$\boldsymbol{x}$是二维的. 我们可以将决策边界$H$的超平面方程式定义为$w_1x_1+w_2x_2+b=0$(标量形式). 将超平面方程式上下移动$c$个单位, 分别是$w_1x_1+w_2x_2+b=c$和$w_1x_1+w_2x_2+b=-c$, 来到对应的间隔上下边界$H_1, H_2$, 由于上下边界一定会经过一些样本数据点, 而这些点距离决策边界最近, 他们决定了间隔距离, 我们称之为支持向量. 我们可以把等式两边分别除以$c$, 得到$\frac{w_1}{c}x_1+\frac{w_2}{c}x_2+\frac{b}{c}=1$, $\frac{w_1}{c}x_1+\frac{w_2}{c}x_2+\frac{b}{c}=-1$和$\frac{w_1}{c}x_1+\frac{w_2}{c}x_2+\frac{b}{c}=0$, 使用$w'_1=\frac{w_1}{c}, w'_2=\frac{w_2}{c}$和$b'=\frac{b}{c}$替换, 可以得到$w'_1x_1+w'_2x_2+b'=1$, $w'_1x_1+w'_2x_2+b'=-1$和$w'_1x_1+w'_2x_2+b'=0$, 由于$w'_1, w'_2$和$b'$只是我们需要求解的代号, 所以将其换为$w_1, w_2$和$b$也不影响计算, 所以最终只需要求解$w_1, w_2$和$b$就可以了, 得到下面三个超平面方程: $w_1x_1+w_2x_2+b=1$, $w_1x_1+w_2x_2+b=-1$和$w_1x_1+w_2x_2+b=0$, 这三个方程式分别称为正超平面方程式(对应正超平面$H_1$), 负超平面方程式(对应负超平面$H_2$)和决策超平面方程式(对应决策超平面$H$).[^1]

<div style="position: relative; padding: 30% 45%;">
<iframe style="position: absolute; width: 100%; height: 100%; left: 0; top: 0;" src="//player.bilibili.com/player.html?isOutside=true&aid=936042727&bvid=BV16T4y1y7qj&cid=494397114&p=1&high_quality=1&autoplay=false&muted=false&t=185&as_wide=1" frameborder="yes" scrolling="no" allowfullscreen="true"></iframe>
</div>

### 最大化拉格朗日函数 {#maximize-lagrange-function}

回到超平面方程式的向量形式, 对于任意一个样本点, 它到超平面$\boldsymbol{w}\cdot \boldsymbol{x} + b=0$的垂直距离可以通过公式$\frac{|\boldsymbol{w}\cdot \boldsymbol{x} + b|}{||\boldsymbol{w}||}$计算. 因此, 支持向量之间的垂直距离可以计算为$d=\frac{|(\boldsymbol{w}\cdot \boldsymbol{x_1}+b)-(\boldsymbol{w}\cdot \boldsymbol{x_2}+b)|}{||\boldsymbol{w}||}$, 由于$\boldsymbol{w}\cdot \boldsymbol{x_1}+b=1$, $\boldsymbol{w}\cdot \boldsymbol{x_2}+b=-1$, 我们可以将其代入上述公式, 得到$d=\frac{2}{||\boldsymbol{w}||}$.

<figure markdown='1'>
![](https://img.ricolxwz.io/71e7706ae085c0fcaf7714a519c50a24.png){ loading=lazy width="400" }
</figure>

我们的目标就是要最大化这个$d$, 也就是最小化$||\boldsymbol{w}||$, 等同于最小化函数$\frac{1}{2}||\boldsymbol{w}||^2.$ 所以线性支持向量机将$\frac{1}{2}||\boldsymbol{w}||^2$作为代价函数, 前提条件是所有的样本必须被正确分类(即硬边际), 即对于任意的$y_i$, 有$y_i(\boldsymbol{w}\cdot \boldsymbol{x_i}+b)\geq 1$. 这是一个凸二次优化问题.

我们可以使用拉格朗日乘数法将约束条件结合到目标函数中, 构造如下: $L(\boldsymbol{w}, b, \boldsymbol{\lambda})=\frac{1}{2}||\boldsymbol{w}||^2-\sum_{i=1}^N \lambda_i[y_i(\boldsymbol{w}\cdot \boldsymbol{x_i}+b)-1]$, 其中$\lambda_i$为拉格朗日乘子, $\boldsymbol{\lambda}=(\lambda_1; \lambda_2; ...; \lambda_N), \lambda_i\geq 0$. 拉格朗日函数是原目标函数的对偶问题, 即求原目标函数的极小值就是求拉格朗日函数的极大值, 令$L(\boldsymbol{w}, b, \boldsymbol{\lambda})$对$\boldsymbol{w}$和$b$求导为$0$可得: $\boldsymbol{w}=\sum_{i=1}^{N}\lambda_iy_i\boldsymbol{x_i}(1)$, $0=\sum_{i=1}^N\lambda_iy_i(2)$. 将$(1), (2)$式代入$L(\boldsymbol{w}, b, \boldsymbol{\lambda})$, 其中前半部分有$\frac{1}{2}\boldsymbol{w}^T{\boldsymbol{w}}=\frac{1}{2}(\sum_{i=1}^N \lambda_iy_i\boldsymbol{x_i})^T(\sum_{j=1}^N\lambda_jy_j\boldsymbol{x_j})=\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \lambda_i\lambda_jy_iy_j\boldsymbol{x_i}^T\boldsymbol{x_j}$, 后半部分有$\sum_{i=1}^N \lambda_i[y_i(\boldsymbol{w}\cdot \boldsymbol{x_i}+b)-1]=\sum_{i=1}^N\lambda_iy_i(\sum_{j=1}^N\lambda_jy_j\boldsymbol{x_j}\cdot \boldsymbol{x_i}+b)-\lambda_i=b\sum_{i=1}^N\lambda_iy_i+\sum_{i=1}^N\sum_{j=1}^N \lambda_i\lambda_jy_iy_j\boldsymbol{x_i}^T\boldsymbol{x_j}-\sum_{i=1}^N\lambda_i$, 故有$max \{L(\boldsymbol{w}, b, \boldsymbol{\lambda})\}=max \sum_{i=1}^N\lambda_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_j\boldsymbol{x_i}\cdot\boldsymbol{x_j}$.[^2]

如何求解这个式子呢? 可以看到, 关键就是要求拉格朗日乘数, 其他的都可以直接读取数据得到. 拉格朗日乘数表示的是每个约束对目标函数的影响大小, 拉格朗日乘数大的样本点(通常是支持向量)对决策边界的贡献较大, 而拉格朗日乘数为$0$的样本点对决策边界没有影响. 这个式子的求解可以使用二次规划(Quadratic Programming, QP)或者其他的技术, QP算法如SMO算法请见西瓜书第124页.

解出拉格朗日乘数之后, 我们就可以计算出系数向量$\boldsymbol{w}=\sum_{i=1}^N\lambda_iy_i\boldsymbol{x_i}$. 最佳决策边界由$\boldsymbol{w}$决定, 而$\boldsymbol{w}$就是一堆向量的线性组合($\sum$拉格朗日乘数\*标签\*特征向量). 在实际中, 大多数的$\lambda_i$的值都是$0$, 也就是说对应的训练样本$\boldsymbol{x_i}$对决策边界的确定没有贡献, 那些对应$\lambda_i$不为$0$的训练样本就是支持向量. 

现在, 如果出现了一个新的样本$z$, 我们可以计算出$f=\boldsymbol{w}\cdot \bm{z}+b=\sum_{i=1}^N\lambda_iy_i\bm{x_i}\cdot \bm{z}+b$, 从而预测结果为$sign(f)$. 如果$f>0$, 会被归类为$1$; 如果$f<0$, 会被归类为$-1$.

好了, 举一个例子. 现在我有$8$个二维的训练样本, 总共分$2$类: $1$和$-1$. 在使用QP求解拉格朗日极值下的乘数$\bm{\lambda}$之后, 我们发现只有$2$个乘数是非零的, 分别是样本$1$和样本$2$, 它们对应的就是支持向量. 现在, 我们可以计算出它们的系数$w_1=\sum_{i=1}^N \lambda_iy_i\bm{x_{i1}}$, $w_2=\sum_{i=1}^N \lambda_iy_i\bm{x_{i2}}$. $b$可以用公式推出(未给出). 现在我们就得到了预测模型$f=w_1x_1+w_2x_2+b$, 根据这个就可以给出预测.

???+ example "例子"

    给出下列样本以及拉格朗日乘数, 计算决策边界表达式.

    | x₁    | x₂    | y  | 拉格朗日乘数 |
    |-------|-------|----|---------------------|
    | 0.3858 | 0.4687 | 1  | 65.5261            |
    | 0.4871 | 0.6118 | -1 | 65.5261            |
    | 0.9218 | 0.4103 | -1 | 0                  |
    | 0.7382 | 0.8936 | -1 | 0                  |
    | 0.1763 | 0.0579 | 1  | 0                  |
    | 0.4057 | 0.3529 | -1 | 0                  |
    | 0.9355 | 0.8132 | 1  | 0                  |
    | 0.2146 | 0.0099 | -1 | 0                  |

    可以看到, 有两个拉格朗日乘数非零, 第一个样本处在正超平面上, 第二个样本处在负超平面上, 它们都是支持向量. 根据公式, 可以得到$w_1=\sum_{i=1}^2\lambda_i y_i \bm{x_{i1}}=65.5261(1*0.3858-1*0.4871)=-6.64$, $w_2=\sum_{i=1}^2\lambda_i y_i \bm{x_{i2}}=65.5261(1*0.4687-1*0.6118)=-9.32$

    对于位于正超平面上的支持向量, 有$y_1(\bm{w}\cdot \bm{x_1}+b)=1$, 故$1\cdot(-6.64\cdot 0.3858-9.32\cdot 0.4687+b)=1$, 可以解出$b=7.93$. 对于位于负超平面上的支持向量, 也有上述式子, 解出来的$b$应该和上面的结果是相等的. 

    综上所述, 决策边界为$y=-6.64x_1-9.32x_2+7.93$.


### 软边际和硬边际 {#软边际和硬边际}

上面我们提到的约束条件是"所有的样本必须正确分类", 这属于硬边际, hard-margin分类问题. 对应的还有软边际, soft-margin分类问题, 我们可以允许一些错误的分类. 软边际分类问题通俗的理解就是允许在正超平面和负超平面之间存在一些样本点. 如下图.

<figure markdown='1'>
![](https://img.ricolxwz.io/8e1f2f2544ec9ab1e218e350cc14322b.png){ loading=lazy width="300" }
</figure>

我们发现, 在边际的距离上B1远胜B2, 但是B1中正负超平面之间存在着两个样本点. 这是否意味着B2就好于B1呢? 其实, 是需要衡量一下的, 相当于边际的距离是加分项, 错误分类是减分项, 要做的就是根据情况掂量掂量最后的收益如何. 一般情况下, 边际的距离更重要, 因为大的边际距离能够让模型更少受噪音干扰, 防止过拟合.

在硬边际分类问题中, 我们要最小化的是$||\boldsymbol{w}||$(对于所有的样本, $y_i(\bm{w}^T\bm{x_i}+b)\geq 1$), 而在软边际问题中, 我们只需要增加一个额外的参数$C$: $||\bm{w}||+C\sum \xi_i$(对于所有的样本, $y_i(\bm{w}^T\bm{x_i}+b)\geq 1-\xi_i$, $\xi_i\geq 0$), $\xi_i$就是每个样本的额外允许的误差, 可以看到当$C$变大的时候, 后面那项在最小化的时候权重就会比较大, 会使得更加注重减小分类错误, 而不是增加边际距离. 

## 非线性支持向量机 {#non-linear-svm}

更为普遍的情况是, 大多数的问题都不是线性可分的, 如图.

<figure markdown='1'>
![](https://img.ricolxwz.io/773f22a1b5e7aad0f4df6bb4707b9bda.png){ loading=lazy width='300' }
</figure>

SVM可以被进一步延伸用来处理这一类问题. 一种思路就是将数据从原始特征空间转换到一个新的特征空间(通常是高纬度的空间). 在新的特征空间中, 学习一个线性决策边界来分离不同类别的数据. 新特征空间中的线性决策边界可以通过逆变换映射回原始特征空间. 在原始特征空间中, 这个映射回来的决策边界通常表现为非线性的决策边界. 例如变换$\phi=(x_1, x_2)\rightarrow(x_1^2-x_1, x_2^2-x_2)$.

<figure markdown='1'>
![](https://img.ricolxwz.io/98b72e6d433d72105d5ada984603dddd.png){ loading=lazy width='600' }
</figure>

再如:

<figure markdown='1'>
![](https://img.ricolxwz.io/db91d70ec41da7485030a459910b2849.png){ loading=lazy width='550' }
</figure>

所以, 我们应该用什么样的映射函数$\phi$? 就算我们知道了映射函数, 我们怎么在新的特征空间里面高效的计算? 因为在新的特征空间里面做点积是非常消耗计算机资源的. 要计算的是$\sum_{i=1}^N\lambda_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_j\boldsymbol{\phi(x_i)}\cdot\boldsymbol{\phi(x_j)}$的最大值, 而$\boldsymbol{\phi(x_i)}$和$\boldsymbol{\phi(x_j)}$都是更高维的向量, 点积非常耗时间.

### 核方法 {#kernel-trick}

解决上述问题的方法是使用核方法, kernel trick. 核方法是一种计算$\boldsymbol{\phi(x_i)}\cdot\boldsymbol{\phi(x_j)}$的方法. 它与之前的方法不同, 它不会先映射$\bm{x_i}$和$\bm{x_j}$然后点积$\boldsymbol{\phi(x_i)}$和$\boldsymbol{\phi(x_j)}$. 它采取的是另一种方式: 我们先计算原始特征向量的点积, 然后使用核函数得出映射后的特征向量的点积. 这个核函数表示了原始特征向量的点积和映射后的特征向量的点积的关系.

例如, 现在我们有两个二维的特征向量$\bm{u}=(u_1, u_2)$, $\bm{v}=(v_1, v_2)$. 和一个映射函数$\phi=(x_1, x_2)\rightarrow(x_1^2, \sqrt{2}x_1x_2, x_2^2)$, 按照原来的方法, 应该先计算$\phi(\bm{u})$和$\phi(\bm{v})$, 然后点积$\boldsymbol{\phi(x_i)}\cdot\boldsymbol{\phi(x_j)}$. $\boldsymbol{\phi(x_i)}\cdot\boldsymbol{\phi(x_j)}=(u_1^2, \sqrt{2}u_1u_2, u_2^2)\cdot(v_1^2, \sqrt{2}v_1v_2, v_2^2)=(u_1v_1+u_2v_2)^2=(\bm{u}\cdot\bm{v})^2$. 这样子费时费力, 消耗大量资源. 但是, 有一个惊喜, 在计算过程中, 我们发现映射后的特征向量的点积居然可以用原始特征向量的点积表示, 即$\phi(\bm{u})\cdot \phi(\bm{v})=(\bm{u}\cdot \bm{v})^2$. 这个关系可以用核函数表示, $K(\bm{u}, \bm{v})=(\bm{u}\cdot\bm{v})^2$. 所以, 我们可以先在低维空间算出$\bm{u}\cdot\bm{v}$, 然后利用核函数直接算出高维空间的点积.

#### 默瑟定理

在上面我们计算的时候我们已经知道了映射函数$\phi$, 但是大多数情况下, 我们是不知道这个映射函数的. 有时候这个映射函数可能非常复杂, 高维空间的维度可能非常高, 显式地定义和计算$\phi$不切实际. 这意味着我们不能先知道$\phi$, 然后从这个$\phi$反推出核函数.

在实际中, 我们会直接选择一个能够使样本线性可分的核函数(知道了核函数相当于知道了映射后的内积, 知道了映射后的内积相当于可以求拉格朗日方程的极大值对应的参数, 得到决策边界), 而不需要关心映射函数$\phi$(因为映射关系对于求解问题可有可无). 

默瑟定理, Mercer定理表明, 一个核函数$K(\bm{u}, \bm{v})$如果满足特定条件, 那么这个核函数一定对应这某一个映射函数$\phi$. 所以, 我们一般会直接选择或者设计一个符合要求的核函数, 它已经隐含了某种高维空间的映射函数, 这种映射函数我们是不用关心的. 

几种常见的符合默瑟定理的核函数为:

- 多项式核: $K(\bm{x}, \bm{y})=(\bm{x}\cdot \bm{y}+1)^p$
- RBF核: $K(\bm{x}, \bm{y})=exp(-\frac{||\bm{x}-\bm{y}||^2}{2\sigma^2})$
- 双曲正切核: $K(x, y)=tanh(k\bm{x}\cdot y-\theta)$ (默瑟定理只对一些$k$和$\theta$成立)

我们要做的就是选出最优的核函数, 使得进行隐式映射后, 得到的样本在高维空间中能够线性可分而且得到的结果不错. 极大化拉格朗日函数可以写为$max \sum_{i=1}^N\lambda_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_jK(\bm{x_i, x_j})$. 预测$z$: $f=\bm{w}\cdot \bm{z}+b=\sum_{i=1}^N \lambda_iy_iK(\bm{x_i}, \bm{z})+b$.

#### RBF核函数

我们来看一种比较常用的核函数: RBF核函数, 这种表达式的函数为$K(\bm{x}, \bm{y})=exp(-\frac{||\bm{x}-\bm{y}||^2}{2\sigma^2})$. 这个函数有一个超参数, Hyper Parameter, 即$\sigma$, 结合我们在[软边际和硬边际](#软边际和硬边际)中讲到的另一个超参数$C$. 我们在训练中常常会调节这两个参数, 尝试将它们组合, 然后评估结果, 如:

<figure markdown='1'>
![](https://img.ricolxwz.io/ffe1d1b4c630060d2bd2b23def06a62f.png){ loading=lazy width='500' }
</figure>

- 超参数$\sigma$控制核函数的曲线形状, 决定了每个点的"影响范围". 当$\sigma$很小的时候, RBF曲线的半径较大, 意味着每个点的影响范围更广, 导致决策边界更加平滑, 接近线性, 这种情况下, 模型可能会欠拟合, 因为它没有捕捉数据中的细节. 当$\sigma$很大的时候, RBF曲线的半径较小, 每个点的影响范围较窄, 模型会试图过多地你和训练数据中的细节, 导致决策边界在个别实例周围出现波动和不规则形, 可能会导致过拟合 
- 超参数$C$控制模型对误分类点的惩罚强度, 参考[软边际和硬边际](#软边际和硬边际). 当$C$很小的时候, 允许更多的点被误分类, 因此决策边界更加平滑, 接近线性, 适用于防止过拟合, 我们可以看左上角的图, 边界基本上是线性的了; 当$C$很大的时候, 模型更加关注误分类的点, 导致决策边界弯曲, 以捕捉到更多的细节, 如果数据复杂, 模型可能会过拟合, 我们可以看到右下角的图, 相比于左下角的图, 两个蓝色的点被正确分类了

控制超参数是非常重要的, 我们可以使用交叉验证结合参数网格的方法来训练模型.

[^1]: FunInCode. (n.d.). 【数之道】支持向量机SVM是什么，八分钟直觉理解其本质_哔哩哔哩_bilibili. Retrieved September 2, 2024, from https://www.bilibili.com/video/BV16T4y1y7qj/
[^2]: 周志华. (n.d.). 机器学习.