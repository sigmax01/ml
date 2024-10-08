---
title: 多维随机变量及其分布
comments: true
---

## 随机变量

### 定义

如果$X_1, X_2, ..., X_n$是定义在同一个样本空间$\Omega$上的$n$个随机变量, 则称$(X_1, X_2, ..., X_n)$为$n$维随机变量或$n$维随机向量, $X_i(i=1, 2, ..., n)$称为第$i$个分量.

当$n=2$时, 记$(X, Y)$为二维随机变量或二维随机向量. 

## 分布函数

### 定义

对任意的$n$个实数$x_1, x_2, ..., x_n$, 称$n$元函数$F(x_1, x_2, ..., x_n)=P\{X_1\leq x_1, X_2\leq x_2, ..., X_n\leq x_n\}$为$n$维随机变量($X_1, X_2, ..., X_n$)的联合分布函数.

当$n=2$时, 则对任意的实数$x, y$, 称二元函数$F(x, y)=P\{X\leq x, Y\leq y\}$为二维随机变量$(X, Y)$的联合分布函数, 简称分布函数, 记为$(X, Y)\sim F(x, y)$.

???+ note "笔记"

    $F(x, y)$是事件$A=\{X\leq x\}$和$B=\{Y\leq y\}$同时发生的概率.

### 性质

- $F(x, y)$是$x, y$的单调不减函数
    - 对任意固定的$y$, 当$x_1< x_2$时, $F(x_1, y)\leq F(x_2, y)$
    - 对任意固定的$x$, 当$y_1< y_2$时, $F(x, y_1)\leq F(x, y_2)$
- $F(x, y)$是$x, y$的右连续函数
    - $\lim_{x\rightarrow x_0^+}F(x, y)=F(x_0+0, y)=F(x_0, y)$
    - $\lim_{y\rightarrow y_0+}F(x, y)=F(x, y_0+0)=F(x, y_0)$
- $F(-\infty, y)=F(x, -\infty)=F(-\infty, -\infty)=0, F(+\infty, +\infty)=1$
- 对于任意$x_1<x_2, y_1<y_2$, 有$P\{x_1<X\leq x_2, y_1<Y\leq y_2\}=F(x_2, y_2)-F(x_2, y_1)-F(x_1, y_2)+F(x_1, y_1)\geq 0$

## 边缘分布函数

### 定义

设二维随机变量$(X, Y)$的联合分布函数为$F(x, y)$, 随机变量$X$与$Y$的分布函数$F_X(x)$与$F_Y(y)$分别称为$(X, Y)$关于$X$和关于$Y$的边缘分布函数, 由概率性质得

$F_X(x)=P\{X\leq x\}=P\{X\leq x, Y<+\infty\}=\lim_{y\rightarrow +\infty}P\{X\leq x, Y\leq y\}=\lim_{y\rightarrow +\infty}F(x, y)=F(x, +\infty)$

同理, 有$F_Y(y)=F(+\infty, y)$.

## 离散型随机变量及其概率分布

### 联合分布率 {#联合分布律}

如果二维随机变量$(X, Y)$只能取有限对值或可列对值$(x_1, y_1), (x_2, y_2), ..., (x_n, y_n), ...$, 则称$(X, Y)$为二维离散型随机变量. 称$p_{ij}=P\{X=x_i, Y=y_i\}, i, j=1, 2, ...$为$(X, Y)$的分布率或随机变量$(X, Y)$的联合分布率, 记为$(X, Y)\sim p_{ij}$, 联合分布率常用表格形式表示.

### 联合分布函数

设$(X, Y)$的概率分布为$p_{ij}, i, j=1, 2, ...$, 则$(X, Y)$的联合分布函数为: $F(x, y)=P\{X\leq x, Y\leq y\}=\sum_{x_i\leq x}\sum_{y_j\leq y}p_{ij}$.

设$G$是平面上的某个区域, 则$P\{(X, Y)\in G\}=\sum_{(x_i, y_j)\in G}p_{ij}$

### 边缘分布函数 {#边缘分布函数}

- $X$的边缘分布函数: $p_{i\cdot}=P\{X=x_i\}=\sum^{\infty}_{j=1}P\{X=x_i, Y=y_j\}=\sum^{\infty}_{j=1}p_{ij}(i=1, 2, ...)$
- $Y$的边缘分布函数: $p_{\cdot j}=P\{Y=y_j\}=\sum^{\infty}_{i=1}P\{X=x_i, Y=y_j\}=\sum^{\infty}_{i=1}p_{ij}(i=1, 2, ...)$

### 条件分布

如果$(X, Y)\sim p_{ij}(i, j=1, 2, ...)$, 对固定的$j$, 如果$p_{\cdot j}=P\{Y=y_j\}>0$, 则称$P\{X=x_i|Y=y_j\}=\frac{P\{X=x_i, Y=y_j\}}{P\{Y=y_j\}}=\frac{p_{ij}}{p_{\cdot j}}(i=1, 2, ...)$为$X$为$Y=y_j$条件下的条件分布.

同理, 对固定的$i$, 如果$p_{i\cdot}>0$, 可定义$Y$在$X=x_i$条件下的条件分布$P\{Y=y_j|X=x_i\}=\frac{p_{ij}}{p_{i\cdot}}(j=1, 2, ...)$.

## 连续型随机变量及其概率密度

### 概率密度/联合分布函数

如果二维随机变量$(X, Y)$的联合分布函数$F(x, y)$可以表示为$F(x, y)=P\{X\leq x, Y\leq y\}=\int_{-\infty}^y\int_{-\infty}^xf(u, v)dudv$, 其中$f(x, y)$是非负可积函数, 则称$(X, Y)$为二维连续型随机变量, 称$f(x, y)$为$(X, Y)$的概率密度, 记为$(X, Y)\sim f(x, y)$.

二元函数$f(x, y)$是概率密度的充分必要条件: $f(x, y)\geq 0, \int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}f(x, y)dxdy = 1$.

设$G$为平面上的某个区域, 则$P\{(X, Y)\in G\}=\iint_G f(x, y)dxdy$.

若$f(x, y)$在点$(x, y)$处连续, 则$\frac{\partial^2F(x, y)}{\partial x\partial y}=f(x, y)$.

若$F(x, y)$连续且可导, 则$(X, Y)$是连续型随机变量, 且$\frac{\partial^2F(x, y)}{\partial x\partial y}$是它的概率密度.

### 边缘概率密度/边缘分布函数

设$(X, Y)\sim f(x, y)$, 则$X$的边缘分布函数为$F_X(x)=F(x, +\infty)=\int^x_{-\infty}[\int^{+\infty}_{-\infty}f(u, v)dv]du$, 其概率密度$f_X(x)=\int_{-\infty}^{+\infty}f(x, y)dy$称$f_X(x)$为$(X, Y)$关于$X$的边缘概率密度.

同理, $Y$也是连续型随机变量, 其概率密度为$f_Y(y)=\int_{-\infty}^{+\infty}f(x, y)dx$.

### 条件分布函数/条件概率密度

设$(X, Y)\sim f(x, y)$, $f_X(x)>0$, 则称$f_{Y|X}(y|x)=\frac{f(x, y)}{f_X(x)}$为$Y$在$X=x$条件下的条件概率密度.

同理可定义$X$在$Y=y$条件下的条件概率密度$f_{X|Y}(x|y)=\frac{f(x, y)}{f_Y(y)}(f_Y(y)>0)$.

称$F_{Y|X}(y|x)=\int^y_{-\infty}f_{Y|X}(y|x)dy=\int^y_{-\infty}\frac{f(x, y)}{f_X(x)}dy$为$Y$在$X=x$条件下的条件分布函数.

同理可定义$X$在$Y=y$条件下的条件分布函数$F_{X|Y}(x|y)=\int^x_{-\infty}f_{X|Y}(x|y)dx=\int_{-\infty}^x\frac{f(x, y)}{f_Y(y)}dx$.

## 常见的随机变量分布类型

### 二维均匀分布

称$(X, Y)$在平面有界区域$D$上服从均匀分布, 如果$(X, Y)$的概率密度为$f(x, y)=\frac{1}{S_D}, (x, y)\in D; f(x, y)=0$, 其他. 其中$S_D$为区域$D$的面积.

### 二维正态分布

如果$(X, Y)$的概率密度为$f(x, y)=\frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2}}exp\{-\frac{1}{2(1-\rho^2)}[(\frac{x-\mu_1}{\sigma_1})^2-2\rho(\frac{x-\mu_1}{\sigma_1})(\frac{y-\mu_2}{\sigma_2})+(\frac{y-\mu_2}{\sigma_2})^2]\}$, 其中$\mu_1\in R, \mu_2\in R, \sigma_1>0, \sigma_2>0, -1<\rho<1$, 则称$(X, Y)$服从参数为$\mu_1, \mu_2, \sigma_1^2, \sigma_2^2, \rho$的二维正态分布, 记为$(X, Y)\sim N(\mu_1, \mu_2; \sigma_1^2, \sigma_2^2; \rho)$, 此时有

- $X\sim N(\mu_1, \sigma_1^2), Y\sim N(\mu_2, \sigma_2^2)$, $\rho$为$X$和$Y$的相关系数, 即$\rho=\frac{Cov(X, Y)}{\sqrt{DX}\sqrt{DY}}=\frac{Cov(X, Y)}{\sigma_1\sigma_2}$
- $X, Y$的条件分布都是正态分布
- $aX+bY(a\neq 0\ or\ b\neq 0)$服从正态分布
- $X$与$Y$相互独立的充要条件是$X$与$Y$不相关, 即$\rho=0$

## 随机变量的相互独立性

### 定义

设二维随机变量$(X, Y)$的联合分布函数为$F(x, y)$, 边缘分布函数为$F_X(x), F_Y(y)$, 如果对任意实数$x, y$都有$F(x, y)=F_X(x)\cdot F_Y(y)$, 则称$X$与$Y$相互独立, 否则称$X$与$Y$不相互独立.

如果$n$维随机变量$(X_1, X_2, ..., X_n)$的联合分布函数等于边缘分布函数的乘积, 即$F(x_1, x_2, ..., x_n)=F_1(x_1)\cdot F_2(x_2)\cdot ... \cdot F_n(x_n)$, 其中$F_i(x_i)(i=1, 2, ..., n)$为$X_i$的边缘分布函数, $x_i$为任意实数, 则称$X_1, X_2, ..., X_n$相互独立.

#### 充要条件

$n$个随机变量$X_1, X_2, ..., X_n$相互独立$\Leftrightarrow$对任意的$n$个实数$x_i(i=1, 2, ..., n)$, $n$个事件$\{X_1\leq x_1\}, \{X_2\leq x_2\}, ..., \{X_n\leq x_n\}$相互独立.

##### 离散型

- 设$(X, Y)$为二维离散型随机变量, 则$X$与$Y$相互独立$\Leftrightarrow$联合分布等于边缘分布相乘, 即$P\{X=x_i, Y=y_i\}=P\{X=x_i\}\cdot P\{Y=y_j\}(i, j=1, 2, ...)$
- 设$n$个离散型随机变量$X_1, X_2, ..., X_n$相互独立$\Leftrightarrow$对任意的$x_i\in D_i=\{X_i$一切可能值$\}(i=1, 2, ..., n)$有$P\{X_1=x_1, ..., X_n=x_n\}=\prod_{i=1}^n P\{X_i=x_i\}$

##### 连续型

- 设#$(X, Y)$为二维连续型随机变量, 则$X$与$Y$相互独立$\Leftrightarrow$概率密度等于边缘概率密度相乘, 即$f(x, y)=f_X(x)\cdot f_Y(y)$
- 设$(X_1, X_2, ..., X_n)$为$n$维连续型随机变量, 则$X_1, X_2, ..., X_n$相互独立$\Leftrightarrow$概率密度等于边缘概率密度相乘, 即$f(x_1, x_2, ..., x_n)=f_1(x_1)\cdot f_2(x_2)\cdot ...\cdot f_n(x_n)$, 其中$f_i(x_i)$为$X_i$的边缘概率密度

### 性质

- 设$X_1, X_2, ..., X_n$相互独立, 则其中任意$k(2\leq k\leq n)$个随机变量也相互独立
- 两个多维随机变量$(X_1, X_2, ..., X_n)$与$(Y_1, Y_2, ..., Y_m)$相互独立, 如果对任意实数$x_i(i=1, 2, ..., n)$与$y_j(j=1, 2, ..., m)$有$P\{X_1\leq x_1, X_2\leq x_2, ..., X_n\leq x_n; Y_1\leq y_1, Y_2\leq y_2, ..., Y_m\leq y_m\}=P\{X_1\leq x_1, X_2\leq x_2, ..., X_n\leq x_n\}\cdot P\{Y_1\leq y_1, Y_2\leq y_2, ..., Y_m\leq y_m\}$. 即联合分布函数等于各自的分布函数相乘$F(x_1, x_2, ..., x_n, y_1, y_2, ..., y_m)=F_1(x_1, x_2, ..., x_n)\cdot F_2(y_1, y_2, ..., y_m)$
- 设$(X, Y)$为二维离散型随机变量, $X$与$Y$独立, 则条件分布等于边缘分布: $P\{X=x_i|Y=y_i\}=P\{X=x_i\}(P\{Y=y_i\}>0)$, $P\{Y=y_i|X=x_i\}=P\{Y=y_i\}(P\{X=x_i\}>0)$
- 设$(X, Y)$为二维连续型随机变量, $X$与$Y$独立, 则条件概率密度等于边缘概率密度: $f_{X|Y}(x|y)=\frac{f(x, y)}{f_Y(y)}=f_X(x)(f_Y(y)>0)$, $f_{Y|X}(y|x)=\frac{f(x, y)}{f_X(x)}=f_Y(y)(f_X(x)>0)$
- 若$X_1, X_2, ..., X_n$相互独立, $g_1(x), g_2(x), ..., g_n(x)$为一元连续函数, 则$g_1(X_1), g_2(X_2), ..., g_n(X_n)$相互独立

## 多维随机变量函数的分布

### 定义

设$X, Y$为随机变量, $g(x, y)$为二元函数, 则以随机变量$X, Y$作为变量的函数$U=g(X, Y)$也是随机变量, 称之为随机变量$X, Y$的函数. 

### 求法

已知$(X, Y)$的联合分布, 求函数$Z=g(X, Y)$的分布, 首先要确定$X, Y$的类型, 而后才用相应的公式计算:

- 如果$(X, Y)$为二维离散型随机变量, 则$Z=g(X, Y)$也是离散型的, 先确定$Z$的值, 而后求其相应的概率, 求其相应的概率, 用一般解题模式(矩阵法)即可求得$Z$的分布
- 如果$X, Y$其中一个离散型的, 另一个是非离散型的, 我们总是将事件对离散型的一切可能值进行全集分解, 而后应用全概率公式求得$Z$的分布
- 如果$(X, Y)$是二维连续型随机变量, 即$(X, Y)\sim f(x, y)$, 则$Z=g(X, Y)$的分布函数$F(z)=P\{g(X, Y)\leq z\}=\iint_{g(x, y)\leq z}f(x, y)dxdy$

### 常见函数的分布及卷积公式

#### 和的分布

设$(X, Y)\sim f(x, y)$, 则$Z=X+Y$的概率密度为$f_Z(z)=\int_{-\infty}^{+\infty}f(x, z-x)dx=\int_{-\infty}^{+\infty}f(z-y, y)dy$.

当$X$与$Y$相互独立时, 有卷积公式$f_Z(z)=\int_{-\infty}^{+\infty}f_X(x)f_Y(z-x)dx=\int_{-\infty}^{+\infty}f_X(z-y)f_Y(y)dy$.

##### 常见分布的可加性

有些相互独立且服从同类型分布的随机变量, 其和的分布也是同类型的, 它们是二项分布, 泊松分布, 正态分布与$\mathcal{X}^2$分布.

设随机变量$X$与$Y$相互独立, 则:

- 若$X\sim B(n, p), Y\sim B(m, p)$, 则$X+Y\sim B(n+m, p)$, 注意$p$相同
- 若$X\sim P(\lambda_1), Y\sim P(\lambda_2)$, 则$X+Y\sim P(\lambda_1+\lambda_2)$
- 若$X\sim N(\mu_1, \sigma_1^2), Y\sim N(\mu_2, \sigma_2^2)$, 则$X+Y\sim N(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$
- 若$X\sim \mathcal{X}^2(n), Y\sim \mathcal{X}^2(m)$, 则$X+Y\sim \mathcal{X}^2(n+m)$

#### 差的分布

设$(X, Y)\sim f(x, y)$, 则$Z=X-Y$的概率密度为$f_Z(z)=\int_{-\infty}^{+\infty}f(x, x-z)dx=\int_{-\infty}^{+\infty}f(y+z, y)dy$.

当$X$与$Y$独立时, 有$f_Z(z)=\int_{-\infty}^{+\infty}f_X(x)(y+z)f_Y(y)dy$.

#### 积的分布

设$(X, Y)\sim f(x, y)$, 则$Z=XY$的概率密度为$f_Z(z)=\int_{-\infty}^{+\infty}\frac{1}{|x|}f(x, \frac{z}{x})dx=\int_{-\infty}^{+\infty}\frac{1}{|y|}f(\frac{z}{y}, y)dy$.

#### 商的分布

设$(X, Y)\sim f(x, y)$, 则$Z=\frac{X}{Y}$的概率密度为$f_Z(z)=\int_{-\infty}^{+\infty}|y|f(yz, y)dy$.

#### $max\{X, Y\}$分布

设$(X, Y)\sim F(X, Y)$, 则$Z=max\{X, Y\}$的分布函数为$F_{max}(z)=P\{max\{X, Y\}\leq z\}=P\{X\leq z, Y\leq z\}=F(z, z)$.

当$X$与$Y$独立时, $F_{max}(z)=F_X(z)\cdot F_Y(z)$.

#### $min\{X, Y\}$分布

设$(X, Y)\sim F(x, y)$, 则$Z=min\{X, Y\}$的分布函数为$F_{min}(z)=P\{min\{X, Y\}\leq z\}=P\{\{X\leq z\}\cup \{Y\leq z\}=P\{X\leq z\}+P\{Y\leq z\}-P\{X\leq z, Y\leq z\}=F_X(z)+F_Y(z)-F(z, z)$.

当$X$与$Y$独立时, $F_{min}(z)=F_X(z)+F_Y(z)-F_X(z)F_y(z)=1-[1-F_X(z)][1-F_Y(z)]$.