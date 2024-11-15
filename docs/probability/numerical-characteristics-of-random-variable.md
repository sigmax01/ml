---
title: 随机变量的数字特征
comments: false
---

## 一维随机变量的数字特征

### 数学期望

#### 概念

设$X$为随机变量, $Y$是$X$的函数, $Y=g(X)$($g$为连续函数).

##### 离散型

如果$X$是离散型随机变量, 其分布列为$p_i=P\{X=x_i\}(i=1, 2, ...)$, 若级数$\sum_{i=1}^{\infty}x_ip_i$绝对收敛, 称随机变量$X$的数学期望存在, 并将级数$\sum_{i=1}^{\infty}x_ip_i$的和称为随机变量$X$的数学期望, 记为$E(X)$或$EX$, 即$EX=\sum_{i=1}^{\infty}x_ip_i$, 否则称$X$的数学期望不存在. 

若级数$\sum_{i=1}^{\infty}g(x_i)p_i$绝对收敛, 则称$Y=g(X)$的数学期望$E[g(X)]$存在, 且$E[g(X)]=\sum_{i=1}^{\infty}g(x_i)p_i$, 否则称$g(X)$的数学期望不存在.

##### 连续型

如果$X$是连续型随机变量, 其概率密度为$f(x)$, 若积分$\int_{-\infty}^{\infty}xf(x)dx$绝对收敛, 则称$X$的数学期望存在, 且$EX=\int_{-\infty}^{+\infty}xf(x)dx$, 否则称$X$的数学期望不存在. 

若积分$\int_{-\infty}^{+\infty}g(x)f(x)dx$绝对收敛, 则称$g(X)$的数学期望存在, 且$E[g(X)]=\int_{-\infty}^{\infty}g(x)f(x)dx$, 否则称$g(X)$的数学期望不存在.

#### 性质

- 对任意常数$a_i$和随机变量$X_i(i=1, 2, ..., n)$有$E(\sum_{i=1}^{n}a_iX_i)=\sum_{i=1}^n a_iEX_i$
- $Ec=c, E(aX+c)=aEX+c, E(X\pm Y)=EX\pm EY$
- 设$X$与$Y$相互独立, 则$E(XY)=EX\cdot EY, E[g_1(X)\cdot g_2(Y)]=E[g_1(X)]\cdot E[g_2(Y)]$
- 一般地, 设$X_1, X_2, ..., X_n$相互独立, 则$E(\prod_{i=1}^{n}X_i)=\prod_{i=1}^{n}EX_i, E[\prod_{i=1}^ng_i(X_i)]=\prod_{i=1}^n E[g_i(X_i)]$

### 方差/标准差/切比雪夫不等式 

#### 概念

设$X$是随机变量, 如果$E[(X-EX)^2]$存在, 则称$E[(X-EX)^2]$为$X$的方差, 记为$DX$, 即$DX=E[(X-EX)^2]=E(X^2)-(EX)^2$, 称$\sqrt{DX}$为$X$的标准差或均方差, 记为$\sigma(X)$. 称随机变量$X^*=\frac{X-EX}{\sqrt{DX}}$为$X$的标准化随机变量, 此时$EX^*=0, DX^*=1$.

#### 性质

- $DX\geq 0, E(X^2)=DX+(EX)^2\geq (EX)^2$
- $Dc = 0$
- $D(aX+b)=a^2DX$
- $D(X\pm Y)=DX+DY\pm 2Cov(X, Y)$
- 如果$X$与$Y$相互独立, 则$D(aX+bY)=a^2DX+b^2DY$
- 一般地, 如果$X_1, X_2, ..., X_n$相互独立, $g_i(x)$为$x$的连续函数, 则$D(\sum_{i=1}^n a_iX_i)=\sum_{i=1}^n a_i^2DX_i$且$D[\sum_{i=1}^ng_i(X_i)]=\sum_{i=1}^n D[g_i(X_i)]$

#### 切比雪夫不等式

如果随机变量$X$的期望$EX$和方差$DX$存在, 则对任意$\epsilon>0$, 有$P\{|X-EX|\geq \epsilon\}\leq \frac{DX}{\epsilon^2}$或$P\{|X-EX|<\epsilon\}\geq 1-\frac{DX}{\epsilon^2}$.

???+ note "笔记"

    由切比雪夫不等式可知, 当$DX$愈小时, 概率$P\{X-EX\}<\epsilon$愈大, 这表明方差是刻画随机变量与其期望值偏离程度的量, 是描述随机变量$X$"分散程度"特征的指标.

常用分布的期望和方差列表如下:

| 分布        | 分布列$P_x$或概率密度$f(x)$                  | 期望           | 方差              |
|-------------|--------------------------------------------|----------------|-------------------|
| 0-1 分布    | \( P\{X=k\} = p^k (1-p)^{1-k}, k=0,1 \)      | \( p \)        | \( p(1-p) \)      |
| 二项分布    | \( P\{X=k\} = C_n^k p^k (1-p)^{n-k}, k=0,1,...,n \) | \( np \)       | \( np(1-p) \)     |
| 泊松分布    | \( P\{X=k\} = \frac{\lambda^k e^{-\lambda}}{k!}, k=0,1,... \) | \( \lambda \)  | \( \lambda \)    |
| 几何分布    | \( P\{X=k\} = (1-p)^{k-1} p, k=1,2,... \)    | \( \frac{1}{p} \) | \( \frac{1-p}{p^2} \) |
| 正态分布    | \( f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \{ - \frac{(x-\mu)^2}{2\sigma^2} \}, -\infty < x < \infty \) | \( \mu \) | \( \sigma^2 \) |
| 均匀分布    | \( f(x) = \frac{1}{b-a}, a < x < b \)      | \( \frac{a+b}{2} \) | \( \frac{(b-a)^2}{12} \) |
| 指数分布    | \( f(x) = \lambda e^{-\lambda x}, x > 0 \) | \( \frac{1}{\lambda} \) | \( \frac{1}{\lambda^2} \) |

## 二维随机变量的数字特征

### 数学期望

#### 概念

设$X, Y$为随机变量, $g(X, Y)$为$X, Y$的函数($g$是连续函数).

##### 离散型

如果$(X, Y)$为离散型随机变量, 其联合分布为$p_{ij}=P\{X=x_i, Y=y_i\}(i, j=1, 2, ...)$, 若级数$\sum_i\sum_j g(x_i, y_j)p_{ij}$绝对收敛, 则定义$E[g(X, Y)]=\sum_i\sum_j g(x_i, y_j)p_{ij}$.

##### 连续型

如果$(X, Y)$为连续型随机变量, 其概率密度为$f(x, y)$, 若积分$\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}g(x, y)f(x, y)dxdy$绝对收敛, 则定义$E[g(X, Y)]=\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}g(x, y)f(x, y)dxdy$.

### 协方差/相关系数

#### 概念

如果随机变量$X$和$Y$的方差存在且$DX>0, DY>0$, 则称$E[(X-EX)(Y-EY)]$为随机变量$X$与$Y$的协方差, 并记为$Cov(X, Y)$, 即$Cov(X, Y)=E[(X-EX)(Y-EY)]=E(XY)-EX\cdot EY$.

其中$E(XY)$为:

- $\sum_i\sum_j x_iy_jP\{X=x_i, Y=y_j\}$ (离散型)
- $\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}xyf(x, y)dxdy$ (连续型)

称$\rho_{XY}=\frac{Cov(X, Y)}{\sqrt{DX}\sqrt{DY}}$为随机变量$X$与$Y$的相关系数, 如果$\rho_{XY}=0$, 则称$X$与$Y$不相关; 如果$\rho_{XY}\neq 0$, 则称$X$与$Y$相关.

#### 性质

- $Cov(X, Y)=Cov(Y, X), \rho_{XY}=\rho_{YX}$, $Cov(X, X)=DX, \rho_{XX}=1$
- $Cov(X, c)=0, Cov(aX+b, Y)=aCov(X, Y)$, $Cov(X_1+X_2, Y)=Cov(X_1, Y)+Cov(X_2, Y)$, 一般的, $Cov(\sum_{i=1}^n a_iX_i, Y)=\sum_{i=1}^n a_iCov(X_i, Y)$
- $|\rho_{XY}|\leq 1$
- 如果$Y=aX+b$, 则$\rho_{XY}=1, a>0$, $\rho_{XY}=-1, a < 0$