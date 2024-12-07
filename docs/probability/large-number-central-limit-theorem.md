---
title: 大数定律与中心极限定理
comments: true
---

## 依概率收敛

设随机变量$X$与随机变量序列$\{X_n\}(n=1, 2, 3, ...)$, 如果对任意的$\epsilon>0$, 有$\lim_{n\rightarrow \infty}P\{|X_n-X|\geq \epsilon\}=0$或$\lim_{n\rightarrow \infty}P\{|X_n-X|<\epsilon\}=1$, 则称随机变量序列$\{X_n\}$依概率收敛于随机变量$X$, 记为$\lim_{n\rightarrow \infty}X_n=X(P)$或$X_n\stackrel{P}{\rightarrow}X(n\rightarrow \infty)$.

## 大数定律

### 切比雪夫大数定律

假设$\{X_n(n=1, 2, ...)\}$是相互独立的随机变量序列, 如果方差$DX_i(i\geq 1)$存在且一致有上界, 即存在常数$C$, 使$DX_i
\leq C$对一切$i\geq 1$均成立, 则$\{X_n\}$服从大数定律: $\frac{1}{n}\sum_{i=1}^{n}X_i\stackrel{P}{\rightarrow} \frac{1}{n}\sum_{i=1}^{n}EX_i$.

### 伯努利大数定律

假设$\mu_n$是$n$重伯努利试验中事件$A$发生的次数, 在每次试验中事件$A$发生的概率为$p(0<p<1)$, 则$\frac{\mu_0}{n}\stackrel{P}{\rightarrow}p$, 即对任意$\epsilon>0$, 有$\lim_{n\rightarrow \infty}P\{|\frac{\mu_0}{n}-p|<\epsilon\}=1$.

### 幸钦大数定律

假设$\{X_n\}$是独立同分布的随机变量序列, 如果$EX_i=\mu(i=1, 2, ...)$存在, 则$\frac{1}{n}\sum_{i=1}^{n}X_i\stackrel{P}{\rightarrow}\mu$, 即对任意$\epsilon>0$, 有$\lim_{n\rightarrow \infty}P\{|\frac{1}{n}\sum_{i=1}^n X_i-\mu|<\epsilon\}=1$.

## 中心极限定理

### 列维-林德伯格定理

假设$\{X_n\}$是独立同分布的随机变量系列, 如果$EX_i=\mu, DX_i=\sigma^2>0(i=1, 2, ...)$存在, 则对任意实数$x$, 有$\lim_{n\rightarrow \infty}P\{\frac{\sum_{i=1}^n X_i-n\mu}{\sqrt{n}\sigma}\leq x\}=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^x e^{-\frac{t^2}{2}}dt=\Phi(x)$.

### 棣莫弗-拉普拉斯定理

假设随机变量$Y_n\sim B(n, p)(0<p<1, n\geq 1)$, 则对任意实数$x$, 有$\lim_{n\rightarrow \infty}P\{\frac{Y_n-np}{\sqrt{np(1-p)}}\leq x\}=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^x e^{-\frac{t^2}{2}}dt=\Phi(x)$.