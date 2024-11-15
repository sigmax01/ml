---
title: 反向传播公式推导
comments: false
---

## 符号设定

- $t$: 真实标签
- $o_q$: 神经元$q$的输出, 即激活函数激活后的输出
- $w_{pq}$: 从神经元$p$到神经元$q$的权重
- $\eta$: 学习率
- $z_q$: 神经元$q$的加权输入, 即激活函数激活前的输出
- $f(z_q)$: 激活函数

## 前向传播

对于每一个神经元$q$, 有加权输入$z_q = \sum_p w_{pq}o_p$. 经过激活函数激活$o_q=f(z_q)=\frac{1}{1+e^{-z_q}}$. 这样一直传播到最后的输出神经元$v$, 选择误差函数为MSE, 计算误差为$L=\frac{1}{2}(t_v-o_v)^2$.

## 反向传播

目标是通过调整权重来最小化损失函数$L$, 核心思想是对损失函数$L$相对于权重$w_{pq}$的偏导数进行计算, 并根据该梯度更新权重. 

### 输出层梯度计算

对于输出层的神经元$v$, 和前一层的某一个神经元$u_t$, 我们需要计算损失函数$L$对于$w_{u_t v}$的导数, 有$\frac{\partial L}{\partial w_{u_t v}}=\frac{\partial L}{\partial o_v}\cdot \frac{\partial o_v}{\partial z_v}\cdot \frac{\partial z_v}{\partial w_{u_tv}}$. 其中, $\frac{\partial L}{\partial o_v}=-(t_v-o_v)$; $\frac{\partial o_v}{\partial z_v}=o_v(1-o_v)$; $\frac{\partial z_v}{\partial w_{u_t v}}=\frac{\partial (w_{u_1v}o_{u_1}+...+w_{u_t v}o_{u_t}+...w_{u_n v}o_{u_n}+b_v)}{\partial w_{u_t}}=o_{u_t}$. 将这三个部分结合起来, 我们得到该权重的导数为: $\frac{\partial L}{\partial w_{u_t v}}=-(t_v-o_v)\cdot o_v(1-o_v)\cdot o_{u_t}$. 我们将$\delta_v$定义为$\delta_v = \frac{\partial L}{\partial o_v}\cdot \frac{\partial o_v}{\partial z_v}=\frac{\partial L}{\partial z_v}=(t_v-o_v)\cdot o_v(1-o_v)$, 因此, 该导数可以写为$\frac{\partial L}{\partial w_{u_t v}}=-\delta_v o_{u_t}$. 而梯度就是$L$对所有的权重求导得到的一个向量: $\nabla L=(\frac{\partial L}{\partial w_{u_1}v},..., \frac{\partial L}{\partial w_{u_t}v},..., \frac{\partial L}{\partial w_{u_n}v})$

### 隐藏层梯度计算

对于隐藏层的某一个神经元$q$, 和前一层的某一个神经元$p_t$, 我们需要计算损失函数$L$对于权重$w_{p_tq}$的导数., 有$\frac{\partial L}{\partial w_{p_tq}}=\frac{\partial L}{\partial z_q}\cdot \frac{\partial z_q}{\partial w_{p_tq}}$. 其中, $\frac{\partial z_q}{\partial w_{p_tq}}=\frac{\partial (w_{p_1q} o_{p_1}+...+w_{p_tq} o_{p_t}+...+w_{p_mq} o_{p_m}+b_q)}{\partial w_{p_t q}}=o_{p_t}$. 现在, 我们需要计算损失$L$对于$z_q$的偏导数$\frac{\partial L}{\partial z_q}$, 注意, 隐藏层的神经元$q$不能直接影响损失$L$, 但是能通过后续的层传递给$L$. 根据链式法则, 我们有$\frac{\partial L}{\partial z_q}=\sum_k \frac{\partial L}{\partial o_k}\cdot \frac{\partial o_k}{\partial z_q}$, 这里的$k$表示$q$的后续一层的神经元$k$. 我们首先来计算$\frac{\partial o_k}{\partial z_q}$, $\frac{\partial o_k}{\partial z_q} = \frac{\partial o_k}{\partial z_k}\cdot \frac{\partial z_k}{\partial z_q}$, 其中, $o_k=f(z_k)$, 有$\frac{\partial o_k}{\partial z_k}=f'(z_k)$; 而$\frac{\partial z_k}{\partial z_q}=\frac{\partial z_k}{\partial o_q}\cdot \frac{\partial o_q}{\partial z_q}=\frac{\partial (w_{q_1k}o_{q_1}+...+w_{qk}o_{q}+...+w_{q_rk}o_{q_r})}{\partial o_q}=w_{qk}\cdot f'(z_q)$, 所以$\frac{\partial o_k}{\partial z_q}=f'(z_k)\cdot w_{qk}\cdot f'(z_q)$. 从而有$\frac{\partial L}{\partial z_q}=\sum_k \frac{\partial L}{\partial o_k}\cdot f'(z_k)\cdot w_{qk}\cdot f'(z_q)$. 我们定义$\delta_q = \frac{\partial L}{\partial z_q}=\sum_k \frac{\partial L}{\partial o_k}\cdot f'(z_k)\cdot w_{qk}\cdot f'(z_q)$. 而$\frac{\partial L}{\partial o_k}\cdot f'(z_k)=\delta_k$, 所以有$\delta_q=\sum_k \delta_k w_{qk}\cdot f'(z_q)$, 而$f'(z_q)=o_q(1-o_q)$, 所以最终有$\delta_q=o_q(1-o_q)\sum_k \delta_kw_{qk}$. 而梯度就是$L$对所有的权重求导得到的一个向量$\nabla L = (\frac{\partial L}{\partial w_{p_1 q}},..., \frac{\partial L}{\partial w_{p_t q}},..., \frac{\partial L}{\partial w_{p_m q}})$. 