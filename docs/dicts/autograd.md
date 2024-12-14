---
title: 自动微分
comments: false
---

## 参考视频[^1]

<div style="position: relative; padding: 30% 45%;">
<iframe style="position: absolute; width: 100%; height: 100%; left: 0; top: 0;" src="//player.bilibili.com/player.html?isOutside=true&bvid=BV1PF411h7Ew&p=1&high_quality=1&autoplay=false&muted=false&t=5&as_wide=1" frameborder="yes" scrolling="no" allowfullscreen="true"></iframe>
</div>

## 微分方法

微分有四种方法: 手动微分, 数值微分, 符号微分和自动微分. 如图[^2]所示.

<figure markdown='1'>
  ![](https://img.ricolxwz.io/e8b99a520b4a39e9675ad00859811bf4.png){ loading=lazy width='500' }
</figure>

- **手动微分**: 一种通过手动推导和计算来求解函数导数的方法. 根据微积分中的求导法则(如链式法则, 乘法法则等), 手动推导出该函数的导数表达式. 并将这个手动推导出来的公式用计算机代码表示, 以便在给定输入值时计算出相应的导数值
- **数值微分**: 数值微分根据离散的数据点估计函数的导数, 这意味着, 它和其他方法的主要区别就是它计算出来的是导数的近似值. 常用的方法包括前向差分算法(使用某个点以及前一个点的值来估计导数), 后向差分算法(使用某个点及后一个点的值来估计导数), 中心差分算法(综合前向和后向差分, 通常提供更高的精度).
- **符号微分**: 符号微分需要对表达式进行符号解析和规则转换, 这可能涉及对复杂函数结构的模式匹配和简化, 将输入的函数转化为其对应的解析形式的导数表达式, 最终结果是一个以符号为基础的解析函数, 如, 使用符号微分对表达式进行求导, 可以直接得到类似$2x+3\sin(x)$这样明确的解析形式. 如果函数很复杂, 解析出来的导数表达式可能极其冗长甚至不便于使用. 它适用于需要明确解析导数公式的场合, 例如数学分析, 公式推导等. **它侧重于给出一个具体的解析导数表达式, 而不是在求出某个点的导数值**
- **自动微分**: 自动微分通过在数值计算的过程中对基本运算步骤的导数链式求导规则进行系统地分解和累积, 最终在一个具体点处高效地求得函数在该点的导数值. **它侧重于对给定输入点求出相应的导数值, 而不是求出导函数的解析式**. 它分为前向模式和反向模式.

[^1]: Deep_Thoughts (导演). (2021, 十一月 15). 13、详细推导自动微分Forward与Reverse模式 [Video recording]. https://www.bilibili.com/video/BV1PF411h7Ew/?spm_id_from=888.80997.embed_other.whitelist&t=5&bvid=BV1PF411h7Ew&vd_source=f86bed5e9ae170543d583b3f354fcaa9
[^2]: Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic differentiation in machine learning: A survey (No. arXiv:1502.05767). arXiv. https://doi.org/10.48550/arXiv.1502.05767
