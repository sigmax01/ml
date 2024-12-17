---
title: 注意力机制
comments: false
---

## 注意力提示[^1]

### 生物学中的注意力提示

注意力是如何应用于视觉世界中的呢? 这要从当前十分普及的双组件(two-component)的框架开始讲起, 在这个框架中, 受试者基于**非自主性提示**和**自主性提示**有选择地引导注意力的焦点.

**非自主性提示**是基于环境中物体的突出性和易见性. 假设我们面前有五个物品, 一份报纸, 一篇论文, 一杯咖啡, 一本笔记和一本书, 所有的纸制品都是黑白印刷的, 但是咖啡杯是红色的. 换句话说, 这个咖啡杯在这种视觉环境中是突出和显眼的, 不由自主地引起注意, 所以我们会把视力最敏锐的地方放到咖啡上.

<figure markdown='1'>
  ![](https://img.ricolxwz.io/b0945597f22366e591a33a8e5bf9e76c.svg){ loading=lazy width='300' }
  <figcaption>由于突出性的非自主性提示(红杯子), 注意力不自主地指向了☕️</figcaption>
</figure>

喝咖啡后, 我们会变得兴奋并想读书, 所以转过了头, 重新聚焦眼睛, 然后看看书, 与上图的突出性导致的选择不同, 此时选择书是受到了认知和意识的控制, 因此注意力在基于**自主性提示**去辅助选择的时候将更为谨慎. 受试者的主观意愿推动, 选择的力量就更加强大.

<figure markdown='1'>
  ![](https://img.ricolxwz.io/cace7a0218a207cf8a11ce3638e1bebc.svg){ loading=lazy width='300' }
  <figcaption>依赖于任务的意志提示(想读一本书), 注意力被自主引导到📖上</figcaption>
</figure>

这两种提示方式会共同作用, 决定了在复杂的视觉场景中, 我们的注意力是如何自动或者有意识地引导到特定的目标上的.

### 查询, 键和值

自主性和非自主性的注意力提示解释了人类的注意力的方式, 下面看看如何通过这两种注意力提示, 用神经网络来设计注意力机制的框架.

首先, 考虑一个相对简单的状况, 即只使用非自主性提示, 要想将选择偏向于感官输入, 可以简单地使用参数化的全连接层, 甚至是非参数化的最大汇聚层或者平均汇聚层.

???+ note "解释上面这句话"

    **参数化的全连接层**: 当我们使用全连接层处理输入特征的时候, 它本质上是对不同的通道或者区域给予不同的权重. 这有点像是系统在底层利用显著特征为输入的各个部分打分, 从而突出某些区域, 抑制另一些区域. 在没有高层语义的情况下, 这种基于简单加权特征的变换就相当于在进行一种“无意识”的注意力分配.

    **非参数化的最大汇聚层**: 最大汇聚层会自动选择局部感受野中的最大值, 这可以看作是一种简单的注意力机制: 无论输入是什么, 我们只挑选出最突出的特征响应, 这其实就模仿了一种底层的非自主注意, 无需学习参数, 只要在特征图中有较强的激活, 就会在下游计算中得到突出.

    **非参数化的平均汇聚层**: 平均汇聚层将局部感受野内的特征平均化处理, 这可以看作是另一种简化的注意力分配, 并不强调单点特征的最大值, 而是对局部区域进行均匀整合, 这在底层注意的框架中可以被理解为一种默认的“温和”的选择策略, 将输入的不同部分在没有特别显著点的情况下平均对待.

    个人理解: Transformer的注意力层学到的是一种提问的方法, 这种提问的方法可能是人无法理解的, 它能够对输入进行query, 类似于“我是不是该读书了”, 然后对应的token的注意力分数就会上升, 而在全连接层或者这些pooling层里面, 无法对输入进行动态的调整, 只是傻傻的无生物性的对token进行权重的分配. CNN中的卷积层也类似, 它query的是“这部分是不是一本书”, 然后输入会产生动态响应. 说白了, 全连接层和pooling就是提取那些特殊的token, 然后给它们更高的权重, 类似于在报纸, 书和红色咖啡杯中选择红色咖啡杯. 而自注意力层就是根据$W_Q, W_K, W_V$提出主观性的疑问, 然后将注意力放在能够响应这些query的token上.

因此, “是否包含自主性提示”将注意力机制和全连接层和汇聚层区别开来. 在注意力机制的背景下, 自主性提示被称为“查询”. 给定任何查询, 注意力机制通过注意力汇聚(attention pooling)将选择引导至感官输入(sensory inputs). 在注意力机制中, 这些感官输入被称为值(value). 更通俗的解释, 每个值都与一个键(key)配对, 这可以想象为感官输入的非自主性提示. 如下图所示, 可以通过设计注意力汇聚的方式, 便于给定的查询(自主性提示)和键(非自主性提示)进行匹配, 这将引导得出最匹配的值(感官输入).

<figure markdown='1'>
![](https://img.ricolxwz.io/cfee07b985832c64b3de088a47ded3a5.svg){ loading=lazy width='400' }
<figcaption>注意力机制通过将注意力汇聚将查询(自主性提示)和键(非自主性提示)结合在一起, 实现对值(感官输入)的选择倾向</figcaption>
</figure>

上面的这种框架在整体结构和性能上占据了核心位置, 因此在下面的分析和讨论中, 这个框架是主要关注的对象. 需要注意的一点是, 注意力框架并非唯一, 研究者们设计了多种不同的注意力机制, 以适应不同的场景和需求. 

### 注意力可视化

平均汇聚层可以被视为输入的加权平均值, 其中各个输入的权重都是一样的. 而注意力汇聚得到的是加权平均的总和值, 其权重是在给定的查询和不同的键之间计算得出来的.

???+ note "为什么平均汇聚层的各个输入权重都是一样的?"

    平均汇聚层虽然是加权的, 但是对于每个输入元素来说, 它们的权重都是1/N, N是输入元素的数量.

为了可视化注意力权重, 需要定义一个`show_heatmaps`函数, 要展示的就是自注意力矩阵, 这个矩阵的每个元素的含义是某个query和某个key之间的注意力分数. 由于我们采用的是多头注意力, 所以输入`matrices`(注意是复数)的形状是(要显示的行数, 要显示的列数, 查询的数目, 键的数目). 例如, 如果有6个注意力头, 可以设置行数为2, 列数为3, 将6个注意力矩阵排列成2行3列的网格.

```py
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """显示矩阵热图"""
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
```

???+ note "解释上述代码"

    `matrices`正如我们所描述的那样, 是一个四维张量, 形状为`(num_rows, num_cols, num_queries, num_keys)`. `d2l.use_svg_display()`用于设置Matplotlib使用SVG格式显示图像. `detach()`用于分离梯度并将其转化为NumPy数组. `sharex=True`, `sharey=True`的含义是所有子图共享同一组x轴或者y轴刻度范围. `axes`是一个用于表示子图的数组, 每个子图对应一个`Axes`对象, 具体来说, 它是一个二维NumPy数组, 形状为`(num_rows, num_cols)`, 每个`axes[i, j]`对应网格中的第`i`行第`j`列的子图, `squeeze=False`用于控制返回`axes`对象的形状, 设置为`False`的时候, 即使只有一行或者一列, `axes`也会被强制转换为二维数组, 即`(1, 1)`的形状.

    第一层循环: 这个`i`对应的是行号, 这个`row_axes`对应的是`axes[0]`, `axes[1]`, ..., 这个`row_matrices`对应的是`matrices[0]`, `matrices[1]`, ...

    第二层循环: 这个`j`对应的是列号, 在第`0`行下, 这个`ax`对应的是`axes[0][0]`, `axes[0][1]`, ..., 即某个子图, 这个`matrix`对应的是`matrix[0][0]`, `matrix[0][1]`, 即某个头的注意力矩阵 

    `i == num_rows - 1`对应的是仅为最底部的子图设置x轴标签, `j == 0`对应的是仅仅为最左侧的子图设置y轴标签.

使用一个简单的例子进行演示. 在这个例子中, 仅当查询和键相同的时候, 注意力权重是1, 否则是0.

```py
attention_weights = torch.eye(10).reshape((1, 1, 10, 10))  # 有10个query和10个key, 并且只有1个头
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
```

<figure markdown='1'>
![](https://img.ricolxwz.io/569074acf4cb1b2c13f5b49179587517.svg){ loading=lazy width='250' }
</figure>

后续的小节会经常调用`show_heatmaps`函数来显示注意力权重.

[^1]: 10.1. 注意力提示—动手学深度学习 2.0.0 documentation. (不详). 取读于 2024年12月17日, 从 https://zh.d2l.ai/chapter_attention-mechanisms/attention-cues.html
