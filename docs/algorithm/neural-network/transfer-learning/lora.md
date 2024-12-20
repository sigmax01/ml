---
title: LoRA
comments: true
---

# LoRA[^1]

## 概要

NLP邻域最重要的一个范式是先使用通用领域的大规模数据进行预训练然后将其搬到特定的任务或者领域上. 重新训练模型所有参数的全量微调方法是不太可行的, 拿GPT-3 175B举个例子, 部署每一个任务特定的微调模型, 每个都要以训练175B参数为代价. 作者提出了低秩适配(Low-Rank Adaptation, LoRA), 它会冻结预训练模型的权重, 并将可训练的秩分解矩阵(trainable rank decomposition matrices)注入到Transformer模型架构的每一层中「怎么感觉和[Adapter](/algorithm/neural-network/transfer-learning/adapter)有点像」, 对于下游任务显著减少了可训练的参数量. 对比使用Adam优化器微调的GPT-3 175B, LoRA可以将可训练的参数量减少了10000倍, 同时, 对于GPU内存的要求减少了3倍. 使用LoRA调优后的模型性能和使用微调的模型性能相当或甚至更好, 但是前者有更少的可训练参数, 更高的训练吞吐量, **并且, 和adapters不一样的是, 没有额外的推理延迟**. ^^作者通过实验发现了语言模型适配中的秩缺失现象, 解释和确认了LoRA方法的有效性.^^ 相关的代码包括在PyTorch上使用LoRA可以在[这里](https://github.com/microsoft/LoRA)找到.

## 背景

### 微调的问题

很多在NLP领域的应用依赖于将一个大型的预训练语言模型搬到许多下游任务中. 这样的adaptation经常是通过微调实现的, 微调会更新预训练模型的所有参数, 然而, 这样的操作不是紧凑的. 随着大模型每隔几个月就被训练一次, 这从对GPT-2或者RoBERTa大模型而言仅仅是“不太方便”, 转变为对拥有1750亿可训练参数的GPT-3而言的部署挑战.

### Adapter/Prompt的问题

许多研究者试图通过仅调整部分参数或者对于每个新的任务都学习一个外部模块. 这样一来, 对于每个任务只需要在预训练模型之外存储和加载一小部分task-specific的参数, 极大提高了在部署阶段的操作效率. 🌟然而, 现有的方法通常存在一些问题, 一方面, 这些方法通过增加模型的深度来实现某些功能(如使用adapter模块[^2]), 另一方面, 这些方法可能会减少模型能够处理的序列长度(如使用Prompt调优, 加入了一些额外的指令), 即模型在处理长文本或长时间序列数据时的能力受到限制, 这两个因素导致推理时间的显著增加. 更重要的是, 这些方法常常无法匹敌微调的基线, 导致需要在效率和性能之间做一个trade-off.🌟

下面是详细探讨.

作者想要解决的这个微调效率低下的问题已经存在很久了. 自动迁移学习出现, 已经有许多工作想要使得模型的调优变得更加parameter&compute高效. 以语言模型举一个例子, 有两种主流的方法: 一个是添加adapter层. 一个是优化输入层激活(其实就是Prompt调优?). 但是, 这些策略都有它们各自的限制, 尤其是在大规模的对延迟很敏感的领域.

**Adapter层会导致推理延迟**. Adapters有很多变种, 作者关注的是由Houlsby等人[^2]提供的原始设计, 这个设计在每个Transformer的block中都有两个adapter模块. Lin等人[^8]提出了一种更为简化的设计, 每个块中只有一个适配器层, 但是有一个额外的LN层. 虽然可以通过剪枝或者利用多任务来降低延迟, 但是由adapter这一层带来的额外计算量是无法绕过的. 这看起来不成问题, 因为adapter被设计成一种"瓶颈结构", 所增加的参数量非常少(有时甚至是原预训练模型的$1\%$之内), 但是, ^^大型神经网络依赖于硬件上的并行性使得延迟降低, 而adapter层是串行处理的, 特别是在线推理的场景下, batch_size通常为1, adapter层推理时间的占比会显著增加.^^ 实验显示, 在没有模型并行的情况下, 在单块GPU上运行GPT-2, 即使adapter的瓶颈维度非常小, 也会显著增加延迟(见下表). 

<figure markdown='1'>
  ![](https://img.ricolxwz.io/b4cce089d30456a9b8007148c07c08ba.webp#only-light){ loading=lazy width='600' }
  ![](https://img.ricolxwz.io/b4cce089d30456a9b8007148c07c08ba_inverted.webp#only-dark){ loading=lazy width='600' }
  <figcaption>GPT-2中型模型单次前向传播的100次实验平均值, 单位毫秒. 使用的是NVIDIA Quadro RTX8000. $|\Theta|$表示的是adapter层的可训练参数数量. AdapterL和AdapterH是两种adapter调优的方法</figcaption>
</figure>

<figure markdown='1'>
![](https://img.ricolxwz.io/51b932b2b2603d0713554d1306e6dd04.webp#only-light){ loading=lazy width='600' }
![](https://img.ricolxwz.io/51b932b2b2603d0713554d1306e6dd04_inverted.webp#only-dark){ loading=lazy width='600' }
<figcaption>以没有adapter(r=0)作为基线, 比较不同seq_len, batch_size, r下adapter延迟增加的百分比. 上面的一条是AdapterH, 下面的一条是AdapterL. 大batch_size和seq_len能够缓解延迟. 在线推理, 短序列的情况下, 这种推理时间增加的百分比高达30%</figcaption>
</figure>
    
???+ note "如何计算attention层和adapter层的复杂度"

    设$N$是序列长度, $D$是特征维度, $B$是batch_size, $D'$是adapter模块的瓶颈维度, 通常$D'\ll D$.

    Attention层复杂度计算: Q和K的维度都是$N\cdot D$. 两个矩阵$m\cdot n$和$n\cdot p$相乘的复杂度是$O(m\cdot p\cdot n)$. 所以QK^T的复杂度是$O(N^2\cdot D)$. Softmax处理QK^T结果的复杂度是$O(N^2)$, 将softmax结果和V进行矩阵乘法, V的维度是$N\cdot D$, 继续运用矩阵乘法的复杂度计算公式, 复杂度是$O(N^2\cdot D)$. 总共有$B$个序列, 所以结果为$O(B\cdot N^2\cdot D)$.

    Adapter层复杂度计算: 对于输入特征维度为$n$和输出特征维度为$m$的线性层, 其时间复杂度为$O(n\times m)$. Adapter的复杂度可以表示为$O(N\cdot D'+D'\cdot N)=O(N\cdot D')$, 总共有$B$个序列, 所以结果为$O(B\cdot N\cdot D')$.

???+ question "为什么adapter层的推理时间占比会随batsh_size减小而增加"

    自注意力层的时间复杂度是$O(B\cdot N^2\cdot D)$. Adapter层的时间复杂度为$O(B\cdot N\cdot D')$. 既然自注意力层和adapter层都和批次大小$B$是线性相关的, 那么为什么adapter层的推理时间占比会随着batch_size的减小而增加呢? 单从复杂度分析感觉这两个占比应该是没有变化的? 可能推理时间和复杂度不是正比关系, 因为不同层在并行性有差异?

    下面是LoRA开发者的回答: https://github.com/microsoft/LoRA/issues/129#issuecomment-1783956680

    > This is because when the bsz is small, we need to parallelize over width to gain the best hardware efficiency. Adapter adds to the depth, which has to be processed sequentially.

???+ note "什么多任务降低延迟"

    多任务降低延迟是指通过同时处理多个任务来减少整体的响应时间和执行延迟. 这种方法通常涉及将一个大任务拆分为多个子任务, 并利用并行处理的方式来提高效率.

???+ note "什么是在线推理"

    在线推理, 也称为实时推理, 是指在机器学习和深度学习应用中, 模型对输入数据进行即时预测或者决策的过程. 与批量推理不同, 在线推理通常处理的是单个或者少量的数据样本, 并且需要在极短的时间内给出结果, 以满足实时性要求. 当每个批次只处理一个请求的时候, 即batch_size为1的时候, 模型几乎完全无法利用硬件的并行能力. 因为GPU一次只能训练一个序列的Q, K, V, 而batch_size为32的时候, 是32个序列的Q, K, V并行训练.

当我们需要对超大规模模型进行切分时, 这个问题会变得更加严重. 因为由于adapter增加的额外的深度需要更多同步GPU的操作, 例如`AllReduce`和`Broadcast`, 除非我们多次冗余地存储adapter的参数, 这偏离了设计初衷.

**直接优化提示词很困难.** 另外一个方向, 例如前缀调优, 面临着不同的挑战. 作者观察到前缀调优难以优化, 并且性能在可训练参数中呈现非单调变化, 这证实了原始论文中的类似观察效果. ^^最重要的是, 序列长度中的大部分会被提示词占据, 大大减少了可用序列长度.^^

### LoRA的提出

作者受到Li等人[^6]和Aghajanyan等人[^7]研究工作的启发. 🌟他们的研究表明, 学习到的过参数化模型(Over-Parameterized Model)实际上存在于一个低内在维度空间(Low Intrinsic Dimension)中. 基于这一发现, 作者假设在模型的**调优/适应**过程中, 权重的变化**也**具有低内在秩(Intrinsic Rank). 这促使作者提出了低秩适应(LoRA)的方法.🌟 ^^LoRA允许我们通过在调优过程中优化某些特定密集层(Dense Layer)的权重变化的秩分解矩阵(Rank Decomposition Matrices)来间接"训练"这些密集层, 同时保持预训练的权重冻结.^^ 如下图所示.

<figure markdown='1'>
![](https://img.ricolxwz.io/1ed1d2f0549dd53c0a7684355d9f56a4.webp#only-light){ loading=lazy width='210' }
![](https://img.ricolxwz.io/1ed1d2f0549dd53c0a7684355d9f56a4_inverted.webp#only-dark){ loading=lazy width='210' }
<figcaption>LoRA. 会对原始权重和调整权重进行相加操作, $d$表示密集层的理论最大秩, $r$表示在调优中的权重变换矩阵的秩</figcaption>
</figure>

???+ note "什么是过参数化模型"

    过参数化, Overparameterization是指模型具有过多的自由度, 即参数数量大于训练数据的样本数量.

拿GPT-3 175B举个例子, 某些密集层的理论最大秩为12288(即图中的$d=12288$, GPT-3 的$d_{model}=12288$), 但是在实际调优的过程中其**权重变化受限于低维度**(根据Li等人的研究), 如图中$r$甚至可以是$1$或者$2$已经足够, 而远远不需要达到理论上的最大秩, 这使得LoRA在存储和计算上非常高效.

### LoRA优点

LoRA有以下的优点:

- 一个预训练模型可以被共享, 并构建用于不同任务的许多小型LoRA模块. 我们可以冻结共享模型, 并通过替换上图的$\bm{A}$和$\bm{B}$来有效切换任务, 从而显著降低存储需求和任务切换开销
- LoRA通过优化注入较小的低秩矩阵, 使得训练过程更加高效, 并降低了硬件门槛, 尤其是在使用自适应优化器时候的效果显著, 提升可达到3倍. 传统的训练方法需要计算大部分参数的梯度并维护优化器状态

    ???+ note "自适应优化器"

        自适应优化器是一类能够根据参数的历史梯度信息动态调整每个参数学习率的优化算法, 常见的自适应优化器包括Adam, AdaGrad, RMSProp, 主要优势在于, 无需手动调节每个参数的学习率, 优化器会根据梯度的一阶(均值)和二阶(方差)矩估计动态调整学习率, 通过合理调整学习率, 可以显著加快模型的收敛速度, 并提高稳定性.

        由于自适应优化器需要为每个参数维护其状态信息(如一阶和二阶矩估计), 由于在使用LoRA的时候, 其参数量更少, 所以要维护的参数的状态信息更少, 所以计算效率更好, 存储需求更低.

- 🌟LoRA在调优的时候对密集层的权重是一个**间接的, 线性的**调整, 在部署到特定任务前, LoRA对于权重的调整会被预先整合到原始密集层的权重中, 即在推理过程中这个LoRA模块就是不存在的. 而Adapter模块在推理的时候仍然是存在的, 因为adapter是在transformer层之间插入独立的神经网络层, 这些层包含独立的权重和偏置参数, 不能被简单地整合到某个层的权重中. 因此, LoRA的延迟和原始网络是相等的, 而Adapter新增了adapter模块内部的延迟🌟
- LoRA和其他方法在功能和实现上相互独立, 互不干扰, 所以可以和其他方法协同工作, 如Prefix-Tuning, 进一步提升模型的性能和效率

### 术语和规范

作者常常会引用Transformer架构并且使用它那里常用的术语. 我们称Transformer层的输入输出维度大小为$d_{model}$. 使用$W_q$, $W_k$, $W_v$, $W_o$指代Q/K/V/Output投影矩阵. $W$或者$W_0$指代预训练的权重矩阵, $\Delta W$指的是在调优过程中累积的权重更新. 使用$r$表示LoRA模块的秩. 并且, 使用Adam优化器优化模型, 在Transformer内部使用一个维度为$d_{fnn}=4\times d_{model}$的MLP(这里的维度指的是隐藏层维度).

## 命题

由于作者的proposal不依赖于训练目标, 所以他们以语言模型作为use case. 下面是一个语言建模问题的简要表述, 特别的, 是针对特定任务提示下条件概率的最大化.

假设我们给定了一个预训练的自回归语言模型$P_{\Phi}(y|x)$, 参数为$\Phi$. 举个例子, $P_{\Phi}(y|x)$可以是一个基于Transformer架构的通用多任务学习器如GPT. 考虑将这个预训练模型用于下游的条件文本生成任务, 如总结, 机器阅读理解(MRC)和自然语言转SQL(NL2SQL). 每个下游任务都由一个上下文-目标对的数据集表示: $\mathcal{Z}=\{(x_i, y_i)\}_{i=1, ..., N}$, 其中$x_i$, $y_i$都是序列. 例如, 在NL2SQL中, $x_i$是自然语言的查询, $y_i$是对应的SQL代码. 对于总结任务, $x_i$是文章的内容, $y_i$是总结.

???+ note "为什么GPT这个预训练模型本身就是多任务的"

    GPT之所以是一个通用多任务模型, 关键在于其在多样化的数据上大规模预训练, 在预训练阶段就已经赋予了模型广泛的多任务学习能力, 使其能够在无需专门调优的情况下, 适应并完成各种不同的任务.

在全量微调中, 模型会被初始化为预训练的权重$\Phi_0$并且通过梯度下降最大化下列条件语言建模的优化目标更新到$\Phi_0+\Delta \Phi$.

$$\max_{\Phi} \sum_{(x, y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log (P_\Phi(y_t | x, y_{<t}))$$

全量微调最大的缺点就是对于每一个下游的任务, 我们都要学习一个不同的$\Delta \Phi$, 并且$\Delta \Phi$的参数量$|\Delta \Phi|$和原始参数量$|\Phi_0|$是相等的. 因此, 如果预训练模型非常大(像GPT-3 $|\Phi_0|\simeq 175$B), 存储和部署许多独立的微调模型是具有挑战性的, 甚至可能根本不可行.

在这篇论文中, 作者采用了一种parameter-efficient的方法: 任务相关的参数增量$\Delta \Phi=\Delta \Phi(\Theta)$由一部分数量较小的参数$\Theta$生成, 参数量远远小于原始参数量$|\Theta|\lll |\Phi_0|$. 寻找$\Delta \Phi$的过程就是在优化$\Theta$.

$$\max_{\Theta} \sum_{(x, y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log (p_{\Phi_0 + \Delta \Phi(\Theta)}(y_t | x, y_{<t}))$$

注意到, 这里拿到的是那一小部分的参数$\Theta$, 而不是整体参数$\Phi$. 在下面的小节中, 作者提出了一种使用低秩表示方法表示$\Delta \Phi$使得在计算和存储上都特别高效. 当他们的预训练模型选择的是GPT-3 175B的时候, 这一小部分可训练参数的量$|\Theta|$是原始参数量$|\Phi_0|$的$0.01\%$.

## 方法

### 低秩参数化更新矩阵

一个神经网络包含许多执行矩阵乘法操作的密集层. 这些密集层的权重矩阵常常是满秩的. 当将其搬到特定任务的时候, Aghajanyan等人的工作显示了预训练模型有低内在维度, 并且即使将模型的高维参数空间通过随机投影降维到一个较小的子空间, 模型仍然能够高效地学习(不是调优). 作者受到他们的启发, 推测在调优的时候对权重的更新也具有低内在秩. 

???+ question "为什么权重矩阵是满秩的, 而模型本身具有低内在秩"

    这可能是因为: 尽管每一层individually满秩, 多个层之间的参数可能高度相关, 导致整个模型的参数空间中是有高度的冗余和相关的, 所以处在低内在秩状态.

对于一个预训练的权重矩阵$W_0\in \mathbb{R}^{d\times k}$. 我们通过将权重的变化表示为一个低秩分解, 即$\Delta W=BA$, 其中, $B\in \mathbb{R}^{d\times r}$, $A\in \mathbb{R}^{r\times k}$, $r\ll \min(d, k)$. 在调优的过程中, $W_0$是被冻结的, 它不会收到权重更新, 而$A$和$B$会被更新, 包含可训练参数. 注意, $W_0$和$\Delta W=BA$都会被相同的输入相乘, 并且它们各自的输出会进行coordinate-wise相加(Adapter就无法直接相加, 非adapter和adapter是两波不同的参数). 对于$h=W_0x$, 我们修改后的前向传播为:

$$h=W_0x+\Delta W_x=W_0x+BAx$$

这个公式已经被表示在第一张图中. ^^作者对$A$进行标准正态分布初始化, 对$B$初始化为零矩阵, 使得$\Delta W=BA=0$, 这个做法类似于adapter的初始化^^, 保证在开始的时候, LoRA模块不会显著改变模型的输出, 这对于保证模型的初始性能和稳定性非常重要. 

作者还对更新量$\Delta Wx$的学习率$\alpha$进行了缩放: 乘以$\frac{1}{r}$, 这是因为, 在没有缩放的情况下, 同样的学习率$\alpha$下改变秩$r$会直接影响权重更新的尺度. 通过引入与$\frac{1}{r}$成比例的缩放, 我们可以对梯度尺度进行归一化, 使其对$r$的变化不太敏感, 这使得超参数$r$的调整更加容易, 因为单个$\alpha$就可以在不同的$r$下表现良好, 即$\frac{\alpha}{r}$才是真正的学习率. 简单的说, 就是通过$r$的改变使得在不同的$r$下Adam优化器能够以相同的策略调整$\alpha$, 省去了不同$r$下调优$\alpha$的必要.

**全量微调的泛化**. 一般的部分微调允许只微调所有预训练权重的一部分, 如只调整顶层的参数. LoRA在这个的基础上更进了一步, **不需要在调优的过程中, 要求权重的累积梯度更新矩阵必须具有满秩**. 这意味着当将LoRA应用于所有的权重矩阵并训练所有的$\Delta W$, 并且将LoRA的秩$r$为对应权重矩阵的秩, 那么我们就得到了一个全量微调, 即两者是等价的. 换句话说, 当我们增加可训练参数量的时候, 训练LoRA会随着秩的增加收敛于全量微调, 训练adapter模块收敛于训练一个隐藏层较大的MLP, 基于前缀的方法收敛于一个无法接受长实际可训练序列的模型.

**没有额外的推理延迟**. 当在production环境下部署的时候, 我们可以特别的计算并存储更新后的权重矩阵$W=W_0+BA$并且按照一个没有添加LoRA模块的原始网络那样进行推理. 注意, $W_0$和$BA$的形状都是$\mathbb{R}^{d\times k}$. ^^这意味着, 当我们要切换到另一个下游任务的时候, 我们得恢复原始的$W=W_0$, 方法就是减去$BA$, 然后加上另外一个$B'A'$^^. 这个方法只需要一点点内存空间, 最重要的是, 这样的操作不会产生额外的推理延迟相比于adapter模块.

### LoRA用于Transformer

在实际运用中, 我们可以将LoRA运用于神经网络的任何权重矩阵上以减少可调优的参数. 在Transformer架构中, 自注意力模块上, 有4个权重矩阵($W_q, W_k, W_v, W_o$); MLP模块上, 有两个权重矩阵. 我们将$W_q$(或$W_k, W_v$视为一个维度为$d_{model}\times d_{model}$的单一矩阵, 即使产生的$Q$, $K$, $V$矩阵会被切分为注意力头). ^^作者仅调整了下游任务的注意力权重, 而不对MLP的权重进行调整, 这是为了简化和提高参数效率.^^ 作者还进一步研究了在Transformer中采用不同类型的注意力权重矩阵的影响. 他们将在后续研究中探讨调整MLP层, LN层的权重和偏置项的影响.

**实践优势和限制.** 最明显的优势来自节约了内存的使用量. 对于一个采用Adam优化器训练的大Transformer模型, 如果$r\ll d_{model}$, VRAM的使用率下降可以高达$2/3$, 因为和全量微调来说, 不需要存储那些冻结参数在优化器中的状态. 例如, 在GPT-3 175B上, 他们将在训练时VRAM的使用量从1.2TB降低到了350GB, 甚至于, 在$r=4$, 并且只对$Q$和$V$投影矩阵进行LoRA低秩矩阵更新的时候, 又减少了大概10000倍, 从350GB减小到35MB. 这允许他们的更少的GPUs上训练并且避免IO瓶颈. 另外一个显著的优势是他们可以通过替换掉LoRA的权重来以极低的成本实现在任务之间的调整, 甚至可以实时调整. 和全量微调相比, 他们还在GPT-3 175B的微调中观察到了25%的加速, 因为不需要计算绝大多数参数的梯度. LoRA也有limitations, 例如, 如果选择将$A$, $B$吸收到$W$中以消除推理延迟, 那么在一个前向传播中批量处理本应由不同$A$, $B$处理的不同任务并非易事. 当然, 在延迟不是那么关键地时候, 也可以动态地选择用于批处理地LoRA模块. 

[^1]: Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-rank adaptation of large language models (No. arXiv:2106.09685). arXiv. https://doi.org/10.48550/arXiv.2106.09685
[^2]: Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., Laroussilhe, Q. de, Gesmundo, A., Attariyan, M., & Gelly, S. (2019). Parameter-efficient transfer learning for NLP (No. arXiv:1902.00751). arXiv. https://doi.org/10.48550/arXiv.1902.00751
[^3]: Rebuffi, S.-A., Bilen, H., & Vedaldi, A. (2017). Learning multiple visual domains with residual adapters (No. arXiv:1705.08045). arXiv. https://doi.org/10.48550/arXiv.1705.08045
[^4]: Li, X. L., & Liang, P. (2021). Prefix-tuning: Optimizing continuous prompts for generation (No. arXiv:2101.00190). arXiv. https://doi.org/10.48550/arXiv.2101.00190
[^5]: Lester, B., Al-Rfou, R., & Constant, N. (2021). The power of scale for parameter-efficient prompt tuning (No. arXiv:2104.08691). arXiv. https://doi.org/10.48550/arXiv.2104.08691
[^6]: Li, Y., & Liang, Y. (2019). Learning overparameterized neural networks via stochastic gradient descent on structured data (No. arXiv:1808.01204). arXiv. https://doi.org/10.48550/arXiv.1808.01204
[^7]: Aghajanyan, A., Zettlemoyer, L., & Gupta, S. (2020). Intrinsic dimensionality explains the effectiveness of language model fine-tuning (No. arXiv:2012.13255). arXiv. https://doi.org/10.48550/arXiv.2012.13255
[^8]: Lin, Z., Madotto, A., & Fung, P. (2020). Exploring versatile generative language model via parameter-efficient transfer learning (No. arXiv:2004.03829). arXiv. https://doi.org/10.48550/arXiv.2004.03829
