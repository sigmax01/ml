---
title: PyTorch
---

## 快速开始

这节会介绍几个在机器学习中比较常用的API.

### 加载数据

在PyTorch中, `touch.utils.data.Dataset`和`torch.utils.data.Dataloader`是两个用于数据加载和处理的核心工具. `Dataset`存储样本及其对应的标签, `Dataloader`在`Dataset`周围包装一个可迭代对象.

PyTorch提供属于特定领域的库, 例如TorchText, TorchVision和TorchAudio, 它们都包含一些已经预定义的`Dataset`数据集. 例如, `torchvision.datasets`模块包含用于许多现实世界视觉数据的`Dataset`对象, 例如COCO, FashionMNIST等等. 在本案例中, 使用CIFAR-10数据集. 每个TorchVision的`Dataset`包含两个参数: `transform`和`target_transform`, 分别用于修改样本和标签.

```py title="输入"
from torchvision import datasets
from torchvision.transforms import ToTensor
# 以CIFAR-10训练数据集为例
training_data = datasets.CIFAR10(
    root="drive/MyDrive/Data/CIFAR10",
    train=True,
    download=True,
    transform=ToTensor()
)
# 以CIFAR-10测试数据集为例
test_data = datasets.CIFAR10(
    root="drive/MyDrive/Data/CIFAR10",
    train=False,
    download=False, # 由于刚才已经下载过了, 所以这里设置为False
    transform=ToTensor()
)
```

``` title="输出"
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to drive/MyDrive/Data/CIFAR10/cifar-10-python.tar.gz
100%|██████████| 170M/170M [00:02<00:00, 62.9MB/s]
Extracting drive/MyDrive/Data/CIFAR10/cifar-10-python.tar.gz to drive/MyDrive/Data/CIFAR10
Files already downloaded and verified
```

其中, `root`指定数据集下载和保存的根目录路径, 例如, 设置`root="data"`, 数据集会被下载到当前工作目录下的`data`文件夹中. 当`download=True`的时候, 如果指定的`root`目录下没有数据集, 就会自动从网上下载, 如果已经下载过, 就不会再次下载. `transform`和`target_transform`是可选参数, 用于对样本和标签进行转换, 如`ToTensor()`会将PIL图像或NumPy数组转换为张量, 除了`ToTensor()`, 还可以进行其他转换操作, 如标准化, 裁剪, 调整大小等.

我们将`Dataset`作为参数传递给`Datalocader`. 这将包装数据集, 并提供对数据集的迭代访问. 在训练模型时, 我们通常会使用`Dataloader`, 因为它支持自动批量处理, 采样, 洗牌多进程数据加载.

```py title="输入"
from torch.utils.data import DataLoader
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```

``` title="输出"
Shape of X [N, C, H, W]: torch.Size([64, 3, 32, 32])
Shape of y: torch.Size([64]) torch.int64
```

`test_dataloader`是一个可迭代对象. 里面包含了整个测试数据集, 并将其分成了很多批次. 每次迭代`test_dataloader`的时候, 它会返回一个批次的数据, 直到遍历完整个数据集. 所以上面的`X`对应的是第一批数据, `N`表示的是批次大小, `C`表示的是通道数, `H`表示的是图像高度, `W`表示的是图像宽度.

## 创建模型

在PyTorch中创建模型的方法是写一个继承`nn.Module`的类. 在`__init__`函数中定义网络的层. 然后在`forward`函数中定义数据如何流经这些层. 为了加速我们的神经网络, 最好把操作转移到GPU或者MPS上面.

???+ tip "MPS是啥"

    MPS是Metal Performace Shader的缩写, 是由Apple提供的一种高性能图形和计算框架, 它是Metal API的一部分, 专为加速图形处理和机器学习任务设计. 它提供了一组高度优化的图形着色器, 用于处理诸如图像滤镜, 卷积操作和其他计算密集任务.

```py title='输入'
import torch
from torch import nn
# 选择CPU, GPU或者MPS设备用于训练
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

``` title='输出'
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

## 优化参数

为了训练模型, 我们需要定义一个损失函数和一个优化器.

???+ note "什么是优化器"

    优化器, Optimizer其实就是深度学习和机器学习中用于调整模型参数(权重和偏置)的算法工具, 以减少损失函数的值, 它是训练神经网络的必须组件, 通过反向传播不断调整参数. 优化器是基于梯度下降的, 可以选择的梯度下降算法有GD, SGD, Mini-batch GD(最常用). 还可以引入动量系数, 权重衰减等概念.

```py
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # lr是学习率
```

如果采用Mini-batch GD的话, 然后通过反向传播更新模型的参数.

```py title='输入'
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) # 返回的是整个数据集的大小
    model.train() # 作用见下方
    for batch, (X, y) in enumerate(dataloader): # dataloader一次迭代会返回一个批, enumerate的第一个参数是批次索引, 第二个参数是批次数据
        X, y = X.to(device), y.to(device)

        # 前向传播并计算误差
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        loss.backward()
        optimizer.step() # 更新参数
        optimizer.zero_grad() # 作用见下方

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

???+ note "`model.tran()`的作用"

    `model.train()`是一个将模型设置为训练模式的方法. 这个是必要的, 因为某些行为在训练和评估期间是不同的. 例如Dropout和BN. Dropout在训练模式下, 会随机将某些神经元的输出置为0, 防止过拟合, 而测试模式下, 不会丢弃任何神经元; BN在训练模式下, 会对当前批次的数据归一化, 在评估模式下, BN对全局的数据归一化, 这个和LN是不一样的, LN在训练和测试模式下没有不同.

???+ note "`optimizer.zero_grad()`的作用"

    这用于清除梯度(将参数的梯度设置为0). 这是由于在PyTorch里面, 梯度不会被自动清除, 需要手动清除. 如果不清除梯度, 会导致新的梯度叠加到之前的梯度上, 这是它的设计特征, 目的是为了支持某些高级功能, 如累积梯度.
