---
title: PyTorch
---

## 快速开始

这节会介绍几个在机器学习中比较常用的API.

### 加载数据

在PyTorch中, `touch.utils.data.Dataset`和`torch.utils.data.Dataloader`是两个用于数据加载和处理的核心工具. `Dataset`存储样本及其对应的标签, `Dataloader`在`Dataset`周围包装一个可迭代对象.

PyTorch提供属于特定领域的库, 例如TorchText, TorchVision和TorchAudio, 它们都包含一些已经预定义的`Dataset`数据集. 例如, `torchvision.datasets`模块包含用于许多现实世界视觉数据的`Dataset`对象, 例如COCO, CIFAR-10等. 在本案例中, 使用FashionMNIST数据集. 每个TorchVision的`Dataset`包含两个参数: `transform`和`target_transform`, 分别用于修改样本和标签.

```py title="输入"
from torchvision import datasets
from torchvision.transforms import ToTensor
# 以FashionMNIST训练数据集为例
training_data = datasets.FashionMNIST(
    root="drive/MyDrive/Data/FashionMNIST",
    train=True,
    download=True,
    transform=ToTensor()
)
# 以FashionMNIST测试数据集为例
test_data = datasets.FashionMNIST(
    root="drive/MyDrive/Data/FashionMNIST",
    train=False,
    download=False, # 由于刚才已经下载过了, 所以这里设置为False
    transform=ToTensor()
)
```

``` title="输出"
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to drive/MyDrive/Data/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz
100%|██████████| 26.4M/26.4M [00:01<00:00, 20.9MB/s]
Extracting drive/MyDrive/Data/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz to drive/MyDrive/Data/FashionMNIST/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to drive/MyDrive/Data/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz
100%|██████████| 29.5k/29.5k [00:00<00:00, 427kB/s]
Extracting drive/MyDrive/Data/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz to drive/MyDrive/Data/FashionMNIST/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to drive/MyDrive/Data/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
100%|██████████| 4.42M/4.42M [00:00<00:00, 6.11MB/s]
Extracting drive/MyDrive/Data/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to drive/MyDrive/Data/FashionMNIST/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to drive/MyDrive/Data/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
100%|██████████| 5.15k/5.15k [00:00<00:00, 18.9MB/s]
Extracting drive/MyDrive/Data/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to drive/MyDrive/Data/FashionMNIST/FashionMNIST/raw
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
Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64]) torch.int64
```

`test_dataloader`是一个可迭代对象. 里面包含了整个测试数据集, 并将其分成了很多批次. 每次迭代`test_dataloader`的时候, 它会返回一个批次的数据, 直到遍历完整个数据集. 所以上面的`X`对应的是第一批数据, `N`表示的是批次大小, `C`表示的是通道数, `H`表示的是图像高度, `W`表示的是图像宽度.

## 创建模型

在PyTorch中创建模型的方法是写一个继承`nn.Module`的类. 在`__init__`函数中定义网络的层. 然后在`forward`函数中定义数据如何流经这些层. 为了加速我们的神经网络, 最好把操作转移到GPU或者MPS上面.

??? tip "MPS是啥"

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

model = NeuralNetwork().to(device) # 将模型送到GPU上(包括其初始化的参数)
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

??? note "什么是优化器"

    优化器, Optimizer其实就是深度学习和机器学习中用于调整模型参数(权重和偏置)的算法工具, 以减少损失函数的值, 它是训练神经网络的必须组件, 通过反向传播不断调整参数. 优化器是基于梯度下降的, 可以选择的梯度下降算法有GD, SGD, Mini-batch GD(最常用). 还可以引入动量系数, 权重衰减等概念.

```py
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # lr是学习率
```

如果采用Mini-batch GD的话, 然后通过反向传播更新模型的参数.

```py
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) # 返回的是整个数据集的大小
    model.train() # 作用见下方
    for batch, (X, y) in enumerate(dataloader): # dataloader一次迭代会返回一个批, enumerate的第一个参数是批次索引, 第二个参数是批次数据
        X, y = X.to(device), y.to(device) # 将数据送到GPU上

        # 前向传播并计算误差
        pred = model(X)
        loss = loss_fn(pred, y) # 可调用对象

        # 反向传播
        loss.backward()
        optimizer.step() # 更新参数
        optimizer.zero_grad() # 作用见下方

        if batch % 100 == 0: # 每100批手动打印一次损失和当前进度(占总体百分比)
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

??? note "`model.train()`的作用"

    `model.train()`是一个将模型设置为训练模式的方法. 这个是必要的, 因为某些行为在训练和评估期间是不同的. 例如Dropout和BN. Dropout在训练模式下, 会随机将某些神经元的输出置为0, 防止过拟合, 而测试模式下, 不会丢弃任何神经元; BN在训练模式下, 会对当前批次的数据归一化, 在评估模式下, BN对全局的数据归一化, 这个和LN是不一样的, LN在训练和测试模式下没有不同. 同理, 可以用`model.eval()`将模型设置为评估模式.

??? note "`optimizer.zero_grad()`的作用"

    这用于清除梯度(将参数的梯度设置为0). 这是由于在PyTorch里面, 梯度不会被自动清除, 需要手动清除. 如果不清除梯度, 会导致新的梯度叠加到之前的梯度上, 这是它的设计特征, 目的是为了支持某些高级功能, 如累积梯度.

??? note "`.item()`的作用"

    在PyTorch中, 经常会看见`.item()`. 这个的作用是将标量tensor, 也就是只包含一个值的张量变成一个普通的Python标量.

??? note "可调用对象"

    在Python中, 如果一个对象定义了`__call__`方法, 则可以像调用函数一样调用这个对象, 这就是所谓的"可调用对象". 在上面的代码中, `nn.CrossEntropyLoss()`返回的对象中就定义了`__call__`方法, 因此可以直接像函数一样调用对象.

    ```py
    loss = loss_fn(pred, y)
    ```

    上述代码等价于:

    ```py
    loss = loss_fn.__call__(pred, y)
    ```

??? note "模型和数据在设备上的分离性"

    PyTorch为了提高灵活性和模块化, 可以选择将模型(包括其参数)和训练/测试数据存放在不同的设备上. 这是因为, 在很多深度学习任务中, 数据预处理(如data augementation, 文本编码, 数据加载)通常更适合在CPU上执行, 因为这些操作对GPU的高并行计算能力利用率不高. 还有原因是GPU的显存容量往往会不够, 需要处理大量数据的时候, 需要将数据分批加载到GPU, 利用普通内存作为缓冲区. 但是真正正在参与计算的数据和模型都必须在同一个设备, 要不是GPU, 要不是CPU.

    在上面的代码中, 我们首先将模型放到了GPU中: `NeuralNetwork().to(device)`. 然后, 分批次将数据加载到GPU中: `X, y = X.to(device), y.to(device)`.


同样, 我们也拿测试集来衡量模型的性能.

```py
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # 将模型设置为评估模式
    test_loss, correct = 0, 0
    with torch.no_grad(): # 作用见下方
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

??? note "`torch.no_grad()`的作用"

    用于禁用梯度计算, 这意味着在这个代码块内, 不会跟踪模型参数的梯度, 因为这是评估模式, 所以不需要更新模型的参数. 这段代码和上下文管理器一起使用, 简化表示.

```py title='输入'
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

``` title='输出'
Epoch 1
-------------------------------
loss: 2.304085  [   64/60000]
loss: 2.294508  [ 6464/60000]
loss: 2.274109  [12864/60000]
loss: 2.270270  [19264/60000]
loss: 2.250886  [25664/60000]
loss: 2.220479  [32064/60000]
loss: 2.225041  [38464/60000]
loss: 2.189506  [44864/60000]
loss: 2.192357  [51264/60000]
loss: 2.157102  [57664/60000]
Test Error:
 Accuracy: 47.3%, Avg loss: 2.149917

Epoch 2
-------------------------------
loss: 2.157969  [   64/60000]
loss: 2.146327  [ 6464/60000]
loss: 2.088529  [12864/60000]
loss: 2.110613  [19264/60000]
loss: 2.043306  [25664/60000]
loss: 1.994351  [32064/60000]
loss: 2.016328  [38464/60000]
loss: 1.932480  [44864/60000]
loss: 1.945457  [51264/60000]
loss: 1.870950  [57664/60000]
Test Error:
 Accuracy: 54.1%, Avg loss: 1.864029

Epoch 3
-------------------------------
loss: 1.894473  [   64/60000]
loss: 1.860638  [ 6464/60000]
loss: 1.745864  [12864/60000]
loss: 1.795800  [19264/60000]
loss: 1.669494  [25664/60000]
loss: 1.634574  [32064/60000]
loss: 1.653545  [38464/60000]
loss: 1.551876  [44864/60000]
loss: 1.585930  [51264/60000]
loss: 1.479611  [57664/60000]
Test Error:
 Accuracy: 60.3%, Avg loss: 1.494985

Epoch 4
-------------------------------
loss: 1.560271  [   64/60000]
loss: 1.524853  [ 6464/60000]
loss: 1.380546  [12864/60000]
loss: 1.458671  [19264/60000]
loss: 1.330814  [25664/60000]
loss: 1.335623  [32064/60000]
loss: 1.347326  [38464/60000]
loss: 1.269260  [44864/60000]
loss: 1.312450  [51264/60000]
loss: 1.211600  [57664/60000]
Test Error:
 Accuracy: 63.7%, Avg loss: 1.235736

Epoch 5
-------------------------------
loss: 1.311352  [   64/60000]
loss: 1.291073  [ 6464/60000]
loss: 1.130169  [12864/60000]
loss: 1.241175  [19264/60000]
loss: 1.110151  [25664/60000]
loss: 1.140992  [32064/60000]
loss: 1.162442  [38464/60000]
loss: 1.094141  [44864/60000]
loss: 1.142886  [51264/60000]
loss: 1.054473  [57664/60000]
Test Error:
 Accuracy: 65.0%, Avg loss: 1.074178

Done!
```
