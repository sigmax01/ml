---
title: PyTorch
---

# PyTorch[^1]

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

??? note "`root`, `download`, `transform`是啥"

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

??? note "`test_dataloader`是啥"

    `test_dataloader`是一个可迭代对象. 里面包含了整个测试数据集, 并将其分成了很多批次. 每次迭代`test_dataloader`的时候, 它会返回一个批次的数据, 直到遍历完整个数据集. 所以上面的`X`对应的是第一批数据, `N`表示的是批次大小, `C`表示的是通道数, `H`表示的是图像高度, `W`表示的是图像宽度.

### 创建模型

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

### 优化参数

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
    size = len(dataloader.dataset) # 返回整个训练集的大小
    num_batches = len(dataloader) # 返回总的批次数
    model.eval() # 将模型设置为评估模式
    test_loss, correct = 0, 0
    with torch.no_grad(): # 作用见下方
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() # 将当前批次的损失值, 累加到test_loss上
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # 将当前批次的正确预测数, 累加到correct上
    test_loss /= num_batches # 计算平均损失
    correct /= size # 计算准度
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

??? note "`torch.no_grad()`的作用"

    用于禁用梯度计算, 这意味着在这个代码块内, 不会跟踪模型参数的梯度, 因为这是评估模式, 所以不需要更新模型的参数. 这段代码和上下文管理器一起使用, 简化表示.

??? note "`pred.argmax(1)`的作用"

    在这里, `pred`是一个形状为(64, 10)的tensor. 表示这一批中所有图片对应10个分类的概率. `pred.argmax(1)`的作用是沿着`pred`tensor的第1个维度查找最大值对应的index, 即在64张图片中查找各自概率最大的分类. `pred.argmax(1) == y`得到的应该是一个形状为(64, )的tensor, 其中的值是`True, False`, 将其转化为`True/False`之后统计一下`True`的个数, 然后用`.item()`将标量tensor转化为Python数值.

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

### 保存模型

一种常见的保存模型的做法是序列化内部状态的相关字典, internal state dictionary, 这个字典用于存储模型或者优化器的内部状态(如权重, 偏置, 学习率等).

```py title='输入'
torch.save(model.state_dict(), "drive/MyDrive/Model/FashionMNIST/model.pth")
print("Saved PyTorch Model State to model.pth")
```

``` title='输出'
Saved PyTorch Model State to model.pth
```

### 加载模型

加载模型的方法是重新创造模型并反序列化得到内部状态的相关字典, 然后用这个字典去覆盖初始化好的模型.

```py title='输入'
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("drive/MyDrive/Model/FashionMNIST/model.pth", weights_only=True))
```

``` title='输出'
<All keys matched successfully>
```

然后这个模型就能够用来预测了.

```py title='输入'
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1] # 只拿出第一个样本出来
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

``` title='输出'
Predicted: "Ankle boot", Actual: "Ankle boot"
```

???+ note "`test_data[0][0]`的含义"

    `training_data`或者是`test_data`是我们的数据集, 其中第一个index表示的是第几个样本, 第二个index表示当前这个样本的特征还是标签. 如`test[0][0]`表示的是第一张照片的特征矩阵.

## Tensors

Tensor是一种和数组和矩阵很像的数据结构, 在PyTorch里面, 使用tensor编码模型的输入和输出, 包括模型的参数. tensor和numpy的nd数组很像, 只是tensor可以跑在GPU和其他硬件加速器上. 实际上, numpy的数组和tensor可以共用一块内存, 而不用复制数据. Tensor也对自动微分进行了优化.

### 初始化Tensor

可以通用多种方式初始化tensor.

1. 直接从数据初始化, 会自动推断数据类型

    ```py
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    ```

2. 从NumPy数组初始化

    ```py
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    ```

3. 从另一个tensor初始化, 会保留另一个tensor的属性(形状, 类型), 除非特别覆盖

    ```py title='输入'
    x_ones = torch.ones_like(x_data) # 保留x_data的形状和数据类型
    print(f"Ones Tensor: \n {x_ones} \n")
    x_rand = torch.rand_like(x_data, dtype=torch.float) # 保留x_data的形状, 但是数据类型是float
    print(f"Random Tensor: \n {x_rand} \n")
    ```

    ``` title='输出'
    Ones Tensor:
     tensor([[1, 1],
            [1, 1]])

    Random Tensor:
     tensor([[0.7053, 0.3019],
            [0.6510, 0.0095]])
    ```

4. 填充随机或者常量

    ```py title='输入'
    shape = (2,3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")
    ```

    ``` title='输出'
    Random Tensor:
     tensor([[0.0596, 0.0417, 0.1678],
            [0.9480, 0.0777, 0.4989]])

    Ones Tensor:
     tensor([[1., 1., 1.],
            [1., 1., 1.]])

    Zeros Tensor:
     tensor([[0., 0., 0.],
            [0., 0., 0.]])
    ```

### Tensor的属性

Tensor属性主要描述了它们的形状, 数据类型, 以及它们存储的设备.

```py title='输入'
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

``` title='输出'
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

### Tensor的操作

超过100种tensor的操作, 包括算术运算, 矩阵乘法(转置, 切片, 索引). 具体可以见[这里](https://pytorch.org/docs/stable/torch.html).

这些操作都能在GPU上运行(通常比CPU快很多), 默认情况下, tensor上创建在CPU上面的. 我们需要特别的使用`.to`函数将tensor转移到GPU上面. 但是记住, 在设备之间拷贝数据的成本是很高的.

```py
# 如果在GPU存在的情况下, 将tensor转移到上面
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
```

下面列举一些常见的操作API.

1. 与NumPy类似的索引和切片

    ```py title='输入'
    tensor = torch.ones(4, 4)
    print(f"First row: {tensor[0]}")
    print(f"First column: {tensor[:, 0]}")
    print(f"Last column: {tensor[..., -1]}")
    tensor[:,1] = 0
    print(tensor)
    ```

    ``` title='输出'
    First row: tensor([1., 1., 1., 1.])
    First column: tensor([1., 1., 1., 1.])
    Last column: tensor([1., 1., 1., 1.])
    tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])
    ```

2. 聚合tensors

    ```py title='输入'
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1)
    ```

    ``` title='输出'
    tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
            [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
            [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
            [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
    ```

    ??? note "这里的`dim`的作用"

        这里的`dim=1`的指的是沿着列的方向聚合.

3. 算术操作

    ```py title='输入'
    # @会进行矩阵乘法, 下面y1, y2, y3的最终结果是一样的
    # tensor.T返回的是tensor这个变量中保存的tensor的转置
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)
    y3 = torch.rand_like(y1) # 这个是一个随机初始化的形状和y1相同的矩阵y3
    torch.matmul(tensor, tensor.T, out=y3)

    # *会进行对应元素的矩阵乘法
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)
    z3 = torch.rand_like(tensor)
    torch.mul(tensor, tensor, out=z3)
    ```

4. 标量tensor

    如果你想要一个只有一个元素的tensor, 例如将tensor中的元素aggregate一下然后使用`.item()`转化成Python的数值变量.

    ```py title='输入'
    agg = tensor.sum()
    agg_item = agg.item()
    print(agg_item, type(agg_item))
    ```

    ``` title='输出'
    12.0 <class 'float'>
    ```

5. 原地操作

    又叫做in-place operation, 指的是直接在原始数据上进行修改, 而不是创建新的副本进行工作, 在PyTorch中, 原地操作通常以下划线`_`结尾.

    ```py title='输入'
    print(f"{tensor} \n")
    tensor.add_(5)
    print(tensor)
    ```

    ``` title='输出'
    tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

    tensor([[6., 5., 6., 6.],
            [6., 5., 6., 6.],
            [6., 5., 6., 6.],
            [6., 5., 6., 6.]])
    ```

    ??? warning "尽量不要使用原地操作"

        原地操作会改变tensor的状态, 可能会影响到其他引用该tensor的代码, 而且还可能影响自动求导.

### 和NumPy的联系

在CPU上的tensor和NumPy的数组可以共享它们的内存空间, 改变一个的同时会改变另一个.

1. Tensor到NumPy数组

    ```py title='输入'
    t = torch.ones(5)
    print(f"t: {t}")
    n = t.numpy()
    print(f"n: {n}")
    ```

    ``` title='输出'
    t: tensor([1., 1., 1., 1., 1.])
    n: [1. 1. 1. 1. 1.]
    ```

    对于tensor的修改会改变NumPy数组.

    ```py title='输入'
    t.add_(1)
    print(f"t: {t}")
    print(f"n: {n}")
    ```

    ``` title='输出'
    t: tensor([2., 2., 2., 2., 2.])
    n: [2. 2. 2. 2. 2.]
    ```

2. NumPy数组到Tensor

    ```py title='输入'
    n = np.ones(5)
    t = torch.from_numpy(n)
    ```

    改变NumPy数组会反映在tensor中.

    ```py title='输入'
    np.add(n, 1, out=n)
    print(f"t: {t}")
    print(f"n: {n}")
    ```

    ``` title='输出'
    t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
    n: [2. 2. 2. 2. 2.]
    ```

## Datasets & DataLoaders

预处理样本的代码可能会变得非常乱并且难以维护, 我们希望数据集代码能够和模型训练代码解耦以实现更好地可读性和模块化. PyTorch提供两种预定义的类`torch.utils.data.DataLoader`和`torch.utils.data.Dataset`, 这两个类允许我们使用预定义的数据集(如CIFAR-10)和自己的数据集. `Dataset`存储的是样本和对应的标签, `DataLoader`在这个基础上把`Dataset`包装成一个可迭代对象, 以便轻松访问样本.

PyTorch提供的一些预定义的数据集可以在这里找到: [图像数据集](https://pytorch.org/vision/stable/datasets.html), [文本数据集](https://pytorch.org/text/stable/datasets.html), [音频数据集](https://pytorch.org/audio/stable/datasets.html)

### 加载数据集

下面是一个从TorchVision导入Fashion-MNIST数据集的方法. Fashion-MNIST是一个来源于Zalando公司的时尚商品图像数据集, 包含70000张28*28像素的灰度图像, 其中60000张用于训练, 10000张用于测试, 该数据集分为10个类别, 包括T桖/上衣, 裤子, 裙子, 外套, 凉鞋, 运动鞋, 包, 长袜, 衬衫和高跟鞋等.

我们通过以下的参数加载FashionMNIST数据集.

- `root`: 是训练集和测试集数据保存的路径
- `train`: 声明是训练集还是测试集
- `download=True`: 从网络下载数据集如果在`root`下没有数据集的话
- `transform`和`target_transform`: 定义特征和标签的转换函数

```py title='输入'
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

``` title='输出'
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0.00/26.4M [00:00<?, ?B/s]
  0%|          | 65.5k/26.4M [00:00<01:12, 361kB/s]
  1%|          | 229k/26.4M [00:00<00:38, 679kB/s]
  3%|3         | 885k/26.4M [00:00<00:10, 2.45MB/s]
  7%|7         | 1.90M/26.4M [00:00<00:05, 4.28MB/s]
 15%|#4        | 3.83M/26.4M [00:00<00:02, 8.19MB/s]
 37%|###7      | 9.80M/26.4M [00:00<00:00, 21.8MB/s]
 50%|####9     | 13.2M/26.4M [00:00<00:00, 22.6MB/s]
 62%|######1   | 16.3M/26.4M [00:01<00:00, 24.8MB/s]
 72%|#######2  | 19.1M/26.4M [00:01<00:00, 25.6MB/s]
 92%|#########2| 24.4M/26.4M [00:01<00:00, 33.1MB/s]
100%|##########| 26.4M/26.4M [00:01<00:00, 19.3MB/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0.00/29.5k [00:00<?, ?B/s]
100%|##########| 29.5k/29.5k [00:00<00:00, 325kB/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0.00/4.42M [00:00<?, ?B/s]
  1%|1         | 65.5k/4.42M [00:00<00:12, 359kB/s]
  4%|4         | 197k/4.42M [00:00<00:05, 725kB/s]
 10%|#         | 459k/4.42M [00:00<00:03, 1.17MB/s]
 38%|###7      | 1.67M/4.42M [00:00<00:00, 4.28MB/s]
 83%|########2 | 3.67M/4.42M [00:00<00:00, 7.57MB/s]
100%|##########| 4.42M/4.42M [00:00<00:00, 6.02MB/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0.00/5.15k [00:00<?, ?B/s]
100%|##########| 5.15k/5.15k [00:00<00:00, 38.2MB/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

### 可视化数据集

我们可以对`Datasets`对象进行索引, 如`training_data[index]`, 这返回的是数据集的第`index`个样本(包括特征和标签, 分别是返回的元祖的第一个元素和第二个元素). 使用matplotlib可以对一些样本进行可视化操作.

```py title='输入'
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item() # 这里的training_data是一个含有60000个元素的对象, 可以像访问列表一样访问它, 随机从里面选择一个样本, 然后使用.item()将标量tensor转化为Python数值
    img, label = training_data[sample_idx] # 后者返回的是一个元祖, 第一个元素是特征, 第二个元素是标签
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```

<figure markdown='1'>
![](https://img.ricolxwz.io/085abdcf90f7eb7f63be72b28979026e.png){ loading=lazy width='500' }
</figure>

### 创建自定义数据集

一个自定义的数据集必须实现以下三个函数, `__init__`, `__len__`和`__getitem__`. 下面有一个例子, 其中, 图片存储在目录`img_dir`中, 它们的标签存储在一个分开的CSV文件`annotations_file`中.

```py
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) # 从img_labels中提取第idx行的第一列内容, 然后和img_dir拼接成完整的路径
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

??? note "`annotations_file`文件样子"

    这个csv文件长得像这样:

    ```csv
    tshirt1.jpg, 0
    tshirt2.jpg, 0
    ......
    ankleboot999.jpg, 9
    ```

### 准备训练数据

我们可以通过`Dataset`对象一个一个取出数据集的样本. 但是, 在训练模型的时候, 我们通常希望进行mini-batch GD, 在每个epoch开始之前都重新排列数据, 然后根据打乱后的顺序生成mini-batch供模型训练, 以减少过拟合, 并使用Python的多进程库`multiprocessing`来加速数据的取回.

`DataLoader`就是一个能够实现上述功能的简单API.

```py
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

### 遍历可迭代对象

我们已经把数据集包装为一个可迭代对象. 下列的每一次迭代都会返回一个batch的`train_features`和`train_labels`, 每个batch的大小为64. 由于我们声明了`shuffle=True`, 所以我们遍历完所有batch之后会打乱所有的数据.

```py title='输入'
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

<figure markdown='1'>
  ![](https://img.ricolxwz.io/c0251616f9dcfd140eda0ec82b82eba5.png){ loading=lazy width='500' }
</figure>

``` title='输出'
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Label: 5
```

### 转换

数据有很大概率不是用于机器学习输入的最终状态, 所以要使用转换(transform)对数据进行一些修改使其适合训练.

所有的TorchVision数据库都有两个参数, `transform`用于修改特征, `target_transform`用于修改标签, 它们接受的是包含逻辑的可调用对象. `torchvision.transforms`提供了一些经常使用的转换函数.

FashionMNIST的数据特征是PIL格式的, 标签是int. 为了训练, 我们需要特征是tensor, 标签是one-hot编码的tensor, 为此, 我们可以使用`ToTensor`和`Lambda`.

```py title='输入'
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

``` title='输出'
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0.00/26.4M [00:00<?, ?B/s]
  0%|          | 65.5k/26.4M [00:00<01:14, 354kB/s]
  1%|          | 197k/26.4M [00:00<00:36, 728kB/s]
  2%|1         | 492k/26.4M [00:00<00:21, 1.23MB/s]
  6%|5         | 1.57M/26.4M [00:00<00:06, 3.97MB/s]
 15%|#4        | 3.83M/26.4M [00:00<00:02, 7.89MB/s]
 31%|###1      | 8.19M/26.4M [00:00<00:01, 17.1MB/s]
 43%|####3     | 11.4M/26.4M [00:00<00:00, 21.1MB/s]
 52%|#####2    | 13.8M/26.4M [00:01<00:00, 21.9MB/s]
 61%|######1   | 16.2M/26.4M [00:01<00:00, 20.6MB/s]
 71%|#######1  | 18.8M/26.4M [00:01<00:00, 22.0MB/s]
 89%|########9 | 23.6M/26.4M [00:01<00:00, 29.1MB/s]
100%|##########| 26.4M/26.4M [00:01<00:00, 17.8MB/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0.00/29.5k [00:00<?, ?B/s]
100%|##########| 29.5k/29.5k [00:00<00:00, 326kB/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0.00/4.42M [00:00<?, ?B/s]
  1%|1         | 65.5k/4.42M [00:00<00:12, 361kB/s]
  5%|5         | 229k/4.42M [00:00<00:06, 682kB/s]
 20%|##        | 885k/4.42M [00:00<00:01, 2.54MB/s]
 44%|####3     | 1.93M/4.42M [00:00<00:00, 4.11MB/s]
100%|##########| 4.42M/4.42M [00:00<00:00, 6.09MB/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0.00/5.15k [00:00<?, ?B/s]
100%|##########| 5.15k/5.15k [00:00<00:00, 38.0MB/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

??? note "`ToTensor()`的作用"

    `ToTensor`将一个PIL图片或者NumPy数组转换为浮点tensor, 并且将图片的像素值归一到[0, 1].

??? note "Lambda函数的作用"

    在这里, 我们定义了一个将int转换为one-hot编码tensor的函数. 首先, 它会创造一个大小为10的零tensor. 然后调用了`scatter_`函数, 作用是将给定值$y$索引上的值设置为1.

## 创建神经网络

神经网络包括对数据进行操作的层/模块, `torch.nn`这个命名空间包含了所有需要构建神经网络的脚手架. PyTorch中所有的模块都是`nn.Module`的子类. 神经网络本身就是一个包含其他模块的模块, 这种嵌套结构使得构建复杂的架构非常简单.

### 获取用于训练的设备

我们希望在GPU或者MPS上训练我们的模型, 可用下列代码检查加速器是否在线, 如果否, 则使用CPU.

```py title='输入'
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
```

``` title='输出'
Using cuda device
```

### 定义类

我们通过继承`nn.Module`的方式定义自己的神经网络. 使用`__init__`函数初始化神经网络层. 每个子类都在`forward`函数中对于输入数据定义操作.

```py
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

我们创建了一个`NeuralNetwork`, 然后将其移动到设备.

```py title='输入'
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

为了使用这个模型, 我们直接向它传递数据, 这将自动触发执行类内部的`__call__`函数, 这个函数会自动调用`forward`函数, 不需要手动调用`forward`函数.

调用模型返回的结果是一个二维的tensor, 第一维对应的是每个样本的输出结果, 第二维对应的是每个类别的原始预测值(raw). 我们将预测结果传给softmax之后可以将第二维的输出转换为对应于每个类别的概率.

```py title='输入'
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1) # 第一维度最大的概率对应的index
print(f"Predicted class: {y_pred}")
```

``` title='输出'
Predicted class: tensor([7], device='cuda:0')
```

### 模型层

我们将之前定义的FashionMNIST模型分解一下, 并看看如果我们传入了一个随机生成的小批量3张28*28的图片看看它是怎么经过网络的.

```py title='输入'
input_image = torch.rand(3,28,28)
print(input_image.size())
```

``` title='输出'
torch.Size([3, 28, 28])
```

1. `nn.Flatten`

    我们初始化了一个`nn.Flatten`层将每个28*28的图片转换成一个连续的数组表示784个像素. 批次的数量3被保留.

    ```py title='输入'
    flatten = nn.Flatten()
    flat_image = flatten(input_image)
    print(flat_image.size())
    ```

    ``` title='输出'
    torch.Size([3, 784])
    ```

2.  `nn.Linear`

    线性层对输入使用当时的权重和截距做一个线性变换.

    ```py title='输入'
    layer1 = nn.Linear(in_features=28*28, out_features=20)
    hidden1 = layer1(flat_image)
    print(hidden1.size())
    ```

    ``` title='输出'
    torch.Size([3, 20])
    ```

3.  `nn.ReLU`

    非线性激活函数用于创造输入和输出之间的复杂映射, 它被放在线性层之后.

    ```py title='输入'
    print(f"Before ReLU: {hidden1}\n\n")
    hidden1 = nn.ReLU()(hidden1)
    print(f"After ReLU: {hidden1}")
    ```

    ``` title='输出'
    Before ReLU: tensor([[ 0.4158, -0.0130, -0.1144,  0.3960,  0.1476, -0.0690, -0.0269,  0.2690,
              0.1353,  0.1975,  0.4484,  0.0753,  0.4455,  0.5321, -0.1692,  0.4504,
              0.2476, -0.1787, -0.2754,  0.2462],
            [ 0.2326,  0.0623, -0.2984,  0.2878,  0.2767, -0.5434, -0.5051,  0.4339,
              0.0302,  0.1634,  0.5649, -0.0055,  0.2025,  0.4473, -0.2333,  0.6611,
              0.1883, -0.1250,  0.0820,  0.2778],
            [ 0.3325,  0.2654,  0.1091,  0.0651,  0.3425, -0.3880, -0.0152,  0.2298,
              0.3872,  0.0342,  0.8503,  0.0937,  0.1796,  0.5007, -0.1897,  0.4030,
              0.1189, -0.3237,  0.2048,  0.4343]], grad_fn=<AddmmBackward0>)


    After ReLU: tensor([[0.4158, 0.0000, 0.0000, 0.3960, 0.1476, 0.0000, 0.0000, 0.2690, 0.1353,
             0.1975, 0.4484, 0.0753, 0.4455, 0.5321, 0.0000, 0.4504, 0.2476, 0.0000,
             0.0000, 0.2462],
            [0.2326, 0.0623, 0.0000, 0.2878, 0.2767, 0.0000, 0.0000, 0.4339, 0.0302,
             0.1634, 0.5649, 0.0000, 0.2025, 0.4473, 0.0000, 0.6611, 0.1883, 0.0000,
             0.0820, 0.2778],
            [0.3325, 0.2654, 0.1091, 0.0651, 0.3425, 0.0000, 0.0000, 0.2298, 0.3872,
             0.0342, 0.8503, 0.0937, 0.1796, 0.5007, 0.0000, 0.4030, 0.1189, 0.0000,
             0.2048, 0.4343]], grad_fn=<ReluBackward0>)
    ```

4. `nn.Sequential`

    `nn.Sequential`是一个有顺序的模块容器, 数据会依照它定义的顺序经过定义在容器内的模块.

    ```py
    seq_modules = nn.Sequential(
        flatten,
        layer1,
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    input_image = torch.rand(3,28,28)
    logits = seq_modules(input_image)
    ```

5. `nn.Softmax`

    神经网络的最后一层返回的是原始数据, 应该被传入softmax层归一化到[0, 1].

    ```py
    softmax = nn.Softmax(dim=1)
    pred_probab = softmax(logits)
    ```

### 模型参数

神经网络中的许多层都是有可训练参数的. 可以通过模型的`parameters()`或者`named_parameters()`方法访问所有的参数.

```py title='输入'
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```

``` title='输出'
Model structure: NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)


Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[ 0.0273,  0.0296, -0.0084,  ..., -0.0142,  0.0093,  0.0135],
        [-0.0188, -0.0354,  0.0187,  ..., -0.0106, -0.0001,  0.0115]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0155, -0.0327], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[ 0.0116,  0.0293, -0.0280,  ...,  0.0334, -0.0078,  0.0298],
        [ 0.0095,  0.0038,  0.0009,  ..., -0.0365, -0.0011, -0.0221]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([ 0.0148, -0.0256], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[-0.0147, -0.0229,  0.0180,  ..., -0.0013,  0.0177,  0.0070],
        [-0.0202, -0.0417, -0.0279,  ..., -0.0441,  0.0185, -0.0268]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([ 0.0070, -0.0411], device='cuda:0', grad_fn=<SliceBackward0>)
```

## 自动微分

前置知识请见[这里](/dicts/autograd).

当我们训练神经网络的时候, 最长使用的就是BP算法. 在这个算法中, 网络的参数会根据输出对于它的梯度调整. 为了计算这些梯度, PyTorch有一个内置的微分引擎叫做`torch.autograd`. 它支持任何DAG的自动微分.

考虑一个最简单的一层神经网络, 输入为`x`, 参数为`w`和`b`, 还有一个损失函数, 那么它在PyTorch中可以被定义为:

```py title='输入'
import torch

x = torch.ones(5)  # 输入tensor
y = torch.zeros(3)  # 预期输出
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```

### Tensor, 函数和计算图

上述代码对应于这个计算图:

<figure markdown='1'>
  ![](https://img.ricolxwz.io/7efcb858120ecb3c0d118ce548c1fa96.png){ loading=lazy width='500' }
</figure>

在这个网络中, `w`和`b`是参数, 是我们要优化的对象, 因此, 我们需要去计算损失函数对应于这些参数的梯度. 为了实现这一点, 我们设置`requires_grad`为`True`.

???+ note "另一种方法"

    可以在开始的时候设置`requires_grad`为`True`, 也可以随后调用`x.requires_grad_(True)`.

用于构建这个计算图的函数其实是`Function`的一个对象, 这个对象知道如何正向计算值, 也知道在反向传播的时候计算参数的微分. 每个tensor的`grad_fn`属性会指向一个用于求导的`Function`对象, 这个对象中记载了该tensor是由哪些tensor通过什么操作生成的. 在反向传播的时候, 从最终的loss开始, 第-1层的`grad_fn`的输出是第-2层`grad_fn`的输入, 一直往前回溯, 就能计算出所有参数的梯度.

```py title='输入'
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
```

``` title='输出'
Gradient function for z = <AddBackward0 object at 0x7f00bbb611b0>
Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0x7f00bd553e50>
```

### 计算梯度

为了计算梯度, 我们可以调用`loss.backward()`函数, 这个会触发链式反应, 及从后向前逐步调用`grad_fucntion`, 然后更新对应参数的`grad`属性.

```py title='输入'
loss.backward()
print(w.grad)
print(b.grad)
```

``` title='输出'
tensor([[0.3313, 0.0626, 0.2530],
        [0.3313, 0.0626, 0.2530],
        [0.3313, 0.0626, 0.2530],
        [0.3313, 0.0626, 0.2530],
        [0.3313, 0.0626, 0.2530]])
tensor([0.3313, 0.0626, 0.2530])
```

???+ tip "只有叶子节点会存储梯度信息"

    **神经网络需要训练的模型参数位于计算图的叶子节点位置(见上图), 中间tensors是在前向传播过程中由参数和输入数据通过计算得到的中间结果.** 在计算图中, 叶子节点通常是由用户直接创建并设定了`requires_grad=True`的tensor. 这些叶子节点会在反向传播后, 会自动把梯度值保存到其`grad`属性中. 对于计算图中的非叶子节点, 它们通常是由叶子节点通过某些运算得到的中间结果. 默认情况下, 这些中间tensor在反向传播的时候不会保存梯度信息. 这是因为在大多数情况下, 我们只需要对参数(叶子节点)进行梯度更新, 而不需要对中间结果存储梯度. 如果确实需要中间结果的梯度, 可以在创建这些中间节点的时候设置`retain_grad`, 这样它们在反向传播结束的时候也会保留梯度信息, 不过这种操作较为少见.

???+ tip "累积梯度的实质"

    累积梯度是指如果不在多次反向传播之间清零(通常使用`oprimizer.zero_grad()`或手动将`.grad`置零), 则参数的`.grad`会累积所有反向传播得到的梯度值. 举个例子:

    - 第一次前向和反向传播后, 参数`W`的梯度`W.grad=g1`
    - 如果此时不清零梯度, 再进行第二次前向和反向传播计算得到梯度`g2`, 那么`W.grad`将变成`g1+g2`
    - 如此一来, 通过多次小批量数据的前向和反向传播, 我们可以累积梯度, 让`W.grad`存储来自多个batch的梯度和

    这种累积在实际中可用于模拟更大的批量训练, 即用小批模拟大批(batch size), 减少显存占用.

???+ tip "重复调用backward需要保留计算图"

    一般来说, 当我们对损失tensor调用一次`backward()`后, 为了节省内存和提升性能, 计算图会被释放(清空). 这意味着如果我们想要再次对同一个图调用`backward()`, 已经不存在可用的计算图来进行第二次反向传播了. 如果确实需要对同一个图多次调用`backward()`, (比如在一些复杂计算中, 我们像重复利用同一批数据的图进行多次梯度计算), 就需要在第一次`backward()`调用的时候传入`retain_graph=True`, 这样做会使得该计算图在反向传播结束的时候依然被保留, 以便后续再次使用, 不过这样会占用较多的内存和计算资源.

### 关闭梯度追踪

默认情况下, 所有`requries_grad=True`的tensor都会追踪它们的梯度计算历史. 然而, 有一些情况下我们并不需要梯度. 例如, 当我们想要测试而不是训练网络的时候. 我们可以使用`torch.no_grad()`块来禁用梯度追踪.

```py title='输入'
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
```

``` title='输出'
True
False
```

另一种方法是对tensor使用`detach()`函数.

```py title='输入'
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
```

``` title='输出'
False
```

### 小小总结

在理论上, PyTorch的自动微分会在一张有向无环图(DAG)中记录tensor和所有操作. 在这个图中, leaves是输入tensor, root是输出tensor. 从root到leaves, 可以使用链式法则自动计算梯度. 本质上, 它用的是自动微分中的反向模式.

在前向传播的时候, 自动微分会做两件事:

- 执行相应的操作
- 在DAG中更新梯度函数

在反向传播的时候, 自动微分会做三件事:

- 调用`.grad_fn`中的函数
- 累积在`.grad`属性中
- 使用链式法则, 将梯度反向传播

### VJP在PyTorch中的应用

首先, 请参考[这里](/dicts/autograd/#VJP).

举个例子, 我们计算的不是损失函数(一个标量, 一个函数)对于输入的梯度, 而是计算20个函数对于输入的梯度. 假设输入tensor`inp`的形状是`(4, 5)`, 也就是说有20个元素, 输入维度`n=20`. 输出tensor`out`的维度是`(5, 4)`, 也就是说有20个元素, 输出维度`m=20`. Jacobian矩阵的大小为20\*20. 假设我需要的是所有输出传回输入的时候产生的灵敏度, 即我们的“种子矩阵”`u`被设置为一个大小5\*4的全一矩阵. 使用PyTorch计算VJP就是设置`backward`的第一个参数等于这个“种子矩阵”.

```py title='输入'
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
```

``` title='输出'
First call
tensor([[4., 2., 2., 2., 2.],
        [2., 4., 2., 2., 2.],
        [2., 2., 4., 2., 2.],
        [2., 2., 2., 4., 2.]])

Second call
tensor([[8., 4., 4., 4., 4.],
        [4., 8., 4., 4., 4.],
        [4., 4., 8., 4., 4.],
        [4., 4., 4., 8., 4.]]) # 梯度累积, 梯度会累积到计算图叶子节点的.grad属性中

Call after zeroing gradients
tensor([[4., 2., 2., 2., 2.],
        [2., 4., 2., 2., 2.],
        [2., 2., 4., 2., 2.],
        [2., 2., 2., 4., 2.]])
```

???+ note "之前损失函数的等价调用方法"

    之前我们使用的是损失函数(即一个标量, 一个函数)对于输入的梯度, 所以等价于种子向量被设置为一个标量`1`, 表示这个输出处于“选中”状态, 所以等价于`backward(torch.tensor(1.0))`.

## 优化参数

现在我们已经有了模型和数据, 是时候进行训练, 验证和测试了. 训练模型是一个重复的过程, 在每一次迭代中, 模型做出推测, 计算错误率, 以及对所有参数的梯度, 使用梯度下降来优化这些参数.

### 超参数

超参数是提供给我们控制模型优化过程的可调整参数. 不同的超参数会影响模型的训练和收敛率. 在这里, 我们定义下列的超参数:

- 批次的多少
- 批次的大小
- 学习率

```py
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

### 优化循环

一旦我们设置了超参数, 我们可以用一个优化循环来训练和优化我们的模型, 每个优化循环叫做一个epoch. 每个epoch包含两个部分:

- 训练循环: 在训练集上迭代, 尝试收敛到最佳参数
- 验证/测试循环: 在验证/测试集上迭代, 测试模型是否有提升

### 损失函数

当我们输入了一些训练数据的时候, 我们的模型可能不会输出正确的预测, 损失函数用于测量输出和输入之间的相似度, 我们希望最小化这个损失函数. 常用的误差函数有对于回归任务有`nn.MSELoss`(均方误差函数), 对于分类任务有`nn.NLLLoss`(负数对数似然函数). `nn.CrossEntropyLoss`结合了`nn.LogSoftmax`和`nn.NLLLoss`, 在这里我们使用的是`nn.CrossEntropyLoss`.

```py
# 初始化损失函数
loss_fn = nn.CrossEntropyLoss()
```

### 优化器

优化是每次迭代训练调整模型参数以减少误差的过程. 优化算法定义了这个过程应该怎么进行(在这个例子中我们使用的是随机梯度下降). 所有的优化逻辑都放在`optimizer`这个对象中. PyTorch中有很多优化器, 如ADAM, RMSProp.

我们通过传入模型的课训练参数和学习率初始化优化器.

```py
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

在训练循环内部, 优化发生在:

- 调用`optimizer.zero_grad()`重置模型参数的梯度的时候, 梯度默认情况下是累积的, 可以在每个batch之后置零
- 将训练误差反向传播的时候, `loss.backward()`, 这个会计算所有参数的梯度
- 拿到梯度之后, 调用`optimizer.step()`调整所有参数

## 保存和加载模型

在这个部分我们会看如何保存和加载模型的状态.

### 保存/加载模型权重

PyTorch模型会将学习到的参数放在一个内部状态字典中, 叫做`state_dict`. 可以使用`torch.save`保存参数.

```py title='输入'
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
```

``` title='输出'
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /var/lib/ci-user/.cache/torch/hub/checkpoints/vgg16-397923af.pth

  0%|          | 0.00/528M [00:00<?, ?B/s]
  4%|3         | 19.2M/528M [00:00<00:02, 202MB/s]
  7%|7         | 39.0M/528M [00:00<00:02, 205MB/s]
 11%|#1        | 58.8M/528M [00:00<00:02, 206MB/s]
 15%|#4        | 78.6M/528M [00:00<00:02, 206MB/s]
 19%|#8        | 98.6M/528M [00:00<00:02, 207MB/s]
 22%|##2       | 119M/528M [00:00<00:02, 208MB/s]
 26%|##6       | 139M/528M [00:00<00:01, 208MB/s]
 30%|###       | 159M/528M [00:00<00:01, 209MB/s]
 34%|###3      | 179M/528M [00:00<00:01, 210MB/s]
 38%|###7      | 199M/528M [00:01<00:01, 210MB/s]
 42%|####1     | 219M/528M [00:01<00:01, 210MB/s]
 45%|####5     | 239M/528M [00:01<00:01, 210MB/s]
 49%|####9     | 260M/528M [00:01<00:01, 210MB/s]
 53%|#####2    | 280M/528M [00:01<00:01, 210MB/s]
 57%|#####6    | 300M/528M [00:01<00:01, 210MB/s]
 61%|######    | 320M/528M [00:01<00:01, 210MB/s]
 64%|######4   | 340M/528M [00:01<00:00, 210MB/s]
 68%|######8   | 360M/528M [00:01<00:00, 211MB/s]
 72%|#######2  | 380M/528M [00:01<00:00, 211MB/s]
 76%|#######5  | 401M/528M [00:02<00:00, 211MB/s]
 80%|#######9  | 421M/528M [00:02<00:00, 211MB/s]
 84%|########3 | 441M/528M [00:02<00:00, 211MB/s]
 87%|########7 | 461M/528M [00:02<00:00, 211MB/s]
 91%|#########1| 481M/528M [00:02<00:00, 211MB/s]
 95%|#########4| 501M/528M [00:02<00:00, 211MB/s]
 99%|#########8| 522M/528M [00:02<00:00, 211MB/s]
100%|##########| 528M/528M [00:02<00:00, 210MB/s]
```

为了加载模型参数, 你需要先创建一个相同模型的instance, 然后使用`load_state_dict()`加载参数.

??? note "`weights_only=True`的作用"

    通过设置`weights_only=True`, 限制了在反序列化(unpickling)过程中仅执行加载权重所需的函数, 这种做法有几个重要的优点. 第一个, 安全性提升, 反序列化过程可能会执行存储在序列化对象中的任意代码, 如果不加限制, 恶意构造的序列化数据可能会导致安全漏洞. 第二个, 性能优化, 在加载模型权重的时候, 通常只需要恢复权重数据, 而不需要重新构建整个模型结构或者执行其他初始化操作.

```py title='输入'
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval() # 将模型设置为处于评估模式, 防止dropout
```

``` title='输出'
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```

### 保存/加载模型

从上面看出, 当我们加载模型参数的时候, 实际上我们还是需要先instantiate模型的类, 因为类定义了模型的结构. 我们可能希望一下子打包模型的参数和结构, 我们可以直接传入`model`, 而不是`model.state_dict()`.

```py
torch.save(model, 'model.pth')
```

然后, 可以通过下列方法加载模型.

```py
model = torch.load('model.pth', weights_only=False),
```

[^1]: Learn the basics—PyTorch tutorials 2.5.0+cu124 documentation. (不详). 取读于 2024年12月13日, 从 https://pytorch.org/tutorials/beginner/basics/intro.html
