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
