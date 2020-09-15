# 网络中的网络（NiN）
:label:`sec_nin`

LeNet、AlexNet和VGG都有一个共同的设计模式：通过一系列的卷积和池化层来提取利用空间结构的特征，然后通过完全连接的层对表示进行后处理。AlexNet和VGG对LeNet的改进主要在于后来的网络如何扩展和深化这两个模块。或者，可以想象在这个过程的早期使用完全连接的层。然而，如果不小心使用密集层，可能会完全放弃表现的空间结构，
*network-in-network*（*NiN*）块提供了另一种选择。
它们是基于一个非常简单的见解提出的：在每个像素的通道上分别使用MLP :cite:`Lin.Chen.Yan.2013`。

## NiN块

回想一下，卷积层的输入和输出由四维张量组成，其轴与示例、通道、高度和宽度相对应。还记得，完全连接层的输入和输出通常是与示例和特征相对应的二维张量。NiN背后的想法是在每个像素位置（对于每个高度和宽度）应用一个完全连接的层。如果我们在每个空间位置绑定权重，我们可以将其视为$1\times 1$卷积层（如:numref:`sec_channels`中所述）或独立于每个像素位置的完全连接层。另一种查看方法是将空间维度中的每个元素（高度和宽度）视为与示例等效，将通道视为等效于要素。

:numref:`fig_nin`说明了VGG和NiN及其区块之间的主要结构差异。NiN块由一个卷积层和两个$1\times 1$卷积层组成，它们作为具有ReLU激活的每像素完全连接层。第一层的卷积窗口形状通常由用户设置。随后的窗造型固定为$1 \times 1$。

![Comparing architectures of VGG and NiN, and their blocks.](../img/nin.svg)
:width:`600px`
:label:`fig_nin`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding,
                      activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def nin_block(num_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(num_channels, kernel_size, strides=strides,
                               padding=padding, activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu')])
```

## 没有模型

最初的NiN网络是在AlexNet之后不久被提出的，显然从中得到了一些启示。NiN使用窗口形状为$11\times 11$、$5\times 5$和$3\times 3$的卷积层，相应的输出信道数与AlexNet中相同。每个NiN块后面是一个最大的池层，步长为2，窗口形状为$3\times 3$。

NiN和AlexNet的一个显著区别是NiN完全避免了完全连接的层。相反，NiN使用一个NiN块，其输出通道数等于label类的数量，后跟一个*global*average pooling layer，生成一个logits向量。NiN设计的一个优点是它显著减少了所需模型参数的数量。然而，在实践中，这种设计有时需要增加模型训练时间。

```{.python .input}
net = nn.Sequential()
net.add(nin_block(96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Dropout(0.5),
        # There are 10 label classes
        nin_block(10, kernel_size=3, strides=1, padding=1),
        # The global average pooling layer automatically sets the window shape
        # to the height and width of the input
        nn.GlobalAvgPool2D(),
        # Transform the four-dimensional output into two-dimensional output
        # with a shape of (batch size, 10)
        nn.Flatten())
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # There are 10 label classes
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # Transform the four-dimensional output into two-dimensional output with a
    # shape of (batch size, 10)
    nn.Flatten())
```

```{.python .input}
#@tab tensorflow
def net():
    return tf.keras.models.Sequential([
        nin_block(96, kernel_size=11, strides=4, padding='valid'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Dropout(0.5),
        # There are 10 label classes
        nin_block(10, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Reshape((1, 1, 10)),
        # Transform the four-dimensional output into two-dimensional output
        # with a shape of (batch size, 10)
        tf.keras.layers.Flatten(),
        ])
```

我们创建一个数据示例来查看每个块的输出形状。

```{.python .input}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

## 培训

像以前一样，我们用时装设计师来训练模特。宁的训练和亚历克内特和VGG的训练相似。

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## 摘要

* NiN使用由一个卷积层和多个$1\times 1$个卷积层组成的块。这可以在卷积堆栈中使用，以允许更多的每像素非线性。
* NiN删除了完全连接的层，并在将通道数量减少到所需的输出数量（例如，时装MNIST为10）后，用全局平均池（即对所有位置求和）替换它们。
* 移除完全连接的层可减少过度装配。NiN的参数少得多。
* NiN的设计影响了许多后来的CNN设计。

## 练习

1. 调整超参数以提高分类精度。
1. 为什么在NiN块中有两个$1\times 1$个卷积层？去掉其中一个，观察分析实验现象。
1. 计算NiN的资源使用情况。
    1. 参数的数量是多少？
    1. 计算量是多少？
    1. 训练期间需要多少记忆？
    1. 预测期间需要多少内存？
1. 一步将$384 \times 5 \times 5$表示法缩减为$10 \times 5 \times 5$表示法可能存在哪些问题？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/79)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/80)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/332)
:end_tab:
