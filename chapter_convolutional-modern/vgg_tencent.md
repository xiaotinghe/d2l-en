# 使用块的网络(VGG)
:label:`sec_vgg`

虽然AlexNet提供了经验证据，证明深度CNN可以取得良好的结果，但它并没有提供一个通用的模板来指导后续的研究人员设计新的网络。在接下来的几节中，我们将介绍几个常用于设计深层网络的启发式概念。

这一领域的进步反映了芯片设计的进步，在芯片设计中，工程师们从放置晶体管到逻辑元件再到逻辑块。同样，神经网络结构的设计也逐渐变得更加抽象，研究人员从从单个神经元到整个层，再到现在的块，重复层的模式。

使用块的想法最早出现在牛津大学的[视觉几何小组](http://www.robots.ox.ac.uk/~vgg/))，在他们同名的*VGG*网络中。通过使用循环和子例程，可以使用任何现代深度学习框架在代码中轻松实现这些重复结构。

## VGG块

经典CNN的基本构建挡路是以下序列：(I)具有填充以保持分辨率的卷积层，(Ii)诸如RELU的非线性，(Iii)诸如最大池层的池层。一个VGG挡路由一系列卷积层组成，紧随其后的是用于空间下采样的最大汇聚层。在最初的VGG论文:cite:`Simonyan.Zisserman.2014`中，作者使用了$3\times3$个核的卷积，填充为1(保持高度和宽度不变)和$2 \times 2$最大合并，步长为2(每次挡路之后分辨率减半)。在下面的代码中，我们定义了一个名为`vgg_block`的函数来实现一个vgg挡路。该函数接受对应于卷积层`num_convs`的数目和输出通道`num_channels`的数目的两个自变量。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def vgg_block(num_convs, in_channels, out_channels):
    layers=[]
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(tf.keras.layers.Conv2D(num_channels,kernel_size=3,
                                    padding='same',activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

## VGG网络

与AlexNet和LeNet一样，VGG网络可以分为两部分：第一部分主要由卷积和池层组成，第二部分由全连通层组成。:numref:`fig_vgg`中描述了这一点。

![From AlexNet to VGG that is designed from building blocks.](../img/vgg.svg)
:width:`400px`
:label:`fig_vgg`

网络的卷积部分连续连接来自:numref:`fig_vgg`(也在`vgg_block`函数中定义)的几个VGG块。以下变量`conv_arch`由元组列表(每个挡路一个)组成，其中每个元组包含两个值：卷积层数和输出通道数，它们恰恰是调用`vgg_block`函数所需的参数。VGG网络的完全连接部分与AlexNet中涵盖的部分完全相同。

原始的VGG网络有5个卷积块，其中前两个各有一个卷积层，后三个各有两个卷积层。第一个挡路有64个输出通道，每个后续的挡路都会使输出通道的数量翻一番，直到这个数字达到512。由于该网络使用8个卷积层和3个全连接层，因此通常称为VGG-11。

```{.python .input}
#@tab all
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
```

以下代码实现VGG-11。这是一个简单的问题，只需在`conv_arch`上执行一个for循环即可。

```{.python .input}
def vgg(conv_arch):
    net = nn.Sequential()
    # The convolutional part
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully-connected part
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net

net = vgg(conv_arch)
```

```{.python .input}
#@tab pytorch
def vgg(conv_arch):
    # The convolutional part
    conv_blks=[]
    in_channels=1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # The fully-connected part
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)
```

```{.python .input}
#@tab tensorflow
def vgg(conv_arch):
    net = tf.keras.models.Sequential()
    # The convulational part
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully-connected part
    net.add(tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)]))
    return net

net = vgg(conv_arch)
```

接下来，我们将构建一个高度为224、宽度为224的单通道数据示例，以观察每一层的输出形状。

```{.python .input}
net.initialize()
X = np.random.uniform(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for blk in net.layers:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t', X.shape)
```

如您所见，我们将每个挡路的高度和宽度减半，最终达到高度和宽度7，然后展平表示以供网络的完全连接部分处理。

## 培训

由于VGG-11比AlexNet计算量大，所以我们构造了一个信道数较少的网络。这对时尚-MNIST的培训来说绰绰有余。

```{.python .input}
#@tab mxnet, pytorch
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
```

```{.python .input}
#@tab tensorflow
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
# Recall that this has to be a function that will be passed to
# `d2l.train_ch6()` so that model building/compiling need to be within
# `strategy.scope()` in order to utilize the CPU/GPU devices that we have
net = lambda: vgg(small_conv_arch)
```

除了使用稍大的学习率外，该模型的训练过程与ALEXNET在:numref:`sec_alexnet`中的过程类似。

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## 摘要

* VGG-11使用可重复使用的卷积块构建网络。根据每个挡路中卷积层数和输出通道的不同，可以定义不同的VGG模型。
* 块的使用使得网络定义的表示非常紧凑。它允许高效地设计复杂网络。
* 在他们的VGG论文中，Simonyan和Ziserman试验了各种架构。特别是，他们发现，几层深而窄的卷积(即$3 \times 3$)比较少层的较宽卷积更有效。

## 练习

1. 当打印出图层的尺寸时，我们只看到了8个结果，而不是11个。剩下的3个图层信息到哪里去了？
1. 与AlexNet相比，VGG的计算速度要慢得多，而且需要更多的GPU内存。分析造成这种情况的原因。
1. 尝试将Fashion-MNIST中图像的高度和宽度从224更改为96。这对实验有什么影响？
1. 参考VGG论文:cite:`Simonyan.Zisserman.2014`中的表1构造其他常见型号，如VGG-16或VGG-19。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/77)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/78)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/277)
:end_tab:
