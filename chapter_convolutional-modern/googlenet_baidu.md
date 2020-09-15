# 具有并行连接的网络（GoogLeNet）
:label:`sec_googlenet`

2014年，*GoogLeNet*赢得了ImageNet挑战赛，提出了一种结合NiN优势和重复块:cite:`Szegedy.Liu.Jia.ea.2015`范例的结构。这篇论文的一个重点是解决什么样的卷积核是最好的问题。毕竟，以前流行的网络使用小到$1 \times 1$和大到$11 \times 11$的选择。本文的一个观点是，有时使用不同大小的内核的组合是有利的。在本节中，我们将介绍GoogLeNet，它提供了原始模型的一个稍微简化的版本：我们省略了一些为稳定训练而添加的特殊特性，但是现在有了更好的训练算法，这些特性是不必要的。

## 初始块

在GoogLeNet中，基本的卷积块被称为“*Inception block*”，很可能是因为电影《盗梦空间》（Inception*）（“我们需要更深入一些”）中的一句话命名的，这部电影启动了一个病毒性模因。

![Structure of the Inception block.](../img/inception.svg)
:label:`fig_inception`

如:numref:`fig_inception`所示，起始块由四条并行路径组成。前三条路径使用窗口大小为$1\times 1$、$3\times 3$和$5\times 5$的卷积层从不同的空间大小提取信息。中间的两条路径对输入执行$1\times 1$卷积，以减少信道数量，降低模型的复杂性。第四条路径使用$3\times 3$最大池化层，然后是$1\times 1$卷积层来更改信道数。这四条路径都使用适当的填充，使输入和输出具有相同的高度和宽度。最后，沿着每个路径的输出沿着信道维度串联起来，并构成块的输出。初始块的通常调整的超参数是每层输出通道的数量。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class Inception(nn.Block):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,
                              activation='relu')
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,
                              activation='relu')
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # Concatenate the outputs on the channel dimension
        return np.concatenate((p1, p2, p3, p4), axis=1)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class Inception(nn.Module):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # Concatenate the outputs on the channel dimension
        return torch.cat((p1, p2, p3, p4), dim=1)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class Inception(tf.keras.Model):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same',
                                           activation='relu')
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same',
                                           activation='relu')
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.p4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')


    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # Concatenate the outputs on the channel dimension
        return tf.keras.layers.Concatenate()([p1, p2, p3, p4])
```

为了理解这个网络为什么工作得这么好，考虑一下过滤器的组合。他们在不同的范围内探索图像。这意味着不同程度的细节可以被不同的过滤器有效地识别。同时，我们可以为不同的范围分配不同数量的参数（例如，更多的参数用于短期，但不能完全忽略长范围）。

## 谷歌模型

如:numref:`fig_inception_full`所示，GoogLeNet使用总共9个初始块和全球平均池的堆栈来生成其估计值。初始块之间的最大池化降低了维度。第一个模块类似于AlexNet和LeNet。块的堆栈是从VGG继承的，全局平均池避免了在末尾出现一堆完全连接的层。

![The GoogLeNet architecture.](../img/inception-full.svg)
:label:`fig_inception_full`

我们现在可以一块一块地实现GoogLeNet。第一模块使用64信道$7\times 7$卷积层。

```{.python .input}
b1 = nn.Sequential()
b1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b1():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

第二个模块使用两个卷积层：首先，64信道$1\times 1$卷积层，然后是$3\times 3$卷积层，它将信道数增加三倍。这对应于初始块中的第二条路径。

```{.python .input}
b2 = nn.Sequential()
b2.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
       nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b2():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 1, activation='relu'),
        tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

第三个模块串联两个完整的初始模块。第一起始块的输出信道数为$64+128+32+32=256$，四条路径中的输出信道比为$64:128:32:32=2:4:1:1$。第二和第三路径首先将输入信道的数目分别减少到$96/192=1/2$和$16/192=1/12$，然后连接第二卷积层。第二起始块的输出信道的数目增加到$128+192+96+64=480$，并且四个路径中的输出信道比的数目为$128:192:96:64 = 4:6:3:2$。第二和第三路径首先将输入信道的数目分别减少到$128/256=1/2$和$32/256=1/8$。

```{.python .input}
b3 = nn.Sequential()
b3.add(Inception(64, (96, 128), (16, 32), 32),
       Inception(128, (128, 192), (32, 96), 64),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b3():
    return tf.keras.models.Sequential([
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

第四个模块更复杂。它串联五个起始块，它们分别有$192+208+48+64=512$、$160+224+64+64=512$、$128+256+64+64=512$、$112+288+64+64=528$和$256+320+128+128=832$个输出通道。分配给这些路径的信道数与第三模块中的信道数相似：具有$3\times 3$卷积层的第二条路径输出最大数量的信道，第二条路径仅具有$1\times 1$卷积层，第三条路径具有$5\times 5$卷积层，第四条路径有$3\times 3$个最大池层。第二和第三路径将首先根据比率减少信道的数量。这些比率在不同的初始阶段略有不同。

```{.python .input}
b4 = nn.Sequential()
b4.add(Inception(192, (96, 208), (16, 48), 64),
       Inception(160, (112, 224), (24, 64), 64),
       Inception(128, (128, 256), (24, 64), 64),
       Inception(112, (144, 288), (32, 64), 64),
       Inception(256, (160, 320), (32, 128), 128),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b4():
    return tf.keras.Sequential([
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

第五个模块有两个起始块，分别有$256+320+128+128=832$和$384+384+128+128=1024$个输出通道。分配给每个路径的信道数与第三和第四模块中的相同，但具体值不同。应该注意的是，第五块后面是输出层。将NiN的平均高度更改为该块中每个层的平均宽度。最后，我们将输出转换成一个二维数组，后面是一个完全连接的层，其输出数量是标签类的数量。

```{.python .input}
b5 = nn.Sequential()
b5.add(Inception(256, (160, 320), (32, 128), 128),
       Inception(384, (192, 384), (48, 128), 128),
       nn.GlobalAvgPool2D())

net = nn.Sequential()
net.add(b1, b2, b3, b4, b5, nn.Dense(10))
```

```{.python .input}
#@tab pytorch
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveMaxPool2d((1,1)),
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
```

```{.python .input}
#@tab tensorflow
def b5():
    return tf.keras.Sequential([
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Flatten()
    ])
# Recall that this has to be a function that will be passed to
# `d2l.train_ch6()` so that model building/compiling need to be within
# `strategy.scope()` in order to utilize the CPU/GPU devices that we have
def net():
    return tf.keras.Sequential([b1(), b2(), b3(), b4(), b5(),
                                tf.keras.layers.Dense(10)])
```

GoogLeNet模型计算复杂，因此不像VGG那样容易修改信道数量。为了有一个合理的训练时间，我们把输入的高度和宽度从224减少到96。这简化了计算。各个模块之间输出形状的变化如下所示。

```{.python .input}
X = np.random.uniform(size=(1, 1, 96, 96))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform(shape=(1, 96, 96, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

## 培训

和以前一样，我们使用时尚MNIST数据集训练我们的模型。在调用训练过程之前，我们将其转换为$96 \times 96$像素分辨率。

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## 摘要

* 初始块相当于具有四条路径的子网。它通过不同窗口形状的卷积层和最大池层并行提取信息。$1 \times 1$卷积降低了每像素级别上的信道维数。最大池会降低分辨率。
* GoogLeNet将多个设计良好的初始块与系列中的其他层连接起来。通过对ImageNet数据集的大量实验，得到了初始块中分配的信道数的比值。
* GoogLeNet，以及它的后续版本，是ImageNet上最有效的模型之一，它以较低的计算复杂度提供了类似的测试精度。

## 练习

1. GoogLeNet有几个迭代。尝试实现并运行它们。其中包括以下内容：
    * 添加批处理规范化层:cite:`Ioffe.Szegedy.2015`，如后面:numref:`sec_batch_norm`中所述。
    * 对起始区块:cite:`Szegedy.Vanhoucke.Ioffe.ea.2016`进行调整。
    * 使用标签平滑进行模型正则化:cite:`Szegedy.Vanhoucke.Ioffe.ea.2016`。
    * 如:numref:`sec_resnet`后面所述，将其包括在剩余连接:cite:`Szegedy.Ioffe.Vanhoucke.ea.2017`中。
1. Glento Gooet的最小尺寸是多少？
1. 比较AlexNet、VGG和NiN与GoogLeNet的模型参数大小。后两种网络结构如何显著减小模型参数的大小？
1. 为什么我们一开始需要长程卷积？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/81)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/82)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/316)
:end_tab:
