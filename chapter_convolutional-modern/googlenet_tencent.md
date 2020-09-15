# 具有并行级联的网络(GoogLeNet)
:label:`sec_googlenet`

2014年，*GoogleNet*赢得了ImageNet挑战赛，提出了一个结合了NIN的优势和重复区块:cite:`Szegedy.Liu.Jia.ea.2015`的范例的结构。本文的一个重点是解决哪种大小的卷积核是最好的问题。毕竟，以前流行的电视网使用的选择小到$1 \times 1$，大到$11 \times 11$。本文中的一个见解是，有时使用不同大小的内核组合可能是有利的。在本节中，我们将介绍GoogLeNet，并提供原始模型的一个稍微简化的版本：我们省略了一些特别的功能，这些功能是为了稳定训练而添加的，但随着更好的训练算法的提供，这些功能现在是不必要的。

## 起始块

谷歌乐网中基本的卷积挡路被称为“盗梦空间挡路”，可能是因为电影“盗梦空间”中的一句话(“我们需要走得更深”)而得名，电影“盗梦空间”引发了病毒式的表情包。

![Structure of the Inception block.](../img/inception.svg)
:label:`fig_inception`

如:numref:`fig_inception`所示，初始挡路由四条平行路径组成。前三条路径使用窗口大小为$1\times 1$、$3\times 3$和$5\times 5$的卷积层从不同的空间大小提取信息。中间两条路径对输入执行$1\times 1$卷积，以减少通道数量，从而降低模型的复杂性。第四条路径使用$3\times 3$的最大池层，然后使用$1\times 1$的卷积层来改变信道的数量。这四条路径都使用适当的填充来赋予输入和输出相同的高度和宽度。最后，沿每条路径的输出沿通道维度串联，构成挡路的输出。盗梦空间挡路通常调优的超参数是每层的输出通道数。

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

要直观了解此网络运行良好的原因，请考虑过滤器的组合。他们在不同的范围内探索图像。这意味着不同程度的细节可以被不同的过滤器有效地识别。同时，我们可以为不同的范围分配不同数量的参数(例如，更多的参数用于短范围，但不能完全忽略长范围)。

## GoogLeNet模式

如:numref:`fig_inception_full`所示，GoogleNet使用总共9个初始块的堆栈和全球平均汇集来生成其估计。起始块之间的最大池化降低了维数。第一个模块类似于AlexNet和LeNet。数据块堆栈继承自VGG，全局平均池避免了末端的完全连接层堆栈。

![The GoogLeNet architecture.](../img/inception-full.svg)
:label:`fig_inception_full`

我们现在可以一块一块地实现GoogLeNet。第一模块使用64通道$7\times 7$卷积层。

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

第二个模块使用两个卷积层：首先是64通道$1\times 1$卷积层，然后是将通道数增加三倍的$3\times 3$卷积层。这对应于“盗梦空间”挡路中的第二条路径。

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

第三个模块串联两个完整的“盗梦空间”模块。第一条盗梦空间挡路的输出通道数为$64+128+32+32=256$，四路输出通道数比为$64:128:32:32=2:4:1:1$。第二和第三路径首先将输入信道的数量分别减少到$96/192=1/2$和$16/192=1/12$，然后连接第二卷积层。第二条盗梦空间挡路的输出通道数增加到$128+192+96+64=480$个，四路输出通道数比为$128:192:96:64 = 4:6:3:2$。第二和第三路径首先将输入通道的数量分别减少到$128/256=1/2$和$32/256=1/8$。

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

第四个模块比较复杂。它串联了五个先发模块，它们分别有$192+208+48+64=512$、$160+224+64+64=512$、$128+256+64+64=512$、$112+288+64+64=528$和$256+320+128+128=832$个输出通道。分配给这些路径的信道数量与第三模块中的类似：具有$3\times 3$卷积层的第二路径输出的信道数量最多，其次是仅具有$1\times 1$卷积层的第一路径，具有$5\times 5$卷积层的第三路径，以及具有$3\times 3$最大池层的第四路径。第二条路径和第三条路径将首先根据该比例减少信道数量。这些比率在不同的“盗梦空间”区块中略有不同。

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

第五个模块有两个初始模块，分别有$256+320+128+128=832$和$384+384+128+128=1024$个输出通道。分配给每条路径的通道数与第三和第四个模块中的相同，但在特定值上有所不同。需要注意的是，第五个挡路之后是输出层。此挡路使用全局平均池层将每个通道的高度和宽度更改为1，就像在nin中一样。最后，我们将输出转换为一个二维数组，后面跟着一个完全连接的层，其输出的数量就是标签分类的数量。

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

GoogLeNet模型计算复杂，因此不像在VGG中那样容易修改通道数。为了在Fashion-MNIST上有一个合理的训练时间，我们将输入的高度和宽度从224降低到96。这简化了计算。不同模块之间输出形状的变化如下所示。

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

与前面一样，我们使用Fashion-MNIST数据集训练我们的模型。在调用训练过程之前，我们将其转换为$96 \times 96$像素分辨率。

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## 摘要

* 先发挡路相当于一个有四条路径的子网。它通过不同窗口形状的卷积层和最大汇聚层并行提取信息。$1 \times 1$卷积降低了每像素级的通道维数。最大池容量会降低分辨率。
* GoogLeNet将多个设计良好的Inception块与其他层串联起来。《盗梦空间挡路》中分配的通道数比例是通过在ImageNet数据集上进行大量实验得到的。
* GoogLeNet及其后续版本是ImageNet上最有效的模型之一，提供了类似的测试精度和较低的计算复杂度。

## 练习

1. GoogLeNet有几次迭代。尝试实现和运行它们。其中一些包括以下内容：
    * 添加批归一化层:cite:`Ioffe.Szegedy.2015`，如稍后在:numref:`sec_batch_norm`中所述。
    * 对盗梦空间挡路:cite:`Szegedy.Vanhoucke.Ioffe.ea.2016`进行调整。
    * 对模型正则化:cite:`Szegedy.Vanhoucke.Ioffe.ea.2016`使用标签平滑。
    * 包括在剩余连接:cite:`Szegedy.Ioffe.Vanhoucke.ea.2017`中，如稍后在:numref:`sec_resnet`中描述的。
1. GoogLeNet工作的最小图像大小是多少？
1. 将AlexNet、VGG和nin的模型参数大小与GoogLeNet进行比较。后两种网络架构如何显著降低模型参数大小？
1. 为什么我们一开始需要长距离卷积呢？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/81)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/82)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/316)
:end_tab:
