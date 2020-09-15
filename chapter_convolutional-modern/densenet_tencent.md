# 密集连接网络(DenseNet)

RESNET极大地改变了人们对如何将深层网络中的功能参数化的看法。*DENSENET*(密集卷积网络)在某种程度上是这个:cite:`Huang.Liu.Van-Der-Maaten.ea.2017`的逻辑扩展。为了理解如何达到这个目的，让我们稍微绕道数学。

## 从ResNet到DenseNet

回想一下函数的泰勒展开。对于点$x = 0$，它可以写为

$$f(x) = f(0) + f'(0) x + \frac{f''(0)}{2!}  x^2 + \frac{f'''(0)}{3!}  x^3 + \ldots.$$

关键是它将一个函数分解成越来越高的高次项。类似地，ResNet将功能分解为

$$f(\mathbf{x}) = \mathbf{x} + g(\mathbf{x}).$$

也就是说，Resnet将$f$分解为简单的线性项和更复杂的非线性项。如果我们希望捕获(不一定要添加)两个术语以外的信息，该怎么办？其中一个解决方案是Densenet :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`。

![The main difference between ResNet (left) and DenseNet (right) in cross-layer connections: use of addition and use of concatenation. ](../img/densenet-block.svg)
:label:`fig_densenet_block`

如:numref:`fig_densenet_block`所示，Resnet和Densenet之间的主要区别在于，在后一种情况下，输出是“连接”的*(用$[,]$表示)，而不是相加。因此，在应用越来越复杂的函数序列之后，我们执行从$\mathbf{x}$到其值的映射：

$$\mathbf{x} \to \left[
\mathbf{x},
f_1(\mathbf{x}),
f_2([\mathbf{x}, f_1(\mathbf{x})]), f_3([\mathbf{x}, f_1(\mathbf{x}), f_2([\mathbf{x}, f_1(\mathbf{x})])]), \ldots\right].$$

最后，将这些功能组合到MLP中，再次减少特征的数量。就实现而言，这相当简单：我们不是添加术语，而是将它们连接起来。DenseNet这个名字源于变量之间的依赖图变得相当密集这一事实。这样的链的最后一层密集地连接到所有前面的层。密集连接如:numref:`fig_densenet`中所示。

![Dense connections in DenseNet.](../img/densenet.svg)
:label:`fig_densenet`

构成DenseNet的主要组件是*密集块*和*过渡层*。前者定义输入和输出如何级联，而后者控制通道的数量，使其不太大。

## 密集区块

DenseNet使用修改后的Resnet的“批标准化、激活和卷积”结构(参见:numref:`sec_resnet`中的练习)。首先，我们实现这种卷积挡路结构。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3), padding='same')

        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x,y], axis=-1)
        return y
```

*密集挡路*由多个卷积块组成，每个卷积块使用相同数量的输出通道。然而，在前向传播中，我们将每个卷积挡路的输入和输出串联在信道维度上。

```{.python .input}
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = np.concatenate((X, Y), axis=1)
        return X
```

```{.python .input}
#@tab pytorch
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = torch.cat((X, Y), dim=1)
        return X
```

```{.python .input}
#@tab tensorflow
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x
```

在下面的示例中，我们定义了一个`DenseBlock`实例，该实例具有10个输出通道的2个卷积块。当使用具有3个通道的输入时，我们将获得具有$3+2\times 10=23$个通道的输出。卷积挡路通道数量控制着输出通道数量相对于输入通道数量的增长。这也被称为*增长率*。

```{.python .input}
blk = DenseBlock(2, 10)
blk.initialize()
X = np.random.uniform(size=(4, 3, 8, 8))
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab pytorch
blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab tensorflow
blk = DenseBlock(2, 10)
X = tf.random.uniform((4, 8, 8, 3))
Y = blk(X)
Y.shape
```

## 过渡层

由于每个密集的挡路都会增加频道数，加得太多会导致模型过于复杂。使用*过渡层*来控制模型的复杂性。它通过使用$1\times 1$卷积层来减少通道数，并以2的步长将平均汇聚层的高度和宽度减半，进一步降低了模型的复杂性。

```{.python .input}
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
#@tab pytorch
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```

```{.python .input}
#@tab tensorflow
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)
```

将具有10个通道的过渡层应用于上一个示例中密集挡路的输出。这会将输出通道的数量减少到10个，并将高度和宽度减半。

```{.python .input}
blk = transition_block(10)
blk.initialize()
blk(Y).shape
```

```{.python .input}
#@tab pytorch
blk = transition_block(23, 10)
blk(Y).shape
```

```{.python .input}
#@tab tensorflow
blk = TransitionBlock(10)
blk(Y).shape
```

## DenseNet模型

接下来，我们将构建DenseNet模型。DenseNet首次使用与ResNet相同的单卷积层和最大池层。

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def block_1():
    return tf.keras.Sequential([
       tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
       tf.keras.layers.BatchNormalization(),
       tf.keras.layers.ReLU(),
       tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

然后，与ResNet使用的由残留块组成的四个模块类似，DenseNet使用四个密集块。与Resnet类似，我们可以设置每个密集挡路使用的卷积层数。这里，我们将其设置为4，与:numref:`sec_resnet`中的Resnet-18模型一致。此外，我们将密集的挡路中卷积层的通道数(即增长率)设置为32个，因此每个密集的挡路将增加128个通道。

在ResNet中，每个模块之间的高度和宽度都减少了一个步长为2的剩余挡路，这里我们使用过渡层将高度和宽度减半，将通道数减半。

```{.python .input}
# `num_channels`: the current number of channels
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    net.add(DenseBlock(num_convs, growth_rate))
    # This is the number of output channels in the previous dense block
    num_channels += num_convs * growth_rate
    # A transition layer that halves the number of channels is added between
    # the dense blocks
    if i != len(num_convs_in_dense_blocks) - 1:
        num_channels //= 2
        net.add(transition_block(num_channels))
```

```{.python .input}
#@tab pytorch
# `num_channels`: the current number of channels
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # This is the number of output channels in the previous dense block
    num_channels += num_convs * growth_rate
    # A transition layer that haves the number of channels is added between
    # the dense blocks
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
```

```{.python .input}
#@tab tensorflow
def block_2():
    net = block_1()
    # `num_channels`: the current number of channels
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs, growth_rate))
        # This is the number of output channels in the previous dense block
        num_channels += num_convs * growth_rate
        # A transition layer that haves the number of channels is added
        # between the dense blocks
        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels //= 2
            net.add(TransitionBlock(num_channels))
    return net
```

与ResNet类似，全局池层和完全连接层在末端连接以产生输出。

```{.python .input}
net.add(nn.BatchNorm(),
        nn.Activation('relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveMaxPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))
```

```{.python .input}
#@tab tensorflow
def net():
    net = block_2()
    net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ReLU())
    net.add(tf.keras.layers.GlobalAvgPool2D())
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(10))
    return net
```

## 培训

由于我们在这里使用的是更深的网络，因此在本节中，我们将把输入高度和宽度从224减少到96，以简化计算。

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## 摘要

* 在跨层连接方面，与输入和输出相加的ResNet不同，DenseNet在通道维度上连接输入和输出。
* 构成DenseNet的主要组件是密集块和过渡层。
* 在组成网络时，我们需要通过添加过渡层来再次缩小通道数量，从而使维度保持在可控范围内。

## 练习

1. 为什么我们在过渡层中使用平均池而不是最大池？
1. DenseNet论文中提到的优点之一是它的模型参数比ResNet的模型参数小。为甚麽会这样呢？
1. DenseNet受到批评的一个问题是它的高内存消耗。
    1. 真的是这样吗？尝试将输入形状更改为$224\times 224$，以查看实际的图形处理器内存消耗。
    1. 您能想出一种降低内存消耗的替代方法吗？您需要如何更改框架？
1. 实施DenseNet论文:cite:`Huang.Liu.Van-Der-Maaten.ea.2017`的表1中列出的各种DenseNet版本。
1. 应用DenseNet的思想设计了一个基于MLP的模型。将其应用于:numref:`sec_kaggle_house`年度房价预测任务。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/87)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/88)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/331)
:end_tab:
