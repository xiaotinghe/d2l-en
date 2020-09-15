# 池化
:label:`sec_pooling`

通常，当我们处理图像时，我们希望逐渐降低隐藏表示的空间分辨率，聚合信息，以便我们在网络中位置越高，每个隐藏节点对其敏感的接受域(在输入中)就越大。

我们的最终任务通常会询问一些关于图像的全局问题，例如，*它是否包含一只猫？*所以通常情况下，我们最后一层的单位应该对整个输入敏感。通过逐步聚合信息，产生越来越粗糙的地图，我们实现了最终学习全局表示的目标，同时在处理的中间层保持了卷积层的所有优势。

此外，当检测较低级别的特征时，例如边(如:numref:`sec_conv_layer`中所讨论的)，我们通常希望我们的表示在一定程度上不受转换的影响。例如，如果我们取具有在黑白之间清晰描绘的图像`X`，并且将整个图像向右移动一个像素，即`Z[i, j] = X[i, j + 1]`，则新图像`Z`的输出可能有很大的不同。边缘将偏移一个像素。在现实中，物体几乎不会完全出现在同一地点。事实上，即使是三脚架和静止物体，快门移动引起的相机振动也可能会使一切移动一个像素左右(高端相机都有特殊功能来解决这个问题)。

本节介绍“合并图层”，它具有降低卷积图层对位置和空间下采样表示的敏感性的双重目的。

## 最大池化和平均池化

与卷积层类似，*Pooling*运算符由固定形状窗口组成，该窗口根据其步幅在输入中的所有区域上滑动，为固定形状窗口(有时称为*Pooling窗口*)遍历的每个位置计算单个输出。然而，与卷积层中输入和核的互相关计算不同，池化层不包含参数(没有*核*)。相反，池运算符是确定性的，通常计算池化窗口中元素的最大值或平均值。这些操作分别称为“最大池化”(简称“最大池化”)和“平均池化”。

在这两种情况下，就像使用互相关运算符一样，我们可以认为合并窗口从输入张量的左上角开始，并从左到右和从上到下在输入张量上滑动。在合并窗口命中的每个位置，它根据使用的是最大还是平均合并来计算窗口中的输入子张量的最大值或平均值。

![Maximum pooling with a pooling window shape of $2\times 2$. The shaded portions are the first output element as well as the input tensor elements used for the output computation: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

:numref:`fig_pooling`中的输出张量的高度为2，宽度为2。这四个元素从每个池窗口中的最大值导出：

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

具有$p \times q$的合用窗口形状的合用图层称为$p \times q$合用图层。池化操作称为$p \times q$池化。

让我们回到本节开头提到的对象边缘检测示例。现在，我们将使用卷积层的输出作为$2\times 2$最大池化的输入。将卷积层输入设置为`X`，将池层输出设置为`Y`。无论是`X[i, j]`和`X[i, j + 1]`的值不同，还是`X[i, j + 1]`和`X[i, j + 2]`的值不同，合并层始终输出`Y[i, j] = 1`。也就是说，使用$2\times 2$的最大汇聚层，我们仍然可以检测由卷积层识别的图案是否在高度或宽度上移动不超过一个元素。

在下面的代码中，我们实现了`pool2d`函数中池层的正向传播。此函数类似于:numref:`sec_conv_layer`中的`corr2d`函数。但是，这里没有内核，将输出计算为输入中每个区域的最大值或平均值。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab mxnet, pytorch
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = d2l.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode =='avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y
```

我们可以在:numref:`fig_pooling`中构造输入张量:numref:`fig_pooling`以验证二维最大汇聚层的输出。

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

此外，我们还使用平均池层进行了实验。

```{.python .input}
#@tab all
pool2d(X, (2, 2), 'avg')
```

## 填充和跨度

与卷积图层一样，合并图层也可以更改输出形状。和前面一样，我们可以通过填充输入和调整步幅来改变操作以获得所需的输出形状。我们可以通过深度学习框架中内置的二维最大汇聚层来演示填充和跨度在汇聚层中的使用。我们首先构造其形状具有四个维度的输入张量`X`，其中示例数量和通道数量都是1。

```{.python .input}
#@tab mxnet, pytorch
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 1, 4, 4))
X
```

```{.python .input}
#@tab tensorflow
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 4, 4, 1))
X
```

默认情况下，来自框架内置类的实例中的Stride和Pooling窗口具有相同的形状。下面，我们使用形状为`(3, 3)`的合并窗口，因此默认情况下我们得到的步幅形状为`(3, 3)`。

```{.python .input}
pool2d = nn.MaxPool2D(3)
# Because there are no model parameters in the pooling layer, we do not need
# to call the parameter initialization function
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
pool2d(X)
```

步距和填充可以手动指定。

```{.python .input}
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='same',
                                   strides=2)
pool2d(X)
```

当然，我们可以指定一个任意的矩形池窗口，并分别指定高度和宽度的填充和跨度。

```{.python .input}
pool2d = nn.MaxPool2D((2, 3), padding=(1, 2), strides=(2, 3))
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d((2, 3), padding=(1, 1), stride=(2, 3))
pool2d(X)
```

```{.python .input}
#@tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='same',
                                   strides=(2, 3))
pool2d(X)
```

## 多渠道

在处理多通道输入数据时，汇聚层分别将每个输入通道汇集在一起，而不是像在卷积层中那样将通道上的输入相加。这意味着用于池化层的输出通道的数量与输入通道的数量相同。下面，我们将连接通道维度上的张量`X`和`X + 1`，以构建具有2个通道的输入。

```{.python .input}
#@tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.stack([X, X+1], 0), (1, 2, 4, 4))
```

如我们所见，合并后的输出通道数仍为2个。

```{.python .input}
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(3, padding='same', strides=2)
pool2d(X)
```

## 摘要

* 对于池化窗口中的输入元素，最大池化操作将最大值指定为输出，而平均池化操作将平均值指定为输出。
* 汇集层的主要优点之一是减轻卷积层对位置的过度敏感性。
* 我们可以指定池化层的填充和跨度。
* 最大合并与大于1的跨度相结合可用于减小空间维度(例如，宽度和高度)。
* 池层的输出通道数与输入通道数相同。

## 练习

1. 您能否将平均池作为卷积层的特例来实现？如果是这样，那就去做吧。
1. 您能实现最大池作为卷积层的特例吗？如果是这样，那就去做吧。
1. 池化层的计算成本是多少？假设汇聚层的输入大小为$c\times h\times w$，则汇聚窗口的形状为$p_h\times p_w$，填充为$(p_h, p_w)$，跨度为$(s_h, s_w)$。
1. 为什么您期望最大池化和平均池化的工作方式不同？
1. 我们是否需要单独的最小池层？你能换成另一台手术吗？
1. 在平均池和最大池之间是否有其他您可以考虑的操作(提示：回想一下Softmax)？为什么它可能不那么受欢迎呢？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/71)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/72)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/274)
:end_tab:
