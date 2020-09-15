# 联营
:label:`sec_pooling`

通常，当我们处理图像时，我们希望逐渐降低隐藏表示的空间分辨率，聚集信息，这样我们在网络中的位置越高，每个隐藏节点对其敏感的接收区域（输入端）就越大。

我们的最终任务通常会问一些关于图像的全局性问题，例如，*它是否包含一只猫？*所以通常我们最后一层的单元应该对整个输入敏感。通过逐渐聚合信息，生成越来越粗的映射，我们最终实现了最终学习全局表示的目标，同时在处理的中间层保留卷积层的所有优点。

此外，当检测较低层次的特征时，例如边缘（如:numref:`sec_conv_layer`中所讨论的），我们通常希望我们的表示在某种程度上对平移保持不变。例如，如果我们拍摄黑白之间轮廓清晰的图像`X`，并将整个图像向右移动一个像素，即`Z[i, j] = X[i, j + 1]`，则新图像`Z`的输出可能大不相同。边缘将移动一个像素。几乎不可能发生在同一个地方。事实上，即使是三脚架和一个静止的物体，由于快门的移动而引起的相机振动可能会使所有物体移动一个像素左右（高端相机配备了特殊功能来解决这个问题）。

本节介绍了*池层*，它具有降低卷积层对位置和空间上下采样表示的敏感性的双重目的。

## 最大池和平均池

与卷积层一样，*pooling*操作符由一个固定形状的窗口组成，该窗口根据其跨距在输入的所有区域上滑动，为固定形状窗口（有时称为*池窗口*）遍历的每个位置计算一个输出。然而，与卷积层中输入和核的互相关计算不同，池层不包含参数（没有*kernel*）。相反，池运算符是确定性的，通常计算池窗口中元素的最大值或平均值。这些操作分别称为*最大池*和*平均池*。

在这两种情况下，与互相关运算符一样，我们可以将池窗口视为从输入张量的左上角开始，从左到右和从上到下在输入张量之间滑动。在池窗口到达的每个位置，它计算窗口中输入子传感器的最大值或平均值，具体取决于使用的是最大值还是平均值。

![Maximum pooling with a pooling window shape of $2\times 2$. The shaded portions are the first output element as well as the input tensor elements used for the output computation: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

:numref:`fig_pooling`中的输出张量的高度为2，宽度为2。这四个元素来自每个池窗口中的最大值：

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

池窗口形状为$p \times q$的池层称为$p \times q$池层。池操作称为$p \times q$池。

让我们回到本节开头提到的对象边缘检测示例。现在我们将使用卷积层的输出作为$2\times 2$最大池的输入。设置卷积层输入为`X`，池层输出为`Y`。无论`X[i, j]`和`X[i, j + 1]`的值是否不同，或`X[i, j + 1]`和`X[i, j + 2]`的值是否不同，池层始终输出`Y[i, j] = 1`。也就是说，使用$2\times 2$最大池层，我们仍然可以检测到卷积层识别的模式在高度或宽度上是否移动不超过一个元素。

在下面的代码中，我们在`pool2d`函数中实现了池层的前向传播。此功能类似于:numref:`sec_conv_layer`中的`corr2d`功能。然而，这里我们没有内核，将输出计算为输入中每个区域的最大值或平均值。

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

我们可以在:numref:`fig_pooling`中构造输入张量`X`来验证二维最大池层的输出。

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

另外，我们对平均池层进行了实验。

```{.python .input}
#@tab all
pool2d(X, (2, 2), 'avg')
```

## 垫步和跨步

与卷积层一样，合并层也可以改变输出形状。和以前一样，我们可以通过填充输入和调整步幅来改变操作以获得所需的输出形状。我们可以通过deep learning框架中内置的二维最大池层来演示池层中填充和跨步的使用。我们首先构造了一个输入张量`X`，它的形状有四个维度，其中例子数和通道数都是1。

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

默认情况下，在同一个窗体中构建了同一个窗体和窗体池。下面，我们使用一个形状为`(3, 3)`的合用窗口，因此默认情况下步幅形状为`(3, 3)`。

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

可以手动指定步幅和填充。

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

当然，我们可以指定任意的矩形池窗口，并分别指定高度和宽度的填充和跨距。

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

## 多通道

当处理多信道输入数据时，池层将每个输入信道单独地汇集在一起，而不是像在卷积层中那样将信道上的输入相加。这意味着池层的输出通道数与输入通道数相同。下面，我们将在通道维度上连接张量`X`和`X + 1`，以构造具有2个通道的输入。

```{.python .input}
#@tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.stack([X, X+1], 0), (1, 2, 4, 4))
```

如我们所见，共用后输出通道的数量仍然是2个。

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

* 以池窗口中的输入元素为输入元素，最大池操作将最大值指定为输出，平均池操作将平均值指定为输出。
* 池层的主要好处之一是减轻卷积层对位置的过度敏感。
* 我们可以指定池层的填充和跨距。
* 最大池，结合一个大于1的步幅可以用来减少空间维度（例如，宽度和高度）。
* 池层的输出通道数与输入通道数相同。

## 练习

1. 你能把平均池作为卷积层的一个特例来实现吗？如果是这样，那就去做。
1. 你能把最大池作为卷积层的一个特例来实现吗？如果是这样，那就去做。
1. 池层的计算成本是多少？假设池层的输入大小为$c\times h\times w$，池窗口的形状为$p_h\times p_w$，填充为$(p_h, p_w)$，跨距为$(s_h, s_w)$。
1. 为什么您期望最大池和平均池的工作方式不同？
1. 我们需要一个单独的最小池层吗？你能用另一个手术代替它吗？
1. 在平均池和最大池之间还有其他操作可以考虑吗（提示：回忆一下softmax）？为什么它不那么受欢迎？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/71)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/72)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/274)
:end_tab:
