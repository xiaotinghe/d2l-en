# 图像的卷积
:label:`sec_conv_layer`

现在我们了解了卷积分层在理论上是如何工作的，现在我们准备看看它们在实践中是如何工作的。基于我们将卷积神经网络作为探索图像数据结构的有效架构的动机，我们坚持以图像为运行示例。

## 互相关运算

回想一下，严格地说，卷积层是一个用词不当的词，因为它们所表达的运算更准确地被描述为互相关。基于我们在:numref:`sec_why-conv`中对卷积层的描述，在这样的层中，输入张量和核张量通过互相关运算来组合以产生输出张量。

让我们暂时忽略通道，看看这是如何处理二维数据和隐藏表示的。在:numref:`fig_correlation`中，输入是高度为3、宽度为3的二维张量。我们将张量的形状标记为$3 \times 3$或($3$,$3$)。内核的高度和宽度都是2。*内核窗口*(或*卷积窗口*)的形状由内核的高度和宽度(这里是$2 \times 2$)决定。

![Two-dimensional cross-correlation operation. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
:label:`fig_correlation`

在二维互相关运算中，我们从位于输入张量左上角的卷积窗口开始，然后从左到右和从上到下跨输入张量滑动。当卷积窗口滑动到某个位置时，该窗口中包含的输入子张量与核张量按元素相乘，并将得到的张量相加得到单个标量值。该结果给出了相应位置的输出张量值。这里，输出张量的高度为2，宽度为2，四个元素从二维互相关运算中导出：

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.
$$

请注意，沿每个轴，输出大小略小于输入大小。因为核的宽度和高度大于1，所以我们只能正确地计算核完全适合于图像内的位置的互相关，输出大小由输入大小$n_h \times n_w$减去卷积核$k_h \times k_w$的大小通过

$$(n_h-k_h+1) \times (n_w-k_w+1).$$

情况就是这样，因为我们需要足够的空间来在图像中“移动”卷积内核。稍后，我们将了解如何通过在图像边界周围填充零来保持大小不变，以便有足够的空间来移动内核。接下来，我们在`corr2d`函数中实现该过程，该函数接受输入张量`X`和核张量`K`，并返回输出张量`Y`。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
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
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y
```

我们可以构造输入张量`X`和核张量`K`从:numref:`fig_correlation`来验证二维互相关运算的上述实现的输出。

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

## 卷积层

卷积层使输入和核互相关，并添加标量偏置以产生输出。卷积层的两个参数是核和标量偏置。当训练基于卷积层的模型时，我们通常会随机初始化内核，就像我们对完全连接层所做的那样。

现在我们准备基于上面定义的`corr2d`函数实现二维卷积层。在`__init__`构造函数中，我们将`weight`和`bias`声明为两个模型参数。正向传播函数调用`corr2d`函数并添加偏置。

```{.python .input}
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

```{.python .input}
#@tab pytorch
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

```{.python .input}
#@tab tensorflow
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias
```

在$h \times w$卷积核或$h \times w$卷积核中，卷积核的高度和宽度分别为$h$和$w$。我们也将具有$h \times w$卷积核的卷积层简称为$h \times w$卷积层。

## 图像中的目标边缘检测

让我们花一点时间来分析卷积层的一个简单应用：通过找到像素变化的位置来检测图像中对象的边缘。首先，我们构建一个$6\times 8$像素的“图像”。中间四列为黑色(0)，睡觉为白色(1)。

```{.python .input}
#@tab mxnet, pytorch
X = d2l.ones((6, 8))
X[:, 2:6] = 0
X
```

```{.python .input}
#@tab tensorflow
X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X
```

接下来，我们构造一个高度为1，宽度为2的核`K`，当我们与输入执行互相关运算时，如果水平相邻的元素相同，则输出为0。否则，输出为非零。

```{.python .input}
#@tab all
K = d2l.tensor([[1.0, -1.0]])
```

我们已经准备好使用参数`X`(我们的输入)和`K`(我们的内核)执行互相关操作。如您所见，我们检测到从白色到黑色的边缘为1，从黑色到白色的边缘为-1。所有其他输出均采用值0。

```{.python .input}
#@tab all
Y = corr2d(X, K)
Y
```

现在，我们可以将内核应用于转置后的图像。不出所料，它消失了。内核`K`仅检测垂直边缘。

```{.python .input}
#@tab all
corr2d(d2l.transpose(X), K)
```

## 学习内核

如果我们知道这正是我们正在寻找的，那么用有限差分`[1, -1]`设计一个边缘检测器是非常巧妙的。然而，当我们查看较大的内核并考虑连续的卷积层时，可能不可能精确地手动指定每个过滤应该做什么。

现在，让我们看看是否可以通过只查看输入-输出对来了解从`Y`生成`X`的内核。我们首先构造卷积层，并将其核初始化为随机张量。接下来，在每次迭代中，我们将使用平方误差将`Y`与卷积层的输出进行比较。然后我们可以计算梯度来更新内核。为简单起见，在下面我们使用二维卷积层的内置类，而忽略偏差。

```{.python .input}
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2D(1, kernel_size=(1, 2), use_bias=False)
conv2d.initialize()

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # Update the kernel
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print(f'batch {i+1}, loss {float(l.sum()):.3f}')
```

```{.python .input}
#@tab pytorch
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'batch {i+1}, loss {l.sum():.3f}')
```

```{.python .input}
#@tab tensorflow
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        # Update the kernel
        update = tf.multiply(3e-2, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'batch {i+1}, loss {tf.reduce_sum(l):.3f}')
```

请注意，在10次迭代之后，误差已下降到一个较小的值。现在我们来看看我们学过的核张量。

```{.python .input}
d2l.reshape(conv2d.weight.data(), (1, 2))
```

```{.python .input}
#@tab pytorch
d2l.reshape(conv2d.weight.data, (1, 2))
```

```{.python .input}
#@tab tensorflow
d2l.reshape(conv2d.get_weights()[0], (1, 2))
```

事实上，学习到的核张量非常接近我们早先定义的核张量`K`。

## 互相关和卷积

回想一下我们从互相关和卷积运算之间:numref:`sec_why-conv`的对应关系中观察到的情况。这里，让我们继续考虑二维卷积层。如果这些层执行:eqref:`eq_2d-conv-discrete`中定义的严格卷积运算，而不是交叉相关，该怎么办？为了得到严格的*卷积*运算的输出，我们只需要水平和垂直翻转二维核张量，然后与输入张量进行*互相关*运算。

值得注意的是，由于核在深度学习中是从数据中学习的，所以无论卷积层执行严格的卷积运算还是互相关运算，卷积层的输出都保持不受影响。

为了说明这一点，假设卷积层执行*互相关‘并且在:numref:`fig_correlation`中学习核，其在这里被表示为矩阵$\mathbf{K}$。假设其他条件保持不变，当该层改为执行严格的*卷积*时，学习到的内核$\mathbf{K}'$将在$\mathbf{K}$被水平和垂直翻转之后与$\mathbf{K}'$相同。也就是说，当卷积层在:numref:`fig_correlation`和$\mathbf{K}'$中对输入执行严格的“卷积”时，将在:numref:`fig_correlation`中获得相同的输出(输入和$\mathbf{K}$的互相关)。

为了与深度学习文献中的标准术语保持一致，我们将继续将互相关运算称为卷积，尽管严格地说，它略有不同。此外，我们使用术语*元素*来指代表示层表示或卷积核的任何张量的条目(或组件)。

## 要素地图和感受场

如在:numref:`subsec_why-conv-channels`中所描述的，在:numref:`fig_correlation`中输出的卷积层有时被称为*特征图*，因为它可以被视为到后续层的空间维度(例如，宽度和高度)中的学习表示(特征)。在CNN中，对于某些层的任何元素$x$，其*接受字段*指的是在前向传播期间可能影响$x$的计算的所有元素(来自所有先前层)。请注意，接受字段可能大于输入的实际大小。

让我们继续使用:numref:`fig_correlation`来解释接受领域。在给定$2 \times 2$卷积核的情况下，着色输出元素(值为$19$)的接受字段是输入的着色部分中的四个元素。现在，让我们将$2 \times 2$的输出表示为$\mathbf{Y}$，并考虑更深层的cnn，该cnn具有附加的$2 \times 2$卷积层，该卷积层将$\mathbf{Y}$作为其输入，输出单个元素$z$。在这种情况下，$z$上的接受字段$\mathbf{Y}$包括$\mathbf{Y}$的所有四个元素，而输入上的接受字段包括所有九个输入元素。因此，当要素地图中的任何元素需要更大的接受域来检测更大范围内的输入要素时，我们可以构建更深层次的网络。

## 摘要

* 二维卷积层的核心计算是二维互相关运算。在其最简单的形式中，这对二维输入数据和内核执行互相关操作，然后添加偏差。
* 我们可以设计一个核来检测图像中的边缘。
* 我们可以从数据中了解内核的参数。
* 使用从数据学习的核，卷积层的输出保持不受影响，而不管这些层执行的操作(严格卷积或互相关)。
* 当要素地图中的任何元素需要更大的接受范围来检测输入上更广泛的要素时，可以考虑更深层次的网络。

## 练习

1. 构建具有对角线边缘的图像`X`。
    1. 如果将本节中的内核`K`应用于它，会发生什么情况？
    1. 如果你调换`X`会发生什么？
    1. 如果你调换`K`会发生什么？
1. 当您尝试自动查找我们创建的`Conv2D`类的渐变时，您看到了哪种错误消息？
1. 如何通过更改输入和核张量将互相关操作表示为矩阵乘法？
1. 手动设计一些内核。
    1. 二阶导数的核的形式是什么？
    1. 积分的核是什么？
    1. 要获得$d$次导数，核的最小大小是多少？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/65)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/66)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/271)
:end_tab:
