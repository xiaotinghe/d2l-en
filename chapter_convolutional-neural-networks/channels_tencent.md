# 多输入多输出通道
:label:`sec_channels`

虽然我们已经在:numref:`subsec_why-conv-channels`中描述了组成每个图像的多个通道(例如，彩色图像具有表示红、绿和蓝的量的标准rgb通道)和多个通道的卷积层，但到目前为止，我们通过仅使用单个输入和单个输出通道简化了所有数值示例。这使得我们可以把我们的输入、卷积核和输出都看作是二维张量。

当我们将通道添加到混合中时，我们的输入和隐藏表示都变成了三维张量。例如，每个rgb输入图像具有形状$3\times h\times w$。我们将这个大小为3的轴称为*channel*维。在本节中，我们将更深入地研究具有多个输入和多个输出通道的卷积内核。

## 多个输入通道

当输入数据包含多个通道时，需要构造一个与输入数据具有相同输入通道数的卷积核，以便与输入数据进行互相关。假设输入数据的通道数为$c_i$，则卷积核的输入通道数也需要为$c_i$。如果我们的卷积核的窗口形状是$k_h\times k_w$，那么当为$c_i=1$时，我们可以认为我们的卷积核只是形状$k_h\times k_w$的二维张量。

但是，当值为$c_i>1$时，我们需要一个核，该核包含*每个*输入通道的形状为$k_h\times k_w$的张量。将这$c_i$个张量连接在一起产生形状为$c_i\times k_h\times k_w$的卷积核。由于输入和卷积核各自具有$c_i$个通道，因此我们可以对每个通道的输入的二维张量和卷积核的二维张量执行互相关操作，将$c_i$个结果相加(对通道求和)以产生二维张量。这是多通道输入和多输入通道卷积核之间二维互相关的结果。

在:numref:`fig_conv_multi_in`中，我们演示了一个具有两个输入通道的二维互相关的例子。阴影部分是第一个输出元素以及用于输出计算的输入和核张量元素：$(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$。

![Cross-correlation computation with 2 input channels.](../img/conv-multi-in.svg)
:label:`fig_conv_multi_in`

为了确保我们真正理解这里发生的事情，我们可以自己实现具有多个输入通道的互相关操作。请注意，我们所做的全部工作就是对每个通道执行一次互相关操作，然后将结果相加。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab mxnet, pytorch
def corr2d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def corr2d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return tf.reduce_sum([d2l.corr2d(x, k) for x, k in zip(X, K)], axis=0)
```

我们可以构造输入张量`X`和核张量`K`，它们对应于:numref:`fig_conv_multi_in`中的值，以验证互相关运算的输出。

```{.python .input}
#@tab all
X = d2l.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = d2l.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
```

## 多个输出通道

不管有多少个输入通道，到目前为止，我们总是得到一个输出通道。但是，正如我们在:numref:`subsec_why-conv-channels`中讨论的那样，每层都必须有多个通道。在最流行的神经网络体系结构中，我们实际上会随着神经网络中更高的位置而增加通道维度，通常是向下采样以牺牲空间分辨率来换取更大的*通道深度*。直观地说，您可以将每个通道视为对某些不同功能集的响应。现实比对这种直觉的最天真的解释要复杂一些，因为表征不是独立学习的，而是经过优化才能共同发挥作用的。因此，可能不是单个通道学习边缘检测器，而是通道空间中的某个方向对应于检测边缘。

分别用$c_i$和$c_o$表示输入和输出通道的数量，并让$k_h$和$k_w$表示内核的高度和宽度。要获得具有多个通道的输出，我们可以为*每个*输出通道创建一个形状为$c_i\times k_h\times k_w$的核张量。我们将它们串联在输出通道维度上，因此卷积核的形状为$c_o\times c_i\times k_h\times k_w$。在互相关运算中，每个输出通道的结果从对应于该输出通道的卷积核中计算，并从输入张量中的所有通道获取输入。

我们实现了一个互相关函数来计算多个通道的输出，如下所示。

```{.python .input}
#@tab all
def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of `K`, and each time, perform
    # cross-correlation operations with input `X`. All of the results are
    # stacked together
    return d2l.stack([corr2d_multi_in(X, k) for k in K], 0)
```

我们通过将核张量`K`与`K+1`(对于`K`中的每个元素加上一个)和`K+2`级联来构造具有3个输出通道的卷积核。

```{.python .input}
#@tab all
K = d2l.stack((K, K + 1, K + 2), 0)
K.shape
```

下面，我们对输入张量`X`和核张量`K`执行互相关操作。现在输出包含3个通道。第一通道的结果与先前输入张量`X`和多输入通道、单输出通道核的结果一致。

```{.python .input}
#@tab all
corr2d_multi_in_out(X, K)
```

## $1\times 1$卷积层

起初，$1 \times 1$的卷积，即$k_h = k_w = 1$，似乎没有多大意义。毕竟，卷积使相邻像素相互关联。$1 \times 1$的卷积显然不会。尽管如此，它们仍然是常见的操作，有时会包含在复杂深层网络的设计中。让我们更详细地看看它的实际作用。

因为使用了最小窗口，所以$1\times 1$卷积失去了较大卷积层识别由高度和宽度维度中相邻元素之间的相互作用组成的图案的能力。仅在通道维度上进行$1\times 1$卷积的计算。

:numref:`fig_conv_1x1`示出了使用具有3个输入通道和2个输出通道的$1\times 1$卷积核进行的互相关计算。请注意，输入和输出具有相同的高度和宽度。输出中的每个元素都是从输入图像中*相同位置*的元素的线性组合中导出的。您可以将$1\times 1$卷积层视为构成一个完全连接的层，该层应用于每个单个像素位置，以将$c_i$个相应的输入值转换为$c_o$个输出值。因为这仍然是一个卷积层，所以权重是跨像素位置绑定的。因此，$1\times 1$卷积层需要$c_o\times c_i$个权重(加上偏差)。

![The cross-correlation computation uses the $1\times 1$ convolution kernel with 3 input channels and 2 output channels. The input and output have the same height and width.](../img/conv-1x1.svg)
:label:`fig_conv_1x1`

让我们检查一下这在实践中是否有效：我们使用完全连接层实现$1 \times 1$的卷积。唯一的问题是，我们需要在矩阵乘法前后对数据形状进行一些调整。

```{.python .input}
#@tab all
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = d2l.reshape(X, (c_i, h * w))
    K = d2l.reshape(K, (c_o, c_i))
    Y = d2l.matmul(K, X)  # Matrix multiplication in the fully-connected layer
    return d2l.reshape(Y, (c_o, h, w))
```

当执行$1\times 1$卷积时，上述函数等同于先前实现的互相关函数`corr2d_multi_in_out`。让我们用一些样本数据来验证这一点。

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0, 1, (3, 3, 3))
K = d2l.normal(0, 1, (2, 3, 1, 1))
```

```{.python .input}
#@tab tensorflow
X = d2l.normal((3, 3, 3), 0, 1)
K = d2l.normal((2, 3, 1, 1), 0, 1)
```

```{.python .input}
#@tab all
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

## 摘要

* 可以使用多个通道来扩展卷积层的模型参数。
* 当以每像素为基础应用时，$1\times 1$卷积层等同于完全连接层。
* $1\times 1$卷积层通常用于调整网络层之间的信道数量并控制模型复杂性。

## 练习

1. 假设我们有两个大小分别为$k_1$和$k_2$的卷积核(其间没有非线性)。
    1. 证明了运算结果可以用一次卷积表示。
    1. 等效单卷积的维数是多少？
    1. 反之亦然吗？
1. 假设形状$c_i\times h\times w$的输入和形状$c_o\times c_i\times k_h\times k_w$的卷积核、填充$(p_h, p_w)$和跨度$(s_h, s_w)$。
    1. 前向传播的计算开销(乘法和加法)是多少？
    1. 内存占用量是多少？
    1. 向后计算的内存占用有多大？
    1. 反向传播的计算成本是多少？
1. 如果我们将输入通道$c_i$的数量和输出通道$c_o$的数量增加一倍，计算的数量会增加多少倍？如果我们把填充物翻一番会怎么样？
1. 如果卷积核的高度和宽度为$k_h=k_w=1$，则前向传播的计算复杂度是多少？
1. 本节最后一个示例中的变量`Y1`和`Y2`是否完全相同？为什么？
1. 当卷积窗口不是$1\times 1$时，如何使用矩阵乘法实现卷积？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/69)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/70)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/273)
:end_tab:
