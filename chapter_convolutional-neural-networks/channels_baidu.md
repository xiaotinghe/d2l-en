# 多输入多输出通道
:label:`sec_channels`

虽然我们已经在:numref:`subsec_why-conv-channels`中描述了构成每个图像的多个信道（例如，彩色图像具有标准的RGB信道来指示红、绿和蓝的数量）和多个信道的卷积层，但是到目前为止，我们通过仅使用单个输入和单个输出信道来简化我们的所有数值示例。这使得我们可以将输入、卷积核和输出看作二维张量。

当我们在混合中添加通道时，我们的输入和隐藏的表示都变成了三维张量。例如，每个RGB输入图像具有$3\times h\times w$的形状。我们将这个尺寸为3的轴称为*通道*维度。在本节中，我们将更深入地研究具有多个输入和多个输出通道的卷积核。

## 多输入通道

当输入数据包含多个通道时，需要构造一个与输入数据具有相同数目输入通道的卷积核，以便与输入数据进行互相关。假设输入数据的信道数为$c_i$，卷积核的输入信道数也需要为$c_i$。如果我们的卷积核的窗口形状是$k_h\times k_w$，那么当$c_i=1$时，我们可以把卷积核看作$k_h\times k_w$形状的二维张量。

然而，当$c_i>1$时，我们需要一个包含形状为$k_h\times k_w$的张量的内核，用于*每个*输入通道。将这些$c_i$张量连接在一起可以得到形状为$c_i\times k_h\times k_w$的卷积核。由于输入和卷积核都有$c_i$个通道，我们可以对每个通道的输入二维张量和卷积核的二维张量进行互相关运算，将$c_i$的结果相加（对通道求和）得到二维张量。这是多通道输入和多输入通道卷积核之间二维互相关的结果。

在:numref:`fig_conv_multi_in`中，我们演示了一个具有两个输入信道的二维互相关的示例。阴影部分是第一个输出元素以及用于输出计算的输入和核张量元素：$(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$。

![Cross-correlation computation with 2 input channels.](../img/conv-multi-in.svg)
:label:`fig_conv_multi_in`

为了确保我们真正理解这里的情况，我们可以自己用多个输入通道实现互相关操作。请注意，我们所做的就是对每个通道执行一个互相关操作，然后将结果相加。

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

我们可以构造与:numref:`fig_conv_multi_in`中的值相对应的输入张量`X`和核张量`K`，以验证互相关运算的输出。

```{.python .input}
#@tab all
X = d2l.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = d2l.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
```

## 多输出通道

不管输入通道的数量如何，到目前为止，我们总是以一个输出通道结束。然而，正如我们在:numref:`subsec_why-conv-channels`中所讨论的，在每一层都有多个信道是至关重要的。在最流行的神经网络架构中，当我们在神经网络中往上走的时候，我们实际上会增加信道的维数，通常是通过下采样来交换空间分辨率以获得更大的信道深度*。直观地说，您可以将每个频道看作是对一些不同功能集的响应。现实比对这种直觉的最天真的解释要复杂一些，因为表象不是独立学习的，而是为了共同使用而优化的。因此，可能不是单通道学习边缘检测器，而是通道空间中的某个方向对应于检测边缘。

用$c_i$和$c_o$分别表示输入和输出通道的数目，并让$k_h$和$k_w$为内核的高度和宽度。为了获得多个通道的输出，我们可以为*每个*输出通道创建一个形状为$c_i\times k_h\times k_w$的内核张量。我们将它们连接到输出通道维上，这样卷积核的形状是$c_o\times c_i\times k_h\times k_w$。在互相关运算中，每个输出通道上的结果都是从对应于该输出通道的卷积核计算出来的，并从输入张量中的所有通道中获取输入。

我们实现了一个互相关函数来计算多个信道的输出，如下所示。

```{.python .input}
#@tab all
def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of `K`, and each time, perform
    # cross-correlation operations with input `X`. All of the results are
    # stacked together
    return d2l.stack([corr2d_multi_in(X, k) for k in K], 0)
```

通过将核张量`K`与`K+1`（`K`中每个元素加一个）和`K+2`连接起来，构造了一个具有3个输出通道的卷积核。

```{.python .input}
#@tab all
K = d2l.stack((K, K + 1, K + 2), 0)
K.shape
```

下面，我们对输入张量`X`与内核张量`K`执行互相关操作。现在输出包含3个通道。第一通道的结果与先前输入张量`X`和多输入单输出通道核的结果一致。

```{.python .input}
#@tab all
corr2d_multi_in_out(X, K)
```

## $1\times 1$卷积层

一开始，$1 \times 1$卷积，即$k_h = k_w = 1$，似乎没有多大意义。毕竟，卷积与相邻像素相关。$1 \times 1$卷积显然不是。尽管如此，它们仍然是流行的操作，有时也包含在复杂的深层网络的设计中。让我们更详细地了解一下它的实际作用。

因为使用了最小窗口，$1\times 1$卷积失去了更大的卷积层识别由高度和宽度维度上相邻元素之间相互作用组成的模式的能力。$1\times 1$卷积的唯一计算发生在信道尺寸上。

:numref:`fig_conv_1x1`显示了使用$1\times 1$卷积核与3个输入通道和2个输出通道的互相关计算。请注意，输入和输出具有相同的高度和宽度。输出中的每个元素都是从输入图像中同一位置*的元素*的线性组合派生的。可以将$1\times 1$卷积层看作是在每个像素位置应用的全连接层，用于将$c_i$对应的输入值转换为$c_o$的输出值。因为这仍然是一个卷积层，所以权重是跨像素位置绑定的。因此，$1\times 1$卷积层需要$c_o\times c_i$权重（加上偏差）。

![The cross-correlation computation uses the $1\times 1$ convolution kernel with 3 input channels and 2 output channels. The input and output have the same height and width.](../img/conv-1x1.svg)
:label:`fig_conv_1x1`

让我们检查一下这在实践中是否有效：我们使用完全连接层实现$1 \times 1$卷积。唯一的问题是我们需要在矩阵乘法之前和之后对数据形状进行一些调整。

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

当执行$1\times 1$卷积时，上述函数相当于先前实现的互相关函数`corr2d_multi_in_out`。让我们用一些样本数据来验证这一点。

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

* 多通道可以用来扩展卷积层的模型参数。
* 当以每像素为基础应用时，$1\times 1$卷积层相当于全连接层。
* $1\times 1$卷积层通常用于调整网络层之间的信道数量和控制模型复杂性。

## 练习

1. 假设我们有两个卷积核，大小分别为$k_1$和$k_2$（中间没有非线性）。
    1. 证明了运算结果可以用一个卷积表示。
    1. 等效单卷积的维数是多少？
    1. 反之亦然吗？
1. 假设输入为$c_i\times h\times w$，卷积核为$c_o\times c_i\times k_h\times k_w$，填充为$(p_h, p_w)$，步长为$(s_h, s_w)$。
    1. 前向传播的计算成本（乘法和加法）是多少？
    1. 内存占用是多少？
    1. 向后计算的内存占用是多少？
    1. 反向传播的计算成本是多少？
1. 如果我们将输入通道$c_i$和输出通道$c_o$的数量加倍，计算数量会增加多少？如果我们把填充物翻一番会怎么样？
1. 如果卷积核的高度和宽度是$k_h=k_w=1$，前向传播的计算复杂度是多少？
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
