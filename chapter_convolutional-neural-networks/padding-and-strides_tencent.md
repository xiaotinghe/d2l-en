# 填充和跨度
:label:`sec_padding`

在前面的:numref:`fig_correlation`示例中，我们的输入高度和宽度都是3，而卷积核的高度和宽度都是2，从而产生了维度为$2\times2$的输出表示。正如我们在:numref:`sec_conv_layer`中概括的那样，假设输入形状为$n_h\times n_w$，卷积核形状为$k_h\times k_w$，则输出形状将为$(n_h-k_h+1) \times (n_w-k_w+1)$。因此，卷积层的输出形状由输入的形状和卷积核的形状确定。

在某些情况下，我们采用了影响输出大小的技术，包括填充和跨步卷积。作为动机，请注意，由于内核的宽度和高度通常大于$1$，因此在应用多次连续卷积之后，我们最终得到的输出往往比输入小得多。如果我们从一幅$240 \times 240$像素的图像开始，$10$层的$5 \times 5$次卷积将图像减少到$200 \times 200$像素，将图像切下$30\$，从而抹去了原始图像边界上任何有趣的信息。
*填充*是处理此问题的最流行工具。

在其他情况下，我们可能想要大幅降低维数，例如，如果我们发现原始输入分辨率很笨拙。
*步进卷积*是一种流行的技术，可以在这些情况下提供帮助。

## 填充

如上所述，应用卷积层时的一个棘手问题是，我们往往会丢失图像周边的像素。由于我们通常使用较小的内核，对于任何给定的卷积，我们可能只丢失几个像素，但当我们应用许多连续的卷积层时，这可能会累积起来。这个问题的一个直接解决方案是在输入图像的边界周围添加额外像素的填充物，从而增加图像的有效大小。通常，我们将额外像素的值设置为零。在:numref:`img_conv_pad`中，我们填充$3 \times 3$个输入，将其大小增加到$5 \times 5$。然后，相应的输出增加到$4 \times 4$矩阵。阴影部分是第一个输出元素以及用于输出计算的输入和核张量元素：$0\times0+0\times1+0\times2+0\times3=0$。

![Two-dimensional cross-correlation with padding.](../img/conv-pad.svg)
:label:`img_conv_pad`

通常，如果我们总共添加$p_h$行填充(大约一半在顶部，一半在底部)和总共$p_w$列填充(大约一半在左边，一半在右边)，输出形状将是

$$(n_h-k_h+p_h+1)\times(n_w-k_w+p_w+1).$$

这意味着输出的高度和宽度将分别增加$p_h$和$p_w$。

在许多情况下，我们需要设置$p_h=k_h-1$和$p_w=k_w-1$以使输入和输出具有相同的高度和宽度。这将使构建网络时更容易预测每一层的输出形状。假设这里$k_h$是奇数，我们将在高度两侧填充$p_h/2$行。如果$k_h$是偶数，一种可能是在输入的顶部填充$\lceil p_h/2\rceil$行，在底部填充$\lfloor p_h/2\rfloor$行。我们将用同样的方式填充宽度的两边。

CNN通常使用高度和宽度值为奇数的卷积核，例如1、3、5或7。选择奇数核大小的好处是，我们可以在填充相同行数和相同行数、相同列数的情况下保留空间维度。

此外，这种使用奇数内核和填充来精确保留维度的做法提供了文书方面的好处。对于任何二维张量`X`，当核的大小是奇数并且所有边的填充行数和列数相同时，产生与输入具有相同高度和宽度的输出时，我们知道输出`Y[i, j]`是通过输入和卷积核的互相关来计算的，窗口以`X[i, j]`为中心。

在下面的示例中，我们创建一个高度和宽度为3的二维卷积层，并在所有侧面应用1像素的填充。给定高度和宽度为8的输入，我们发现输出的高度和宽度也是8。

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# For convenience, we define a function to calculate the convolutional layer.
# This function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return Y.reshape(Y.shape[2:])

# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = np.random.uniform(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

# We define a convenience function to calculate the convolutional layer. This
# function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return Y.reshape(Y.shape[2:])
# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

# We define a convenience function to calculate the convolutional layer. This
# function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    X = tf.reshape(X, (1, ) + X.shape + (1, ))
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return tf.reshape(Y, Y.shape[1:3])
# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

当卷积核的高度和宽度不同时，通过设置不同的高度和宽度填充数，可以使输出和输入具有相同的高度和宽度。

```{.python .input}
# Here, we use a convolution kernel with a height of 5 and a width of 3. The
# padding numbers on either side of the height and width are 2 and 1,
# respectively
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
# Here, we use a convolution kernel with a height of 5 and a width of 3. The
# padding numbers on either side of the height and width are 2 and 1,
# respectively
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
# Here, we use a convolution kernel with a height of 5 and a width of 3. The
# padding numbers on either side of the height and width are 2 and 1,
# respectively
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(5, 3), padding='valid')
comp_conv2d(conv2d, X).shape
```

## 大步走

在计算互相关时，我们从输入张量左上角的卷积窗口开始，然后将其向下和向右滑动到所有位置。在前面的示例中，我们默认一次滑动一个图元。然而，有时，为了提高计算效率，或者因为我们希望下采样，我们一次移动窗口的元素超过一个，跳过中间位置。

我们将每张幻灯片遍历的行数和列数称为*跨度*。到目前为止，我们已经对高度和宽度使用了步幅1。有时候，我们可能想用更大的步幅。:numref:`img_conv_stride`表示垂直跨度为3，水平跨度为2的二维互相关运算。阴影部分是输出元素以及用于输出计算的输入和核张量元素：$0\times0+0\times1+1\times2+2\times3=8$、$0\times0+6\times1+0\times2+0\times3=6$。我们可以看到，当输出第一列的第二个元素时，卷积窗口向下滑动三行。输出第一行的第二个元素时，卷积窗口向右滑动两列。当卷积窗口在输入上继续向右滑动两列时，没有输出，因为输入元素不能填满窗口(除非我们添加另一列填充)。

![Cross-correlation with strides of 3 and 2 for height and width, respectively.](../img/conv-stride.svg)
:label:`img_conv_stride`

通常，当高度的步幅为$s_h$，宽度的步幅为$s_w$时，输出形状为

$$\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor.$$

如果设置为$p_h=k_h-1$和$p_w=k_w-1$，则输出形状将简化为$\lfloor(n_h+s_h-1)/s_h\rfloor \times \lfloor(n_w+s_w-1)/s_w\rfloor$。更进一步，如果输入高度和宽度可以被高度和宽度上的跨度整除，那么输出形状将是$(n_h/s_h) \times (n_w/s_w)$。

下面，我们将高度和宽度上的步幅都设置为2，从而将输入高度和宽度减半。

```{.python .input}
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
comp_conv2d(conv2d, X).shape
```

接下来，我们将看一个稍微复杂一些的示例。

```{.python .input}
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(3,5), padding='valid', strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

为简洁起见，当输入高度和宽度两侧的填充数分别为$p_h$和$p_w$时，我们称填充数为$(p_h, p_w)$。具体地说，如果为$p_h = p_w = p$，则填充为$p$。当高度和宽度上的步幅分别为$s_h$和$s_w$时，我们称步幅为$(s_h, s_w)$。具体地说，值为$s_h = s_w = s$时，步幅为$s$。默认情况下，填充为0，跨度为1。实际上，我们很少使用不均匀的跨度或填充，即通常为$p_h = p_w$和$s_h = s_w$。

## 摘要

* 填充可以增加输出的高度和宽度。这通常用于使输出具有与输入相同的高度和宽度。
* 步幅可能会降低输出的分辨率，例如，将输出的高度和宽度减少到仅为输入高度和宽度的$1/n$($n$是大于$1$的整数)。
* 填充和跨度可以有效地调整数据的维数。

## 练习

1. 对于本节中的最后一个示例，使用数学计算输出形状，以查看其是否与实验结果一致。
1. 在本节的实验中尝试其他填充和跨距组合。
1. 对于音频信号，步幅2对应的是什么？
1. 步幅大于1的计算优势是什么？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/67)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/68)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/272)
:end_tab:
