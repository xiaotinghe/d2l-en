# 多层网络
:label:`sec_mlp`

在:numref:`chap_linear`中，我们引入了softmax回归（:numref:`sec_softmax`），实现了从头开始的算法（:numref:`sec_softmax_scratch`）和使用高级api（:numref:`sec_softmax_concise`），并训练分类器从低分辨率图像中识别出10类服装。在此过程中，我们学习了如何对数据进行纠结，将输出强制为有效的概率分布，应用适当的损失函数，并根据模型参数最小化它。既然我们已经在简单线性模型的背景下掌握了这些机制，我们就可以开始探索深层神经网络，这是本书主要关注的一类相对丰富的模型。

## 隐藏层

我们已经在:numref:`subsec_linear_model`中描述了仿射变换，它是一个带有偏差的线性变换。首先，回顾一下与我们的softmax回归示例相对应的模型体系结构，如:numref:`fig_softmaxreg`所示。这个模型通过一个单一的仿射变换直接将我们的输入映射到我们的输出，然后是一个softmax操作。如果我们的标签确实通过仿射变换与我们的输入数据相关，那么这种方法就足够了。但是仿射变换中的线性是一个很强的假设。

### 线性模型可能会出错

例如，线性意味着*较弱*单调性*假设：我们特征的任何增加必须总是导致模型输出的增加（如果相应的权重是正的），或者总是导致模型输出的减少（如果相应的权重是负的）。有时候这是有道理的。例如，如果我们试图预测一个人是否会偿还贷款，我们可以合理地设想，在其他条件不变的情况下，收入较高的申请人总是比收入较低的申请人更有可能偿还贷款。虽然是单调的，但这种关系可能与偿还概率没有线性关系。收入从0万增加到5万，与从100万增加到105万相比，还款可能性的增加可能更大。处理这一问题的一种方法可能是对我们的数据进行预处理，使线性变得更合理，比如说，使用收入对数作为我们的特征。

请注意，我们可以很容易地想出违反单调性的例子。比如说，我们要根据体温来预测死亡概率。对于体温高于37°C（98.6°F）的人来说，温度越高风险越大。但是，对于体温低于37°C的人来说，温度越高风险越低！在这种情况下，我们也可以通过一些巧妙的预处理来解决这个问题。也就是说，我们可以使用37°C的距离作为我们的特征。

但是，如何对猫和狗的图像进行分类呢？在位置（13，17）增加像素的强度是否应该总是增加（或总是降低）图像描绘狗的可能性？对线性模型的依赖对应于一个隐含的假设：区分猫和狗的唯一要求是评估单个像素的亮度。这种方法注定要失败，在一个反转图像保留类别的世界中。

然而，尽管与我们之前的例子相比，这里的线性显然是荒谬的，但是我们可以用一个简单的预处理修复来解决这个问题就不那么明显了。这是因为任何像素的重要性都以复杂的方式依赖于其上下文（周围像素的值）。虽然我们的数据可能存在一种表示形式，它将考虑到我们的特征之间的相关交互作用，在此基础上建立一个线性模型是合适的，但我们根本不知道如何手工计算它。对于深度神经网络，我们使用观测数据来共同学习通过隐藏层的表示和作用于该表示的线性预测器。

### 合并隐藏层

我们可以克服线性模型的这些局限性，通过合并一个或多个隐藏层来处理更一般的函数类。最简单的方法是将许多完全连接的层堆叠在一起。每一层输入到它上面的层，直到我们产生输出。我们可以把前$L-1$层看作我们的表示，最后一层作为我们的线性预测器。这种结构通常称为*多层感知器*，通常缩写为*MLP*。下面，我们用图表描述了一个MLP（:numref:`fig_mlp`）。

![An MLP with a hidden layer of 5 hidden units. ](../img/mlp.svg)
:label:`fig_mlp`

该MLP有4个输入，3个输出，其隐藏层包含5个隐藏单元。由于输入层不涉及任何计算，使用该网络生成输出需要同时实现隐藏层和输出层的计算；因此，该MLP中的层数为2。请注意，这些层都是完全连接的。每一个输入都会影响隐藏层中的每一个神经元，而每一个输入又会影响输出层中的每一个神经元。

### 从线性到非线性

如前所述，通过矩阵$\mathbf{X} \in \mathbb{R}^{n \times d}$，我们表示$n$的小批量示例，其中每个示例有$d$个输入（特征）。对于隐藏层具有$h$个隐藏单元的一个隐藏层MLP，用$\mathbf{H} \in \mathbb{R}^{n \times h}$表示隐藏层的输出，这些输出是
*隐藏表示*。
在数学或代码中，$\mathbf{H}$也被称为*隐藏层变量*或*隐藏变量*。由于隐藏层和输出层都是完全连接的，我们有隐藏层权重$\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$和偏移$\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$，输出层权重$\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$和偏移$\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$。形式上，我们计算一个隐藏层MLP的输出$\mathbf{O} \in \mathbb{R}^{n \times q}$如下：

$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$

请注意，在添加隐藏层之后，我们的模型现在要求我们跟踪和更新其他参数集。那么我们从中得到了什么呢？你可能会惊讶地发现——在上面定义的模型中——*我们从麻烦中得不到任何好处*！原因很简单。上面的隐藏单元由输入的仿射函数给出，而输出（pre softmax）只是隐藏单元的仿射函数。仿射函数的仿射函数本身就是仿射函数。此外，我们的线性模型已经能够表示任何仿射函数。

我们可以通过证明，对于任何权重值，我们都可以折叠隐藏层，得到一个参数为$\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$和$\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$的等效单层模型：

$$
\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.
$$

为了实现多层体系结构的潜力，我们还需要一个更关键的因素：在仿射变换之后，将一个非线性*激活函数*$\sigma$应用于每个隐藏单元。激活函数（例如$\sigma(\cdot)$）的输出称为*激活*。一般来说，在激活函数到位的情况下，不再可能将MLP压缩为线性模型：

$$
\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
\end{aligned}
$$

由于$\mathbf{X}$中的每一行都对应于minibatch中的一个示例，因此我们定义了非线性$\sigma$以行方式应用于其输入，即一次一个示例。注意，在:numref:`subsec_softmax_vectorization`中，我们以同样的方式使用softmax的符号来表示行操作。通常，如本节所述，我们应用于隐藏层的激活函数不仅仅是行方式的，而是元素方面的。这意味着在计算了层的线性部分之后，我们可以计算每个激活，而不必查看其他隐藏单元的值。大多数激活函数都是这样。

为了构建更一般的mlp，我们可以继续堆叠这样的隐藏层，例如$\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$和$\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$，一个接一个，产生更具表现力的模型。

### 通用逼近器

mlp可以通过它们的隐藏神经元捕捉我们输入之间的复杂交互作用，这些神经元依赖于每个输入的值。我们可以很容易地设计隐藏节点来执行任意计算，例如，对一对输入进行基本逻辑运算。此外，对于激活函数的某些选择，众所周知，mlp是通用逼近器。即使只有一个隐藏层网络，给定足够多的节点（可能有荒谬的多）和正确的权重集，我们也可以为任何函数建模，尽管实际上学习该函数是困难的部分。你可能会认为你的神经网络有点像C编程语言。这种语言和其他现代语言一样，能够表达任何可计算程序。但实际上，制定出一个符合你的规范的程序是很困难的。

而且，仅仅因为一个单一的隐藏层网络
*可以学习任何函数
但这并不意味着你应该用一个隐藏层网络来解决所有的问题。事实上，我们可以通过使用更深（而不是更广）的网络来更紧凑地近似许多函数。我们将在后面的章节中讨论更严格的论点。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

## 激活函数

激活函数通过计算加权和并进一步加上偏差来决定神经元是否应该被激活。它们是将输入信号转换为输出信号的可微算子，但大多数算子都加入了非线性。因为激活函数是深度学习的基础，让我们简单介绍一些常见的激活函数。

### ReLU函数

由于实现的简单性和它在各种预测任务上的良好性能，最流行的选择是*校正线性单元*（*ReLU*）。ReLU提供了一个非常简单的非线性变换。给定一个元素$x$，函数定义为该元素和$0$的最大值：

$$\operatorname{ReLU}(x) = \max(x, 0).$$

非正式地说，ReLU函数只保留正元素，并通过将相应的激活设置为0来丢弃所有负元素。为了获得一些直觉，我们可以画出函数。如你所见，激活函数是分段线性的。

```{.python .input}
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'relu(x)', figsize=(5, 2.5))
```

当输入为负时，ReLU函数的导数为0；当输入为正时，ReLU函数的导数为1。注意，当输入值精确等于0时，ReLU函数是不可微的。在这些情况下，我们默认使用左边的导数，当输入为0时，我们假设导数为0。我们可以摆脱这种情况，因为输入可能永远不会是零。有句老话说，如果微妙的边界条件很重要，我们可能在做数学，而不是工程。传统的智慧在这里可能适用。我们绘制了下面绘制的ReLU函数的导数。

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.relu(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of relu',
         figsize=(5, 2.5))
```

使用ReLU的原因是它的导数表现得特别好：要么消失，要么就让争论通过。这使得优化表现得更好，并且它减轻了困扰以前版本的神经网络的梯度消失的问题（稍后将详细介绍）。

请注意，ReLU函数有许多变体，包括*参数化ReLU*（*pReLU*）函数:cite:`He.Zhang.Ren.ea.2015`。这种变化为ReLU添加了一个线性项，因此即使参数为负数，某些信息仍然可以通过：

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

### 乙状窦函数

*sigmoid函数*将其输入（其值位于域$\mathbb{R}$）转换为间隔（0，1）上的输出。因此，sigmoid通常被称为*压缩函数*：它将范围（-inf，inf）中的任何输入压缩为范围（0，1）中的某个值：

$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

在最早的神经网络中，科学家们对模拟生物神经元很感兴趣，这些神经元要么发射，要么不发射。因此，这一领域的先驱者，一直追溯到人工神经元的发明者McCulloch和Pitts，把重点放在阈值单元上。阈值激活在输入低于某个阈值时取值0，当输入超过阈值时取值1。

当注意力转向基于梯度的学习时，sigmoid函数是一个自然的选择，因为它是一个平滑的，可微的阈值单元近似。当我们想将输出解释为二进制分类问题的概率时，sigmoid仍然被广泛用作输出单元的激活函数（您可以将sigmoid看作softmax的一个特例）。然而，sigmoid大部分已经被更简单和更容易训练的ReLU所取代，大多数用于隐藏层。在后面关于递归神经网络的章节中，我们将描述利用sigmoid单元来控制跨时间的信息流的体系结构。

下面，我们绘制sigmoid函数。注意，当输入接近0时，sigmoid函数接近线性变换。

```{.python .input}
with autograd.record():
    y = npx.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

sigmoid函数的导数由以下方程给出：

$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$

sigmoid函数的导数如下所示。注意，当输入为0时，sigmoid函数的导数达到最大值0.25。当输入从0向任一方向发散时，导数接近0。

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
# Clear out previous gradients
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of sigmoid',
         figsize=(5, 2.5))
```

### Tanh函数

与sigmoid函数一样，tanh（双曲正切）函数也会压缩其输入，将其转换为-1和1之间间隔上的元素：

$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

我们在下面绘制tanh函数。注意，当输入接近0时，tanh函数接近线性变换。尽管函数的形状类似于sigmoid函数，tanh函数在坐标系原点处表现出点对称性。

```{.python .input}
with autograd.record():
    y = np.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
y = tf.nn.tanh(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

tanh函数的导数为：

$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$

tanh函数的导数绘制如下。当输入接近0时，tanh函数的导数接近最大值1。正如我们在sigmoid函数中看到的，当输入从0向任意方向移动时，tanh函数的导数接近0。

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
# Clear out previous gradients.
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.tanh(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of tanh',
         figsize=(5, 2.5))
```

总之，我们现在知道了如何结合非线性来构建具有表现力的多层神经网络结构。顺便说一句，你的知识已经让你在1990年左右掌握了一个类似于从业者的工具箱。在某些方面，您比90年代的任何人都有优势，因为您可以利用强大的开源深度学习框架快速构建模型，只需几行代码。以前，训练这些网络需要研究人员编写数千行C和Fortran代码。

## 摘要

* MLP在输出层和输入层之间添加一个或多个完全连接的隐藏层，并通过激活函数转换隐藏层的输出。
* 常用的激活函数包括ReLU函数、sigmoid函数和tanh函数。

## 练习

1. 计算pReLU激活函数的导数。
1. 证明了只使用ReLU（或pReLU）的MLP构造了一个连续的分段线性函数。
1. 显示$\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x)$。
1. 假设我们有一个非线性，一次适用于一个小批量。你认为这会导致什么样的问题？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/90)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/91)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/226)
:end_tab:
