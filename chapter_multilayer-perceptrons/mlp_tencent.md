# 多层感知器
:label:`sec_mlp`

在:numref:`chap_linear`，我们引入了Softmax回归(:numref:`sec_softmax`)，从头开始实现算法(:numref:`sec_softmax_scratch`)，使用高级API(:numref:`sec_softmax_concise`)，并训练分类器从低分辨率图像中识别10类服装。在此过程中，我们学习了如何处理数据，将输出强制为有效的概率分布，应用适当的损失函数，并根据模型参数将其最小化。既然我们已经在简单的线性模型的背景下掌握了这些力学，我们就可以开始探索深度神经网络，这是本书主要涉及的相对丰富的模型类别。

## 隐藏层

我们在:numref:`subsec_linear_model`中描述了仿射变换，它是一个加了偏差的线性变换。首先，回想一下与我们的Softmax回归示例相对应的模型体系结构，如:numref:`fig_softmaxreg`中所示。该模型通过单个仿射变换将我们的输入直接映射到我们的输出，然后进行Softmax操作。如果我们的标签确实通过仿射变换与我们的输入数据相关，那么这种方法就足够了。但是，仿射变换中的线性是一个“强有力的”假设。

### 线性模型可能会出错

例如，线性意味着*单调性*的“较弱”假设：特征的任何增加都必须总是导致模型输出的增加(如果对应的权重为正)，或者总是导致模型的输出减少(如果对应的权重为负)。有时候这是有道理的。例如，如果我们试图预测一个人是否会偿还贷款，我们可能会合理地认为，在其他条件不变的情况下，收入较高的申请人总是比收入较低的申请人更有可能偿还贷款。虽然单调，但这种关系可能与还款概率不是线性相关的。收入从0增加到5万，可能比从100万增加到105万更有可能带来更大的还款可能性。处理这一问题的一种方法可能是对我们的数据进行预处理，使线性变得更可信，比如说，通过使用收入的对数作为我们的特征。

请注意，我们可以很容易地找出违反单调性的示例。例如，我们想要根据体温预测死亡概率。对于体温高于37摄氏度(98.6华氏度)的个人来说，温度越高风险越大。然而，对于体温低于37摄氏度的人来说，温度越高，风险就越低！在这种情况下，我们也可以通过一些巧妙的预处理来解决问题。也就是说，我们可以使用37°C的距离作为我们的特征。

但是，如何对猫和狗的图像进行分类呢？增加位置(13，17)处像素的强度是否总是增加(或总是降低)图像描绘狗的可能性？对线性模型的依赖对应于一个隐含的假设，即区分猫和狗的唯一要求是评估单个像素的亮度。在一个倒置图像保留类别的世界里，这种方法注定要失败。

然而，尽管这里的线性显然是荒谬的，与我们前面的示例相比，我们可以通过简单的预处理修复来解决这个问题就不那么明显了。这是因为任何像素的重要性都以复杂的方式取决于其上下文(周围像素的值)。虽然我们的数据可能会有一种表示，它会考虑到我们的特征之间的相关交互作用，在此基础上建立一个线性模型将是合适的，但我们只是不知道如何手动计算它。对于深度神经网络，我们使用观测数据来联合学习通过隐藏层的表示和作用于该表示的线性预测器。

### 合并隐藏图层

我们可以克服线性模型的这些限制，通过合并一个或多个隐藏层来处理更一般的函数类。要做到这一点，最简单的方法是将许多完全连接的层堆叠在一起。每一层都提供给它上面的层，直到我们生成输出。我们可以把前$L-1$层看作我们的表示，把最后一层看作我们的线性预测器。这种架构通常称为“多层感知器”，通常缩写为*MLP*。下面，我们以图表方式描述了一个mlp(:numref:`fig_mlp`)。

![An MLP with a hidden layer of 5 hidden units. ](../img/mlp.svg)
:label:`fig_mlp`

该MLP有4个输入，3个输出，其隐藏层包含5个隐藏单元。由于输入层不涉及任何计算，因此使用此网络产生输出需要同时实现隐藏层和输出层的计算；因此，此MLP中的层数为2。请注意，这两个层都是完全连接的。每一次输入都会影响隐层中的每一个神经元，而每一次输入又会影响输出层中的每一个神经元。

### 从线性到非线性

如前所述，通过矩阵$\mathbf{X} \in \mathbb{R}^{n \times d}$，我们表示$n$个示例的小批量，其中每个示例具有$d$个输入(特征)。对于其隐藏层具有$h$个隐藏单元的单隐藏层mlp，用$\mathbf{H} \in \mathbb{R}^{n \times h}$表示隐藏层的输出，它们是
*隐藏表示*。
在数学或代码中，$\mathbf{H}$也称为“隐藏层变量”或“隐藏变量”。因为隐藏层和输出层都是完全连接的，所以我们具有隐藏层权重$\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$和偏置$\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$以及输出层权重$\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$和偏置$\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$。形式上，我们按如下方式计算单隐层最大似然比的输出$\mathbf{O} \in \mathbb{R}^{n \times q}$：

$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$

请注意，在添加隐藏层之后，我们的模型现在需要跟踪和更新其他参数集。那么我们在交换中得到了什么呢？您可能会惊讶地发现-在上面定义的模型中-*我们的麻烦一无所获*！原因很简单。上面的隐藏单元由输入的仿射函数给出，而输出(Pre-Softmax)只是隐藏单元的仿射函数。仿射函数的仿射函数本身就是仿射函数。此外，我们的线性模型已经能够表示任何仿射函数。

我们可以正式地查看等价性，方法是证明对于任意权重值，我们只需折叠隐藏层，即可产生具有参数$\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$和$\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$的等效单层模型：

$$
\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.
$$

为了实现多层结构的潜力，我们还需要一个关键因素：在仿射变换之后对每个隐藏单元应用非线性*激活函数*$\sigma$。激活功能(例如，$\sigma(\cdot)$)的输出被称为*激活*。一般来说，有了激活函数，就不可能再将我们的MLP折叠成线性模型：

$$
\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
\end{aligned}
$$

由于$\mathbf{X}$中的每一行对应于小批量中的一个示例，具有一些符号的滥用，所以我们定义非线性$\sigma$以行的方式应用于其输入，即，一次一个示例。请注意，在:numref:`subsec_softmax_vectorization`中，我们以相同的方式使用了SoftMAX的符号来表示行式操作。通常，如本节所述，我们应用于隐藏层的激活函数不仅是按行的，而且是按元素的。这意味着在计算层的线性部分之后，我们可以计算每个激活，而不需要查看其他隐藏单元所取的值。对于大多数激活功能都是如此。

为了构建更通用的MLP，我们可以继续堆叠这样的隐藏层，例如，$\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$和$\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$，一个在另一个之上，产生更具表现力的模型。

### 通用逼近器

MLP可以通过它们隐藏的神经元捕捉到我们输入之间的复杂相互作用，这取决于每个输入的值。我们可以很容易地设计隐藏节点来执行任意计算，例如，对一对输入进行基本逻辑操作。此外，对于激活函数的某些选择，众所周知，MLP是万能逼近器。即使是单隐层网络，给定足够的节点(可能非常多)和正确的权重集，我们也可以对任何函数建模，尽管实际上学习该函数是困难的部分。您可能认为您的神经网络有点像C编程语言。这种语言和任何其他现代语言一样，能够表达任何可计算的程序。但实际上要想出一个符合您的规范的程序才是最困难的部分。

而且，仅仅因为一个单隐层网络
*可以*学习任何函数
并不意味着您应该尝试使用单隐藏层网络来解决所有问题。事实上，通过使用更深(而不是更广)的网络，我们可以更紧凑地逼近许多函数。我们将在接下来的章节中涉及更严格的论点。

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

激活函数通过计算加权和并进一步加上偏差来决定神经元是否应该被激活。它们是将输入信号转换为输出的可微运算符，而它们中的大多数都增加了非线性。由于激活函数是深度学习的基础，让我们简要介绍一些常见的激活函数。

### RELU函数

最受欢迎的选择是*校正线性单元*(*RELU*)，因为它既实现简单，又在各种预测任务中表现良好。RELU提供了一种非常简单的非线性变换。给定元素$x$，函数被定义为该元素和$0$的最大值：

$$\operatorname{ReLU}(x) = \max(x, 0).$$

非正式地，RELU函数通过将相应的激活设置为0来仅保留正元素并丢弃所有负元素。为了获得一些直觉，我们可以画出函数的曲线图。如你所见，激活函数是分段线性的。

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

当输入为负时，RELU函数的导数为0，而当输入为正时，RELU函数的导数为1。请注意，当输入取值精确等于0时，RELU函数不可微。在这些情况下，我们缺省为左侧导数，并假设当输入为0时导数为0。我们可以逃脱惩罚，因为输入可能永远不会是零。有一句古老的谚语说，如果微妙的边界条件很重要，我们很可能是在做(*真正*)数学，而不是工程。这一传统观点可能适用于这里。我们绘制下图所示的RELU函数的导数。

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

使用RELU的原因是，它的衍生品表现得特别好：要么它们消失了，要么它们只是让论点通过了。这使得优化表现得更好，并缓解了困扰神经网络以前版本(稍后将详细介绍)的已有文献记载的梯度消失问题。

注意，RELU函数有许多变体，包括*参数化RELU*(*pReLU*)函数:cite:`He.Zhang.Ren.ea.2015`。此变体为relu添加了一个线性项，因此即使参数是否定的，某些信息仍然可以通过：

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

### Sigmoid函数

*Sigmoid函数*将其值位于域$\mathbb{R}$中的输入变换为位于区间(0，1)上的输出。因此，Sigmoid通常称为*挤压函数*：它将范围(-inf，inf)中的任何输入压缩到范围(0，1)中的某个值：

$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

在最早的神经网络中，科学家们感兴趣的是对“有火”或“不火”的生物神经元进行建模。因此，这一领域的先驱，从人工神经元的发明者麦卡洛克和皮茨开始，专注于阈值单位。阈值激活在其输入低于某个阈值时取值0，当输入超过阈值时取值1。

当注意力转移到基于梯度的学习时，Sigmoid函数是一个自然而然的选择，因为它是一个平滑的、可微分的阈值单元近似值。当我们想要将输出解释为二进制分类问题的概率时，Sigmoid仍然被广泛用作输出单元上的激活函数(您可以将Sigmoid视为Softmax的特例)。然而，乙状窦大部分已经被更简单、更容易训练的REU所取代，以便在隐藏的层中使用。在后面关于递归神经网络的章节中，我们将描述利用S型单元来控制跨时间信息流的体系结构。

下面，我们绘制Sigmoid函数。请注意，当输入接近0时，Sigmoid函数接近线性变换。

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

Sigmoid函数的导数由以下公式给出：

$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$

Sigmoid函数的导数如下所示。请注意，当输入为0时，Sigmoid函数的导数达到最大值0.25。当输入在任一方向上从0发散时，导数接近0。

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

### TANH函数

与Sigmoid函数类似，tanh(双曲正切)函数也压缩其输入，将其转换为-1和1之间的间隔上的元素：

$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

我们绘制下面的tanh函数。请注意，当输入接近0时，tanh函数接近线性变换。虽然函数的形状类似于Sigmoid函数，但tanh函数关于坐标系的原点表现出点对称。

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

tanh函数的导数是：

$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$

tanh函数的导数如下所示。当输入接近0时，tanh函数的导数接近最大值1。正如我们在Sigmoid函数中看到的，当输入在任一方向远离0时，tanh函数的导数接近0。

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

总而言之，我们现在知道如何结合非线性来构建富有表现力的多层神经网络结构。顺便说一句，您的知识已经让您掌握了一个类似于1990年左右的实践者的工具包。在某些方面，您比在20世纪90年代工作的任何人都有优势，因为您可以利用功能强大的开源深度学习框架快速构建模型，只需使用几行代码。以前，训练这些网络需要研究人员编写数千行C和Fortran代码。

## 摘要

* MLP在输出层和输入层之间增加一个或多个完全连接的隐藏层，并通过激活功能转换隐藏层的输出。
* 常用的激活功能包括RELU功能、Sigmoid功能和TOH功能。

## 练习

1. 计算pReLU激活函数的导数。
1. 证明了仅使用RELU(或pReLU)的MLP构造了一个连续的分段线性函数。
1. 拿出$\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x)$美元。
1. 假设我们有一个非线性，一次只适用于一个小批量。您预计这会造成什么样的问题？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/90)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/91)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/226)
:end_tab:
