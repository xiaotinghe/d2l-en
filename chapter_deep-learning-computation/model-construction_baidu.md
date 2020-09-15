# 层和块
:label:`sec_model_construction`

当我们第一次引入神经网络时，我们关注的是具有单一输出的线性模型。在这里，整个模型只由一个神经元组成。请注意，单个神经元（i）接受一些输入；（ii）生成相应的标量输出；（iii）具有一组相关参数，这些参数可以更新以优化某些感兴趣的目标函数。然后，一旦我们开始考虑具有多个输出的网络，我们就利用矢量化算法来描述整个神经元层。就像单个神经元一样，层（i）接受一组输入，（ii）生成相应的输出，（iii）由一组可调参数描述。当我们使用softmax回归时，一个单层本身就是模型。然而，即使我们随后引入了MLPs，我们仍然可以认为该模型保留了相同的基本结构。

有趣的是，对于mlp，整个模型及其组成层都共享这种结构。整个模型接受原始输入（特征），生成输出（预测），并拥有参数（所有组成层的组合参数）。同样，每个单独的层接收输入（由前一层提供）生成输出（到下一层的输入），并且具有一组可调参数，这些参数根据从下一层向后流动的信号进行更新。

虽然您可能认为神经元、层和模型为我们的业务提供了足够的抽象，但事实证明，我们经常发现谈论比单个层大但比整个模型小的组件更方便。例如，ResNet-152体系结构在计算机视觉中非常流行，它拥有数百层。这些层由*层*组*的重复模式组成。一次只实现一层这样的网络会变得很乏味。这种担心不仅仅是假设性的——这种设计模式在实践中很常见。上面提到的ResNet架构赢得了2015年ImageNet和COCO计算机视觉比赛的识别和检测:cite:`He.Zhang.Ren.ea.2016`，仍然是许多视觉任务的首选架构。在其他领域，包括自然语言处理和语音，类似的体系结构以不同的重复模式排列在一起。

为了实现这些复杂的网络，我们引入了神经网络*块*的概念。块可以描述单个层、由多个层组成的组件或整个模型本身！使用块抽象的一个好处是可以将它们组合成更大的构件，通常是递归的。:numref:`fig_blocks`对此进行了说明。通过定义代码来按需生成任意复杂度的块，我们可以编写出奇紧凑的代码，同时仍然实现复杂的神经网络。

![Multiple layers are combined into blocks, forming repeating patterns of larger models.](../img/blocks.svg)
:label:`fig_blocks`

从编程的角度来看，块由*类*表示。它的任何子类都必须定义一个将其输入转换为输出的前向传播函数，并且必须存储任何必需的参数。请注意，有些块根本不需要任何参数。最后，为了计算梯度，块必须具有反向传播函数。幸运的是，在定义我们自己的块时，由于自动微分（在:numref:`sec_autograd`中引入）提供了一些幕后魔术，我们只需要担心参数和前向传播函数。

首先，我们回顾一下用于实现MLPs（:numref:`sec_mlp_concise`）的代码。下面的代码生成一个网络，其中一个完全连接的隐藏层有256个单元和ReLU激活，然后是一个具有10个单元的完全连接的输出层（没有激活功能）。

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

X = tf.random.uniform((2, 20))
net(X)
```

:begin_tab:`mxnet`
在这个例子中，我们通过实例化`nn.Sequential`来构建我们的模型，将返回的对象分配给`net`变量。接下来，我们反复调用它的`add`函数，按照应该执行的顺序添加层。简而言之，`nn.Sequential`定义了一种特殊类型的`Block`，这个类用胶子表示块。它维护一个有序的组成部分`Block`的列表，`add`函数只是方便将每个连续的`Block`添加到列表中。请注意，每个层都是`Dense`类的一个实例，该类本身就是`Block`的子类。前向传播（`forward`）函数也非常简单：它将列表中的每个`Block`链接在一起，将每个`Block`的输出作为下一个的输入传递给下一个。注意，到目前为止，我们一直在通过构造`net(X)`调用我们的模型来获得它们的输出。这实际上只是`net.forward(X)`的简写，这是通过`Block`类的`__call__`函数实现的一个巧妙的Python技巧。
:end_tab:

:begin_tab:`pytorch`
在这个例子中，我们通过实例化一个`nn.Sequential`来构建我们的模型，层的执行顺序是作为参数传递的。简而言之，`nn.Sequential`定义了一种特殊类型的`Module`，该类在PyTorch中表示一个块。它维护了一个有序的组成部分`Module`的列表，注意两个完全连接的层中的每一个都是`Linear`类的一个实例，这个类本身就是`Module`的子类。前向传播（`forward`）函数也非常简单：它将列表中的每个块链接在一起，将每个块的输出作为下一个块的输入。注意，到目前为止，我们一直在通过构造`net(X)`调用我们的模型来获得它们的输出。这实际上只是`net.forward(X)`的简写，这是通过Block类的`__call__`函数实现的一个巧妙的Python技巧。
:end_tab:

:begin_tab:`tensorflow`
在这个例子中，我们通过实例化一个`keras.models.Sequential`来构建我们的模型，层的执行顺序是作为参数传递的。简而言之，`Sequential`定义了一种特殊类型的`keras.Model`，该类在Keras中表示一个块。它维护一个有序的组成部分`Model`s的列表，注意两个完全连接的层中的每一个都是`Model`类的一个实例，这个类本身就是`Model`的子类。前向传播（`call`）函数也非常简单：它将列表中的每个块链接在一起，将每个块的输出作为下一个块的输入。注意，到目前为止，我们一直在通过构造`net(X)`调用我们的模型来获得它们的输出。这实际上只是`net.call(X)`的简写，这是通过Block类的`__call__`函数实现的一个巧妙的Python技巧。
:end_tab:

## 自定义块

也许发展关于块如何工作的直觉的最简单的方法是自己实现一个块。在实现我们自己的自定义块之前，我们简要总结一下每个块必须提供的基本功能：

1. 将输入数据作为其前向传播函数的参数。
1. 通过使前向传播函数返回值来生成输出。请注意，输出的形状可能与输入的形状不同。例如，我们上面模型中的第一个完全连接的层接收任意维的输入，但是返回一个维度256的输出。
1. 计算其输出相对于其输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的。
1. 存储并提供对执行前向传播计算所需的参数的访问。
1. 根据需要初始化模型参数。

在下面的代码片段中，我们从头开始编写一个块，对应于一个具有256个隐藏单元的隐藏层和一个10维输出层的MLP。注意，下面的`MLP`类继承了表示块的类。我们将严重依赖父类的函数，只提供我们自己的构造函数（Python中的`__init__`函数）和前向传播函数。

```{.python .input}
class MLP(nn.Block):
    # Declare a layer with model parameters. Here, we declare two
    # fully-connected layers
    def __init__(self, **kwargs):
        # Call the constructor of the `MLP` parent class `Block` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # Hidden layer
        self.out = nn.Dense(10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def forward(self, X):
        return self.out(self.hidden(X))
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the `MLP` parent class `Block` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # Hidden layer
        self.out = nn.Linear(256, 10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def forward(self, X):
        # Note here we use the funtional version of ReLU defined in the
        # nn.functional module.
        return self.out(F.relu(self.hidden(X)))
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the `MLP` parent class `Block` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__()
        # Hidden layer
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def call(self, X):
        return self.out(self.hidden((X)))
```

让我们首先关注前向传播函数。注意，它以`X`作为输入，计算应用激活函数的隐藏表示，并输出其逻辑。在这个`MLP`实现中，两个层都是实例变量。为了了解为什么这是合理的，设想实例化两个mlp，`net1`和`net2`，并在不同的数据上训练它们。当然，我们希望它们代表两种不同的学习模式。

我们在构造函数中实例化MLP的层，然后在每次调用前向传播函数时调用这些层。注意一些关键细节。首先，我们定制的`__init__`函数通过`super().__init__()`调用父类的`__init__`函数，从而避免了重新编写适用于大多数块的样板代码的痛苦。然后我们实例化两个完全连接的层，将它们分配给`self.hidden`和`self.out`。注意，除非我们实现一个新的运算符，否则我们不必担心反向传播函数或参数初始化。系统将自动生成这些功能。让我们试试这个。

```{.python .input}
net = MLP()
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch
net = MLP()
net(X)
```

```{.python .input}
#@tab tensorflow
net = MLP()
net(X)
```

块抽象的一个主要优点是它的多功能性。我们可以对块进行子类化以创建层（如完全连接的层类）、整个模型（如上面的`MLP`类）或具有中等复杂度的各种组件。我们在接下来的章节中充分利用了这种多功能性，比如在处理卷积神经网络时。

## 顺序块

现在我们可以更仔细地看看`Sequential`类是如何工作的。回想一下`Sequential`的设计是为了把其他模块串起来。为了构建我们自己的简化`MySequential`，我们只需要定义两个关键功能：
1. 一种将块逐个追加到列表中的函数。
2. 一种前向传播函数，用于将输入按与附加块相同的顺序传递给块链。

下面的`MySequential`类提供了与默认`Sequential`类相同的功能。

```{.python .input}
class MySequential(nn.Block):
    def add(self, block):
        # Here, `block` is an instance of a `Block` subclass, and we assume 
        # that it has a unique name. We save it in the member variable
        # `_children` of the `Block` class, and its type is OrderedDict. When
        # the `MySequential` instance calls the `initialize` function, the
        # system automatically initializes all members of `_children`
        self._children[block.name] = block

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._children.values():
            X = block(X)
        return X
```

```{.python .input}
#@tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            # Here, `block` is an instance of a `Module` subclass. We save it
            # in the member variable `_modules` of the `Module` class, and its
            # type is OrderedDict
            self._modules[block] = block

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._modules.values():
            X = block(X)
        return X
```

```{.python .input}
#@tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        for block in args:
            # Here, `block` is an instance of a `tf.keras.layers.Layer`
            # subclass
            self.modules.append(block)

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

:begin_tab:`mxnet`
`add`函数向有序字典`_children`添加一个块。您可能会想知道为什么每个胶子`Block`都有一个`_children`属性，以及为什么我们使用它而不只是自己定义一个Python列表。简而言之，`_children`的主要优点是在我们的块的参数初始化过程中，Gluon知道在`_children`字典中查找参数也需要初始化的子块。
:end_tab:

:begin_tab:`pytorch`
在`__init__`方法中，我们将每个块逐个添加到有序字典`_modules`中。您可能会想知道为什么每个`Module`都有一个`_modules`属性，以及为什么我们使用它而不只是自己定义一个Python列表。简而言之，`_modules`的主要优点是在我们的块参数初始化过程中，系统知道要在`_modules`字典中查找其参数也需要初始化的子块。
:end_tab:

当我们的`MySequential`的前向传播函数被调用时，每个添加的块都按照它们被添加的顺序执行。我们现在可以使用我们的`MySequential`类重新实现MLP。

```{.python .input}
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```

```{.python .input}
#@tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(X)
```

注意，`MySequential`的这种用法与我们之前为`Sequential`类编写的代码相同（如:numref:`sec_mlp_concise`中所述）。

## 在前向传播函数中执行代码

`Sequential`类使模型构造变得简单，允许我们组装新的体系结构，而不必定义自己的类。然而，并不是所有的架构都是简单的菊花链。当需要更大的灵活性时，我们需要定义自己的块。例如，我们可能希望在前向传播函数中执行Python的控制流。此外，我们可能希望执行任意的数学运算，而不是简单地依赖预定义的神经网络层。

您可能已经注意到，到目前为止，我们网络中的所有操作都对网络的激活及其参数起作用。然而，有时我们可能希望合并既不是前一层的结果也不是可更新参数的术语。我们称之为常数参数。例如，我们需要一个计算函数$f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$的层，其中$\mathbf{x}$是输入，$\mathbf{w}$是我们的参数，$c$是某个在优化过程中没有更新的指定常量。所以我们实现了一个`FixedHiddenMLP`类，如下所示。

```{.python .input}
class FixedHiddenMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Random weight parameters created with the `get_constant` function
        # are not updated during training (i.e., constant parameters)
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, X):
        X = self.dense(X)
        # Use the created constant parameters, as well as the `relu` and `dot`
        # functions
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        # Reuse the fully-connected layer. This is equivalent to sharing
        # parameters with two fully-connected layers
        X = self.dense(X)
        # Control flow
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # Use the created constant parameters, as well as the `relu` and `mm`
        # functions
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # Reuse the fully-connected layer. This is equivalent to sharing
        # parameters with two fully-connected layers
        X = self.linear(X)
        # Control flow
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab tensorflow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # Random weight parameters created with `tf.constant` are not updated
        # during training (i.e., constant parameters)
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        X = self.flatten(inputs)
        # Use the created constant parameters, as well as the `relu` and
        # `matmul` functions
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # Reuse the fully-connected layer. This is equivalent to sharing
        # parameters with two fully-connected layers
        X = self.dense(X)
        # Control flow
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
```

在这个`FixedHiddenMLP`模型中，我们实现了一个隐藏层，其权重（`self.rand_weight`）在实例化时被随机初始化，然后是常量。这个权重不是一个模型参数，因此它永远不会被反向传播更新。然后，网络将这个“固定”层的输出通过一个完全连接的层。

注意，在返回输出之前，我们的模型做了一些不寻常的事情。我们运行了一个while循环，在$L_1$范数大于$1$的条件下进行测试，并将输出向量除以$2$，直到它满足条件为止。最后，我们在`X`中返回了条目的总和。据我们所知，没有标准的神经网络执行这种操作。请注意，此特定操作在任何实际任务中可能都没有用处。我们的重点只是向您展示如何将任意代码集成到神经网络计算的流程中。

```{.python .input}
net = FixedHiddenMLP()
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch, tensorflow
net = FixedHiddenMLP()
net(X)
```

我们可以混合和匹配各种组装块的方法。在下面的示例中，我们以一些创造性的方式嵌套块。

```{.python .input}
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, X):
        return self.dense(self.net(X))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())
chimera.initialize()
chimera(X)
```

```{.python .input}
#@tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```

```{.python .input}
#@tab tensorflow
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(X)
```

## 汇编

:begin_tab:`mxnet, tensorflow`
热心的读者可能会开始担心其中一些操作的效率。毕竟，我们深度学习lib库，执行代码，还有很多其他的Python事情，它们都应该是一个高性能的深度学习库。Python的[global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock)的问题是众所周知的。在深入学习的背景下，我们担心我们的极快的GPU可能要等到一个微不足道的CPU运行Python代码之后，才能运行另一个作业。加速Python的最好方法是完全避免它。
:end_tab:

:begin_tab:`mxnet`
胶子这样做的一个方法是允许
*杂交*，这将在后面描述。
这里，Python解释器在第一次调用块时执行它。胶子运行时记录正在发生的事情，以及下一次它将对Python的调用短路。在某些情况下，这可以大大加快速度，但当控制流（如上所述）在不同的网络通道上引导不同的分支时，需要小心。我们建议感兴趣的读者在读完本章后，查看混合部分（:numref:`sec_hybridize`）来了解编译。
:end_tab:

## 摘要

* 层就是块。
* 许多层可以组成一个块。
* 一个块可以由许多块组成。
* 块可以包含代码。
* 块负责大量的内务处理，包括参数初始化和反向传播。
* 层和块的顺序连接由`Sequential`块处理。

## 练习

1. 如果将`MySequential`更改为在Python列表中存储块，会出现什么样的问题？
1. 实现一个以两个块为参数的块，例如`net1`和`net2`，并返回前向传播中两个网络的级联输出。这也被称为并行块。
1. 假设您想要连接同一网络的多个实例。实现一个工厂函数，该函数生成同一块的多个实例，并从中构建更大的网络。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/54)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/55)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/264)
:end_tab:
