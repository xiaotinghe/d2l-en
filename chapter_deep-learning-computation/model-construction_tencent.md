# 图层和块
:label:`sec_model_construction`

当我们第一次引入神经网络时，我们关注的是具有单一输出的线性模型。在这里，整个模型只由一个神经元组成。注意，单个神经元(I)接受某一组输入；(Ii)生成相应的标量输出；以及(Iii)具有一组可以更新以优化某些感兴趣的目标函数的相关参数。然后，一旦我们开始考虑具有多个输出的网络，我们就利用矢量化算法来表征整层神经元。就像单个神经元一样，层(I)接受一组输入，(Ii)产生相应的输出，(Iii)由一组可调参数描述。当我们使用SoftMax回归时，单层本身就是模型。然而，即使我们随后引入了MLP，我们仍然可以认为该模型保留了相同的基本结构。

有趣的是，对于MLP，整个模型及其组成层都共享此结构。整个模型接受原始输入(特征)，生成输出(预测)，并拥有参数(来自所有组成层的组合参数)。同样，每个单独的层摄取输入(由前一层提供)生成输出(到后一层的输入)，并拥有一组根据从后一层向后流动的信号更新的可调参数。

虽然您可能认为神经元、层和模型为我们的业务提供了足够的抽象，但事实证明，我们经常发现谈论比单个层大但比整个模型小的组件更方便。例如，在计算机视觉中广泛流行的ResNet-152体系结构就有数百层。这些层由*层组*的重复图案组成。一次实施一层这样的网络可能会变得单调乏味。这个问题不仅仅是假设的-这样的设计模式在实践中很常见。上述Resnet架构赢得了2015年ImageNet和COCO计算机视觉竞赛的识别和检测:cite:`He.Zhang.Ren.ea.2016`，并且仍然是许多视觉任务的首选架构。层以各种重复模式排列的类似体系结构现在在其他领域普遍存在，包括自然语言处理和语音。

为了实现这些复杂的网络，我们引入了神经网络*挡路*的概念。挡路可以描述单层，可以描述由多层组成的组件，也可以描述整个模型本身！使用挡路抽象的一个好处是它们可以组合成更大的工件，通常是递归的。这一点在:numref:`fig_blocks`中进行了说明。通过定义代码来按需生成任意复杂度的块，我们可以编写出令人惊讶的紧凑代码，同时仍然可以实现复杂的神经网络。

![Multiple layers are combined into blocks, forming repeating patterns of larger models.](../img/blocks.svg)
:label:`fig_blocks`

从编程的角度来看，挡路用*类*来表示。它的任何子类都必须定义将其输入转换为输出的前向传播函数，并且必须存储任何必要的参数。请注意，有些块根本不需要任何参数。最后，为了计算梯度，挡路必须具有反向传播函数。幸运的是，由于自动微分(:numref:`sec_autograd`引入)提供的一些幕后魔力，在定义我们自己的挡路时，我们只需要担心参数和前向传播函数。

首先，我们回顾用于实现MLP的代码(:numref:`sec_mlp_concise`)。下面的代码生成一个具有256个单元的全连接隐藏层和REU激活的网络，然后是一个具有10个单元的全连接输出层(无激活功能)。

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
在本例中，我们通过实例化一个`nn.Sequential`，将返回的对象赋给`net`变量来构建我们的模型。接下来，我们重复调用它的`add`函数，按照应该执行的顺序附加层。简而言之，`nn.Sequential`定义了一种特殊的`Block`，即用胶子表示挡路的类。它维护成分`Block`s的有序列表。`add`功能简单地便于将每个连续的`Block`添加到列表。请注意，每一层都是`Dense`类的实例，而该类本身就是`Block`的子类。前向传播(`forward`)函数也非常简单：它将列表中的每个`Block`链接在一起，将每个的输出作为输入传递给下一个。注意，到目前为止，我们一直通过构造`net(X)`调用我们的模型以获得它们的输出。这实际上只是`net.forward(X)`的简写，这是通过`Block`类的`__call__`函数实现的一个巧妙的Python技巧。
:end_tab:

:begin_tab:`pytorch`
在本例中，我们通过实例化一个`nn.Sequential`来构建我们的模型，这些层按照它们应该被执行的顺序作为参数传递。简而言之，`nn.Sequential`定义了一种特殊的`Module`，即在PyTorch中表示挡路的类。它维护成分`Module`的有序列表。请注意，两个完全连接的层中的每一个都是`Linear`类的实例，而该类本身就是`Module`的子类。前向传播(`forward`)函数也非常简单：它将列表中的每个挡路链接在一起，将每个挡路的输出作为输入传递给下一个。注意，到目前为止，我们一直通过构造`net(X)`调用我们的模型以获得它们的输出。这实际上只是`net.forward(X)`的简写，这是通过挡路类的`__call__`函数实现的一个巧妙的Python技巧。
:end_tab:

:begin_tab:`tensorflow`
在本例中，我们通过实例化一个`keras.models.Sequential`来构建我们的模型，这些层按照它们应该被执行的顺序作为参数传递。简而言之，`Sequential`定义了一种特殊的`keras.Model`，即在凯拉斯呈现挡路的类。它维护成分`Model`的有序列表。请注意，两个完全连接的层中的每一个都是`Dense`类的实例，而该类本身就是`Model`的子类。前向传播(`call`)函数也非常简单：它将列表中的每个挡路链接在一起，将每个挡路的输出作为输入传递给下一个。注意，到目前为止，我们一直通过构造`net(X)`调用我们的模型以获得它们的输出。这实际上只是`net.call(X)`的简写，这是通过挡路类的`__call__`函数实现的一个巧妙的Python技巧。
:end_tab:

## 自定义挡路

要想直观地了解挡路是如何工作的，最简单的方法可能就是自己实现一个。在实施我们自己的自定义挡路之前，我们先简要总结一下每个挡路必须提供的基本功能：

1. 将输入数据作为参数接收到其前向传播函数。
1. 通过使前向传播函数返回值来生成输出。请注意，输出可能具有与输入不同的形状。例如，上面模型中的第一个完全连接层接受任意维度的输入，但返回维度256的输出。
1. 计算其输出相对于其输入的梯度，该梯度可通过其反向传播功能访问。通常这是自动发生的。
1. 存储并提供对执行前向传播计算所需的那些参数的访问。
1. 根据需要初始化模型参数。

在下面的代码片段中，我们从头开始编写一个挡路，对应于一个具有256万个隐藏单元的隐藏层和一个10维输出层。请注意，下面的`MLP`个类继承了表示挡路的类。我们将严重依赖父类的函数，只提供我们自己的构造函数(Python语言中的`__init__`函数)和前向传播函数。

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

让我们首先关注前向传播函数。请注意，它将`X`作为输入，计算应用了激活函数的隐藏表示，并输出其日志。在此`MLP`实现中，两个层都是实例变量。要了解这为什么是合理的，可以想象实例化两个MLP(`net1`和`net2`)，并根据不同的数据对它们进行训练。当然，我们希望它们代表两个不同的学习模型。

我们在构造函数中实例化MLP的层，然后在每次调用转发传播函数时调用这些层。请注意几个关键细节。首先，我们定制的`__init__`函数通过`__init__`调用父类的`super().__init__()`函数，省去了重复适用于大多数块的样板代码的痛苦。然后，我们实例化两个完全连接的层，将它们分配给`self.hidden`和`self.out`。请注意，除非我们实现一个新操作符，否则我们不需要担心反向传播函数或参数初始化。系统将自动生成这些函数。让我们试试看吧。

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

挡路抽象的一个关键优点是它的多功能性。我们可以子类化挡路来创建层(如全连通层类)、整个模型(如上面的`MLP`类)或各种中等复杂度的组件。我们将在接下来的章节中利用这种多功能性，例如在处理卷积神经网络时。

## “挡路”系列丛书

现在我们可以仔细看看`Sequential`级是如何工作的。回想一下，`Sequential`旨在将其他数据块以菊花链形式链接在一起。要构建我们自己的简化`MySequential`，我们只需要定义两个关键函数：
1. 将块逐个附加到列表中的函数。
2. 一种前向传播函数，用于以与附加输入的顺序相同的顺序，通过区块链传递输入。

下面的`MySequential`类提供与默认`Sequential`类相同的功能。

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
`add`函数将单个挡路添加到有序词典`_children`。您可能想知道为什么每个胶子`Block`都有一个`_children`属性，为什么我们要使用它，而不是自己定义一个Python列表。简而言之，`_children`的主要优势是，在我们挡路的参数初始化期间，胶子知道要在`_children`字典中查找参数也需要初始化的子块。
:end_tab:

:begin_tab:`pytorch`
在`__init__`方法中，我们将每个挡路逐个添加到有序词典`_modules`中。您可能想知道为什么每个`Module`都有一个`_modules`属性，为什么我们要使用它，而不是自己定义一个Python列表。简而言之，`_modules`的主要优势在于，在我们挡路的参数初始化期间，系统知道要在`_modules`字典中查找参数也需要初始化的子块。
:end_tab:

当我们的`MySequential`的前向传播函数被调用时，每个添加的挡路都会按照它们被添加的顺序执行。我们现在可以使用`MySequential`类重新实现一个mlp。

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

请注意，`MySequential`的这种用法与我们之前为`Sequential`类编写的代码相同(如:numref:`sec_mlp_concise`中所述)。

## 在前向传播函数中执行代码

`Sequential`类使模型构建变得容易，允许我们组装新的体系结构，而不必定义我们自己的类。但是，并不是所有的架构都是简单的菊花链。当需要更大的灵活性时，我们会希望定义我们自己的块。例如，我们可能希望在前向传播函数中执行Python的控制流。此外，我们可能希望执行任意的数学运算，而不是简单地依赖预定义的神经网络层。

您可能已经注意到，到目前为止，我们网络中的所有操作都是根据我们网络的激活及其参数进行操作的。但是，有时我们可能希望合并既不是先前层的结果也不是可更新参数的术语。我们称这些为“常数参数”。例如，假设我们想要一个计算函数$f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$的层，其中$\mathbf{x}$是输入，$\mathbf{w}$是我们的参数，$c$是在优化期间不更新的某个指定常量。因此，我们实现一个`FixedHiddenMLP`类，如下所示。

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

在这个`FixedHiddenMLP`模型中，我们实现了一个隐藏层，其权重(`self.rand_weight`)在实例化时被随机初始化，并且此后是恒定的。该权重不是模型参数，因此不会通过反向传播进行更新。然后，网络将这个“固定”层的输出通过一个完全连接的层。

请注意，在返回输出之前，我们的模型做了一些不寻常的事情。我们运行了WHILE循环，在其$L_1$范数大于$1$的条件下进行测试，并将输出向量除以$2$，直到它满足条件。最后，我们返回`X`中条目的总和。据我们所知，没有标准的神经网络执行这种操作。请注意，此特定操作可能在任何实际任务中都没有用处。我们的重点只是向您展示如何将任意代码集成到您的神经网络计算流程中。

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

我们可以混合搭配各种方式把积木组装在一起。在下面的示例中，我们以一些创造性的方式嵌套块。

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

## 编译

:begin_tab:`mxnet, tensorflow`
狂热的读者可能会开始担心其中一些操作的效率。毕竟，我们在一个应该是高性能的深度学习库中进行了大量的字典查找、代码执行和许多其他Python式的事情。巨蟒[global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock)的问题是众所周知的。在深度学习环境中，我们担心速度极快的GPU可能要等到微不足道的CPU运行Python代码后才能运行另一个作业。提高Python速度的最好方法是完全避免使用它。
:end_tab:

:begin_tab:`mxnet`
胶子做到这一点的一种方式是允许
*杂交*，这将在后面描述。
在这里，Python解释器在第一次调用挡路时执行它。GLUON运行时记录正在发生的事情，并在下一次运行时缩短对Python的调用。在某些情况下，这可以大大加快速度，但是当控制流(如上所述)在不同通道上引导不同的分支通过网络时，需要注意这一点。我们建议感兴趣的读者在读完本章后查看杂交部分(:numref:`sec_hybridize`)来了解编译。
:end_tab:

## 摘要

* 图层是块。
* 许多层可以组成一个挡路。
* 许多区块可以组成挡路。
* 挡路可以包含代码。
* 块负责大量内务工作，包括参数初始化和反向传播。
* 层和块的顺序连接由`Sequential`挡路处理。

## 练习

1. 如果您将`MySequential`更改为在Python列表中存储块，会出现什么问题？
1. 实现一个挡路，它以两个块作为参数，例如`net1`和`net2`，并在正向传播中返回两个网络的串联输出。这也叫平行挡路。
1. 假设您要串联同一网络的多个实例。实现一个工厂函数，该函数可以生成同一挡路的多个实例，并在此基础上构建更大的网络。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/54)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/55)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/264)
:end_tab:
