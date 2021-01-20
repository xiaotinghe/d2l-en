# 参数管理

一旦我们选择了一个架构并设置了超参数，我们就进入了训练循环，我们的目标是找到使损失函数最小化的参数值。在训练之后，我们需要这些参数来进行未来的预测。此外，我们有时会希望提取参数，以便在其他上下文中重用它们，将我们的模型保存到磁盘以便可以在其他软件中执行，或者用于检查，以期获得科学的理解。

大多数时候，我们将能够忽略参数如何声明和操作的基本细节，依靠深度学习框架来完成繁重的工作。然而，当我们离开带有标准层的层叠架构时，我们有时需要深入到声明和操作参数的麻烦中。在本节中，我们将介绍以下内容：

* 访问用于调试、诊断和可视化的参数。
* 参数初始化。
* 在不同的模型构件之间共享参数。

我们首先关注一个隐藏层的MLP。

```{.python .input}
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X)  # Forward computation
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
import numpy as np

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X)
```

## 参数访问

让我们从如何从您已经知道的模型中访问参数开始。当一个模型通过`Sequential`类定义时，我们可以首先通过索引到模型中来访问任何层，就像它是一个列表一样。每个层的参数都可以方便地定位在其属性中。我们可以检查第二个完全连通层的参数，如下所示。

```{.python .input}
print(net[1].params)
```

```{.python .input}
#@tab pytorch
print(net[2].state_dict())
```

```{.python .input}
#@tab tensorflow
print(net.layers[2].weights)
```

输出告诉我们一些重要的事情。首先，这个完全连接的层包含两个参数，分别对应于该层的权重和偏移。两者都存储为单精度浮点（float32）。请注意，参数的名称允许我们唯一地标识每个层的参数，即使在包含数百个层的网络中也是如此。

### 目标参数

请注意，每个参数都表示为参数类的一个实例。要对参数做任何有用的事情，我们首先需要访问底层的数值。有几种方法可以做到这一点。有些比较简单，有些则更一般。下面的代码从第二个神经网络层提取偏差，后者返回一个参数类实例，并进一步访问该参数的值。

```{.python .input}
print(type(net[1].bias))
print(net[1].bias)
print(net[1].bias.data())
```

```{.python .input}
#@tab pytorch
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

```{.python .input}
#@tab tensorflow
print(type(net.layers[2].weights[1]))
print(net.layers[2].weights[1])
print(tf.convert_to_tensor(net.layers[2].weights[1]))
```

:begin_tab:`mxnet,pytorch`
参数是复杂的对象，包含值、渐变和其他信息。这就是为什么我们需要显式地请求值。

除了值之外，每个参数还允许我们访问渐变。因为我们还没有为这个网络调用反向传播，所以它处于初始状态。
:end_tab:

```{.python .input}
net[1].weight.grad()
```

```{.python .input}
#@tab pytorch
net[2].weight.grad == None
```

### 一次所有参数

当我们需要对所有参数执行操作时，一个接一个地访问它们可能会变得很乏味。当我们处理更复杂的块（例如，嵌套块）时，这种情况会变得特别棘手，因为我们需要递归整个树来提取每个子块的参数。下面我们将演示如何访问第一个完全连接层的参数与访问所有层的参数。

```{.python .input}
print(net[0].collect_params())
print(net.collect_params())
```

```{.python .input}
#@tab pytorch
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

```{.python .input}
#@tab tensorflow
print(net.layers[1].weights)
print(net.get_weights())
```

这为我们提供了另一种访问网络参数的方法，如下所示。

```{.python .input}
net.collect_params()['dense1_bias'].data()
```

```{.python .input}
#@tab pytorch
net.state_dict()['2.bias'].data
```

```{.python .input}
#@tab tensorflow
net.get_weights()[1]
```

### 从嵌套块收集参数

让我们看看，如果我们在彼此内部嵌套多个块，参数命名约定是如何工作的。为此，我们首先定义一个生成块的函数（可以说是块工厂），然后在更大的块中组合这些块。

```{.python .input}
def block1():
    net = nn.Sequential()
    net.add(nn.Dense(32, activation='relu'))
    net.add(nn.Dense(16, activation='relu'))
    return net

def block2():
    net = nn.Sequential()
    for _ in range(4):
        # Nested here
        net.add(block1())
    return net

rgnet = nn.Sequential()
rgnet.add(block2())
rgnet.add(nn.Dense(10))
rgnet.initialize()
rgnet(X)
```

```{.python .input}
#@tab pytorch
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # Nested here
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```

```{.python .input}
#@tab tensorflow
def block1(name):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation=tf.nn.relu)],
        name=name)

def block2():
    net = tf.keras.Sequential()
    for i in range(4):
        # Nested here
        net.add(block1(name=f'block-{i}'))
    return net

rgnet = tf.keras.Sequential()
rgnet.add(block2())
rgnet.add(tf.keras.layers.Dense(1))
rgnet(X)
```

现在我们已经设计了网络，让我们看看它是如何组织的。

```{.python .input}
print(rgnet.collect_params)
print(rgnet.collect_params())
```

```{.python .input}
#@tab pytorch
print(rgnet)
```

```{.python .input}
#@tab tensorflow
print(rgnet.summary())
```

由于层是分层嵌套的，所以我们也可以访问它们，就像通过嵌套列表建立索引一样。例如，我们可以访问第一个主块，其中的第二个子块，以及第一层的偏移，如下所示。

```{.python .input}
rgnet[0][1][0].bias.data()
```

```{.python .input}
#@tab pytorch
rgnet[0][1][0].bias.data
```

```{.python .input}
#@tab tensorflow
rgnet.layers[0].layers[1].layers[1].weights[1]
```

## 参数初始化

既然我们知道了如何访问参数，那么让我们看看如何正确初始化它们。我们在:numref:`sec_numerical_stability`中讨论了正确初始化的必要性。深度学习框架为其层提供默认的随机初始化。然而，我们通常希望根据各种其他协议初始化权重。该框架提供了最常用的协议，还允许创建自定义初始值设定项。

:begin_tab:`mxnet`
默认情况下，MXNet通过从均匀分布中随机提取$U(-0.07, 0.07)$初始化权重参数，将偏差参数清除为零。MXNet的`init`模块提供了各种预置的初始化方法。
:end_tab:

:begin_tab:`pytorch`
默认情况下，PyTorch通过从根据输入和输出维度计算的范围中绘制来统一初始化权重和偏差矩阵。Pythorch的`nn.init`模块提供了各种预置的初始化方法。
:end_tab:

:begin_tab:`tensorflow`
默认情况下，Keras通过从根据输入和输出维度计算的范围中绘制来统一初始化权重矩阵，并且偏差参数都设置为零。TensorFlow在根模块和`keras.initializers`模块中提供了多种初始化方法。
:end_tab:

### 内置初始化

让我们首先调用内置的初始化器。下面的代码将所有权重参数初始化为标准偏差为0.01的高斯随机变量，而偏差参数被清除为零。

```{.python .input}
# Here `force_reinit` ensures that parameters are freshly initialized even if
# they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(X)
net.weights[0], net.weights[1]
```

我们还可以将所有参数初始化为给定的常量值（比如1）。

```{.python .input}
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
net.weights[0], net.weights[1]
```

我们还可以为某些块应用不同的初始值设定项。例如，下面我们用Xavier初始化器初始化第一层，并将第二层初始化为常量值42。

```{.python .input}
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input}
#@tab pytorch
def xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        torch.nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(1)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

### 自定义初始化

有时，我们需要的初始化方法不是由深度学习框架提供的。在下面的示例中，我们使用以下奇怪分布为任何权重参数$w$定义初始值设定项：

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ with probability } \frac{1}{4} \\
            0    & \text{ with probability } \frac{1}{2} \\
        U(-10, -5) & \text{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
这里我们定义了`Initializer`类的一个子类。通常，我们只需要实现`_init_weight`函数，该函数接受张量参数（`data`）并为其分配所需的初始化值。
:end_tab:

:begin_tab:`pytorch`
同样，我们实现了一个`my_init`函数来应用于`net`。
:end_tab:

:begin_tab:`tensorflow`
这里我们定义了`Initializer`的一个子类，并实现了`__call__`函数，该函数返回给定形状和数据类型的期望张量。
:end_tab:

```{.python .input}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input}
#@tab pytorch
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) 
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input}
#@tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        return tf.random.uniform(shape, dtype=dtype)

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])
```

请注意，我们始终可以选择直接设置参数。

```{.python .input}
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input}
#@tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

:begin_tab:`mxnet`
高级用户注意：如果要在`autograd`范围内调整参数，则需要使用`set_data`以避免混淆自动微分机制。
:end_tab:

## 约束参数

通常，我们希望跨多个层共享参数。让我们看看如何优雅地做这件事。下面我们分配一个稠密层，然后使用它的参数来设置另一层的参数。

```{.python .input}
net = nn.Sequential()
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)

# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input}
#@tab pytorch
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
#@tab tensorflow
# tf.keras behaves a bit differently. It removes the duplicate layer
# automatically
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
# Check whether the parameters are different
print(len(net.layers) == 3)
```

:begin_tab:`mxnet,pytorch`
此示例显示第二层和第三层的参数是绑定的。它们不仅是相等的，它们是用相同的张量表示的。因此，如果我们改变其中一个参数，另一个参数也会改变。您可能会想知道，当参数绑定时，渐变会发生什么？由于模型参数包含梯度，所以在反向传播过程中，第二个隐藏层和第三个隐藏层的梯度被加在一起。
:end_tab:

## 摘要

* 我们有几种方法来访问、初始化和绑定模型参数。
* 我们可以使用自定义初始化。

## 练习

1. 使用:numref:`sec_model_construction`中定义的`FancyMLP`模型，访问各个层的参数。
1. 查看初始化模块文档以了解不同的初始化器。
1. 构造一个包含共享参数层的MLP并对其进行训练。在训练过程中，观察各层模型参数和梯度。
1. 为什么共享参数是个好主意？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/269)
:end_tab:
