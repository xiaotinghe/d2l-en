# 自定义图层

深度学习成功背后的一个因素是，可以用创造性的方式组合广泛的层，以设计适合于各种任务的体系结构。例如，研究人员发明了专门用于处理图像、文本、循环顺序数据和执行动态编程的层。迟早，你会遇到或发明一个在深度学习框架中还不存在的层。在这些情况下，您必须构建自定义层。在本节中，我们将向您展示如何操作。

## 不带参数的图层

首先，我们构造一个自己没有任何参数的自定义层。如果你还记得我们在:numref:`sec_model_construction`对挡路的介绍，这应该看起来很眼熟。下面的`CenteredLayer`个类只需从其输入中减去平均值。要构建它，我们只需继承基础层类并实现前向传播功能。

```{.python .input}
from mxnet import gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
import torch.nn.functional as F

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)
```

让我们通过向其提供一些数据来验证我们的层是否按预期工作。

```{.python .input}
layer = CenteredLayer()
layer(np.array([1, 2, 3, 4, 5]))
```

```{.python .input}
#@tab pytorch
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
```

```{.python .input}
#@tab tensorflow
layer = CenteredLayer()
layer(tf.constant([1, 2, 3, 4, 5]))
```

现在，我们可以将层作为组件合并到构建更复杂的模型中。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
```

```{.python .input}
#@tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

作为额外的健全性检查，我们可以通过网络发送随机数据，并检查平均值实际上是否为0。因为我们处理的是浮点数，所以由于量化，我们可能仍然会看到一个非常小的非零数。

```{.python .input}
Y = net(np.random.uniform(size=(4, 8)))
Y.mean()
```

```{.python .input}
#@tab pytorch
Y = net(torch.rand(4, 8))
Y.mean()
```

```{.python .input}
#@tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

## 带参数的图层

现在我们知道了如何定义简单的层，让我们继续使用可以通过培训调整的参数来定义层。我们可以使用内置函数创建参数，这些参数提供一些基本的内务管理功能。具体地说，它们管理访问、初始化、共享、保存和加载模型参数。这样，除了其他好处外，我们将不需要为每个自定义层编写自定义序列化例程。

现在，让我们实现我们自己版本的完全连接层。回想一下，该层需要两个参数，一个用于表示权重，另一个用于偏移。在此实现中，我们默认烘焙RELU激活。这一层需要输入参数：`in_units`和`units`，分别表示输入和输出的数量。

```{.python .input}
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

```{.python .input}
#@tab pytorch
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

```{.python .input}
#@tab tensorflow
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        return tf.matmul(X, self.weight) + self.bias
```

接下来，我们实例化`MyDense`类并访问其模型参数。

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
#@tab pytorch
dense = MyLinear(5, 3)
dense.weight
```

```{.python .input}
#@tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

我们可以使用自定义层直接执行前向传播计算。

```{.python .input}
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
#@tab pytorch
dense(torch.rand(2, 5))
```

```{.python .input}
#@tab tensorflow
dense(tf.random.uniform((2, 5)))
```

我们还可以使用自定义层构建模型。一旦我们拥有了它，我们就可以像使用内置的完全连接层一样使用它。

```{.python .input}
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

## 摘要

* 我们可以通过Basic Layer类设计自定义层。这允许我们定义灵活的新层，其行为不同于库中的任何现有层。
* 一旦定义，自定义层就可以在任意上下文和体系结构中调用。
* 层可以有本地参数，这些参数可以通过内置函数创建。

## 练习

1. 设计一个接受输入并计算张量缩减的层，即，它返回$y_k = \sum_{i, j} W_{ijk} x_i x_j$。
1. 设计一个返回数据傅立叶系数前半部分的层。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/58)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/59)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/279)
:end_tab:
