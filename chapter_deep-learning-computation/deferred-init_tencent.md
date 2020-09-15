# 延迟初始化
:label:`sec_deferred_init`

到目前为止，似乎我们在建立网络方面的草率行为没有受到影响。具体地说，我们做了以下不直观的事情，它们可能看起来不应该起作用：

* 我们定义了网络体系结构，但没有指定输入维度。
* 我们在没有指定上一层的输出尺寸的情况下添加层。
* 我们甚至在提供足够的信息来确定我们的模型应该包含多少参数之前“初始化”了这些参数。

您可能会对我们的代码运行感到惊讶。毕竟，深度学习框架无法判断网络的输入维度是多少。这里的诀窍是，框架*推迟初始化*，等到我们第一次通过模型传递数据时，才推断飞翔上每一层的大小。

稍后，当与卷积神经网络一起工作时，该技术将变得更加方便，因为输入维数(即，图像的分辨率)将影响每个后续层的维数。因此，在编写代码时无需知道维度是什么而设置参数的能力可以极大地简化指定和随后修改模型的任务。接下来，我们将更深入地了解初始化机制。

## 实例化网络

首先，让我们实例化一个MLP。

```{.python .input}
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    return net

net = get_net()
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

在这一点上，网络不可能知道输入层的权重的维度，因为输入维度仍然未知。因此，框架尚未初始化任何参数。我们通过尝试访问以下参数进行确认。

```{.python .input}
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
#@tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
请注意，当参数对象存在时，每层的输入尺寸列为-1。MXNet使用特定值-1表示参数尺寸仍然未知。在这一点上，访问`net[0].weight.data()`的尝试将触发运行时错误，声明在可以访问参数之前必须初始化网络。现在让我们看看当我们试图通过`initialize`函数初始化参数时会发生什么。
:end_tab:

:begin_tab:`tensorflow`
请注意，每个层对象都存在，但权重为空。使用`net.get_weights()`会引发错误，因为权重尚未初始化。
:end_tab:

```{.python .input}
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
正如我们所见，一切都没有改变。当输入尺寸未知时，调用Initialize不会真正初始化参数。相反，此调用注册到MXNet，这是我们希望的(并且可选地，根据哪个分布)来初始化参数。
:end_tab:

接下来，让我们通过网络传递数据，使框架最终初始化参数。

```{.python .input}
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

一旦我们知道输入维数20，框架就可以通过插入值20来识别第一层的权重矩阵的形状。在识别了第一层的形状之后，框架继续进行到第二层，依此类推，通过计算图直到知道所有的形状。请注意，在这种情况下，只有第一层需要延迟初始化，但框架会按顺序初始化。一旦所有参数形状都已知，框架最终可以初始化参数。

## 摘要

* 延迟初始化可能很方便，允许框架自动推断参数形状，使修改体系结构变得容易，并消除了一个常见的错误来源。
* 我们可以通过模型传递数据，使框架最终初始化参数。

## 练习

1. 如果将输入尺寸指定给第一层而不指定给后续层，会发生什么情况？您可以立即进行初始化吗？
1. 如果指定不匹配的尺寸，会发生什么情况？
1. 如果您有不同维度的输入，您需要做什么？提示：请看参数绑定。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/281)
:end_tab:
