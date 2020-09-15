# 延迟初始化
:label:`sec_deferred_init`

到目前为止，似乎我们在建立人际网络时表现得很马虎。具体来说，我们做了以下不直观的事情，这些事情似乎不应该奏效：

* 我们定义了网络架构，但没有指定输入维度。
* 我们添加层时没有指定前一层的输出维度。
* 我们甚至在提供足够的信息来确定我们的模型应该包含多少参数之前“初始化”了这些参数。

您可能会惊讶于我们的代码运行。毕竟，深度学习框架无法判断网络的输入维度是什么。这里的诀窍是框架*推迟初始化*，等到我们第一次通过模型传递数据时，才能动态地推断出每个层的大小。

随后，当使用卷积神经网络时，由于输入维度（即图像的分辨率）将影响每个后续层的维数，因此该技术将变得更加方便。因此，在编写代码时无需知道维度是什么而设置参数的能力可以大大简化指定和随后修改模型的任务。接下来，我们将更深入地研究初始化机制。

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

此时，网络不可能知道输入层权重的维数，因为输入维数仍然未知。因此，框架尚未初始化任何参数。我们通过尝试访问以下参数进行确认。

```{.python .input}
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
#@tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
请注意，当参数对象存在时，每个层的输入维度列为-1。MXNet使用特殊值-1表示参数维度仍然未知。此时，尝试访问`net[0].weight.data()`将触发运行时错误，指出必须先初始化网络，然后才能访问参数。现在让我们看看当我们试图通过`initialize`函数初始化参数时会发生什么。
:end_tab:

:begin_tab:`tensorflow`
请注意，每个层对象都存在，但权重为空。使用`net.get_weights()`将抛出一个错误，因为权重尚未初始化。
:end_tab:

```{.python .input}
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
如我们所见，一切都没有改变。当输入维度未知时，调用initialize不会真正初始化参数。相反，这个调用注册到我们希望初始化参数的MXNet（可选，根据哪个分布）。
:end_tab:

接下来让我们通过网络传递数据，使框架最终初始化参数。

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

一旦我们知道输入维数20，框架就可以通过插入值20来识别第一层权重矩阵的形状。识别出第一层的形状后，框架进入第二层，依此类推，直到所有形状都已知为止。注意，在这种情况下，只有第一层需要延迟初始化，但是框架按顺序初始化。一旦知道了所有参数形状，框架就可以最终初始化参数。

## 摘要

* 延迟初始化可以很方便，允许框架自动推断参数形状，使修改体系结构变得容易，并消除了一个常见的错误源。
* 我们可以通过模型传递数据，使框架最终初始化参数。

## 练习

1. 如果将输入尺寸指定给第一个图层，而不是指定给后续图层，会发生什么情况？你能立即初始化吗？
1. 如果指定不匹配的维度，会发生什么情况？
1. 如果你有不同维度的输入，你需要做什么？提示：看看参数绑定。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/281)
:end_tab:
