# Softmax回归的简明实现
:label:`sec_softmax_concise`

正如深度学习框架的高级api使得在:numref:`sec_linear_concise`中实现线性回归更加容易，我们将发现实现分类模型同样（或者可能更方便）方便。让我们继续使用时尚MNIST数据集，并保持批量大小为256，如:numref:`sec_softmax_scratch`所示。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 初始化模型参数

如:numref:`sec_softmax`所述，softmax回归的输出层是完全连接的层。因此，为了实现我们的模型，我们只需要在`Sequential`上添加一个具有10个输出的完全连接层。同样，在这里，`Sequential`并不是真正必要的，但是我们最好养成这个习惯，因为它在实现深层模型时无处不在。同样，我们随机初始化权重，平均值为零，标准偏差为0.01。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
# PyTorch does not implicitly reshape the inputs. Thus we define a layer to
# reshape the inputs in our network
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1,784)

net = nn.Sequential(Reshape(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
net.add(tf.keras.layers.Dense(10, kernel_initializer=weight_initializer))
```

## 重新审视Softmax实施
:label:`subsec_softmax-implementation-revisited`

在前面的:numref:`sec_softmax_scratch`示例中，我们计算了模型的输出，然后通过交叉熵损失运行这个输出。从数学上讲，这是一件非常合理的事情。然而，从计算的角度来看，指数可能是数值稳定性问题的一个来源。

回想一下，softmax函数计算$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$，其中$\hat y_j$是预测概率分布$\hat{\mathbf{y}}$的$j^\mathrm{th}$元素，$o_j$是logits $\mathbf{o}$的$j^\mathrm{th}$元素。如果$o_k$中的一些非常大（即，非常正），那么$\exp(o_k)$可能比某些数据类型的最大值（即*溢出*）大。这将使分母（和/或分子）`inf`（无穷大），我们最终遇到`inf`或`nan`（不是数字）。在这些情况下，我们不能得到定义良好的交叉熵返回值。

解决这个问题的一个技巧是在继续进行softmax计算之前，首先从$o_k$中减去$\max(o_k)$。您可以验证每个$o_k$按常数因子的移位不会改变softmax的返回值。在减法和归一化步骤之后，可能一些$o_j$具有较大的负值，因此相应的$\exp(o_j)$将取接近于零的值。由于精度有限（即*下溢*），这些值可能四舍五入为零，使$\hat y_j$为零，而$\log(\hat y_j)$为`-inf`。在反向传播的道路上走几步，我们可能会发现自己面临着一个可怕的`nan`结果的屏幕。

幸运的是，即使我们在计算指数函数，我们最终还是打算取它们的对数（当计算交叉熵损失时）。通过将这两个算子softmax和交叉熵结合在一起，我们可以避免在反向传播过程中可能困扰我们的数值稳定性问题。如下式所示，我们避免计算$\exp(o_j)$，由于$\log(\exp(\cdot))$中的取消，可以直接使用$o_j$。

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j)}{\sum_k \exp(o_k)}\right) \\
& = \log{(\exp(o_j))}-\log{\left( \sum_k \exp(o_k) \right)} \\
& = o_j -\log{\left( \sum_k \exp(o_k) \right)}.
\end{aligned}
$$

我们希望保留传统的softmax函数，以防我们需要通过我们的模型来评估输出概率。但是，我们没有将softmax概率传递到新的损失函数中，而是在交叉熵损失函数中传递logits并同时计算softmax及其日志，这会做一些聪明的事情，比如[“LogSumExp trick”](https://en.wikipedia.org/wiki/LogSumExp).

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss()
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

## 优化算法

在这里，我们使用学习率为0.1的小批量随机梯度下降作为优化算法。注意，这与我们在线性回归示例中应用的相同，它说明了优化器的一般适用性。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=.1)
```

## 培训

接下来我们调用:numref:`sec_softmax_scratch`中定义的训练函数来训练模型。

```{.python .input}
#@tab all
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

和以前一样，这个算法收敛到一个可以达到相当高精度的解决方案，尽管这次的代码行比以前少了。

## 摘要

* 使用高级api，我们可以更简洁地实现softmax回归。
* 从计算的角度来看，实现softmax回归具有复杂性。请注意，在许多情况下，深度学习框架在这些最著名的技巧之外采取了额外的预防措施，以确保数值的稳定性，从而使我们避免了在实践中从零开始编写所有模型时可能遇到的更多陷阱。

## 练习

1. 尝试调整超参数，例如批处理大小、时间段数和学习速率，以查看结果。
1. 增加历代训练的次数。为什么一段时间后测试精度会下降？我们怎么能解决这个问题？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/260)
:end_tab:
