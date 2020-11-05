# SoftMax回归的简明实现
:label:`sec_softmax_concise`

正如深度学习框架的高级API使得在:numref:`sec_linear_concise`中实现线性回归变得容易得多一样，我们会发现实现分类模型也同样(或者可能更方便)。让我们坚持使用Fashion-MNIST数据集，并将批大小保持在:numref:`sec_softmax_scratch`中的256%。

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

## 正在初始化模型参数

如:numref:`sec_softmax`中所述，SoftMAX回归的输出层是完全连接层。因此，要实现我们的模型，我们只需要在`Sequential`中添加一个具有10个输出的完全连接层。同样，在这里，`Sequential`并不是真正必要的，但我们不妨养成习惯，因为在实现深度模型时，它将无处不在。同样，我们用零均值和标准差0.01随机初始化权重。

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

## 重新实施SoftMAX
:label:`subsec_softmax-implementation-revisited`

在前一个示例:numref:`sec_softmax_scratch`中，我们计算了模型的输出，然后通过交叉熵损失运行此输出。从数学上讲，这是完全合理的做法。然而，从计算的角度来看，幂运算可能是数值稳定性问题的一个来源。

回想一下，SOFTMAX函数计算$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$，其中$\hat y_j$是预测概率分布$\hat{\mathbf{y}}$的$j^\mathrm{th}$个元素，$o_j$是Logits $\mathbf{o}$的$j^\mathrm{th}$个元素。如果$o_k$中的一些非常大(即非常正)，则$\exp(o_k)$可能大于某些数据类型的最大值(即*溢出*)。这将使分母(和/或分子)为`inf`(无穷大)，我们最终会遇到0、`inf`或`nan`(不是数字)表示$\hat y_j$。在这些情况下，我们得不到明确定义的交叉熵返回值。

解决此问题的一个技巧是，在进行Softmax计算之前，首先从全部$\max(o_k)$中减去$o_k$。您可以验证每个$o_k$的这种恒定因子移位不会更改Softmax的返回值。在减法和归一化步骤之后，可能有一些$o_j$具有较大的负值，因此相应的$\exp(o_j)$将采用接近于零的值。由于精度有限(即*下溢*)，这些值可能四舍五入为零，$\hat y_j$为零，`-inf`为$\log(\hat y_j)$。在反向传播的道路上再走几步，我们可能会发现自己面对的是令人恐惧的`nan`个结果的屏幕。

幸运的是，即使我们在计算指数函数，我们最终也要取它们的对数(在计算交叉熵损失时)，这一事实拯救了我们。通过将这两个运算符Softmax和交叉熵结合在一起，我们可以避免在反向传播过程中可能会困扰我们的数值稳定性问题。如下面的公式所示，我们避免了计算$\exp(o_j)$，而可以直接使用$o_j$，因为在$\log(\exp(\cdot))$中取消了。

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j)}{\sum_k \exp(o_k)}\right) \\
& = \log{(\exp(o_j))}-\log{\left( \sum_k \exp(o_k) \right)} \\
& = o_j -\log{\left( \sum_k \exp(o_k) \right)}.
\end{aligned}
$$

我们希望保留传统的Softmax函数，以防我们想要通过我们的模型评估输出概率。但是，我们不会将Softmax概率传递到我们的新损失函数中，而是只传递logit，并在交叉熵损失函数内同时计算Softmax及其日志，该函数执行像[“LogSumExp trick”](https://en.wikipedia.org/wiki/LogSumExp).]这样的智能操作

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

这里，我们使用学习率为0.1的小批量随机梯度下降作为优化算法。请注意，这与我们在线性回归示例中应用的情况相同，它说明了优化器的一般适用性。

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

接下来，我们调用:numref:`sec_softmax_scratch`中定义的训练函数来训练模型。

```{.python .input}
#@tab all
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

与以前一样，该算法收敛到一种解决方案，该解决方案实现了相当高的精度，尽管这一次使用的代码行比以前少。

## 摘要

* 使用高级API，我们可以更简洁地实现Softmax回归。
* 从计算的角度来看，实施Softmax回归是错综复杂的。请注意，在许多情况下，深度学习框架在这些最广为人知的技巧之外采取了额外的预防措施，以确保数值稳定性，从而使我们避免了在实践中尝试从头开始编写所有模型时会遇到的更多陷阱。

## 练习

1. 尝试调整超参数，如批处理大小、历元数和学习率，以查看结果。
1. 增加训练的纪元数。为什么一段时间后测试的准确度可能会下降呢？我们怎么才能解决这个问题呢？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/260)
:end_tab:
