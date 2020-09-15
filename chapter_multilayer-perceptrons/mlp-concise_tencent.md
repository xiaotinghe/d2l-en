# 多层感知器的简明实现
:label:`sec_mlp_concise`

正如您可能预期的那样，通过依赖高级API，我们可以更简洁地实现MLP。

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

## 型号

与我们的Softmax回归实现(:numref:`sec_softmax_concise`)的简明实现相比，唯一的区别是我们添加了
*两个*完全连接的层
(之前我们增加了*一*)。第一个是我们的隐藏层，它包含256个隐藏单元，并应用了RELU激活功能。第二个是我们的输出层。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)])
```

训练循环与我们实施Softmax回归时完全相同。这种模块性使我们能够将有关模型体系结构的事项从正交考虑中分离出来。

```{.python .input}
batch_size, lr, num_epochs = 256, 0.1, 10
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
```

```{.python .input}
#@tab pytorch
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
```

```{.python .input}
#@tab tensorflow
batch_size, lr, num_epochs = 256, 0.1, 10
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
```

```{.python .input}
#@tab all
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 摘要

* 使用高级API，我们可以更简洁地实现MLP。
* 对于相同的分类问题，MLP的实现与Softmax回归的实现相同，只是增加了具有激活函数的隐含层。

## 练习

1. 尝试添加不同数量的隐藏层(您也可以修改学习速率)。哪种设置效果最好？
1. 尝试不同的激活功能。哪一个效果最好？
1. 尝试不同的初始化权重方案。什么方法最有效？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/94)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/95)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/262)
:end_tab:
