# 文件I/O

到目前为止，我们讨论了如何处理数据以及如何构建、培训和测试深度学习模型。然而，在某个时刻，我们希望对所学的模型足够满意，我们希望保存结果以备将来在各种环境中使用（甚至可能在部署中进行预测）。此外，当运行一个长时间的培训过程时，最佳实践是定期保存中间结果（检查点），以确保在服务器电源线被绊倒时不会损失几天的计算量。因此，现在是时候学习如何加载和存储单独的权重向量和整个模型。本节讨论这两个问题。

## 加载和保存张量

对于单个张量，我们可以直接调用`load`和`save`函数分别读写它们。这两个函数都要求我们提供一个名称，`save`要求将要保存的变量作为输入。

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
import numpy as np

x = tf.range(4)
np.save("x-file.npy", x)
```

我们现在可以将存储文件中的数据读回内存。

```{.python .input}
x2 = npx.load('x-file')
x2
```

```{.python .input}
#@tab pytorch
x2 = torch.load("x-file")
x2
```

```{.python .input}
#@tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

我们可以存储一个张量列表，然后把它们读回内存。

```{.python .input}
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
#@tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
#@tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

我们甚至可以编写和阅读从字符串映射到张量的字典。当我们要读取或写入模型中的所有权重时，这很方便。

```{.python .input}
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
#@tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
#@tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

## 加载和保存模型参数

保存单个权重向量（或其他张量）是有用的，但是如果我们想保存（并在以后加载）整个模型，则会变得非常乏味。毕竟，我们可能有数百个参数组散布在各处。因此，深度学习框架提供了内置功能来加载和保存整个网络。需要注意的一个重要细节是，这将保存模型*参数*而不是整个模型。例如，如果我们有一个3层MLP，我们需要单独指定体系结构。原因是模型本身可以包含任意代码，因此它们不能自然地序列化。因此，为了恢复模型，我们需要用代码生成体系结构，然后从磁盘加载参数。让我们从我们熟悉的MLP开始。

```{.python .input}
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

接下来，我们将模型的参数存储为一个名为“mlp参数".

```{.python .input}
net.save_parameters('mlp.params')
```

```{.python .input}
#@tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
#@tab tensorflow
net.save_weights('mlp.params')
```

为了恢复模型，我们实例化了原始MLP模型的一个克隆。我们没有随机初始化模型参数，而是直接读取文件中存储的参数。

```{.python .input}
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
#@tab pytorch
clone = MLP()
clone.load_state_dict(torch.load("mlp.params"))
clone.eval()
```

```{.python .input}
#@tab tensorflow
clone = MLP()
clone.load_weights("mlp.params")
```

由于两个实例具有相同的模型参数，相同输入`X`的计算结果应该相同。让我们来验证一下。

```{.python .input}
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
#@tab pytorch
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
#@tab tensorflow
Y_clone = clone(X)
Y_clone == Y
```

## 摘要

* `save`和`load`函数可用于执行张量对象的文件I/O。
* 我们可以通过参数字典保存和加载网络的全部参数集。
* 保存架构必须在代码中完成，而不是在参数中。

## 练习

1. 即使不需要将经过训练的模型部署到不同的设备上，存储模型参数的实际好处是什么？
1. 假设我们只想重用网络的一部分，以将其合并到不同体系结构的网络中。你会如何使用，比如说在一个新的网络中使用前两层网络？
1. 如何保存网络架构和参数？你会对架构施加什么限制？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/327)
:end_tab:
