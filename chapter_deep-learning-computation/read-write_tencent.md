# 文件I/O

到目前为止，我们讨论了如何处理数据以及如何构建、训练和测试深度学习模型。然而，在某一时刻，我们希望对学习到的模型足够满意，以便保存结果以供以后在各种上下文中使用(甚至可能在部署中进行预测)。此外，在运行较长的培训过程时，最佳实践是定期保存中间结果(检查点)，以确保如果我们在服务器的电源线上绊倒，我们不会损失几天的计算时间。因此，现在是学习如何加载和存储单个权重向量和整个模型的时候了。本节将解决这两个问题。

## 加载和保存张量

对于单个张量，我们可以直接调用`load`和`save`函数来分别读取和写入它们。这两个函数都要求我们提供一个名称，而`save`要求将变量作为输入保存。

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

现在，我们可以将存储文件中的数据读回内存。

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

我们可以存储张量列表，并将它们读回内存。

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

我们甚至可以编写和阅读从字符串映射到张量的字典。当我们想要读取或写入模型中的所有权重时，这是很方便的。

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

保存单个权重向量(或其他张量)很有用，但如果我们要保存(并在以后加载)整个模型，则会变得非常单调乏味。毕竟，我们可能有数百个参数组散布在各处。为此，深度学习框架提供了加载和保存整个网络的内置功能。需要注意的一个重要细节是，这将保存模型*参数*，而不是整个模型。例如，如果我们有一个3层的MLP，我们需要单独指定架构。这是因为模型本身可以包含任意代码，因此它们不能自然地序列化。因此，为了恢复模型，我们需要在代码中生成体系结构，然后从磁盘加载参数。让我们从我们熟悉的MLP开始。

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

接下来，我们将模型的参数存储为名为“mlp.params”的文件。

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

为了恢复模型，我们实例化了原始MLP模型的克隆。我们不是随机初始化模型参数，而是直接读取文件中存储的参数。

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

由于两个实例具有相同的模型参数，因此相同输入`X`的计算结果应该是相同的。让我们验证一下这一点。

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
* 我们可以通过参数字典保存和加载网络的整个参数集。
* 保存体系结构必须在代码中完成，而不是在参数中完成。

## 练习

1. 即使不需要将经过训练的模型部署到不同的设备，存储模型参数的实际好处是什么？
1. 假设我们只想重用要合并到不同架构的网络中的网络的一部分。比如说，您将如何在新网络中使用以前网络的前两层？
1. 您将如何着手保存网络架构和参数？您会对架构施加哪些限制？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/327)
:end_tab:
