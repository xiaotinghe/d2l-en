# 多层感知器的从头开始实现
:label:`sec_mlp_scratch`

既然我们已经从数学上描述了多层感知器(MLP)的特征，让我们尝试自己实现一个多层感知器(MLP)。为了与我们之前使用Softmax回归(:numref:`sec_softmax_scratch`)获得的结果进行比较，我们将继续使用Fashion-MNIST图像分类数据集(:numref:`sec_fashion_mnist`)。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
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

回想一下，Fashion-MNIST包含10个类，每个图像由$28 \times 28 = 784$个灰度像素值网格组成。同样，我们现在将忽略像素之间的空间结构，因此我们可以将其视为具有784个输入特征和10个类的简单分类数据集。首先，我们将实现一个具有一个隐藏层和256个隐藏单元的MLP。请注意，我们可以将这两个量都视为超参数。通常，我们选择2的幂的层宽度，由于内存在硬件中的分配和寻址方式，这往往在计算上是高效的。

同样，我们将用几个张量来表示我们的参数。请注意，*对于每一层*，我们必须跟踪一个权重矩阵和一个偏差向量。一如既往，我们为相对于这些参数的损耗梯度分配内存。

```{.python .input}
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)
W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```

```{.python .input}
#@tab tensorflow
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = tf.Variable(tf.random.normal(
    shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
b1 = tf.Variable(tf.zeros(num_hiddens))
W2 = tf.Variable(tf.random.normal(
    shape=(num_hiddens, num_outputs), mean=0, stddev=0.01))
b2 = tf.Variable(tf.random.normal([num_outputs], stddev=.01))

params = [W1, b1, W2, b2]
```

## 激活函数

为了确保我们知道一切是如何工作的，我们将使用Maximum函数自己实现RELU激活，而不是直接调用内置的`relu`函数。

```{.python .input}
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input}
#@tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input}
#@tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

## 型号

因为我们忽略了空间结构，所以我们将每个二维图像`reshape`成一个长度为`num_inputs`的平面矢量。最后，我们只需几行代码就可以实现我们的模型。

```{.python .input}
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(np.dot(X, W1) + b1)
    return np.dot(H, W2) + b2
```

```{.python .input}
#@tab pytorch
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(X@W1 + b1)  # Here '@' stands for matrix multiplication
    return (H@W2 + b2)
```

```{.python .input}
#@tab tensorflow
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(tf.matmul(X, W1) + b1)
    return tf.matmul(H, W2) + b2
```

## 损失函数

为了确保数值稳定性，并且由于我们已经从头开始实施了Softmax函数(:numref:`sec_softmax_scratch`)，因此我们利用来自高级API的集成函数来计算Softmax和交叉熵损失。回想一下我们早些时候在:numref:`subsec_softmax-implementation-revisited`中对这些错综复杂的问题的讨论。我们鼓励感兴趣的读者查看Loss函数的源代码，以加深对实现细节的了解。

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss()
```

```{.python .input}
#@tab tensorflow
def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(
        y, y_hat, from_logits=True)
```

## 培训

幸运的是，MLP的训练循环与Softmax回归的训练循环完全相同。再次利用`d2l`包，我们调用`train_ch3`函数(参见:numref:`sec_softmax_scratch`)，将纪元数设置为10，并将学习率设置为0.1.

```{.python .input}
num_epochs, lr = 10, 0.1
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

```{.python .input}
#@tab pytorch
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 10, 0.1
updater = d2l.Updater([W1, W2, b1, b2], lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

为了对学习到的模型进行评估，我们将其应用于一些测试数据。

```{.python .input}
#@tab all
d2l.predict_ch3(net, test_iter)
```

## 摘要

* 我们看到实现一个简单的MLP是很容易的，即使手动完成也是如此。
* 然而，对于大量的层，从头开始实现MLP仍然会变得杂乱无章(例如，命名和跟踪我们模型的参数)。

## 练习

1. 更改超参数`num_hiddens`的值，并查看此超参数对结果有何影响。确定此超参数的最佳值，使所有其他参数保持不变。
1. 尝试添加额外的隐藏层以查看它对结果有何影响。
1. 改变学习速度会如何改变你的成绩？固定模型架构和其他超参数(包括纪元数)，多大的学习率会给您带来最好的结果？
1. 通过对所有超参数(学习率、历元数、隐藏层数、每层隐藏单元数)进行联合优化，可以得到什么最佳结果？
1. 描述为什么处理多个超参数更具挑战性。
1. 对于在多个超参数上构建搜索，您能想到的最聪明的策略是什么？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/227)
:end_tab:
