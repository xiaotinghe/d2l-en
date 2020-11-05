# 从零开始实现线性回归
:label:`sec_linear_scratch`

既然您了解了线性回归背后的关键思想，我们就可以开始在代码中动手实现了。在这一部分中，我们将从头开始实现整个方法，包括数据管道、模型、损失函数和小批量随机梯度下降优化器。虽然现代的深度学习框架几乎可以自动化所有这些工作，但是从零开始实现是确保您真正知道自己在做什么的唯一方法。此外，当需要定制模型、定义我们自己的层或损失函数时，了解引擎盖下的工作原理将非常方便。在这一节中，我们将只依赖于张量和自微分。然后，我们将介绍一个更简洁的实现，利用深度学习框架的各种各样的好处。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

## 生成数据集

为了简单起见，我们将根据具有加性噪声的线性模型构造一个人工数据集。我们的任务是使用数据集中包含的有限的一组示例来恢复这个模型的参数。我们将使数据保持低维，这样我们就可以很容易地将其可视化。在下面的代码片段中，我们生成了一个包含1000个示例的数据集，每个示例由从标准正态分布中采样的2个特性组成。因此，我们的合成数据集将是一个矩阵$\mathbf{X}\in \mathbb{R}^{1000 \times 2}$。

生成数据集的真实参数将为$\mathbf{w} = [2, -3.4]^\top$和$b = 4.2$，我们的合成标签将根据以下线性模型分配，噪声项为$\epsilon$：

$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$$

您可以将$\epsilon$看作是捕获特性和标签上的潜在测量误差。我们假设标准假设成立，因此$\epsilon$服从正态分布，平均值为0。为了使问题容易，我们将其标准偏差设置为0.01。下面的代码生成我们的合成数据集。

```{.python .input}
#@tab mxnet, pytorch
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = d2l.normal(0, 1, (num_examples, len(w)))
    y = d2l.matmul(X, w) + b
    y += d2l.normal(0, 0.01, y.shape)
    return X, d2l.reshape(y, (-1, 1))
```

```{.python .input}
#@tab tensorflow
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = d2l.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = d2l.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = d2l.reshape(y, (-1, 1))
    return X, y
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

注意`features`中的每一行都包含一个二维数据示例，`labels`中的每一行都包含一个一维标签值（标量）。

```{.python .input}
#@tab all
print('features:', features[0],'\nlabel:', labels[0])
```

通过使用第二个特征`features[:, 1]`和`labels`生成散点图，我们可以清楚地观察到两者之间的线性相关性。

```{.python .input}
#@tab all
d2l.set_figsize()
# The semicolon is for displaying the plot only
d2l.plt.scatter(d2l.numpy(features[:, 1]), d2l.numpy(labels), 1);
```

## 读取数据集

回想一下，训练模型包括对数据集进行多次传递，每次获取一小批示例，并使用它们来更新我们的模型。由于这个过程对于训练机器学习算法是如此的基础，所以有必要定义一个实用函数来洗牌数据集并以小批量方式访问它。

在下面的代码中，我们定义`data_iter`函数来演示此功能的一个可能实现。该函数接受一个批大小、一个特征矩阵和一个标签向量，生成大小为`batch_size`的小批量。每个小批量包含一个特性和标签的元组。

```{.python .input}
#@tab mxnet, pytorch
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = d2l.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

```{.python .input}
#@tab tensorflow
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)
```

一般来说，请注意，我们希望使用大小合理的小批量来利用GPU硬件，它擅长并行化操作。因为每个例子都可以并行地通过我们的模型，并且每个例子的损失函数的梯度也可以被并行地获得，gpu允许我们在几乎不比处理一个例子花费更多的时间来处理数百个例子。

为了建立一些直觉，让我们阅读并打印第一批数据示例。每个minibatch中特性的形状告诉我们minibatch的大小和输入特性的数量。同样，我们的小批量标签的形状由`batch_size`给出。

```{.python .input}
#@tab all
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

当我们运行迭代时，我们会连续获得不同的小批量，直到整个数据集用完为止（试试这个）。虽然上面实现的迭代对于教学目的来说是很好的，但是它的效率很低，可能会在实际问题上给我们带来麻烦。例如，它要求我们将所有数据加载到内存中，并执行大量随机内存访问。在深度学习框架中实现的内置迭代器效率要高得多，它们可以处理存储在文件中的数据和通过数据流馈送的数据。

## 初始化模型参数

在我们开始通过小批量随机梯度下降来优化我们的模型参数之前，我们首先需要有一些参数。我们用0.01的标准偏差初始化一个0.01标准偏差的随机分布。

```{.python .input}
w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01),
                trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)
```

在初始化参数之后，我们的下一个任务是更新它们，直到它们与我们的数据完全匹配为止。每次更新都需要取损失函数相对于参数的梯度。给定这个梯度，我们可以在可能减少损失的方向上更新每个参数。

因为没有人想显式地计算梯度（这是一个冗长且容易出错的过程），我们使用自动微分法（如:numref:`sec_autograd`中所介绍的）来计算梯度。

## 定义模型

下一步，我们必须定义我们的模型，将其输入和参数与其输出相关联。回想一下，为了计算线性模型的输出，我们只需取输入特征$\mathbf{X}$和模型权重$\mathbf{w}$的矩阵向量点乘，并将偏移量$b$添加到每个示例中。注意$\mathbf{Xw}$以下是一个向量，$b$是一个标量。调用:numref:`subsec_broadcasting`中描述的广播机制。当我们添加一个向量和一个标量时，标量被加到向量的每个分量上。

```{.python .input}
#@tab all
def linreg(X, w, b):  #@save
    """The linear regression model."""
    return d2l.matmul(X, w) + b
```

## 定义损失函数

由于模型的更新需要考虑损失函数的梯度，因此我们应该首先定义损失函数。这里我们将使用:numref:`sec_linear_regression`中描述的平方损失函数。在实现中，需要将真值`y`转换为预测值的形状73229365。由以下函数返回的结果也将具有与73229365相同的形状。

```{.python .input}
#@tab all
def squared_loss(y_hat, y):  #@save
    """Squared loss."""
    return (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
```

## 定义优化算法

正如我们在:numref:`sec_linear_regression`中所讨论的，线性回归有一个闭合形式的解。然而，这不是一本关于线性回归的书：这是一本关于深入学习的书。由于本书介绍的其他模型都无法解析求解，我们将借此机会介绍您的第一个小批量随机梯度下降的工作实例。

在每一步，使用从我们的数据集随机抽取的一个小批量，我们将根据我们的参数估计损失的梯度。下一步，我们将在可能减少损失的方向上更新参数。下面的代码应用小批量随机梯度下降更新，给定一组参数、一个学习速率和一个批大小。更新步骤的大小由学习速率`lr`确定。因为我们的损失是作为一个小批实例的总和来计算的，所以我们用批量大小（`batch_size`）来规范化我们的步长，这样典型步长的大小不会严重依赖于我们对批量大小的选择。

```{.python .input}
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

```{.python .input}
#@tab pytorch
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param in params:
        param.data.sub_(lr*param.grad/batch_size)
        param.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param, grad in zip(params, grads):
        param.assign_sub(lr*grad/batch_size)
```

## 训练

现在我们已经准备好了所有的部分，我们准备好实施主训练循环。理解这些代码是至关重要的，因为在你的职业生涯中，你将看到几乎完全相同的培训循环。

在每一次迭代中，我们将获取一小批训练示例，并将它们传递到我们的模型中，以获得一组预测。计算完损耗后，我们开始反向通过网络，存储每个参数的梯度。最后，我们将调用优化算法`sgd`来更新模型参数。

总之，我们将执行以下循环：

* 初始化参数$(\mathbf{w}, b)$
* 完成直到重复
    * 计算坡度$\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * 更新参数$(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$

在每个*epoch*中，我们将遍历整个数据集（使用`data_iter`函数），遍历一次训练数据集中的每个示例（假设示例的数量可以被批大小整除）。时代数`num_epochs`和学习率`lr`都是超参数，我们在这里分别设置为3和0.03。不幸的是，设置超参数是很棘手的，需要通过反复试验进行一些调整。我们暂时省略了这些细节，但稍后将在:numref:`chap_optimization`中修改它们。

```{.python .input}
#@tab all
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
```

```{.python .input}
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Because `l` has a shape (`batch_size`, 1) and is not a scalar
        # variable, the elements in `l` are added together to obtain a new
        # variable, on which gradients with respect to [`w`, `b`] are computed
        l.backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab pytorch
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on `l` with respect to [`w`, `b`]
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab tensorflow
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape() as g:
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on l with respect to [`w`, `b`]
        dw, db = g.gradient(l, [w, b])
        # Update parameters using their gradient
        sgd([w, b], [dw, db], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')
```

在本例中，因为我们自己合成了数据集，所以我们确切地知道真正的参数是什么。因此，我们可以通过将真实参数与通过训练循环学习到的参数进行比较来评估我们在训练中的成功。事实证明，他们彼此非常亲近。

```{.python .input}
#@tab all
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
print(f'error in estimating b: {true_b - b}')
```

注意，我们不应该想当然地认为我们能够完美地恢复参数。然而，在机器学习中，我们通常不太关心恢复真实的底层参数，而更关心那些能够导致高精度预测的参数。幸运的是，即使是在困难的优化问题上，随机梯度下降也常常能找到非常好的解，部分原因是，对于深部网络，存在着许多参数配置，这导致了高精度的预测。

## 摘要

* 我们看到了deep网络是如何从零开始实现和优化的，只使用张量和自动微分，而不需要定义层或花哨的优化器。
* 这一部分只触及了可能的表面。在下面的部分中，我们将根据我们刚刚介绍的概念描述其他模型，并学习如何更简洁地实现它们。

## 练习

1. 如果我们将权重初始化为零会发生什么。算法还能工作吗？
1. 假设你是[Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm)，试图建立一个介于电压和电流之间的模型。你能用自动微分来学习你的模型参数吗？
1. 你能用普朗克定律吗(https://en.wikipedia.org/wiki/Planck%27s_法律)用光谱能量密度确定物体的温度？
1. 如果你想计算二阶导数，你可能会遇到什么问题？你要怎么修理它们？
1.  为什么`squared_loss`功能中需要`reshape`功能？
1. 使用不同的学习速率进行实验，以找出损失函数值下降的速度。
1. 如果示例数不能除以批处理大小，那么`data_iter`函数的行为会发生什么变化？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/201)
:end_tab:
