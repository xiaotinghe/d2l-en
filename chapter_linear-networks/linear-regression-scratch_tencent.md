# 从头开始实施线性回归
:label:`sec_linear_scratch`

既然您了解了线性回归背后的关键思想，我们就可以开始在代码中进行动手实现了。在这一部分，我们将从头开始实施整个方法，包括数据管道、模型、损失函数和小批量随机梯度下降优化器。虽然现代深度学习框架几乎可以自动化所有这些工作，但从头开始实现是确保您真正知道自己在做什么的唯一方法。此外，当涉及到定制模型、定义我们自己的层或损失函数时，了解事情是如何在引擎盖下工作的将被证明是很方便的。在本节中，我们将仅依靠张量和自动微分。之后，我们将利用深度学习框架的华而不实的优势，推出更简明的实现。

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

为了简单起见，我们将根据带有加性噪声的线性模型来构建人工数据集。我们的任务将是使用数据集中包含的有限示例集恢复此模型的参数。我们将保持数据的低维，这样我们就可以很容易地将其可视化。在下面的代码片段中，我们生成一个包含1000个示例的数据集，每个示例由从标准正态分布中采样的2个特征组成。因此，我们的合成数据集将是矩阵$\mathbf{X}\in \mathbb{R}^{1000 \times 2}$。

生成我们的数据集的真实参数将是$\mathbf{w} = [2, -3.4]^\top$和$b = 4.2$，我们的合成标签将根据以下带有噪声项$\epsilon$的线性模型进行分配：

$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$$

您可以将$\epsilon$视为捕获要素和标签上的潜在测量错误。我们将假设标准假设成立，因此$\epsilon$服从均值为0的正态分布。为了使问题简单，我们将其标准差设置为0.01。下面的代码生成我们的合成数据集。

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

注意，`features`中的每行由2维数据示例组成，`labels`中的每行由1维标签值(标量)组成。

```{.python .input}
#@tab all
print('features:', features[0],'\nlabel:', labels[0])
```

通过使用第二特征`features[:, 1]`和`labels`生成散点图，我们可以清楚地观察到两者之间的线性相关性。

```{.python .input}
#@tab all
d2l.set_figsize()
# The semicolon is for displaying the plot only
d2l.plt.scatter(d2l.numpy(features[:, 1]), d2l.numpy(labels), 1);
```

## 正在读取数据集

回想一下，训练模型包括多次遍历数据集，一次抓取一小批示例，并使用它们来更新我们的模型。由于这一过程对于训练机器学习算法非常重要，因此有必要定义一个实用函数来洗牌数据集并以小批量方式访问它。

在下面的代码中，我们定义`data_iter`函数来演示此功能的一种可能实现。该函数采用批次大小、特征矩阵和标签向量，产生大小为`batch_size`的小批次。每个小批量由一组特征和标签组成。

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

通常，请注意，我们希望使用大小合理的小批来利用GPU硬件，它擅长并行化操作。由于每个示例可以通过我们的模型并行馈送，并且每个示例的损失函数的梯度也可以并行获取，因此GPU使我们能够在几乎不比处理单个示例多的时间内处理数百个示例。

为了建立一些直觉，让我们阅读并打印第一小批数据示例。每个小批量中的特征的形状告诉我们小批量的大小和输入特征的数量。同样，我们的小批量标签将会有一个`batch_size`的形状。

```{.python .input}
#@tab all
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

当我们运行迭代时，我们会连续获得不同的小批，直到整个数据集都用完为止(试试这个)。虽然上面实现的迭代对于说教目的是很好的，但它的效率很低，可能会在实际问题上给我们带来麻烦。例如，它要求我们将所有数据加载到内存中，并执行大量随机内存访问。深度学习框架中实现的内置迭代器效率要高得多，它们既可以处理存储在文件中的数据，也可以处理通过数据流馈送的数据。

## 正在初始化模型参数

在我们可以开始通过小批量随机梯度下降来优化我们的模型参数之前，我们首先需要一些参数。在下面的代码中，我们通过从均值为0且标准差为0.01的正态分布中抽样随机数，并将偏差设置为0来初始化权重。

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

在初始化参数之后，我们的下一个任务是更新它们，直到它们完全符合我们的数据。每次更新都需要取我们的损失函数相对于参数的梯度。在给定该梯度的情况下，我们可以更新方向上的每个参数，以减少损失。

由于没有人想显式地计算梯度(这很乏味且容易出错)，我们使用:numref:`sec_autograd`中引入的自动微分来计算梯度。

## 定义模型

接下来，我们必须定义我们的模型，将其输入和参数与其输出相关联。回想一下，为了计算线性模型的输出，我们简单地取输入特征$\mathbf{X}$和模型权重$\mathbf{w}$的矩阵向量点积，并将偏移量$b$添加到每个示例。请注意，低于$\mathbf{Xw}$是向量，$b$是标量。回想一下:numref:`subsec_broadcasting`中描述的广播机制。当我们将向量和标量相加时，标量会被加到向量的每个分量上。

```{.python .input}
#@tab all
def linreg(X, w, b):  #@save
    """The linear regression model."""
    return d2l.matmul(X, w) + b
```

## 定义损失函数

因为更新我们的模型需要取损失函数的梯度，所以我们应该首先定义损失函数。这里我们将使用:numref:`sec_linear_regression`中描述的平方损失函数。在实现中，我们需要将真值`y`转换成预测值的形状`y_hat`。以下函数返回的结果也将具有与`y_hat`相同的形状。

```{.python .input}
#@tab all
def squared_loss(y_hat, y):  #@save
    """Squared loss."""
    return (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
```

## 定义优化算法

正如我们在:numref:`sec_linear_regression`中所讨论的，线性回归有一个闭合形式的解。然而，这不是一本关于线性回归的书，而是一本关于深度学习的书。由于本书介绍的其他模型都不能解析求解，我们将借此机会介绍您的第一个小批量随机梯度下降的工作示例。

在每个步骤中，使用从我们的数据集中随机抽取的一个小批量，我们将估计损失相对于我们的参数的梯度。接下来，我们将向可能减少损失的方向更新参数。以下代码在给定一组参数、学习率和批大小的情况下应用小批量随机梯度下降更新。更新步骤的大小由学习率`lr`确定。因为我们的损失是以小批量示例的总和来计算的，所以我们用批量大小(`batch_size`)来归一化步长，这样典型步长的大小就不会很大程度上依赖于我们对批量大小的选择。

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

## 培训

既然我们已经准备好了所有的部分，我们就可以准备实现主训练循环了。理解此代码至关重要，因为在您的深度学习职业生涯中，您将一遍又一遍地看到几乎相同的培训循环。

在每一次迭代中，我们将抓取一小批训练示例，并将它们通过我们的模型来获得一组预测。在计算损耗之后，我们开始反向通过网络，存储关于每个参数的梯度。最后，我们将调用优化算法`sgd`来更新模型参数。

总之，我们将执行以下循环：

* 初始化参数$(\mathbf{w}, b)$
* 重复操作，直到完成为止
    * 计算梯度$\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * 更新参数$(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$

在每个*纪元*中，我们将遍历整个数据集(使用`data_iter`函数)一次，遍历训练数据集中的每个示例(假设示例数量可以被批大小整除)。历元数`num_epochs`和学习率`lr`都是超参数，我们在这里分别将其设置为3和0.03.不幸的是，设置超参数很棘手，需要通过反复试验进行一些调整。我们现在省略这些细节，但稍后在:numref:`chap_optimization`修改它们。

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

在本例中，因为我们自己合成了数据集，所以我们确切地知道真正的参数是什么。因此，我们可以通过将真实参数与我们通过培训循环学到的参数进行比较来评估我们在培训中的成功。事实上，事实证明他们彼此非常亲近。

```{.python .input}
#@tab all
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
print(f'error in estimating b: {true_b - b}')
```

请注意，我们不应该想当然地认为我们能够完美地恢复参数。然而，在机器学习中，我们通常不太关心恢复真实的潜在参数，而更关心导致高精度预测的参数。幸运的是，即使在困难的优化问题上，随机梯度下降往往也能找到非常好的解，部分原因是，对于深层网络，存在许多导致高精度预测的参数配置。

## 摘要

* 我们了解了如何从头开始实现和优化深层网络，只需使用张量和自动微分，而不需要定义层或花哨的优化器。
* 这一节只触及了可能性的皮毛。在接下来的几节中，我们将基于刚才介绍的概念描述其他模型，并学习如何更简洁地实现它们。

## 练习

1. 如果我们将权重初始化为零，会发生什么呢？这个算法还会起作用吗？
1. 假设你正[Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm)地试图想出一个介于电压和电流之间的模型。你能使用自动微分来学习你的模型的参数吗？
1. 你能用[普朗克的Law](https://en.wikipedia.org/wiki/Planck%27s_law)]用光谱能量密度来确定物体的温度吗？
1. 如果你想计算二阶导数，你可能会遇到什么问题？你会怎么修理它们呢？
1.  为什么`reshape`函数需要在`squared_loss`函数中使用？
1. 使用不同的学习率进行实验，找出损失函数值下降的速度有多快。
1. 如果示例的数量不能除以批处理大小，那么`data_iter`函数的行为会发生什么呢？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/201)
:end_tab:
