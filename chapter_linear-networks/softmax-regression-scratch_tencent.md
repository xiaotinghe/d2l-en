# 软最大回归从头开始的实现
:label:`sec_softmax_scratch`

就像我们从头开始实现线性回归一样，我们相信Softmax回归也是类似的基础，您应该知道如何自己实现它的血淋淋的细节。我们将使用在:numref:`sec_fashion_mnist`中引入的Fashion-MNIST数据集，设置一个批大小为256时的数据迭代器。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
from IPython import display
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from IPython import display
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from IPython import display
```

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 正在初始化模型参数

与我们的线性回归示例一样，这里的每个示例都将由一个固定长度的向量表示。原始数据集中的每个示例都是一个$28 \times 28$图像。在本节中，我们将展平每个图像，将其视为长度为784的向量。在未来，我们将讨论更复杂的策略来利用图像中的空间结构，但现在我们只是将每个像素位置视为另一个特征。

回想一下，在Softmax回归中，我们有多少个类就有多少个输出。因为我们的数据集有10个类，所以我们的网络的输出维数将为10。因此，我们的权重将构成一个$784 \times 10$矩阵，而偏差将构成一个$1 \times 10$行向量。与线性回归一样，我们将使用高斯噪声初始化权重`W`，并将偏差设为初始值0。

```{.python .input}
num_inputs = 784
num_outputs = 10

W = np.random.normal(0, 0.01, (num_inputs, num_outputs))
b = np.zeros(num_outputs)
W.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
num_inputs = 784
num_outputs = 10

W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs),
                                 mean=0, stddev=0.01))
b = tf.Variable(tf.zeros(num_outputs))
```

## 定义SoftMax操作

在实施SoftMax回归模型之前，让我们简要回顾一下SUM运算符是如何沿张量中的特定维度工作的，如:numref:`subseq_lin-alg-reduction`和:numref:`subseq_lin-alg-non-reduction`中所述。给定矩阵`X`，我们可以对所有元素(缺省情况下)求和，或者仅对同一轴中的元素求和，即，相同的列(轴0)或相同的行(轴1)。请注意，如果`X`是形状为(2，3)的张量，并且我们对列求和，则结果将是形状为(3，)的向量。在调用SUM运算符时，我们可以指定保留原始张量中的轴数，而不是折叠出我们求和后的维度。这将产生形状为(1，3)的二维张量。

```{.python .input}
#@tab pytorch
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdim=True), d2l.reduce_sum(X, 1, keepdim=True)
```

```{.python .input}
#@tab mxnet, tensorflow
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

现在我们准备好实施Softmax操作。回想一下，Softmax由三个步骤组成：i)我们对每一项求幂(使用`exp`)；ii)我们对每行求和(批中的每个示例都有一行)，以获得每个示例的归一化常数；iii)我们将每行除以其归一化常数，以确保结果之和为1。在查看代码之前，让我们回想一下它是如何表示为等式的：

$$
\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.
$$

分母或归一化常数有时也称为*配分函数*(其对数称为对数配分函数)。这个名字的起源是在[统计分析(Statistics physics](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)))]中，其中一个相关的方程对粒子系综上的分布进行了建模。

```{.python .input}
#@tab mxnet, tensorflow
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

```{.python .input}
#@tab pytorch
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdim=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

如您所见，对于任何随机输入，我们将每个元素转换为非负数。此外，每一行的总和为1，这是概率所需的。

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
#@tab tensorflow
X = tf.random.normal((2, 5), 0, 1)
X_prob = softmax(X)
X_prob, tf.reduce_sum(X_prob, 1)
```

请注意，虽然这在数学上看起来是正确的，但我们在实现中有点草率，因为我们没有采取措施防止由于矩阵的大元素或非常小的元素而导致的数值溢出或下溢。

## 定义模型

既然我们已经定义了Softmax操作，我们就可以实施Softmax回归模型了。下面的代码定义了如何通过网络将输入映射到输出。请注意，在通过我们的模型传递数据之前，我们使用`reshape`函数将批次中的每个原始图像展平为向量。

```{.python .input}
#@tab all
def net(X):
    return softmax(d2l.matmul(d2l.reshape(X, (-1, W.shape[0])), W) + b)
```

## 定义损失函数

接下来，我们需要实现:numref:`sec_softmax`中引入的交叉熵损失函数。这可能是所有深度学习中最常见的损失函数，因为目前，分类问题远远多于回归问题。

回想一下，交叉熵取分配给真实标签的预测概率的负对数似然。我们可以使用一个操作符挑选所有元素，而不是使用Python for-loop迭代预测(这往往效率很低)。下面，我们创建具有3个类别上的预测概率的2个示例的玩具数据`y_hat`。然后，我们在第一个示例中选取第一类的概率，在第二个示例中选取第三类的概率。

```{.python .input}
#@tab mxnet, pytorch
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
#@tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

现在，我们只需一行代码就可以高效地实现交叉熵损失函数。

```{.python .input}
#@tab mxnet, pytorch
def cross_entropy(y_hat, y):
    return - d2l.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
```

```{.python .input}
#@tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1])))

cross_entropy(y_hat, y)
```

## 分类准确率

给定预测概率分布`y_hat`，每当我们必须输出硬预测时，我们通常选择具有最高预测概率的类。事实上，许多应用程序都需要我们做出选择。Gmail必须将电子邮件分类为“主要”、“社交”、“更新”或“论坛”。它可能会在内部估计概率，但在一天结束时，它必须从类中选择一个。

当预测与标签分类`y`一致时，它们是正确的。分类精度是所有正确预测的分数。虽然直接优化精确度可能很困难(它是不可微的)，但它通常是我们最关心的性能度量，并且我们在训练分类器时几乎总是会报告它。

为了计算精度，我们执行以下操作。首先，如果`y_hat`是一个矩阵，我们假设第二维存储每个类别的预测分数。我们使用`argmax`根据每行中最大条目的索引来获得预测的类别。然后，我们将预测的类别与地面真实的`y`进行元素比较。由于相等运算符`==`对数据类型敏感，因此我们转换`y_hat`的数据类型以匹配`y`的数据类型。结果是一个包含0(假)和1(真)条目的张量。求和得出正确预测的数量。

```{.python .input}
#@tab all
def accuracy(y_hat, y):  #@save
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)        
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))
```

我们将继续使用前面定义的变量`y_hat`和`y`分别作为预测概率分布和标签。我们可以看到，第一个示例的预测类是2(行中最大的元素是0.6，索引为2)，这与实际标签0不一致。第二个示例的预测类别为2(该行最大元素为0.5，索引为2)，与实际标签2一致，因此这两个示例的分类准确率为0.5。

```{.python .input}
#@tab all
accuracy(y_hat, y) / len(y)
```

类似地，我们可以评估任何模型`net`对经由数据迭代器`data_iter`访问的数据集的准确性。

```{.python .input}
#@tab mxnet, tensorflow
def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for _, (X, y) in enumerate(data_iter):
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for _, (X, y) in enumerate(data_iter):
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

这里的`Accumulator`是一个实用程序类，用于累加多个变量的总和。在上面的`evaluate_accuracy`函数中，我们在`Accumulator`实例中创建了2个变量，分别用于存储正确预测的数量和预测的数量。当我们迭代数据集时，两者都将随着时间的推移而累积。

```{.python .input}
#@tab all
class Accumulator:  #@save
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

因为我们用随机权重初始化了`net`模型，所以这个模型的精度应该接近随机猜测，即10个类的精度是0.1%。

```{.python .input}
#@tab all
evaluate_accuracy(net, test_iter)
```

## 培训

如果您通读我们在:numref:`sec_linear_scratch`中线性回归的实现，那么SoftMax回归的训练循环应该会看起来非常熟悉。在这里，我们重构实现以使其可重用。首先，我们定义一个函数来训练一个时期。请注意，`updater`是更新模型参数的通用函数，它接受批大小作为参数。它可以是`d2l.sgd`函数的包装器，也可以是框架的内置优化函数。

```{.python .input}
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """Train a model within one epoch (defined in Chapter 3)."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        # Compute gradients and update parameters
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.size)
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab pytorch
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """The training loop defined in Chapter 3."""
    # Set the model to training mode
    if isinstance(net, torch.nn.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # Using PyTorch in-built optimizer & loss criterion
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y),
                       y.size().numel())
        else:
            # Using custom built optimizer & loss criterion
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab tensorflow
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """The training loop defined in Chapter 3."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        with tf.GradientTape() as tape:
            y_hat = net(X)
            # Keras implementations for loss takes (labels, predictions)
            # instead of (predictions, labels) that users might implement
            # in this book, e.g. `cross_entropy` that we implemented above
            if isinstance(loss, tf.keras.losses.Loss):
                l = loss(y, y_hat)
            else:
                l = loss(y_hat, y)
        if isinstance(updater, tf.keras.optimizers.Optimizer):
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            updater.apply_gradients(zip(grads, params))
        else:
            updater(X.shape[0], tape.gradient(l, updater.params))
        # Keras loss by default returns the average loss in a batch
        l_sum = l * float(tf.size(y)) if isinstance(
            loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

在展示训练函数的实现之前，我们定义了一个在动画中绘制数据的实用程序类。同样，它的目标是简化书中睡觉中的代码。

```{.python .input}
#@tab all
class Animator:  #@save
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

然后，下面的训练函数在通过`train_iter`访问的训练数据集上训练模型`net`，用于多个历元，其由`num_epochs`指定。在每个时期结束时，在通过`test_iter`访问的测试数据集上对模型进行评估。我们将利用`Animator`级来可视化培训进度。

```{.python .input}
#@tab all
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """Train a model (defined in Chapter 3)."""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

作为一种从头开始的实现，我们使用:numref:`sec_linear_scratch`中定义的小批量随机梯度下降来优化模型的损失函数，学习率为0.1%。

```{.python .input}
#@tab mxnet, pytorch
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

```{.python .input}
#@tab tensorflow
class Updater():  #@save
    """For updating parameters using minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, batch_size, grads):
        d2l.sgd(self.params, grads, self.lr, batch_size)

updater = Updater([W, b], lr=0.1)
```

现在我们用10个历元训练模型。请注意，历元数(`num_epochs`)和学习率(`lr`)都是可调整的超参数。通过改变它们的值，我们可以提高模型的分类精度。

```{.python .input}
#@tab all
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

## 预测

既然训练已经完成，我们的模型就可以对一些图像进行分类了。给定一系列图像，我们将比较它们的实际标签(文本输出的第一行)和模型的预测(文本输出的第二行)。

```{.python .input}
#@tab all
def predict_ch3(net, test_iter, n=6):  #@save
    """Predict labels (defined in Chapter 3)."""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(X), axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

## 摘要

* 使用Softmax回归，我们可以训练多类分类的模型。
* Softmax回归的训练循环与线性回归非常相似：检索和读取数据，定义模型和损失函数，然后使用优化算法训练模型。您很快就会发现，大多数常见的深度学习模型都有类似的培训过程。

## 练习

1. 在本节中，我们将根据Softmax运算的数学定义直接实现Softmax函数。这会造成什么问题呢？提示：试着计算一下$\exp(50)$的大小。
1. 本节中的函数`cross_entropy`根据交叉熵损失函数的定义来实现。此实施可能会出现什么问题？提示：考虑对数域。
1. 你能想出什么解决方案来解决上述两个问题？
1. 退回最有可能的标签总是一个好主意吗？例如，您会因为医学诊断而这样做吗？
1. 假设我们想要使用Softmax回归根据一些特征预测下一个单词。词汇量大可能会产生哪些问题？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/225)
:end_tab:
