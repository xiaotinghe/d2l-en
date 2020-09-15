# 辍学
:label:`sec_dropout`

在:numref:`sec_weight_decay`中，我们通过惩罚$L_2$权重范数引入了正则化统计模型的经典方法。从概率的角度来看，我们可以通过论证我们已经假设了一个先验的观点，即权重从均值为零的高斯分布中取值，从而证明这一技术的合理性。更直观地说，我们可能会认为，我们鼓励模型在许多特性之间分散权重，而不是过分依赖于少量潜在的虚假关联。

## 重新审视过拟合

面对更多的特征而不是示例，线性模型往往过于拟合。但是给出的例子比特性多，我们通常可以指望线性模型不会过度拟合。不幸的是，线性模型推广的可靠性是有代价的。简单地说，线性模型不考虑特性之间的交互作用。对于每个特征，线性模型必须指定正权重或负权重，忽略上下文。

在传统文本中，归纳性和灵活性之间的这种基本张力被描述为“偏差-方差权衡”。线性模型有很高的偏差：它们只能代表一小类函数。然而，这些模型的方差很低：它们在不同的随机数据样本中给出了相似的结果。

深层神经网络位于偏差方差谱的另一端。与线性模型不同，神经网络不局限于单独观察每个特征。他们可以学习功能组之间的交互。例如，他们可能会推断“尼日利亚”和“西联”一起出现在一封电子邮件中表示垃圾邮件，但分开来看则不是。

即使当我们的例子比特征多得多的时候，深层神经网络也有过度拟合的能力。2017年，一组研究人员通过在随机标记的图像上训练深度网络，展示了神经网络的极端灵活性。尽管没有将输入与输出联系起来的真实模式，他们发现通过随机梯度下降优化的神经网络可以完美地标记训练集中的每一幅图像。想想这意味着什么。如果标签是均匀随机分配的，并且有10个类，那么没有一个分类器能对保留数据的准确率超过10%。这里的泛化差距高达90%。如果我们的模型表现力如此之强，以至于它们会严重过度拟合，那么我们什么时候才能期望它们不会过度拟合呢？

深层网络令人费解的泛化特性的数学基础仍然是一个开放的研究问题，我们鼓励以理论为导向的读者深入研究这个主题。现在，我们转向实际工具的调查，这些工具往往在经验上改进深网的泛化。

## 扰动鲁棒性

让我们简单地想想我们对一个好的预测模型的期望。我们希望它能在看不见的数据上表现良好。经典的泛化理论认为，为了缩小列车性能和试验性能之间的差距，我们应该建立一个简单的模型。简单可以以少量维度的形式出现。我们在:numref:`sec_model_selection`中讨论线性模型的单项基函数时探讨了这一点。此外，正如我们在:numref:`sec_weight_decay`中讨论权重衰减（$L_2$正则化）时所看到的，参数的（逆）范数也代表了一种有用的简化度量。简单性的另一个有用的概念是平滑性，即函数对输入的微小变化不敏感。例如，当我们对图像进行分类时，我们希望在像素中添加一些随机噪声基本上是无害的。

1995年，Christopher Bishop证明了带输入噪声的训练等价于Tikhonov正则化:cite:`Bishop.1995`。这项工作在要求函数平滑（因而简单）和要求它对输入中的扰动具有弹性之间建立了清晰的数学联系。

2014年，Srivastava等人。:cite:`Srivastava.Hinton.Krizhevsky.ea.2014`开发了一个聪明的想法，如何将毕晓普的想法应用到网络的内部层。也就是说，他们建议在训练时先在网络的每一层中注入噪声，然后再计算下一层。他们意识到，当训练具有多层的深层网络时，注入噪声只会在输入输出映射上强制实现平滑。

他们的想法叫做“dropout”，在前向传播过程中，在计算每个内部层的同时注入噪声，这已经成为训练神经网络的标准技术。这个方法叫做“dropout”，因为我们
*在训练过程中去掉一些神经元。
在整个训练过程中，在每次迭代中，标准的丢失包括在计算下一层之前将每层中的一些节点归零。

明确地说，我们把自己的叙述与毕肖普联系起来。关于辍学的原始论文通过对有性生殖的惊人类比提供了直觉。作者认为，神经网络过度拟合的特征是每一层都依赖于前一层的特定激活模式，称之为“协同适应”。他们声称，辍学破坏了共适应，正如有性生殖被认为是破坏共适应基因一样。

关键的挑战是如何注入这种噪音。一种想法是以一种“无偏”的方式注入噪声，这样每一层的期望值——在固定其他层时——等于它在没有噪声的情况下的期望值。

在毕晓普的工作中，他在线性模型的输入中加入了高斯噪声。在每次训练迭代中，他将从均值为零的分布$\epsilon \sim \mathcal{N}(0,\sigma^2)$采样的噪声添加到输入$\mathbf{x}$，得到一个扰动点$\mathbf{x}' = \mathbf{x} + \epsilon$。预计$E[\mathbf{x}'] = \mathbf{x}$。

在标准的辍学正则化中，每一层都通过被保留（而不是退出）的节点的分数标准化而去借方。换句话说，在*退出概率*$p$的情况下，每个中间激活$h$被随机变量$h'$替换，如下所示：

$$
\begin{aligned}
h' =
\begin{cases}
    0 & \text{ with probability } p \\
    \frac{h}{1-p} & \text{ otherwise}
\end{cases}
\end{aligned}
$$

根据设计，期望值保持不变，即$E[h'] = h$。

## 实践中辍学

回忆一下在:numref:`fig_mlp`中有一个隐藏层和5个隐藏单元的MLP。当我们将dropout应用于一个隐藏层，以$p$的概率将每个隐藏单元归零，结果可以看作是一个只包含原始神经元子集的网络。在:numref:`fig_dropout2`中，$h_2$和$h_5$被移除。因此，输出的计算不再依赖于$h_2$或$h_5$，并且在执行反向传播时，它们各自的梯度也消失了。这样，输出层的计算不能过度依赖于$h_1, \ldots, h_5$的任何一个元素。

![MLP before and after dropout.](../img/dropout2.svg)
:label:`fig_dropout2`

通常，我们在测试时禁用退出。给定一个经过训练的模型和一个新的例子，我们不需要删除任何节点，因此不需要规范化。然而，也有一些例外：一些研究人员将测试时的辍学作为一种启发式方法来估计神经网络预测的不确定性：如果预测在许多不同的辍学掩码上一致，那么我们可以说网络更自信。

## 从头开始实施

为了实现单个层的丢失函数，我们必须从Bernoulli（二进制）随机变量中提取与我们层的维数相同的样本，其中随机变量的值为$1$（保持），概率为$1-p$（下降），$p$（下降）。实现这一点的一个简单方法是首先从均匀分布$U[0, 1]$中提取样本。然后我们可以保留那些对应的样本大于$p$的节点，去掉其余的节点。

在下面的代码中，我们实现了一个`dropout_layer`函数，该函数以概率`X`删除张量输入`X`中的元素，并按上述方式重新缩放余数：将幸存者除以`1.0-dropout`。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return np.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = np.random.uniform(0, 1, X.shape) > dropout
    return mask.astype(np.float32) * X / (1.0 - dropout)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return torch.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = (torch.Tensor(X.shape).uniform_(0, 1) > dropout).float()
    return mask * X / (1.0 - dropout)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return tf.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = tf.random.uniform(
        shape=tf.shape(X), minval=0, maxval=1) < 1 - dropout
    return tf.cast(mask, dtype=tf.float32) * X / (1.0 - dropout)
```

我们可以用几个例子来测试`dropout_layer`函数。在下面的代码行中，我们通过dropout操作传递输入`X`，概率分别为0、0.5和1。

```{.python .input}
X = np.arange(16).reshape(2, 8)
print(dropout_layer(X, 0))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1))
```

```{.python .input}
#@tab pytorch
X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(16, dtype=tf.float32), (2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
```

### 定义模型参数

同样，我们使用:numref:`sec_fashion_mnist`中引入的时尚MNIST数据集。我们定义了一个包含两个隐藏层的MLP，每个层包含256个单元。

```{.python .input}
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens1))
b1 = np.zeros(num_hiddens1)
W2 = np.random.normal(scale=0.01, size=(num_hiddens1, num_hiddens2))
b2 = np.zeros(num_hiddens2)
W3 = np.random.normal(scale=0.01, size=(num_hiddens2, num_outputs))
b3 = np.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
```

```{.python .input}
#@tab tensorflow
num_outputs, num_hiddens1, num_hiddens2 = 10, 256, 256
```

### 定义模型

下面的模型将dropout应用于每个隐藏层的输出（遵循激活函数）。我们可以分别为每一层设置丢失概率。一个常见的趋势是将较低的辍学概率设置为靠近输入层。下面我们分别为第一层和第二层设置0.2和0.5。我们确保退学只在训练期间有效。

```{.python .input}
dropout1, dropout2 = 0.2, 0.5

def net(X):
    X = X.reshape(-1, num_inputs)
    H1 = npx.relu(np.dot(X, W1) + b1)
    # Use dropout only when training the model
    if autograd.is_training():
        # Add a dropout layer after the first fully connected layer
        H1 = dropout_layer(H1, dropout1)
    H2 = npx.relu(np.dot(H1, W2) + b2)
    if autograd.is_training():
        # Add a dropout layer after the second fully connected layer
        H2 = dropout_layer(H2, dropout2)
    return np.dot(H2, W3) + b3
```

```{.python .input}
#@tab pytorch
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()

        self.num_inputs = num_inputs
        self.training = is_training

        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)

        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # Use dropout only when training the model
        if self.training == True:
            # Add a dropout layer after the first fully connected layer
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # Add a dropout layer after the second fully connected layer
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
```

```{.python .input}
#@tab tensorflow
dropout1, dropout2 = 0.2, 0.5

class Net(tf.keras.Model):
    def __init__(self, num_outputs, num_hiddens1, num_hiddens2):
        super().__init__()
        self.input_layer = tf.keras.layers.Flatten()
        self.hidden1 = tf.keras.layers.Dense(num_hiddens1, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(num_hiddens2, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_outputs)

    def call(self, inputs, training=None):
        x = self.input_layer(inputs)
        x = self.hidden1(x)
        if training:
            x = dropout_layer(x, dropout1)
        x = self.hidden2(x)
        if training:
            x = dropout_layer(x, dropout2)
        x = self.output_layer(x)
        return x

net = Net(num_outputs, num_hiddens1, num_hiddens2)
```

### 培训和测试

这与前面描述的MLP培训和测试类似。

```{.python .input}
num_epochs, lr, batch_size = 10, 0.5, 256
loss = gluon.loss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

```{.python .input}
#@tab pytorch
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr, batch_size = 10, 0.5, 256
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 简明实施

对于高级api，我们只需要在每个完全连接的层之后添加一个`Dropout`层，将丢失概率作为惟一的参数传递给它的构造函数。在训练过程中，`Dropout`层将根据指定的丢失概率随机丢弃前一层的输出（或相当于下一层的输入）。当不处于训练模式时，`Dropout`层只在测试期间传递数据。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation="relu"),
        # Add a dropout layer after the first fully connected layer
        nn.Dropout(dropout1),
        nn.Dense(256, activation="relu"),
        # Add a dropout layer after the second fully connected layer
        nn.Dropout(dropout2),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # Add a dropout layer after the first fully connected layer
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # Add a dropout layer after the second fully connected layer
        nn.Dropout(dropout2),
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
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # Add a dropout layer after the first fully connected layer
    tf.keras.layers.Dropout(dropout1),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # Add a dropout layer after the second fully connected layer
    tf.keras.layers.Dropout(dropout2),
    tf.keras.layers.Dense(10),
])
```

接下来，我们对模型进行训练和测试。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 摘要

* 除了控制维数和权重向量的大小之外，dropout是另一种避免过度拟合的工具。它们通常是联合使用的。
* Dropout用期望值为$h$的随机变量替换激活$h$。
* 退学只在训练期间使用。

## 练习

1. 如果改变第一层和第二层的退出概率会怎么样？特别是，如果切换两个层的值会怎么样？设计一个实验来回答这些问题，定量地描述你的结果，并总结出定性的结论。
1. 增加epoch的数量，并将使用dropout和不使用dropout时获得的结果进行比较。
1. 当应用和不应用dropout时，每个隐藏层中激活的变化是多少？绘制一个图来显示这两个模型的数量随时间的变化情况。
1. 为什么在考试时通常不使用辍学？
1. 以本节中的模型为例，比较使用dropout和weight decay的效果。当同时使用“辍学”和“体重衰减”时会发生什么情况？结果是相加的吗？回报是否减少（或更糟）？他们互相抵消了吗？
1. 如果我们将dropout应用于权重矩阵的各个权重而不是激活，会发生什么？
1. 发明另一种技术，在每层注入随机噪声，这与标准的丢失技术不同。你能开发出一种在时尚MNIST数据集上表现优于dropout的方法吗（对于一个固定的架构）？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/100)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/101)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/261)
:end_tab:
