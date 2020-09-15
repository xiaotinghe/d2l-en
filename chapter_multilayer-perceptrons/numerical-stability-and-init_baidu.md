# 数值稳定性和初始化
:label:`sec_numerical_stability`

到目前为止，我们实现的每个模型都要求我们根据预先指定的分布初始化其参数。到目前为止，我们认为初始化方案是理所当然的，掩盖了如何做出这些选择的细节。你甚至可能会觉得这些选择并不特别重要。相反，初始化方案的选择在神经网络学习中起着非常重要的作用，对保持数值稳定性至关重要。此外，这些选择可以与非线性激活函数的选择以有趣的方式联系在一起。我们选择哪个函数以及如何初始化参数可以决定优化算法收敛的速度。在这里糟糕的选择会导致我们在训练时遇到爆炸或消失的梯度。在本节中，我们将更详细地探讨这些主题，并讨论一些有用的启发式方法，您将发现这些方法在您的职业生涯中对深度学习非常有用。

## 消失和爆炸梯度

考虑一个有$L$层的深层网络，输入$\mathbf{x}$，输出$\mathbf{o}$。每一层$l$由一个由权重$f_l$参数化的变换$f_l$定义，其隐藏变量为$\mathbf{h}^{(l)}$（设$\mathbf{h}^{(0)} = \mathbf{x}$），我们的网络可以表示为：

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \text{ and thus } \mathbf{o} = f_L \circ \ldots \circ f_1(\mathbf{x}).$$

如果所有隐藏变量和输入都是向量，我们可以将$\mathbf{o}$相对于任何一组参数$\mathbf{W}^{(l)}$的梯度写如下：

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\mathrm{def}}{=}} \cdot \ldots \cdot \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\mathrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\mathrm{def}}{=}}.$$

换句话说，这个梯度是$L-l$矩阵$\mathbf{M}^{(L)} \cdot \ldots \cdot \mathbf{M}^{(l+1)}$和梯度向量$\mathbf{v}^{(l)}$的乘积。因此，我们容易受到同样的数值下溢问题的影响，这些问题往往是在将太多的概率相乘时突然出现的。在处理概率时，一个常见的技巧是切换到对数空间，即将压力从尾数转换为数值表示的指数。不幸的是，我们上面的问题更严重：最初，矩阵$\mathbf{M}^{(l)}$可能有各种各样的特征值。他们的产品可能很大，也可能非常小。

不稳定梯度带来的风险超出了数值表示。不可预测的梯度也威胁到我们的优化算法的稳定性。我们可能面临的参数更新要么（i）过大，破坏我们的模型（爆炸梯度*问题）；要么（ii）太小（消失梯度*问题），使得学习变得不可能，因为参数几乎每次更新都很难移动。

### 消失梯度

导致消失梯度问题的一个常见的罪魁祸首是在每个层的线性运算之后附加的激活函数$\sigma$的选择。从历史上看，sigmoid函数$1/(1 + \exp(-x))$（:numref:`sec_mlp`年引入）很受欢迎，因为它类似于阈值函数。由于早期的人工神经网络受到生物神经网络的启发，神经元发出“完全”或“完全不”的想法（就像生物神经元）似乎很吸引人。让我们仔细看看乙状结肠，看看它为什么会导致梯度消失。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

如你所见，sigmoid的梯度在它的输入大和小的时候都会消失。此外，当反向传播通过许多层时，除非我们在金发姑娘区，那里许多乙状体的输入接近于零，否则整个产品的梯度可能会消失。当我们的网络拥有许多层时，除非我们小心，否则在某一层可能会切断梯度。事实上，这个问题曾经困扰着深度网络培训。因此，相对稳定（但在神经上不太可信）的ReLUs已经成为从业者的默认选择。

### 爆炸梯度

相反的问题，当梯度爆炸时，同样令人烦恼。为了更好地说明这一点，我们画了100个高斯随机矩阵，并与一些初始矩阵相乘。对于我们选择的尺度（方差$\sigma^2=1$的选择），矩阵乘积爆炸。当这是由于深层网络的初始化而发生的，我们没有机会得到一个梯度下降优化器来收敛。

```{.python .input}
M = np.random.normal(size=(4, 4))
print('a single matrix', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))

print('after multiplying 100 matrices', M)
```

```{.python .input}
#@tab pytorch
M = torch.normal(0, 1, size=(4,4))
print('a single matrix \n',M)
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))

print('after multiplying 100 matrices\n', M)
```

```{.python .input}
#@tab tensorflow
M = tf.random.normal((4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = tf.matmul(M, tf.random.normal((4, 4)))

print('after multiplying 100 matrices\n', M.numpy())
```

### 打破对称

神经网络设计中的另一个问题是参数化过程中固有的对称性。假设我们有一个简单的MLP，有一个隐藏层和两个单元。在这种情况下，我们可以对第一层的权重$\mathbf{W}^{(1)}$进行置换，同样地，也可以对输出层的权重进行置换以获得相同的函数。第一个隐藏单元和第二个隐藏单元没有什么特别的区别。换句话说，我们在每一层的隐藏单元之间有排列对称性。

这不仅仅是理论上的麻烦。考虑前面提到的带有两个隐藏单元的一个隐藏层MLP。为了便于说明，假设输出层将两个隐藏单元转换为一个输出单元。我们想象一下，如果93层的某个常数被初始化了，会发生什么。在这种情况下，在前向传播过程中，任何一个隐藏单元采用相同的输入和参数，产生相同的激活，并将其馈送给输出单元。在反向传播期间，根据参数$\mathbf{W}^{(1)}$对输出单元进行微分，得到一个梯度，其元素都取相同的值。因此，在基于梯度的迭代（例如，小批量随机梯度下降）之后，$\mathbf{W}^{(1)}$的所有元素仍然取相同的值。这样的迭代永远不会打破对称性，我们可能永远也无法实现网络的表达能力。隐藏层的行为就好像只有一个单元。请注意，虽然小批量随机梯度下降不会打破这种对称性，辍学正则化会！

## 参数初始化

解决（或至少减轻）上述问题的一种方法是通过仔细初始化。优化过程中的额外注意和适当的正则化可以进一步提高稳定性。

### 默认初始化

在前面的章节中，例如在:numref:`sec_linear_concise`中，我们使用正态分布来初始化权重值。如果我们不指定初始化方法，框架将使用默认的随机初始化方法，对于中等规模的问题，这种方法通常很有效。

### Xavier初始化
:label:`subsec_xavier`

让我们看看某个完全连接层的输出（例如，隐藏变量）$o_{i}$的比例分布
*没有非线性*。
对于该层$n_\mathrm{in}$输入$x_j$及其相关权重$w_{ij}$，输出由

$$o_{i} = \sum_{j=1}^{n_\mathrm{in}} w_{ij} x_j.$$

重量$w_{ij}$都是独立于同一分布绘制的。此外，我们假设这个分布具有零均值和方差$\sigma^2$。注意，这并不意味着分布必须是高斯分布，只是平均值和方差必须存在。现在，让我们假设层$x_j$的输入也具有零均值和方差$\gamma^2$，并且它们独立于$w_{ij}$并且彼此独立。在这种情况下，我们可以计算$o_i$的平均值和方差，如下所示：

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\mathrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\mathrm{in}} E[w_{ij}] E[x_j] \\&= 0, \\
    \mathrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\mathrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

保持方差不变的一种方法是设置$n_\mathrm{in} \sigma^2 = 1$。现在考虑反向传播。在那里，我们面临着一个类似的问题，尽管梯度是从离输出更近的层传播的。使用与正向传播相同的推理，我们可以看到梯度的方差可以放大，除非$n_\mathrm{out} \sigma^2 = 1$，其中$n_\mathrm{out}$是该层的输出数量。这使我们陷入两难境地：我们不可能同时满足这两个条件。相反，我们只需满足：

$$
\begin{aligned}
\frac{1}{2} (n_\mathrm{in} + n_\mathrm{out}) \sigma^2 = 1 \text{ or equivalently }
\sigma = \sqrt{\frac{2}{n_\mathrm{in} + n_\mathrm{out}}}.
\end{aligned}
$$

这就是现在标准且实用的Xavier初始化*的基础，它以其创建者:cite:`Glorot.Bengio.2010`的第一作者命名。通常，Xavier初始化从均值和方差为零的高斯分布中采样权重$\sigma^2 = \frac{2}{n_\mathrm{in} + n_\mathrm{out}}$。我们也可以利用Xavier的直觉来选择从均匀分布中抽样权重时的方差。注意均匀分布$U(-a, a)$的方差为$\frac{a^2}{3}$。将$\frac{a^2}{3}$插入到$\sigma^2$的条件中，可以根据

$$U\left(-\sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}, \sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}\right).$$

虽然上述数学推理中不存在非线性的假设在神经网络中很容易被违背，但Xavier初始化方法在实际应用中效果良好。

### 超越

上面的推理仅仅触及了参数初始化的现代方法的表面。深度学习框架通常实现十几种不同的启发式方法。此外，参数初始化一直是深度学习基础研究的热点。其中包括专门用于绑定（共享）参数、超分辨率、序列模型和其他情况的启发式方法。例如，肖等。通过使用精心设计的初始化方法:cite:`Xiao.Bahri.Sohl-Dickstein.ea.2018`，证明了无需构造技巧训练10000层神经网络的可能性。

如果您对该主题感兴趣，我们建议您深入研究本模块的内容，阅读提出并分析每种启发式方法的论文，然后探索有关该主题的最新出版物。也许你会偶然发现甚至发明一个聪明的想法，并为深度学习框架提供一个实现。

## 摘要

* 梯度消失和爆炸是深部网络中常见的问题。在参数初始化时需要非常小心，以确保梯度和参数保持良好的控制。
* 需要初始化启发式来确保初始梯度既不太大也不太小。
* ReLU激活函数缓解了消失梯度问题。这可以加速收敛。
* 随机初始化是保证优化前对称性被破坏的关键。
* Xavier初始化表明，对于每一层，任何输出的方差不受输入数目的影响，任何梯度的方差不受输出数量的影响。

## 练习

1. 你能设计出其他的情况吗？除了MLP层的排列对称性之外，神经网络可能会表现出需要破坏的对称性吗？
1. 我们是否可以将线性回归或softmax回归中的所有权重参数初始化为相同的值？
1. 求两个矩阵乘积特征值的解析界。这说明了如何确保渐变条件良好？
1. 如果我们知道有些术语有分歧，我们能在事后解决吗？看看关于分层自适应速率缩放的论文:cite:`You.Gitman.Ginsburg.2017`。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/235)
:end_tab:
