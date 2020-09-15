# 数值稳定性和初始化
:label:`sec_numerical_stability`

到目前为止，我们实现的每个模型都要求我们根据某个预先指定的分布来初始化它的参数。到目前为止，我们认为初始化方案是理所当然的，忽略了这些选择是如何做出的细节。你甚至可能会得到这样的印象：这些选择并不是特别重要。相反，初始化方案的选择在神经网络学习中起着非常重要的作用，对保持数值稳定性至关重要。此外，这些选择可以通过选择非线性激活函数以有趣的方式捆绑在一起。我们选择哪个函数以及如何初始化参数可以决定我们的优化算法收敛的速度有多快。这里的糟糕选择可能会导致我们在训练时遇到爆炸或消失的梯度。在本节中，我们将更详细地研究这些主题，并讨论一些有用的启发式方法，您会发现这些启发式方法在您的整个深度学习生涯中都很有用。

## 消失和爆炸梯度

考虑一个具有$L$层、输入$\mathbf{x}$和输出$\mathbf{o}$的深层网络。每一层$l$由变换$f_l$定义，该变换由权重$\mathbf{W}^{(l)}$参数化，其隐藏变量是$\mathbf{h}^{(l)}$(令$\mathbf{h}^{(0)} = \mathbf{x}$)，我们的网络可以表示为：

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \text{ and thus } \mathbf{o} = f_L \circ \ldots \circ f_1(\mathbf{x}).$$

如果所有隐藏变量和输入都是向量，我们可以将$\mathbf{o}$相对于任何一组参数$\mathbf{W}^{(l)}$的梯度写如下：

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\mathrm{def}}{=}} \cdot \ldots \cdot \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\mathrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\mathrm{def}}{=}}.$$

换言之，该梯度是$L-l$个矩阵$\mathbf{M}^{(L)} \cdot \ldots \cdot \mathbf{M}^{(l+1)}$和梯度向量$\mathbf{v}^{(l)}$的乘积。因此，我们容易受到相同的数值下溢问题的影响，当将太多的概率乘在一起时，这些问题经常会出现。在处理概率时，一个常见的技巧是切换到对数空间，即将压力从尾数转移到数值表示的指数。不幸的是，我们上面的问题更为严重：最初，矩阵$\mathbf{M}^{(l)}$可能具有各种各样的特征值。他们可能很小，也可能很大，他们的产品可能“非常大”，也可能“非常小”。

不稳定梯度带来的风险超出了数值表示的范围。不可预测的梯度也威胁到我们优化算法的稳定性。我们可能面临的参数更新要么(I)过大，破坏了我们的模型(*爆炸梯度*问题)；要么(Ii)过小(*消失梯度*问题)，使得学习变得不可能，因为参数几乎不会在每次更新时移动。

### 消失梯度

导致消失梯度问题的一个常见罪魁祸首是选择附加在每层的线性运算之后的激活函数$\sigma$。从历史上看，S形函数$1/(1 + \exp(-x))$(:numref:`sec_mlp`引入)很流行，因为它类似于阈值函数。由于早期的人工神经网络受到生物神经网络的启发，神经元要么“完全”激发，要么“完全不激发”(就像生物神经元)的想法似乎很有吸引力。让我们仔细看看乙状窦，看看它为什么会导致渐变消失。

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

正如你所看到的，无论是当它的输入很大，还是当它们很小时，乙状窦的梯度都会消失。此外，当反向传播通过许多层时，除非我们在金发地带，在那里许多S型线的输入接近于零，否则整个乘积的梯度可能会消失。当我们的网络夸耀有很多层时，除非我们小心，否则梯度很可能会在某一层被切断。事实上，这个问题曾经困扰着深度网络培训。因此，更稳定(但在神经上不太可信)的RELU已经成为从业者的默认选择。

### 爆炸梯度

相反的问题，当梯度爆炸时，可能同样令人烦恼。为了更好地说明这一点，我们绘制了100个高斯随机矩阵，并将它们与一些初始矩阵相乘。对于我们选择的比例(选择方差$\sigma^2=1$)，矩阵乘积爆炸。当由于深度网络的初始化而发生这种情况时，我们没有机会让梯度下降优化器收敛。

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

神经网络设计中的另一个问题是其参数化所固有的对称性。假设我们有一个简单的MLP，它有一个隐藏层和两个单元。在这种情况下，我们可以对第一层的权重$\mathbf{W}^{(1)}$进行置换，并且同样对输出层的权重进行置换以获得相同的函数。第一隐藏单元与第二隐藏单元没有什么特别的区别。换句话说，我们在每一层的隐藏单元之间具有排列对称性。

这不仅仅是理论上的麻烦。考虑前述具有两个隐藏单元的单隐藏层MLP。为便于说明，假设输出层将两个隐藏单元转换为仅一个输出单元。想象一下，如果我们将隐藏层的所有参数初始化为$\mathbf{W}^{(1)} = c$，而某个常量为$c$，会发生什么情况。在这种情况下，在前向传播期间，两个隐藏单元采用相同的输入和参数，产生相同的激活，该激活被馈送到输出单元。在反向传播期间，相对于参数$\mathbf{W}^{(1)}$对输出单元进行微分给出其元素全部取相同值的梯度。因此，在基于梯度的迭代(例如，小批量随机梯度下降)之后，$\mathbf{W}^{(1)}$的所有元素仍然采用相同的值。这样的迭代本身永远不会“打破对称性”，我们可能永远不会意识到网络的表现力。隐藏层的行为就像它只有一个单元一样。请注意，虽然小批量随机梯度下降不会打破这种对称性，但辍学正则化将打破这种对称性！

## 参数初始化

解决-或者至少缓解-上面提出的问题的一种方法是通过仔细的初始化。优化过程中的额外注意和适当的规则化可以进一步提高稳定性。

### 默认初始化

在前面的部分中，例如在:numref:`sec_linear_concise`中，我们使用正态分布来初始化权重值。如果我们不指定初始化方法，框架将使用默认的随机初始化方法，这种方法在实践中通常适用于中等大小的问题。

### Xavier初始化
:label:`subsec_xavier`

让我们看一下某些完全连接层的输出(例如，隐藏变量)$o_{i}$的比例分布
*没有非线性*。
利用该层的$n_\mathrm{in}$个输入$x_j$及其关联权重$w_{ij}$，通过以下方式给出输出

$$o_{i} = \sum_{j=1}^{n_\mathrm{in}} w_{ij} x_j.$$

权重$w_{ij}$都独立于相同的分布绘制。此外，让我们假设该分布具有零均值和方差$\sigma^2$。请注意，这并不意味着分布必须是高斯的，只是均值和方差需要存在。现在，让我们假设层$x_j$的输入也具有零均值和方差$\gamma^2$，并且它们独立于$w_{ij}$并且彼此独立。在这种情况下，我们可以按如下方式计算$o_i$的平均值和方差：

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\mathrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\mathrm{in}} E[w_{ij}] E[x_j] \\&= 0, \\
    \mathrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\mathrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

保持方差固定的一种方法是设置$n_\mathrm{in} \sigma^2 = 1$。现在考虑反向传播。在那里，我们面临着类似的问题，尽管梯度是从更靠近输出的层传播的。使用与正向传播相同的推理，我们可以看到，除非达到$n_\mathrm{out} \sigma^2 = 1$，否则梯度的方差可能会增大，其中$n_\mathrm{out}$是该层的输出数。这使我们进退两难：我们不可能同时满足这两个条件。相反，我们只是试图满足：

$$
\begin{aligned}
\frac{1}{2} (n_\mathrm{in} + n_\mathrm{out}) \sigma^2 = 1 \text{ or equivalently }
\sigma = \sqrt{\frac{2}{n_\mathrm{in} + n_\mathrm{out}}}.
\end{aligned}
$$

这就是以其创建者:cite:`Glorot.Bengio.2010`的第一作者的名字命名的现在标准且实际有益的“泽维尔初始化”背后的推理。通常，泽维尔初始化从具有零均值和零方差的高斯分布中采样权重$\sigma^2 = \frac{2}{n_\mathrm{in} + n_\mathrm{out}}$。当从均匀分布中抽样权重时，我们也可以采用Xavier的直觉来选择方差。注意，均匀分布$U(-a, a)$具有方差$\frac{a^2}{3}$。将$\frac{a^2}{3}$插入我们在$\sigma^2$上的条件会产生根据以下内容进行初始化的建议

$$U\left(-\sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}, \sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}\right).$$

虽然上述数学推理中不存在非线性的假设在神经网络中很容易被违反，但Xavier初始化方法在实践中证明是有效的。

### 超越

上面的推理仅仅触及了现代参数初始化方法的皮毛。深度学习框架通常实现十几种不同的启发式方法。此外，参数初始化一直是深度学习基础研究的热点领域。其中包括专门用于绑定(共享)参数、超分辨率、序列模型和其他情况的启发式算法。例如，肖等人。演示了通过使用精心设计的初始化方法10000来训练无体系结构技巧的:cite:`Xiao.Bahri.Sohl-Dickstein.ea.2018`层神经网络的可能性。

如果您对该主题感兴趣，我们建议您深入研究本模块提供的内容，阅读提出并分析每个启发式方法的论文，然后浏览有关该主题的最新出版物。也许您会偶然发现甚至发明一个聪明的想法，并为深度学习框架贡献一个实现。

## 摘要

* 梯度的消失和爆炸是深层网络中普遍存在的问题。参数初始化需要非常小心，以确保梯度和参数保持良好控制。
* 需要初始化试探法来确保初始梯度既不太大也不太小。
* REU激活函数缓解了消失梯度问题。这可以加速融合。
* 随机初始化是确保在优化之前对称被打破的关键。
* Xavier初始化表明，对于每一层，任何输出的方差不受输入数目的影响，任何梯度的方差也不受输出数目的影响。

## 练习

1. 除了MLP层中的排列对称性之外，您还能设计出神经网络可能表现出对称性需要破缺的其他情况吗？
1. 我们可以将线性回归或Softmax回归中的所有权重参数初始化为相同的值吗？
1. 查找两个矩阵乘积的特征值的解析界。这对确保渐变条件良好有什么启示？
1. 如果我们知道有些条款有分歧，我们能在事后解决这个问题吗？请参阅有关分层自适应速率缩放的论文，了解Inspiration :cite:`You.Gitman.Ginsburg.2017`。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/235)
:end_tab:
