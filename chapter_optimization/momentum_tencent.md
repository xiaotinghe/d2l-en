# 动量
:label:`sec_momentum`

在:numref:`sec_sgd`中，我们回顾了执行随机梯度下降时会发生什么，即，在执行仅有噪声的梯度变体的优化时会发生什么。特别是，我们注意到，对于噪声梯度，当面对噪声选择学习率时，我们需要格外谨慎。如果我们把它减得太快，收敛就会停滞不前。如果我们过于宽松，我们就不能收敛到一个足够好的解决方案，因为噪音不断地驱使我们远离最优。

## 基础知识

在这一部分中，我们将探索更有效的优化算法，特别是针对实践中常见的某些类型的优化问题。

### 漏水平均数

在上一节中，我们讨论了作为加速计算的一种方法的小型批处理SGD。它还有一个很好的副作用，即平均梯度减少了方差。小批量SGD可通过以下方式计算：

$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

为了保持符号的简单性，这里我们使用$\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$作为样本$i$的sgd，使用在时间$t-1$更新的权重。如果我们能从减少方差的效果中获益，甚至超过平均小批量的梯度，那就太好了。完成此任务的一种选择是将渐变计算替换为“泄漏平均值”：

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$

大约$\beta \in (0, 1)$美元。这有效地将瞬时渐变替换为在多个“过去”渐变上平均的渐变。$\mathbf{v}$被称为*动量*。它累积过去的渐变，类似于一个沉重的球沿着目标函数景观滚落，如何在过去的力上整合。为了更详细地了解正在发生的情况，让我们递归地将$\mathbf{v}_t$扩展为

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$

较大的$\beta$相当于长期平均值，而较小的$\beta$只相当于相对于梯度法的轻微修正。新的梯度替换不再指向特定实例上的最陡下降方向，而是指向过去梯度的加权平均方向。这使我们能够实现在一批中求平均值的大部分好处，而无需实际计算其上的渐变。我们稍后将更详细地回顾此平均过程。

上述推理形成了现在所知的“加速*梯度法”的基础，例如动量梯度法。在优化问题是病态的情况下(即，有些方向的进展比另一些方向慢得多，类似于狭窄的峡谷)，它们还享受着更有效的额外好处(即在某些情况下，进展要比其他方向慢得多，就像一个狭窄的峡谷)。此外，它们还允许我们对随后的梯度进行平均，以获得更稳定的下降方向。事实上，即使对于无噪音的凸性问题，加速度方面也是动量起作用的关键原因之一，也是它起到如此好的作用的原因之一。

正如人们所预期的那样，由于它的有效性，动量是深度学习优化以及以后的一个很好的研究课题。例如，参见精美的[说明性文章](Https://distill.pub/2017/momentum/) by :cite:`Goh.2017`)以获取深入分析和互动动画。它是由:cite:`Polyak.1964`人提出的。:cite:`Nesterov.2018`在凸优化的背景下进行了详细的理论讨论。很长一段时间以来，深度学习的动力一直被认为是有益的。有关详细信息，请参见例如:cite:`Sutskever.Martens.Dahl.ea.2013`的讨论。

### 一个病态问题

为了更好地理解动量法的几何性质，我们重温了梯度下降，尽管目标函数明显不那么令人愉快。回想一下，在:numref:`sec_gd`中，我们使用了$f(\mathbf{x}) = x_1^2 + 2 x_2^2$，即适度扭曲的椭球物镜。我们通过在$x_1$方向上伸展该函数来进一步扭曲该函数

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

和以前一样，$f$的最低标准是$(0, 0)$。这个函数在$x_1$的方向是“非常”平坦的。让我们看看当我们像以前一样对这个新函数执行梯度下降时会发生什么。我们选择$0.4$的学习率。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

通过施工，$x_2$方向的坡度比水平$x_1$方向的坡度要高得多，变化也快得多。因此，我们在两个不受欢迎的选择之间左右为难：如果我们选择一个较小的学习率，我们将确保解决方案不会在$x_2$方向发散，但我们将背负着在$x_1$方向缓慢收敛的负担。相反，随着学习率的提高，我们在$x_1$的方向上进展很快，但在$x_2$的方向上出现了分歧。下面的例子说明了即使在学习率从$0.4$略微提高到$0.6$之后也会发生什么情况。$x_1$方向的收敛有所改善，但整体解决方案质量要差得多。

```{.python .input}
#@tab all
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

### 动量法

动量法使我们可以解决上述梯度下降问题。查看上面的优化轨迹，我们可能会直觉地认为，对过去的梯度进行平均会很有效。毕竟，在$x_1$方向上，这将聚合对齐良好的渐变，从而增加我们每一步覆盖的距离。相反，在梯度振荡的$x_2$方向上，由于相互抵消的振荡，聚合梯度将减小步长。使用$\mathbf{v}_t$而不是梯度$\mathbf{g}_t$产生以下更新公式：

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

请注意，对于$\beta = 0$，我们恢复了规则的梯度下降。在深入研究数学属性之前，让我们快速了解一下算法在实践中是如何运行的。

```{.python .input}
#@tab all
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

正如我们所看到的，即使有我们以前使用的相同的学习速度，势头仍然会很好地收敛。让我们看看当我们减小动量参数时会发生什么。将其减半至$\beta = 0.25$会导致轨迹几乎不收敛。尽管如此，这比没有动力(当解决方案出现分歧时)要好得多。

```{.python .input}
#@tab all
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

请注意，我们可以将Momentum与SGD相结合，特别是MiniBatch-SGD。唯一的变化是，在这种情况下，我们将渐变$\mathbf{g}_{t, t-1}$替换为$\mathbf{g}_t$。最后，为方便起见，我们在时间$\mathbf{v}_0 = 0$初始化$t=0$。让我们看看泄漏平均实际上对更新有什么影响。

### 有效样本量

回想一下那个$\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$。在限制中，条款加起来是$\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$。换句话说，我们不是在GD或SGD中采取$\eta$大小的步骤，而是采用$\frac{\eta}{1-\beta}$大小的步骤，同时处理可能表现得更好的下降方向。这两个好处合而为一。为了说明$\beta$的不同选项的加权行为，请看下图。

```{.python .input}
#@tab all
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

## 实践性实验

让我们看看动量在实践中是如何工作的，也就是说，当在适当的优化器的上下文中使用时。为此，我们需要更具伸缩性的实现。

### 从头开始实施

与(小批量)SGD相比，动量法需要保持一组辅助变量，即速度。它具有与梯度(和优化问题的变量)相同的形状。在下面的实现中，我们将这些变量称为`states`。

```{.python .input}
#@tab mxnet,pytorch
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)
```

```{.python .input}
#@tab tensorflow
def init_momentum_states(features_dim):
    v_w = tf.Variable(d2l.zeros((features_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    return (v_w, v_b)
```

```{.python .input}
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v
```

```{.python .input}
#@tab pytorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
            v[:].assign(hyperparams['momentum'] * v + g)
            p[:].assign(p - hyperparams['lr'] * v)
```

让我们看看这在实践中是如何运作的。

```{.python .input}
#@tab all
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

当我们将动量超参数`momentum`增加到0.9时，它相当于一个明显更大的有效样本大小$\frac{1}{1 - 0.9} = 10$。我们将学习率略微降低到$0.01$，以使情况得到控制。

```{.python .input}
#@tab all
train_momentum(0.01, 0.9)
```

降低学习率进一步解决了任何非平滑优化问题。将其设置为$0.005$会产生良好的收敛特性。

```{.python .input}
#@tab all
train_momentum(0.005, 0.9)
```

### 简明实施

胶子没有什么可做的，因为标准的`sgd`解算器已经内置了动量。设置匹配参数会产生非常相似的轨迹。

```{.python .input}
d2l.train_concise_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

## 理论分析

到目前为止，$f(x) = 0.1 x_1^2 + 2 x_2^2$的2D示例似乎相当做作。现在我们将看到，至少在最小化凸二次目标函数的情况下，这实际上相当代表人们可能遇到的问题的类型。

### 二次凸函数

考虑一下函数

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$

这是一个一般的二次函数。对于正定矩阵$\mathbf{Q} \succ 0$，即对于具有正特征值的矩阵，这在$\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$具有最小值$b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$的最小化。因此，我们可以将$h$重写为

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$

渐变是以$\partial_{\mathbf{x}} f(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$给出的。也就是说，它是由$\mathbf{x}$到最小化器之间的距离乘以$\mathbf{Q}$得出的。因此，动量也是项$\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$的线性组合。

由于$\mathbf{Q}$是正定的，因此对于正特征值的正交(旋转)矩阵$\mathbf{O}$和对角矩阵$\boldsymbol{\Lambda}$，可以通过$\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$将其分解成其特征系统。这允许我们将变量从$\mathbf{x}$更改为$\mathbf{z} := \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$，以获得一个非常简单的表达式：

$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$

这里是$c' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$。因为$\mathbf{O}$只是一个正交矩阵，所以这不会以有意义的方式干扰梯度。以$\mathbf{z}$的梯度下降表示为

$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$

这个表达式中的重要事实是梯度下降*不会在不同的特征空间之间混合。也就是说，当用$\mathbf{Q}$的特征系统表示时，优化问题以坐标方式进行。对于势头来说，这也是成立的。

$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$

在此过程中，我们证明了如下定理：凸二次函数有动量和无动量的梯度下降沿二次矩阵的特征向量方向分解为坐标方向的优化。

### 标量函数

给出上述结果，让我们看看当我们最小化函数$f(x) = \frac{\lambda}{2} x^2$时会发生什么。对于梯度下降，我们有

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$

无论何时$|1 - \eta \lambda| < 1$，此优化都会以指数速度收敛，因为在执行$t$步之后，我们就会得到$x_t = (1 - \eta \lambda)^t x_0$步。这显示了当我们将学习率提高$\eta$到$\eta \lambda = 1$时，收敛速度最初是如何提高的。除此之外，情况会有所不同，对于$\eta \lambda > 2$，优化问题也会有所不同。

```{.python .input}
#@tab all
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

为了分析动量情况下的收敛性，我们首先用两个标量重写更新方程：一个用于$x$，另一个用于动量$v$。这将产生：

$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$

我们使用$\mathbf{R}$表示$2 \times 2$的主导收敛行为。在$t$步之后，初始选择$[v_0, x_0]$变为$\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$。因此，决定收敛速度的是特征值$\mathbf{R}$。参见[蒸馏帖子](HTTPS://distill.pub/2017/momentum/) of :cite:`Goh.2017`查看精彩动画，:cite:`Flammarion.Bach.2015`查看详细分析。可以证明$0 < \eta \lambda < 2 + 2 \beta$的动量汇聚在一起。与梯度下降的$0 < \eta \lambda < 2$相比，这是一个更大的可行参数范围。它还表明，一般来说，$\beta$的大值是可取的。进一步的细节需要相当多的技术细节，我们建议感兴趣的读者查阅原始出版物。

## 摘要

* 动量用过去梯度上的漏水平均值代替梯度。这显著加快了融合速度。
* 这对于无噪声梯度下降和(有噪声)随机梯度下降都是理想的。
* 动量可以防止随机梯度下降更有可能发生的优化过程的停滞。
* 由于过去数据的指数减权，有效梯度数给出了$\frac{1}{1-\beta}$。
* 在凸二次问题的情况下，这可以被显式地详细地分析。
* 实现相当简单，但它需要我们存储一个额外的状态向量(动量$\mathbf{v}$)。

## 练习

1. 使用动量、超参数和学习率的其他组合，观察和分析不同的实验结果。
1. 试着用GD和Momentum解一个二次问题，其中你有多个特征值，即$f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$，例如$\lambda_i = 2^{-i}$。绘制初始化$x$的值如何降低的图$x_i = 1$。
1. 导出$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$的最小值和最小值。
1. 当我们有势头地执行SGD时，会有什么变化？当我们使用迷你批量SGD时会发生什么？用这些参数做实验？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/354)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1070)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1071)
:end_tab:
