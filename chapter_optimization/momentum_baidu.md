# 动量
:label:`sec_momentum`

在:numref:`sec_sgd`中，我们回顾了当执行随机梯度下降时会发生什么，即当执行优化时，只有一个噪声的梯度变量可用。特别是，我们注意到，对于噪声梯度，在面对噪声时选择学习速率时，我们需要格外谨慎。如果我们减少得太快，收敛就会停滞。如果我们太过宽容，我们就无法收敛到一个足够好的解，因为噪音不断地把我们从最优性中赶走。

## 基础

在本节中，我们将探讨更有效的优化算法，特别是对于实践中常见的某些类型的优化问题。

### 漏报平均数

在上一节中，我们讨论了小批量SGD作为加速计算的一种手段。它还有一个很好的副作用，平均梯度减少了方差的数量。小批量SGD可通过以下公式计算：

$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

为了保持符号简单，这里我们使用$\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$作为样本$i$的SGD，使用在$t-1$时更新的权重。如果我们能从方差减少的效果中获益，甚至超过对小批量的平均梯度，那就太好了。完成此任务的一个选择是用“泄漏平均值”替换梯度计算：

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$

大约$\beta \in (0, 1)$人。这有效地将瞬时梯度替换为多个*过去*梯度的平均值。$\mathbf{v}$被称为*动量*。它积累了过去的梯度类似于一个沉重的球滚动下来的目标函数景观整合过去的力量。为了更详细地了解正在发生的事情，让我们递归地将$\mathbf{v}_t$扩展到

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$

大$\beta$相当于一个长期平均值，而小$\beta$相当于一个相对于梯度法的轻微修正。新的梯度替换不再指向某一特定实例的最陡下降方向，而是指向过去梯度的加权平均方向。这使我们能够实现对一批进行平均的大部分好处，而不需要实际计算梯度。稍后我们将更详细地重新讨论这个平均过程。

上面的推理形成了现在被称为“加速”梯度法的基础，比如动量梯度法。在优化问题病态的情况下（例如，在某些方向上，进度比其他方向慢得多，类似于狭窄的峡谷），它们还享有更有效的额外好处。此外，它们允许我们对随后的梯度进行平均，以获得更稳定的下降方向。事实上，即使对于无噪声的凸问题，加速度也是动量能起作用的关键原因之一。

正如人们所期望的，由于动量的有效性，它是一个深入学习和超越优化研究的课题。如需深入分析和交互式动画，请参阅美丽的[说明性文章]（https://distill.pub/2017/momentum/) by :cite:`Goh.2017`）。它是由:cite:`Polyak.1964`提出的。:cite:`Nesterov.2018`在凸优化的背景下进行了详细的理论讨论。很长一段时间以来，深度学习的动力一直被认为是有益的。详见:cite:`Sutskever.Martens.Dahl.ea.2013`的讨论。

### 病态的问题

为了更好地理解动量法的几何性质，我们重新讨论了梯度下降法，尽管目标函数不太令人愉快。回想一下，在:numref:`sec_gd`中，我们使用了$f(\mathbf{x}) = x_1^2 + 2 x_2^2$，即一个中等扭曲的椭球物镜。我们通过在$x_1$方向上通过

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

与之前一样，$f$的最小值为$(0, 0)$。此函数在$x_1$方向上非常平坦。让我们看看在这个新函数上执行梯度下降时会发生什么。我们选择的学习率为$0.4$。

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

通过构造，$x_2$方向的坡度比水平$x_1$方向的坡度高得多，变化也快得多。因此，我们被困在两个不受欢迎的选择之间：如果我们选择一个小的学习率，我们可以确保解决方案不会在$x_2$方向发散，但我们在$x_1$方向的收敛速度缓慢。相反地，在学习率较高的情况下，我们在$x_1$方向上进展很快，但在$x_2$方向上出现了分歧。下面的例子说明了即使在学习率从$0.4$略微提高到$0.6$之后会发生什么。$x_1$方向的收敛性有所提高，但整体解的质量要差得多。

```{.python .input}
#@tab all
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

### 动量法

动量法使我们能够解决上述的梯度下降问题。从上面的优化轨迹来看，我们可能会直觉地认为，平均过去的梯度效果会很好。毕竟，在$x_1$方向上，这将聚集良好对齐的梯度，从而增加我们每一步覆盖的距离。相反，在梯度振荡的$x_2$方向，由于相互抵消的振荡，聚合梯度将减小步长。使用$\mathbf{v}_t$代替梯度$\mathbf{g}_t$产生以下更新方程：

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

请注意，对于$\beta = 0$，我们恢复常规梯度下降。在深入研究数学性质之前，让我们快速了解一下算法在实践中的表现。

```{.python .input}
#@tab all
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

正如我们所看到的，即使学习速度和以前一样，动量仍然收敛得很好。让我们看看当我们减小动量参数时会发生什么。将其减半至$\beta = 0.25$将导致几乎不收敛的轨迹。尽管如此，这比没有动力要好得多（当解决方案出现分歧时）。

```{.python .input}
#@tab all
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

请注意，我们可以结合动量与新元，特别是，小批量新元。唯一的变化是，在这种情况下，我们将梯度$\mathbf{g}_{t, t-1}$替换为$\mathbf{g}_t$。最后，为了方便起见，我们在时间$t=0$初始化$\mathbf{v}_0 = 0$。让我们看看漏平均对更新的实际影响。

### 有效样品重量

回想一下$\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$。在限额中，这些条款加起来是$\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$。换言之，在GD或SGD中，我们采取的不是$\eta$大小的步骤，而是$\frac{\eta}{1-\beta}$大小的步骤，同时处理可能表现更好的下降方向。这是两个好处合一。为了说明$\beta$的不同选择的加权行为，请考虑下图。

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

## 实际实验

让我们看看动量在实践中是如何工作的，即在适当的优化器上下文中使用动量时。为此，我们需要一个更具伸缩性的实现。

### 从头开始实施

与小批量SGD相比，动量法需要保持一组辅助变量，即速度。它的形状与梯度（以及优化问题的变量）相同。在下面的实现中，我们称这些变量为`states`。

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

当我们将动量超参数`momentum`增加到0.9时，它相当于一个更大的有效样本量$\frac{1}{1 - 0.9} = 10$。我们将学习率略微降低到$0.01$，以控制事态发展。

```{.python .input}
#@tab all
train_momentum(0.01, 0.9)
```

降低学习率进一步解决了任何非光滑优化问题。将其设置为$0.005$将产生良好的收敛特性。

```{.python .input}
#@tab all
train_momentum(0.005, 0.9)
```

### 简明实现

因为标准的`sgd`解算器已经内置了动量，所以在胶子中几乎没有什么可以做的。设置匹配参数会产生非常相似的轨迹。

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

到目前为止，$f(x) = 0.1 x_1^2 + 2 x_2^2$的2D示例似乎相当做作。我们现在将看到，这实际上是相当有代表性的类型的问题，一个人可能会遇到，至少在最小化凸二次目标函数的情况下。

### 二次凸函数

考虑函数

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$

这是一个一般的二次函数。对于正定矩阵$\mathbf{Q} \succ 0$，即，对于具有正特征值的矩阵，这在$\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$处具有最小值$b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$。因此，我们可以将$h$重写为

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$

坡度由$\partial_{\mathbf{x}} f(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$给出。也就是说，它由$\mathbf{x}$和最小值之间的距离乘以$\mathbf{Q}$得到。因此动量也是项$\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$的线性组合。

由于$\mathbf{Q}$是正定的，因此可以通过$\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$将其分解为正特征值的正交（旋转）矩阵$\mathbf{O}$和对角矩阵$\boldsymbol{\Lambda}$的特征系统。这允许我们将变量从$\mathbf{x}$更改为$\mathbf{z} := \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$，以获得一个非常简化的表达式：

$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$

这里是$c' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$。由于$\mathbf{O}$只是一个正交矩阵，这不会以一种有意义的方式扰动梯度。表示为$\mathbf{z}$梯度下降

$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$

这个表达式中的重要事实是梯度下降*不会在不同的特征空间之间混合。也就是说，当用$\mathbf{Q}$的特征系统表示时，优化问题以坐标方式进行。这也有助于保持势头。

$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$

本文证明了一个定理：凸二次函数的带动量和不带动量的梯度下降分解为二次矩阵特征向量方向上的坐标优化。

### 标量函数

给出上述结果，让我们看看当我们最小化函数$f(x) = \frac{\lambda}{2} x^2$时会发生什么。对于梯度下降，我们有

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$

每当$|1 - \eta \lambda| < 1$这个优化收敛于一个指数率，因为$t$步骤后，我们有$x_t = (1 - \eta \lambda)^t x_0$。这显示了当我们增加学习率$\eta$直到$\eta \lambda = 1$时，收敛速度是如何最初提高的。除此之外，事情的分歧和$\eta \lambda > 2$的优化问题分歧。

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

为了分析动量情况下的收敛性，我们首先用两个标量重写更新方程：一个用于$x$，一个用于$v$。这将产生：

$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$

我们使用$\mathbf{R}$来表示$2 \times 2$控制收敛行为。在$t$步骤之后，初始选择$[v_0, x_0]$变为$\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$。因此，由$\mathbf{R}$的特征值决定收敛速度。请参阅[District post]（https://distill.pub/2017/momentum/) of :cite:`Goh.2017`）以获得一个出色的动画，:cite:`Flammarion.Bach.2015`以获得详细的分析。可以证明$0 < \eta \lambda < 2 + 2 \beta$动量收敛。与梯度下降的$0 < \eta \lambda < 2$相比，这是一个更大的可行参数范围。它还表明，一般而言，$\beta$的大值是可取的。进一步的细节需要大量的技术细节，我们建议感兴趣的读者查阅原始出版物。

## 摘要

* 动量用过去梯度的漏平均值代替梯度。这大大加快了收敛速度。
* 它既适用于无噪声梯度下降，也适用于（噪声）随机梯度下降。
* 动量防止了优化过程的停滞，这对于随机梯度下降更可能发生。
* 由于对过去数据的指数降权，有效的梯度数由$\frac{1}{1-\beta}$给出。
* 在凸二次问题的情况下，这可以显式地详细分析。
* 实现非常简单，但它需要我们存储一个额外的状态向量（momentum $\mathbf{v}$）。

## 练习

1. 使用动量超参数和学习率的其他组合，观察和分析不同的实验结果。
1. 对一个有多重特征值的二次型问题，比如$f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$，比如$\lambda_i = 2^{-i}$，试试GD和动量。绘制$x$的值在初始化$x_i = 1$时如何减小。
1. 导出$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$的最小值和最小值。
1. 当我们用动量执行SGD时会发生什么变化？当我们使用带动量的小批量SGD时会发生什么？试验参数？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/354)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1070)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1071)
:end_tab:
