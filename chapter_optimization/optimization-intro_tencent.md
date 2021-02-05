# 优化与深度学习

在这一部分，我们将讨论优化和深度学习之间的关系，以及在深度学习中使用优化的挑战。对于深度学习问题，我们通常首先定义一个损失函数。一旦我们有了损失函数，我们就可以使用优化算法来尝试使损失最小化。在优化中，损失函数通常被称为优化问题的目标函数。按照传统和惯例，大多数优化算法都关注“最小化”。如果我们需要最大化一个目标，有一个简单的解决方案：只需将目标上的符号反转即可。

## 优化估算

虽然优化为深度学习提供了一种最小化损失函数的方法，但从本质上讲，优化和深度学习的目标是完全不同的。前者主要关注最小化目标，而后者关注的是在给定有限数据量的情况下找到合适的模型。在:numref:`sec_model_selection`中，我们详细讨论了这两个目标之间的区别。例如，训练误差和泛化误差通常是不同的：由于优化算法的目标函数通常是基于训练数据集的损失函数，所以优化的目标是减少训练误差。然而，统计推理(因此也是深度学习)的目标是减少泛化误差。要实现后者，除了使用优化算法减少训练误差外，还需要注意过拟合。我们从为本章导入几个库开始。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

接下来，我们定义两个函数，期望函数$f$和经验函数$g$，以说明该问题。这里，$g$没有$f$平滑，因为我们只有有限的数据量。

```{.python .input}
#@tab all
def f(x): return x * d2l.cos(np.pi * x)
def g(x): return f(x) + 0.2 * d2l.cos(5 * np.pi * x)
```

下图说明训练误差的最小值可以位于与预期误差的最小值(或测试误差的最小值)不同的位置。

```{.python .input}
#@tab all
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('empirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('expected risk', (1.1, -1.05), (0.95, -0.5))
```

## 深度学习中的优化挑战

在这一章中，我们将特别关注优化算法在最小化目标函数方面的性能，而不是模型的泛化误差。在:numref:`sec_linear_regression`中，我们区分了最优化问题的解析解和数值解。在深度学习中，大多数目标函数比较复杂，没有解析解。相反，我们必须使用数值优化算法。下面的优化算法都属于这一类。

深度学习优化面临诸多挑战。其中一些最令人烦恼的问题是局部极小值、鞍点和消失梯度。让我们看看其中的几个。

### 本地最小映像

对于目标函数$f(x)$，如果在$x$的值$f(x)$小于在$x$附近的任何其他点的值$f(x)$，则$f(x)$可以是局部最小值。如果$f(x)$在$x$处的值是整个域上的目标函数的最小值，则$f(x)$是全局最小值。

例如，给定函数

$$f(x) = x \cdot \text{cos}(\pi x) \text{ for } -1.0 \leq x \leq 2.0,$$

我们可以近似该函数的局部最小值和全局最小值。

```{.python .input}
#@tab all
x = d2l.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```

深度学习模型的目标函数通常有许多局部最优解。当优化问题的数值解在局部最优附近时，当目标函数解的梯度趋于或变为零时，最终迭代得到的数值解可能只会局部最小化目标函数，而不是全局最小化。只有一定程度的噪声可能会使参数超出局部最小值。事实上，这是随机梯度下降的有益性质之一，其中小批量上的梯度的自然变化能够将参数从局部极小值中剔除。

### 鞍点

除了局部极小值，鞍点是渐变消失的另一个原因。[saddle point](https://en.wikipedia.org/wiki/Saddle_point)是函数的所有梯度消失但既不是全局最小值也不是局部最小值的任何位置。考虑函数$f(x) = x^3$。它的一阶和二阶导数以$x=0$英镑的价格消失。优化可能会在该点停止，即使这不是最低要求。

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```

更高维度中的鞍点甚至更隐蔽，如下面的示例所示。考虑函数$f(x, y) = x^2 - y^2$。它的鞍点在$(0, 0)$。对于$y$，这是最大值，对于$x$，这是最小值。此外，它“看起来”像一个马鞍，这就是这个数学性质得名的地方。

```{.python .input}
#@tab all
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

我们假设一个函数的输入是一个$k$维向量，它的输出是一个标量，那么它的海森矩阵将有$k$个特征值(参见:numref:`sec_geometry-linear-algebraic-ops`)。函数的解可以是函数梯度为零的位置处的局部最小值、局部最大值或鞍点：

* 当函数的海森矩阵在零梯度位置的特征值都为正时，函数存在局部最小值。
* 当函数的海森矩阵在零梯度位置的特征值都为负时，函数存在局部极大值。
* 当函数的海森矩阵在零梯度位置的特征值为负和正时，函数存在鞍点。

对于高维问题，至少部分特征值为负的可能性相当高。这使得鞍点比局部最小值更有可能出现。我们将在下一节介绍凸性时讨论这种情况的一些例外情况。简而言之，凸函数是黑森函数的特征值从不为负的函数。然而，可悲的是，大多数深度学习问题并不属于这一类。尽管如此，它仍然是研究优化算法的一个很好的工具。

### 消失梯度

可能遇到的最隐蔽的问题是渐变消失。例如，假设我们想要最小化函数$f(x) = \tanh(x)$，并且我们碰巧从$x = 4$开始。我们可以看到，$f$的梯度接近于零。更具体地说是$f'(x) = 1 - \tanh^2(x)$，因此是$f'(4) = 0.0013$。因此，在我们取得进展之前，优化将会停滞不前很长一段时间。这被证明是在引入REU激活功能之前训练深度学习模型相当棘手的原因之一。

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```

正如我们所看到的，深度学习的优化充满了挑战。幸运的是，存在一系列健壮的算法，它们执行得很好，即使对于初学者也很容易使用。此外，没有必要真的去寻找“最佳解决方案”。局部最优解甚至近似解仍然是非常有用的。

## 摘要

* 最小化训练误差并不能保证我们找到使预期误差最小化的最佳参数集。
* 优化问题可能存在许多局部极小值。
* 这个问题可能有更多的鞍点，因为问题通常不是凸的。
* 渐变消失可能会导致优化停止。通常，问题的重新参数化会有所帮助。参数的良好初始化也是有益的。

## 练习

1. 考虑一个简单的多层感知器，该感知器具有一个隐藏层，比方说，在隐藏层中有$d$维的单个隐藏层和单个输出。证明了对于任何局部最小值，至少存在行为相同的$d！$等价解。
1. 假设我们具有对称随机矩阵$\mathbf{M}$，其中条目$M_{ij} = M_{ji}$均取自某个概率分布$p_{ij}$。此外，假设$p_{ij}(x) = p_{ij}(-x)$，即分布是对称的(参见例如:cite:`Wigner.1958`的细节)。
    * 证明了特征值上的分布也是对称的。即，对于任何特征向量$\mathbf{v}$，关联的特征值$\lambda$满足$P(\lambda > 0) = P(\lambda < 0)$的概率。
    * 为什么上面的“不”意味着$P(\lambda > 0) = 0.5$？
1. 你还能想到深度学习优化还涉及哪些挑战？
1. 假设您想要在(真实的)鞍座上平衡一个(真实的)球。
    * 为什么这很难呢？
    * 你能在优化算法中也利用这个效应吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/349)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/487)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/489)
:end_tab:
