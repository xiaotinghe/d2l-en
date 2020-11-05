# 微积分
:label:`sec_calculus`

直到至少2500年前，古希腊人把一个多边形分成三角形，并把它们的面积相加，才发现多边形的面积。为了找到弯曲形状的面积，例如圆，古希腊人将多边形刻成这样的形状。如:numref:`fig_circle_area`所示，具有更多等长边的内接多边形更接近圆。这个过程也被称为*耗尽法*。

![Find the area of a circle with the method of exhaustion.](../img/polygon-circle.svg)
:label:`fig_circle_area`

事实上，穷举法就是积分学*（将在:numref:`sec_integral_calculus`中描述）的起源。2000多年后，微积分的另一个分支，微积分*诞生了。在微分学最关键的应用中，最优化问题考虑的是如何做到最好。正如:numref:`subsec_norms_and_objectives`中所讨论的，这些问题在深度学习中普遍存在。

在深度学习中，我们*训练*模型，不断地更新它们，以便它们在看到越来越多的数据时变得越来越好。通常，变好意味着最小化损失函数，这个分数可以回答“我们的模型有多糟糕？”这个问题比看上去更微妙。归根结底，我们真正关心的是生成一个在我们从未见过的数据上表现良好的模型。但我们只能将模型与我们实际看到的数据相匹配。（2）我们可以将数学模型的有效性扩展到所观察到的数据集中，并以此来指导我们的数学模型的有效性*：我们可以把这些模型的有效性扩展到所观察到的数据集中。

为了帮助您理解后面章节中的优化问题和方法，这里我们提供了一个非常简短的微分学入门，这是在深入学习中常用的。

## 导数与微分

我们首先讨论导数的计算，这是几乎所有深度学习优化算法的关键步骤。在深度学习中，我们通常选择相对于模型参数可微的损失函数。简单地说，这意味着，对于每个参数，我们可以确定损失增加或减少的速度，如果我们把这个参数增加*或*减少*一个无穷小的量。

假设我们有一个函数$f: \mathbb{R} \rightarrow \mathbb{R}$，它的输入和输出都是标量。$f$的*导数*定义为

$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h},$$
:eqlabel:`eq_derivative`

如果这个限制存在。如果$f'(a)$存在，则$f$在$a$处被称为*可微*。如果$f$在一个区间的每一个数处都是可微的，那么这个函数在这个区间上是可微的。我们可以将:eqref:`eq_derivative`中的导数$f'(x)$解释为$f(x)$相对于$x$的*瞬时*变化率。所谓的瞬时变化率是基于$x$中$h$的变化，接近$0$。

为了说明导数，让我们用一个例子来做实验。定义$u = f(x) = 3x^2-4x$。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

通过设置$x=1$并让732293612接近$0$，:eqref:`eq_derivative`中$\frac{f(x+h) - f(x)}{h}$的数值结果接近$2$。虽然这个实验不是一个数学证明，但我们稍后会看到，当$x=1$时，导数$u'$是$u'$。

```{.python .input}
#@tab all
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1
```

让我们熟悉一下导数的几个等价符号。给定$y = f(x)$，其中$x$和$y$分别是函数$f$的自变量和因变量。以下表达式等效：

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

其中，符号$\frac{d}{dx}$和$D$是*微分运算符*，表示*微分*的操作。我们可以使用以下规则来区分常用函数：

* $DC = 0$（$C$是一个常数），
* $Dx^n = nx^{n-1}$（*幂律*，$n$是任何实数），
* $De^x = e^x$,
* $D\ln(x) = 1/x.$

为了区分一个函数是由一些更简单的函数组成的，例如上面的常见函数，下面的规则对我们很方便。假设函数$f$和$g$都是可微的，$C$是常数，我们有*常数倍数规则*

$$\frac{d}{dx} [Cf(x)] = C \frac{d}{dx} f(x),$$

*求和规则*

$$\frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx} f(x) + \frac{d}{dx} g(x),$$

*产品规则*

$$\frac{d}{dx} [f(x)g(x)] = f(x) \frac{d}{dx} [g(x)] + g(x) \frac{d}{dx} [f(x)],$$

商法则*

$$\frac{d}{dx} \left[\frac{f(x)}{g(x)}\right] = \frac{g(x) \frac{d}{dx} [f(x)] - f(x) \frac{d}{dx} [g(x)]}{[g(x)]^2}.$$

现在我们可以应用上面的一些规则来找到$u' = f'(x) = 3 \frac{d}{dx} x^2-4\frac{d}{dx}x = 6x-4$。因此，通过设置$x = 1$，我们得到了$u' = 2$：这一点得到了我们在本节中的早期实验的支持，其中数值结果接近$2$。此导数也是$x = 1$时曲线$u = f(x)$切线的斜率。

为了可视化对导数的这种解释，我们将使用`matplotlib`，这是Python中一个流行的绘图库。为了配置`matplotlib`生成的图形的属性，我们需要定义几个函数。在下面，`use_svg_display`函数指定`matplotlib`包以输出svg图形以获得更清晰的图像。

```{.python .input}
#@tab all
def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')
```

我们定义`set_figsize`函数来指定图形大小。请注意，这里我们直接使用`d2l.plt`，因为进口声明`from matplotlib import pyplot as plt`已标记为保存在前言中的`d2l`包中。

```{.python .input}
#@tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

下面的`set_axes`函数设置`matplotlib`生成的图形的轴的属性。

```{.python .input}
#@tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

有了这三个用于图形配置的函数，我们定义了`plot`函数来简洁地绘制多条曲线，因为我们需要在整本书中可视化许多曲线。

```{.python .input}
#@tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

现在我们可以把函数$u = f(x)$和它的切线$y = 2x - 3$画在$x=1$处，其中系数$2$是切线的斜率。

```{.python .input}
#@tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## 偏导数

到目前为止，我们只讨论了一个变量函数的微分。在学习中，函数往往依赖于很多变量。因此，我们需要将微分的思想推广到这些多元函数上。

设$y = f(x_1, x_2, \ldots, x_n)$是一个有$n$个变量的函数。$y$相对于其$i^\mathrm{th}$参数$x_i$的*偏导数*为

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$

为了计算$\frac{\partial y}{\partial x_i}$，我们可以简单地将$x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$作为常数，并计算$y$相对于$x_i$的导数。对于偏导数的表示法，以下内容等效：

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

## 坡度

我们可以将一个多元函数的所有变量的偏导数串联起来，得到函数的梯度向量。假设函数$f: \mathbb{R}^n \rightarrow \mathbb{R}$的输入是$n$维向量$\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$，输出是标量。函数$f(\mathbf{x})$相对于$\mathbf{x}$的梯度是$n$偏导数的向量：

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,$$

其中$\nabla_{\mathbf{x}} f(\mathbf{x})$通常在没有歧义的情况下被$\nabla f(\mathbf{x})$代替。

设$\mathbf{x}$为$n$维向量，在对多元函数进行微分时，通常使用以下规则：

* 所有$\mathbf{A} \in \mathbb{R}^{m \times n}$、$\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$，
* 所有$\mathbf{A} \in \mathbb{R}^{n \times m}$、$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$，
* 所有$\mathbf{A} \in \mathbb{R}^{n \times n}$、$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$，
* $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$.

同样，对于任何矩阵$\mathbf{X}$，我们有$\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$。我们将在后面看到，梯度对于设计深度学习中的优化算法非常有用。

## 链式规则

然而，这样的梯度很难找到。这是因为深度学习中的多元函数通常是*复合的，所以我们可能不应用上述任何规则来区分这些函数。幸运的是，*链规则*使我们能够区分复合函数。

让我们首先考虑单变量的函数。假设函数$y=f(u)$和$u=g(x)$都是可微的，那么链式规则说明

$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$

现在让我们把注意力转向一个更一般的场景，其中函数有任意数量的变量。假设可微函数$y$有变量$u_1, u_2, \ldots, u_m$，其中每个可微函数$u_i$都有变量$x_1, x_2, \ldots, x_n$。注意$y$是$x_1, x_2, \ldots, x_n$的函数。然后链式法则给出

$$\frac{dy}{dx_i} = \frac{dy}{du_1} \frac{du_1}{dx_i} + \frac{dy}{du_2} \frac{du_2}{dx_i} + \cdots + \frac{dy}{du_m} \frac{du_m}{dx_i}$$

任何$i = 1, 2, \ldots, n$。

## 摘要

* 微积分和微积分是微积分学的两个分支，微积分可以应用于深度学习中普遍存在的优化问题。
* 导数可以解释为函数相对于其变量的瞬时变化率。它也是函数曲线切线的斜率。
* 梯度是一个向量，其分量是多元函数对其所有变量的偏导数。
* 链式规则使我们能够区分复合函数。

## 练习

1. 当$x = 1$时，绘制函数$y = f(x) = x^3 - \frac{1}{x}$及其切线。
1. 求函数的梯度。
1. 函数$f(\mathbf{x}) = \|\mathbf{x}\|_2$的梯度是多少？
1. 你能写出$u = f(x, y, z)$和$u = f(x, y, z)$、$y = y(a, b)$和$z = z(a, b)$的链式法则吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/32)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/33)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/197)
:end_tab:
