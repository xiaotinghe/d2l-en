# 梯度下降
:label:`sec_gd`

在本节中，我们将介绍梯度下降的基本概念。这是必要的。如需有关凸优化的深入介绍，请参阅:cite:`Boyd.Vandenberghe.2004`。尽管后者很少直接用于深度学习，但了解梯度下降是理解随机梯度下降算法的关键。例如，由于学习速率过大，优化问题可能会出现分歧。这种现象已经可以在梯度下降中看到。同样地，预处理是梯度下降中的一种常用技术，并延续到更高级的算法中。让我们从一个简单的特例开始。

## 一维梯度下降

一维梯度下降是一个很好的例子来解释为什么梯度下降算法可以减少目标函数的值。考虑一类连续可微实值函数$f: \mathbb{R} \rightarrow \mathbb{R}$。利用泰勒展开式（:numref:`sec_single_variable_calculus`），我们得到

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

也就是说，在一次近似中，$f(x+\epsilon)$由函数值$f(x)$和$f'(x)$在$x$处的一阶导数$f'(x)$给出。假设小$\epsilon$在负梯度方向上移动将减少$f$是合理的。为了简单起见，我们选择固定步长$\eta > 0$，然后选择$\epsilon = -\eta f'(x)$。把这个代入泰勒展开式

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$

如果$f'(x) \neq 0$的衍生品没有消失，我们将从$\eta f'^2(x)>0$开始取得进展。此外，我们总是可以选择$\eta$小到足以使高阶项变得不相关。因此我们到达

$$f(x - \eta f'(x)) \lessapprox f(x).$$

这意味着，如果我们使用

$$x \leftarrow x - \eta f'(x)$$

要迭代$x$，函数$f(x)$的值可能会下降。因此，在梯度下降中，我们首先选择初始值$x$和常量$\eta > 0$，然后使用它们连续迭代$x$，直到达到停止条件，例如，当梯度$|f'(x)|$的幅度足够小或迭代次数达到某个值时。

为了简单起见，我们选择目标函数$f(x)=x^2$来说明如何实现梯度下降。虽然我们知道$x=0$是最小化$f(x)$的解决方案，但是我们仍然使用这个简单的函数来观察$x$的变化。一如既往，我们从导入所有必需的模块开始。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
f = lambda x: x**2  # Objective function
gradf = lambda x: 2 * x  # Its derivative
```

接下来，我们使用$x=10$作为初始值，并假设$\eta=0.2$。使用梯度下降法迭代$x$ 10次，我们可以看到，最终$x$的值接近最优解。

```{.python .input}
#@tab all
def gd(eta):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * gradf(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

res = gd(0.2)
```

优化$x$的进度可以绘制如下。

```{.python .input}
#@tab all
def show_trace(res):
    n = max(abs(min(res)), abs(max(res)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, res], [[f(x) for x in f_line], [f(x) for x in res]],
             'x', 'f(x)', fmts=['-', '-o'])

show_trace(res)
```

### 学习率
:label:`section_gd-learningrate`

学习率$\eta$可由算法设计者设置。如果我们使用的学习率太小，它将导致$x$更新非常缓慢，需要更多的迭代才能得到更好的解决方案。为了说明在这种情况下会发生什么，请考虑$\eta = 0.05$的同一优化问题的进度。如我们所见，即使经过10个步骤，我们仍然离最优解很远。

```{.python .input}
#@tab all
show_trace(gd(0.05))
```

相反，如果我们使用过高的学习率，$\left|\eta f'(x)\right|$对于一阶泰勒展开式可能太大。也就是说，:eqref:`gd-taylor`中的$\mathcal{O}(\eta^2 f'^2(x))$一词可能变得重要。在这种情况下，我们不能保证$x$的迭代能够降低$f(x)$的值。例如，当我们将学习率设置为$\eta=1.1$时，$x$超出了最优解$x=0$并逐渐发散。

```{.python .input}
#@tab all
show_trace(gd(1.1))
```

### 局部最小值

为了说明非凸函数的情况，考虑$f(x) = x \cdot \cos c x$的情况。这个函数有无穷多个局部极小值。根据我们对学习速度的选择以及问题的条件，我们最终可能会得到许多解决方案中的一个。下面的例子说明了（不切实际的）高学习率如何导致当地最低水平的下降。

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)
f = lambda x: x * d2l.cos(c * x)
gradf = lambda x: d2l.cos(c * x) - c * x * d2l.sin(c * x)
show_trace(gd(2))
```

## 多元梯度下降

现在我们对单变量情况有了更好的直觉，让我们考虑$\mathbf{x} \in \mathbb{R}^d$的情况。即，目标函数$f: \mathbb{R}^d \to \mathbb{R}$将向量映射成标量。相应地，它的梯度也是多元的。它是一个由$d$个偏导数组成的向量：

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

梯度中的每个偏导数元素$\partial f(\mathbf{x})/\partial x_i$指示$\mathbf{x}$处$f$相对于输入$x_i$的变化率。和以前一样，在单变量的情况下，我们可以对多变量函数使用相应的Taylor近似来了解我们应该做什么。特别是，我们有

$$f(\mathbf{x} + \mathbf{\epsilon}) = f(\mathbf{x}) + \mathbf{\epsilon}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\mathbf{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

换句话说，在$\mathbf{\epsilon}$的二阶项中，最陡下降的方向由负梯度$-\nabla f(\mathbf{x})$给出。选择合适的学习率$\eta > 0$产生典型的梯度下降算法：

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

为了了解算法在实践中的表现，让我们构造一个目标函数$f(\mathbf{x})=x_1^2+2x_2^2$，其中二维向量$\mathbf{x} = [x_1, x_2]^\top$作为输入，标量作为输出。坡度由$\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$给出。我们将从初始位置$[-5, -2]$通过梯度下降观察$\mathbf{x}$的轨迹。我们还需要两个助手函数。第一个使用update函数并将其应用于初始值$20$次。第二个助手显示$\mathbf{x}$的轨迹。

```{.python .input}
#@tab all
def train_2d(trainer, steps=20):  #@save
    """Optimize a 2-dim objective function with a customized trainer."""
    # s1 and s2 are internal state variables and will
    # be used later in the chapter
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    return results

def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

接下来，我们观察优化变量$\mathbf{x}$对于学习率$\eta = 0.1$的轨迹。我们可以看到，经过20步之后，$\mathbf{x}$的值接近$[0, 0]$的最小值。进展相当顺利，尽管相当缓慢。

```{.python .input}
#@tab all
f = lambda x1, x2: x1 ** 2 + 2 * x2 ** 2  # Objective
gradf = lambda x1, x2: (2 * x1, 4 * x2)  # Gradient

def gd(x1, x2, s1, s2):
    (g1, g2) = gradf(x1, x2)  # Compute gradient
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)  # Update variables

eta = 0.1
show_trace_2d(f, train_2d(gd))
```

## 自适应方法

正如我们在:numref:`section_gd-learningrate`中所看到的，让$\eta$的学习率“恰到好处”是很棘手的。如果我们把它选得太小，我们就没有进展。如果我们把它选得太大，解就会振荡，在最坏的情况下，它甚至可能发散。如果我们可以自动确定$\eta$或者完全不必选择步长怎么办？在这种情况下，二阶方法不仅要考虑目标的值和梯度，还要考虑其曲率。虽然由于计算成本的原因，这些方法不能直接应用于深度学习，但它们为如何设计高级优化算法提供了有用的直觉，这些算法模拟了下面概述的算法的许多理想特性。

### 牛顿法

回顾泰勒的$f$扩展没有必要停止后，第一个任期。事实上，我们可以把它写成

$$f(\mathbf{x} + \mathbf{\epsilon}) = f(\mathbf{x}) + \mathbf{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \mathbf{\epsilon}^\top \nabla \nabla^\top f(\mathbf{x}) \mathbf{\epsilon} + \mathcal{O}(\|\mathbf{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`

为了避免繁琐的符号，我们将$H_f := \nabla \nabla^\top f(\mathbf{x})$定义为$f$的*Hessian*。这是$d \times d$矩阵。对于小问题$d$和简单问题$H_f$很容易计算。另一方面，对于深度网络，由于存储$\mathcal{O}(d^2)$个条目的成本，$H_f$可能大得令人望而却步。此外，由于我们需要将反向传播应用于反向传播调用图，因此通过反向传播进行计算的成本可能太高。现在让我们忽略这些考虑，看看我们会得到什么算法。

毕竟，$f$的最小值满足$\nabla f(\mathbf{x}) = 0$。取:eqref:`gd-hot-taylor`对$\mathbf{\epsilon}$的导数，忽略我们得到的高阶项

$$\nabla f(\mathbf{x}) + H_f \mathbf{\epsilon} = 0 \text{ and hence }
\mathbf{\epsilon} = -H_f^{-1} \nabla f(\mathbf{x}).$$

也就是说，作为优化问题的一部分，我们需要反转Hessian $H_f$。

对于$f(x) = \frac{1}{2} x^2$，我们有$\nabla f(x) = x$和$H_f = 1$。因此，对于任何$x$，我们获得$\epsilon = -x$。换句话说，一步就足以完美地收敛，而无需任何调整！唉，我们有点幸运，因为泰勒展开式是精确的。让我们看看其他问题会发生什么。

```{.python .input}
#@tab all
c = d2l.tensor(0.5)
f = lambda x: d2l.cosh(c * x)  # Objective
gradf = lambda x: c * d2l.sinh(c * x)  # Derivative
hessf = lambda x: c**2 * d2l.cosh(c * x)  # Hessian

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * gradf(x) / hessf(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton())
```

现在让我们看看当我们有一个非凸函数时会发生什么，比如$f(x) = x \cos(c x)$。毕竟，请注意，在牛顿的方法中，我们最终除以黑森函数。这意味着，如果二阶导数是负的，我们会朝着增加的方向走。这是算法的致命缺陷。让我们看看实践中会发生什么。

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)
f = lambda x: x * d2l.cos(c * x)
gradf = lambda x: d2l.cos(c * x) - c * x * d2l.sin(c * x)
hessf = lambda x: - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton())
```

这出了惊人的错误。我们怎样才能修好它？一种方法是用黑森函数的绝对值来“修正”黑森函数。另一个策略是恢复学习速度。这似乎违背了目的，但不完全是。拥有二阶信息可以使我们在曲率较大时保持谨慎，在物镜平坦时采取较长的步骤。让我们看看在学习率稍低的情况下是如何工作的，比如说$\eta = 0.5$。如我们所见，我们有一个相当有效的算法。

```{.python .input}
#@tab all
show_trace(newton(0.5))
```

### 收敛性分析

我们只分析了凸三次可微$f$的收敛速度，其中二阶导数最小值为$x^*$，即$f''(x^*) > 0$。多元证明是下面论点的一个直接扩展，省略了，因为它在直觉方面对我们帮助不大。

用$x_k$表示第$k$次迭代时$x$的值，$e_k := x_k - x^*$表示与最优性的距离。通过泰勒级数展开，我们得到条件$f'(x^*) = 0$可以写成

$$0 = f'(x_k - e_k) = f'(x_k) - e_k f''(x_k) + \frac{1}{2} e_k^2 f'''(\xi_k).$$

这适用于$\xi_k \in [x_k - e_k, x_k]$人。回想一下，我们有$x_{k+1} = x_k - f'(x_k) / f''(x_k)$的更新。将上述扩张除以$f''(x_k)$得到

$$e_k - f'(x_k) / f''(x_k) = \frac{1}{2} e_k^2 f'''(\xi_k) / f''(x_k).$$

插入更新公式将导致以下边界$e_{k+1} \leq e_k^2 f'''(\xi_k) / f'(x_k)$。因此，每当我们在一个有界区域$f'''(\xi_k) / f''(x_k) \leq c$，我们就有一个平方递减误差$e_{k+1} \leq c e_k^2$。

另一方面，优化研究人员称之为“线性”收敛，而像$e_{k+1} \leq \alpha e_k$这样的条件称为“恒定”收敛速度。请注意，此分析附带了一些警告：当我们到达快速收敛区域时，我们实际上没有太多的保证。相反，我们只知道，一旦我们达到它，收敛将非常快。其次，这要求$f$在高阶导数上表现良好。归根结底，就是要确保$f$在如何改变其值方面没有任何“令人惊讶”的特性。

### 预处理

毫不奇怪，计算和存储完整的Hessian非常昂贵。因此，寻找替代品是可取的。一种改进方法是避免计算整个Hessian，而只计算*对角线*项。虽然这不如完整的牛顿法好，但它仍然比不使用要好得多。此外，主要对角线元素的估计是推动随机梯度下降优化算法创新的原因。这将导致表单的更新算法

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \mathrm{diag}(H_f)^{-1} \nabla f(\mathbf{x}).$$

要了解为什么这可能是一个好主意，请考虑一个变量以毫米表示高度，另一个变量以公里表示高度的情况。假设这两个自然尺度都是以米为单位的，我们在参数化上有一个可怕的不匹配。使用预处理可消除此问题。梯度下降的有效预处理相当于为每个坐标选择不同的学习速率。

### 直线搜索梯度下降法

梯度下降的一个关键问题是我们可能会超出目标或进展不足。解决这个问题的一个简单方法是结合使用直线搜索和梯度下降。也就是说，我们使用$\nabla f(\mathbf{x})$给出的方向，然后对$\eta$使$f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$最小化的步长执行二进制搜索。

此算法收敛迅速（有关分析和证明，请参见:cite:`Boyd.Vandenberghe.2004`）。然而，为了深入学习，这并不太可行，因为行搜索的每一步都需要我们评估整个数据集上的目标函数。这是太昂贵的方式来完成。

## 摘要

* 学习率很重要。太大，我们分歧，太小，我们没有取得进展。
* 梯度下降会陷入局部极小值。
* 在高维中，学习率的调整是复杂的。
* 预处理有助于调节比例。
* 牛顿的方法要快得多，一旦它开始在凸问题中正常工作。
* 对于非凸问题，不要使用牛顿法而不作任何调整。

## 练习

1. 用不同的学习率和目标函数进行梯度下降实验。
1. 在区间$[a, b]$中实现线搜索以最小化凸函数。
    * 您是否需要导数进行二进制搜索，即，决定是选择$[a, (a+b)/2]$还是$[(a+b)/2, b]$。
    * 算法的收敛速度有多快？
    * 实现了该算法，并将其应用于$\log (\exp(x) + \exp(-2*x -3))$的最小化问题。
1. 设计一个目标函数定义在$\mathbb{R}^2$梯度下降是非常缓慢的。提示：缩放不同的坐标不同。
1. 使用预处理实现牛顿方法的轻量级版本：
    * 使用对角Hessian作为预条件。
    * 使用该值的绝对值，而不是实际值（可能有符号）。
    * 将此应用于上述问题。
1. 将上述算法应用于多个目标函数（凸或非凸）。如果你把坐标旋转$45$度会怎么样？

[Discussions](https://discuss.d2l.ai/t/351)
