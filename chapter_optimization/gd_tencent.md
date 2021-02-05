# 渐变下降
:label:`sec_gd`

在本节中，我们将介绍梯度下降的基本概念。这是必要的简短。例如，有关凸优化的深入介绍，请参见:cite:`Boyd.Vandenberghe.2004`。虽然后者很少直接用于深度学习，但理解梯度下降是理解随机梯度下降算法的关键。例如，优化问题可能会因为过大的学习率而发散。这种现象在梯度下降中已经可以看到。同样，预处理是梯度下降中的一种常见技术，并延续到更高级的算法中。让我们从一个简单的特例开始。

## 一维中的梯度下降

一维梯度下降是一个很好的例子，解释了为什么梯度下降算法可能会降低目标函数的值。考虑一些连续可微的实值函数$f: \mathbb{R} \rightarrow \mathbb{R}$。使用泰勒展开(:numref:`sec_single_variable_calculus`)，我们得到

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

也就是说，在一次近似中，$f(x+\epsilon)$由函数值$f(x)$和一阶导数$f'(x)$在$x$处给出。假设对于较小的$\epsilon$，向负梯度方向移动将减少$f$，这并不是不合理的。为简单起见，我们选择固定步长$\eta > 0$，然后选择$\epsilon = -\eta f'(x)$。把这个代入上面的泰勒展开式，我们就会得到

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$

如果导数$f'(x) \neq 0$没有消失，我们从$\eta f'^2(x)>0$开始就取得了进展。此外，我们总是可以选择$\eta$个足够小的项，使高阶项变得无关紧要。因此我们到达了

$$f(x - \eta f'(x)) \lessapprox f(x).$$

这意味着，如果我们使用

$$x \leftarrow x - \eta f'(x)$$

要迭代$x$，函数$f(x)$的值可能会下降。因此，在梯度下降中，我们首先选择初始值$x$和常数$\eta > 0$，然后使用它们来连续迭代$x$，直到达到停止条件，例如当梯度$|f'(x)|$的大小足够小或者迭代次数已经达到某一值时。

为简单起见，我们选择目标函数$f(x)=x^2$来说明如何实现梯度下降。虽然我们知道$x=0$是最小化$f(x)$的解决方案，但是我们仍然使用这个简单的函数来观察$x$是如何变化的。一如既往，我们从导入所有需要的模块开始。

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

接下来，我们使用$x=10$作为初始值，并假定为$\eta=0.2$。使用梯度下降迭代$x$ 10次，我们可以看到，最终$x$的值接近最优解。

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

$x$以上的优化进度如下所示。

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

学习率$\eta$可以由算法设计者设置。如果我们使用太小的学习率，会导致$x$更新非常慢，需要更多迭代才能得到更好的解决方案。要显示在这种情况下会发生什么，请考虑$\eta = 0.05$的同一优化问题的进度。正如我们所看到的，即使在10个步骤之后，我们仍然离最优解决方案很远。

```{.python .input}
#@tab all
show_trace(gd(0.05))
```

相反，如果我们使用过高的学习率，对于一阶泰勒展开公式来说，$\left|\eta f'(x)\right|$可能太大了。也就是说，术语$\mathcal{O}(\eta^2 f'^2(x))$在:eqref:`gd-taylor`可能会变得重要。在这种情况下，我们不能保证$x$的迭代能够降低$f(x)$的值。例如，当我们将学习率设置为$\eta=1.1$时，$x$超过最优解$x=0$并逐渐发散。

```{.python .input}
#@tab all
show_trace(gd(1.1))
```

### 本地最小映像

为了说明非凸函数会发生什么情况，请考虑$f(x) = x \cdot \cos c x$的情况。这个函数有无限多个局部极小值。根据我们对学习速度的选择和问题条件的好坏，我们可能最终会得到许多解决方案中的一种。下面的例子说明了(不切实际的)高学习率将如何导致较差的局部最小值。

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)
f = lambda x: x * d2l.cos(c * x)
gradf = lambda x: d2l.cos(c * x) - c * x * d2l.sin(c * x)
show_trace(gd(2))
```

## 多元梯度下降

现在我们对单变量的情况有了更好的直觉，让我们考虑一下$\mathbf{x} \in \mathbb{R}^d$的情况。也就是说，目标函数$f: \mathbb{R}^d \to \mathbb{R}$将矢量映射成标量。相应地，它的梯度也是多元的。它是由$d$个偏导数组成的向量：

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

梯度中的每个偏导数元素$\partial f(\mathbf{x})/\partial x_i$表示$\mathbf{x}$处相对于输入$x_i$的$f$的变化率。就像以前在单变量情况下一样，我们可以使用对应的多变量函数的泰勒近似来了解我们应该做什么。特别值得一提的是，我们有

$$f(\mathbf{x} + \mathbf{\epsilon}) = f(\mathbf{x}) + \mathbf{\epsilon}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\mathbf{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

换言之，在$\mathbf{\epsilon}$中直到二次项的最陡下降方向由负梯度$-\nabla f(\mathbf{x})$给出。选择合适的学习率$\eta > 0$产生典型的梯度下降算法：

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

为了查看算法在实践中的行为，让我们构造目标函数$f(\mathbf{x})=x_1^2+2x_2^2$，其中二维矢量$\mathbf{x} = [x_1, x_2]^\top$作为输入，标量作为输出。渐变是以$\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$给出的。我们将通过从初始位置$\mathbf{x}$梯度下降来观察$[-5, -2]$的轨迹。我们还需要两个帮助器函数。第一个使用更新函数，并将其应用于初始值$20$次。第二个辅助对象可视化$\mathbf{x}$的轨迹。

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

接下来，我们观察优化变量$\mathbf{x}$对于学习率$\eta = 0.1$的轨迹。我们可以看到，经过20步之后，$\mathbf{x}$的值接近其最低值$[0, 0]$。尽管进展相当缓慢，但进展相当良好。

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

正如我们在:numref:`section_gd-learningrate`中看到的，让$\eta$的学习率“恰到好处”是很棘手的。如果我们把它挑得太小，我们就没有进展。如果我们把它选得太大，解就会振荡，在最坏的情况下，它甚至可能会发散。如果我们可以自动确定$\eta$，或者根本不需要选择步长，会怎么样？在这种情况下，不仅要看目标的值和梯度，还要看它的*曲率*的二阶方法会有所帮助。虽然由于计算成本的原因，这些方法不能直接应用于深度学习，但它们为如何设计模仿下面概述的算法的许多所需特性的高级优化算法提供了有用的直观。

### 牛顿法

回顾泰勒展开的$f$，第一任期过后没有必要停下来。事实上，我们可以把它写成

$$f(\mathbf{x} + \mathbf{\epsilon}) = f(\mathbf{x}) + \mathbf{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \mathbf{\epsilon}^\top \nabla \nabla^\top f(\mathbf{x}) \mathbf{\epsilon} + \mathcal{O}(\|\mathbf{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`

为了避免繁琐的记法，我们将$H_f := \nabla \nabla^\top f(\mathbf{x})$定义为$f$的“黑森*”。这是一个$d \times d$的矩阵。对于小的$d$和简单的问题，$H_f$很容易计算。另一方面，对于深度网络，由于存储$H_f$个条目的成本，$\mathcal{O}(d^2)$个条目可能大得令人望而却步。此外，通过反向传播进行计算可能过于昂贵，因为我们需要将反向传播应用于反向传播调用图。现在，让我们忽略这些考虑，看看我们会得到什么算法。

毕竟，最低的$f$就满足了$\nabla f(\mathbf{x}) = 0$。取:eqref:`gd-hot-taylor`对$\mathbf{\epsilon}$的导数，忽略更高阶项，我们得到

$$\nabla f(\mathbf{x}) + H_f \mathbf{\epsilon} = 0 \text{ and hence }
\mathbf{\epsilon} = -H_f^{-1} \nabla f(\mathbf{x}).$$

也就是说，作为优化问题的一部分，我们需要反转黑森$H_f$。

$f(x) = \frac{1}{2} x^2$元我们有$\nabla f(x) = x$元和$H_f = 1$元。因此，对于任何$x$，我们都可以得到$\epsilon = -x$。换句话说，一步就足以完美地收敛，不需要任何调整！唉，我们这里有点幸运，因为泰勒展开式是准确的。让我们看看在其他问题上会发生什么。

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

现在让我们看看当我们有一个*非凸*函数时会发生什么，比如$f(x) = x \cos(c x)$。毕竟，请注意，在牛顿的方法中，我们最终被黑森除以。这意味着如果二阶导数为“负”，我们就会走向“增加”$f$的方向。这是该算法的一个致命缺陷。让我们看看实践中会发生什么。

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)
f = lambda x: x * d2l.cos(c * x)
gradf = lambda x: d2l.cos(c * x) - c * x * d2l.sin(c * x)
hessf = lambda x: - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton())
```

这件事出了很大的差错。我们怎么才能把它修好呢？一种方法是“修复”黑森，取而代之的是它的绝对值。另一种策略是恢复学习率。这似乎违背了目的，但也不完全是。有了二阶信息，我们就可以在曲率大的时候小心翼翼，在物镜平坦的时候走更长的步。让我们看看在稍微小一点的学习率(比方说$\eta = 0.5$)的情况下，这是如何运作的。正如我们所看到的，我们有一个相当有效的算法。

```{.python .input}
#@tab all
show_trace(newton(0.5))
```

### 收敛分析

我们只分析了凸的和三次可微的$f$的收敛速度，其中二阶导数在其最小值$x^*$是非零的，即其中$f''(x^*) > 0$。多元证明是下面论点的直接扩展，由于它对我们的直觉帮助不大，所以被省略了。

用$x_k$表示$k$次迭代时的值$x$，让$e_k := x_k - x^*$表示离最优值的距离。通过泰勒级数展开，我们得到条件$f'(x^*) = 0$可以写成

$$0 = f'(x_k - e_k) = f'(x_k) - e_k f''(x_k) + \frac{1}{2} e_k^2 f'''(\xi_k).$$

这适用于大约$\xi_k \in [x_k - e_k, x_k]$人。回想一下，我们有更新$x_{k+1} = x_k - f'(x_k) / f''(x_k)$。将上述扩张除以$f''(x_k)$的收益率

$$e_k - f'(x_k) / f''(x_k) = \frac{1}{2} e_k^2 f'''(\xi_k) / f''(x_k).$$

插入更新方程式导致以下界限$e_{k+1} \leq e_k^2 f'''(\xi_k) / f'(x_k)$。因此，无论何时我们在有界$f'''(\xi_k) / f''(x_k) \leq c$的区域内，我们都会有一个二次递减的误差$e_{k+1} \leq c e_k^2$。

另外，优化研究人员将这种情况称为“线性”收敛，而像$e_{k+1} \leq \alpha e_k$这样的条件则称为“恒定”收敛速度。请注意，这一分析附带了一些警告：我们真的不能很好地保证我们什么时候能达到快速收敛的区域。相反，我们只知道，一旦我们到达它，收敛将会非常快。其次，这要求$f$对于更高阶导数表现良好。归根结底是要确保$f$在如何更改其值方面没有任何“令人惊讶”的属性。

### 预处理

毫不奇怪，计算和存储完整的黑森语是非常昂贵的。因此，寻找替代方案是可取的。改善问题的一种方法是避免计算整个黑森系数，而只计算*对角线*条目。虽然这不如完整的牛顿方法好，但总比不用要好得多。此外，主对角线元素的估计是驱动随机梯度下降优化算法的一些创新的原因。这将导致更新表单的算法

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \mathrm{diag}(H_f)^{-1} \nabla f(\mathbf{x}).$$

要了解为什么这可能是个好主意，请考虑这样一种情况，其中一个变量以毫米为单位表示高度，另一个变量以公里为单位表示高度。假设两个自然尺度都以米为单位，我们在参数化方面会有严重的不匹配。使用预处理可以消除这一点。利用梯度下降进行有效的预处理相当于为每个坐标选择不同的学习率。

### 带线搜索的梯度下降

梯度下降的关键问题之一是我们可能超过目标或进展不足。解决这个问题的一个简单方法是将线搜索与梯度下降结合使用。也就是说，我们使用$\nabla f(\mathbf{x})$给出的方向，然后执行关于哪个步长$\eta$最小化$f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$的二进制搜索。

该算法快速收敛(有关分析和证明，请参见例如:cite:`Boyd.Vandenberghe.2004`)。然而，为了深入学习的目的，这并不是很可行，因为线搜索的每一步都需要我们评估整个数据集上的目标函数。这太昂贵了，无法实现。

## 摘要

* 学习率很重要。太大了，我们就会分道扬镳，太小了，我们就不会取得进展。
* 梯度下降可能会陷入局部极小值。
* 在高维中，调整学习率是很复杂的。
* 预处理可以帮助调整比例。
* 牛顿的方法要快得多，一旦它开始在凸问题中正常工作。
* 注意，对于非凸问题，不要使用牛顿方法而不作任何调整。

## 练习

1. 实验采用不同的学习速率和目标函数进行梯度下降。
1. 实现线搜索以最小化区间$[a, b]$中的凸函数。
    * 二分搜索是否需要导数，即决定选择$[a, (a+b)/2]$还是$[(a+b)/2, b]$。
    * 算法的收敛速度有多快？
    * 实现该算法，并将其应用于最小化$\log (\exp(x) + \exp(-2*x -3))$。
1. 设计一个在$\mathbb{R}^2$上定义的目标函数，其中梯度下降非常缓慢。提示：以不同方式缩放不同的坐标。
1. 使用预处理实现牛顿方法的轻量级版本：
    * 使用对角线黑森作为预处理。
    * 使用该值的绝对值，而不是实际的(可能是有符号的)值。
    * 将此应用于上面的问题。
1. 将上述算法应用于多个目标函数(凸或非凸)。如果将坐标旋转$45$度会发生什么情况？

[Discussions](https://discuss.d2l.ai/t/351)
