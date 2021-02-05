# 随机梯度下降
:label:`sec_sgd`

在这一部分，我们将介绍随机梯度下降的基本原理。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

## 随机梯度更新

在深度学习中，目标函数通常是训练数据集中每个示例的损失函数的平均值。假设$f_i(\mathbf{x})$是具有$n$个示例的训练数据集的损失函数，索引为$i$，参数向量为$\mathbf{x}$，则我们得到目标函数

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$

在$\mathbf{x}$处的目标函数的梯度被计算为

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

如果使用梯度下降，则每次自变量迭代的计算成本为$\mathcal{O}(n)$，与$n$呈线性增长。因此，当模型训练数据集较大时，每次迭代的梯度下降代价会很高。

随机梯度下降(SGD)算法减少了每次迭代的计算量。在随机梯度下降的每一次迭代中，我们随机对数据示例的指标$i\in\{1,\ldots, n\}$进行均匀采样，并计算梯度$\nabla f_i(\mathbf{x})$以更新$\mathbf{x}$：

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}).$$

这里，$\eta$是学习率。我们可以看到，每次迭代的计算成本从梯度下降的$\mathcal{O}(n)$下降到常数$\mathcal{O}(1)$。应该指出的是，随机梯度$\nabla f_i(\mathbf{x})$是梯度$\nabla f(\mathbf{x})$的无偏估计。

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

这意味着，平均而言，随机梯度是梯度的良好估计。

现在，我们将通过向梯度添加均值为0和方差为1的随机噪声来模拟SGD，将其与梯度下降进行比较。

```{.python .input}
f = lambda x1, x2: x1 ** 2 + 2 * x2 ** 2  # Objective
gradf = lambda x1, x2: (2 * x1, 4 * x2)  # Gradient

def sgd(x1, x2, s1, s2):
    global lr  # Learning rate scheduler
    (g1, g2) = gradf(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()  # Learning rate at time t
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)  # Update variables

eta = 0.1
lr = (lambda: 1)  # Constant learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50))
```

```{.python .input}
#@tab pytorch
f = lambda x1, x2: x1 ** 2 + 2 * x2 ** 2  # Objective
gradf = lambda x1, x2: (2 * x1, 4 * x2)  # Gradient

def sgd(x1, x2, s1, s2):
    global lr  # Learning rate scheduler
    (g1, g2) = gradf(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()  # Learning rate at time t
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)  # Update variables

eta = 0.1
lr = (lambda: 1)  # Constant learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50))
```

```{.python .input}
#@tab tensorflow
f = lambda x1, x2: x1 ** 2 + 2 * x2 ** 2  # Objective
gradf = lambda x1, x2: (2 * x1, 4 * x2)  # Gradient

def sgd(x1, x2, s1, s2):
    global lr  # Learning rate scheduler
    (g1, g2) = gradf(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal([1], 0.0, 1)
    g2 += d2l.normal([1], 0.0, 1)
    eta_t = eta * lr()  # Learning rate at time t
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)  # Update variables

eta = 0.1
lr = (lambda: 1)  # Constant learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50))
```

正如我们所看到的，SGD中变量的轨迹比我们在上一节中观察到的梯度下降中的轨迹噪音要大得多。这是由于梯度的随机性造成的。也就是说，即使当我们到达最小值附近时，我们仍然受到通过$\eta \nabla f_i(\mathbf{x})$的瞬时梯度注入的不确定性的影响。即使走了50步，质量还是不太好。更糟糕的是，在额外的步骤之后，它不会有所改善(我们鼓励读者自己尝试更多的步骤来证实这一点)。这给我们留下了唯一的选择-改变$\eta$的学习率。然而，如果我们把这个选得太小，我们一开始就不会取得任何有意义的进展。另一方面，如果我们把它选得太大，我们将得不到一个好的解决方案，如上所述。解决这些相互冲突的目标的唯一方法是随着优化进程“动态”降低学习率。

这也是将学习率函数`lr`添加到`sgd`阶跃函数中的原因。在上述示例中，当我们将关联的`lr`功能设置为常数(即`lr = (lambda: 1)`)时，用于学习速率调度的任何功能都处于休眠状态。

## 动态学习速率

将$\eta$替换为依赖于时间的学习率$\eta(t)$增加了控制优化算法的收敛性的复杂性。特别是，需要计算出$\eta$应该以多快的速度腐烂。如果太快，我们将过早停止优化。如果我们把它降低得太慢，我们在优化上浪费了太多的时间。随着时间的推移，有几种基本策略可用于调整$\eta$(我们将在后面的章节中讨论更高级的策略)：

$$
\begin{aligned}
    \eta(t) & = \eta_i \text{ if } t_i \leq t \leq t_{i+1}  && \mathrm{piecewise~constant} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \mathrm{exponential} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \mathrm{polynomial}
\end{aligned}
$$

在第一种情况下，我们降低学习速率，例如，每当优化进展停滞时。这是训练深度网络的常用策略。或者，我们可以通过指数衰减来更积极地减少它。不幸的是，这导致在算法收敛之前过早停止。一种流行的选择是$\alpha = 0.5$的多项式衰减。在凸优化的情况下，有大量的证明表明这个速度是很好的。让我们看看这在实践中是什么样子的。

```{.python .input}
#@tab all
def exponential():
    global ctr
    ctr += 1
    return math.exp(-0.1 * ctr)

ctr = 1
lr = exponential  # Set up learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000))
```

正如预期的那样，参数的方差显著减小。然而，这是以不能收敛到最优解$\mathbf{x} = (0, 0)$为代价的。即使过了1000步，我们离最优解决方案还有很远的路要走。事实上，该算法根本无法收敛。另一方面，如果我们使用多项式衰减，其中学习速率随着步数的平方根的倒数而衰减，则收敛性是好的。

```{.python .input}
#@tab all
def polynomial():
    global ctr
    ctr += 1
    return (1 + 0.1 * ctr)**(-0.5)

ctr = 1
lr = polynomial  # Set up learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50))
```

对于如何设置学习速率，有更多的选择。例如，我们可以从一个很小的比率开始，然后迅速增加，然后再次降低，尽管速度较慢。我们甚至可以在更小和更大的学习率之间交替。这样的时间表种类繁多。现在让我们把重点放在可以进行全面理论分析的学习速率时间表上，即在凸设置下的学习速率。对于一般的非凸问题，要获得有意义的收敛保证是非常困难的，因为最小化非线性非凸问题一般是NP困难的。有关调查，请参见优秀的[Tibishani 2015年的讲座notes](https://www.stat.cmu.edu/~ryantibs/convexopt-F15/lectures/26-nonconvex.pdf)]。

## 凸目标的收敛性分析

以下内容是可选的，主要用于传达对问题的更多直觉。我们将自己限制在最简单的证明之一，如:cite:`Nesterov.Vial.2000`所述。例如，只要目标函数表现得特别好，就存在更先进的证明技术。:cite:`Hazan.Rakhlin.Bartlett.2008`示出，对于强凸函数，即对于可以从下面以$\mathbf{x}^\top \mathbf{Q} \mathbf{x}$为界的函数，可以在像$\eta(t) = \eta_0/(\beta t + 1)$这样降低学习率的同时，以少量的步骤最小化它们。不幸的是，这种情况在深度学习中从未真正发生过，我们在实践中的下降速度要慢得多。

考虑一下这样的情况，$\mathbf{w}_{t+1} = \mathbf{w}_{t} - \eta_t \partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w}).$美元

具体地说，假设$\mathbf{x}_t$取自某个分布$P(\mathbf{x})$，且$l(\mathbf{x}, \mathbf{w})$是$\mathbf{w}$中所有$\mathbf{x}$的凸函数。最后用来表示

$$R(\mathbf{w}) = E_{\mathbf{x} \sim P}[l(\mathbf{x}, \mathbf{w})]$$

预期风险降低了$R^*$，其最低风险为$\mathbf{w}$。最后，让$\mathbf{w}^*$为最小化(我们假设它存在于定义了$\mathbf{w}$的域内)。在这种情况下，我们可以跟踪当前参数$\mathbf{w}_t$和风险最小化程序$\mathbf{w}^*$之间的距离，并查看其是否随着时间的推移而改善：

$$\begin{aligned}
    \|\mathbf{w}_{t+1} - \mathbf{w}^*\|^2 & = \|\mathbf{w}_{t} - \eta_t \partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w}) - \mathbf{w}^*\|^2 \\
    & = \|\mathbf{w}_{t} - \mathbf{w}^*\|^2 + \eta_t^2 \|\partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w})\|^2 - 2 \eta_t
    \left\langle \mathbf{w}_t - \mathbf{w}^*, \partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w})\right\rangle.
   \end{aligned}
$$

梯度$\partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w})$可以从上方由某个李普希茨常数$L$限定，因此我们有

$$\eta_t^2 \|\partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w})\|^2 \leq \eta_t^2 L^2.$$

我们最感兴趣的是$\mathbf{w}_t$和$\mathbf{w}^*$之间的距离是如何变化的*预期*。事实上，对于任何特定的步骤序列，距离可能会增加，这取决于我们遇到的$\mathbf{x}_t$个步骤中的哪一个。因此，我们需要绑定内积。通过凸性，我们得到了

$$
l(\mathbf{x}_t, \mathbf{w}^*) \geq l(\mathbf{x}_t, \mathbf{w}_t) + \left\langle \mathbf{w}^* - \mathbf{w}_t, \partial_{\mathbf{w}} l(\mathbf{x}_t, \mathbf{w}_t) \right\rangle.
$$

使用这两个不等式并将其插入上面，我们在时间$t+1$获得关于参数之间距离的界限，如下所示：

$$\|\mathbf{w}_{t} - \mathbf{w}^*\|^2 - \|\mathbf{w}_{t+1} - \mathbf{w}^*\|^2 \geq 2 \eta_t (l(\mathbf{x}_t, \mathbf{w}_t) - l(\mathbf{x}_t, \mathbf{w}^*)) - \eta_t^2 L^2.$$

这意味着只要当前损失和最优损失之间的预期差值超过$\eta_t L^2$，我们就会取得进展。由于前者必然会收敛到$0$，因此$\eta_t$的学习率也需要消失。

接下来，我们来看一下这个表达式的期望值。这就产生了

$$E_{\mathbf{w}_t}\left[\|\mathbf{w}_{t} - \mathbf{w}^*\|^2\right] - E_{\mathbf{w}_{t+1}\mid \mathbf{w}_t}\left[\|\mathbf{w}_{t+1} - \mathbf{w}^*\|^2\right] \geq 2 \eta_t [E[R[\mathbf{w}_t]] - R^*] -  \eta_t^2 L^2.$$

最后一步是对$t \in \{t, \ldots, T\}$的不平等进行求和。由于和望远镜，通过去掉较低的项，我们得到了

$$\|\mathbf{w}_{0} - \mathbf{w}^*\|^2 \geq 2 \sum_{t=1}^T \eta_t [E[R[\mathbf{w}_t]] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$

请注意，我们利用了给定的$\mathbf{w}_0$，因此可以降低期望值。上次定义

$$\bar{\mathbf{w}} := \frac{\sum_{t=1}^T \eta_t \mathbf{w}_t}{\sum_{t=1}^T \eta_t}.$$

那么根据凸性，它就会得出这样的结论

$$\sum_t \eta_t E[R[\mathbf{w}_t]] \geq \sum \eta_t \cdot \left[E[\bar{\mathbf{w}}]\right].$$

将其插入到上面的不等式中会得到下界

$$
\left[E[\bar{\mathbf{w}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t}.
$$

这里$r^2 := \|\mathbf{w}_0 - \mathbf{w}^*\|^2$是初始参数选择和最终结果之间距离的界限。简而言之，收敛速度取决于损失函数通过李普希兹常数$L$改变的速度有多快，以及初始值离最优值$r$有多远。请注意，界限是以$\bar{\mathbf{w}}$为单位，而不是以$\mathbf{w}_T$为单位。这是因为$\bar{\mathbf{w}}$是优化路径的平滑版本。现在让我们来分析一下$\eta_t$的一些选择。

* **已知时间域**。只要知道$r, L$和$T$，我们就可以选$\eta = r/L \sqrt{T}$。这将产生上限$r L (1 + 1/T)/2\sqrt{T} < rL/\sqrt{T}$。也就是说，我们以$\mathcal{O}(1/\sqrt{T})$的速率收敛到最优解。
* **未知时间域**。每当我们想要对*任何*时间$T$有一个好的解决方案时，我们都可以选择$\eta = \mathcal{O}(1/\sqrt{T})$。这会耗费额外的对数因子，并导致表单$\mathcal{O}(\log T / \sqrt{T})$的上界。

注意，对于强凸损失$l(\mathbf{x}, \mathbf{w}') \geq l(\mathbf{x}, \mathbf{w}) + \langle \mathbf{w}'-\mathbf{w}, \partial_\mathbf{w} l(\mathbf{x}, \mathbf{w}) \rangle + \frac{\lambda}{2} \|\mathbf{w}-\mathbf{w}'\|^2$，我们可以设计更快收敛的优化调度。事实上，$\eta$的指数衰减导致形式$\mathcal{O}(\log T / T)$的界限。

## 随机梯度与有限样本

到目前为止，当谈到随机梯度下降时，我们玩得有点快和松。我们假设我们绘制实例$x_i$，通常具有来自某个分布$p(x, y)$的标签$y_i$，并且我们使用它以某种方式更新权重$w$。特别地，对于有限样本量，我们简单地认为离散分布$p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$允许我们在其上执行sgd。

然而，这并不是我们真正做的事情。在当前部分的玩具示例中，我们简单地将噪声添加到非随机梯度，即，我们假装具有对$(x_i, y_i)$。事实证明，这在这里是合理的(有关详细讨论，请参阅练习)。更令人不安的是，在之前的所有讨论中，我们显然没有这样做。相反，我们只迭代了所有实例一次。要了解为什么这是可取的，请考虑相反的情况，即我们从离散分布中抽样$n$个观测值，并进行替换。随机选择元素$i$的概率为$N^{-1}$。因此，至少选择一次就是

$$P(\mathrm{choose~} i) = 1 - P(\mathrm{omit~} i) = 1 - (1-N^{-1})^N \approx 1-e^{-1} \approx 0.63.$$

一个类似的推理表明，恰好挑选一个样本一次的概率是${N \choose 1} N^{-1} (1-N^{-1})^{N-1} = \frac{N-1}{N} (1-N^{-1})^{N} \approx e^{-1} \approx 0.37$。与没有替换的采样相比，这会导致方差增加和数据效率降低。因此，在实践中，我们执行后者(这是本书的默认选择)。最后请注意，通过数据集的重复遍历以*不同的*随机顺序遍历它。

## 摘要

* 对于凸问题，我们可以证明对于广泛的学习率选择，随机梯度下降将收敛于最优解。
* 对于深度学习来说，通常情况并非如此。然而，对凸问题的分析给了我们关于如何接近最优化的有用的洞察力，即逐步降低学习速度，尽管不是太快。
* 当学习速度太小或太大时，就会出现问题。在实践中，只有经过多次实验才能找到合适的学习速度。
* 当训练数据集中的样本较多时，梯度下降的每次迭代计算成本较高，因此SGD在这些情况下是首选的。
* SGD的最优性保证通常在非凸情况下不可用，因为需要检查的局部极小值的数量很可能是指数的。

## 练习

1. 针对SGD使用不同的学习速率计划，并使用不同的迭代次数进行实验。具体地说，将距最优解$(0, 0)$的距离绘制为迭代次数的函数。
1. 证明对于函数$f(x_1, x_2) = x_1^2 + 2 x_2^2$，将正态噪声添加到梯度等同于最小化损失函数$l(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$，其中$x$是从正态分布得出的。
    * 导出$\mathbf{x}$的分布的均值和方差。
    * 表明对于目标函数$f(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{\mu})^\top Q (\mathbf{x} - \mathbf{\mu})$对于$Q \succeq 0$，该属性通常成立。
1. 比较$\{(x_1, y_1), \ldots, (x_m, y_m)\}$有替换样本和无替换样本时SGD的收敛性。
1. 如果某个渐变(或者更确切地说，某个坐标)始终大于所有其他渐变，您将如何更改SGD解算器？
1. 假设是$f(x) = x^2 (1 + \sin x)$。$f$有多少个局部极小值？你能以这样一种方式改变$f$，使其最小化，需要计算所有的局部最小值吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/352)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/497)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1067)
:end_tab:
