# 随机梯度下降法
:label:`sec_sgd`

在本节中，我们将介绍随机梯度下降的基本原理。

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

在深度学习中，目标函数通常是训练数据集中每个例子损失函数的平均值。假设$f_i(\mathbf{x})$是训练数据集的损失函数，有$n$个例子，索引为$i$，参数向量为$\mathbf{x}$，则得到目标函数

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$

目标函数在$\mathbf{x}$处的梯度计算如下

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

如果使用梯度下降法，则每个自变量迭代的计算成本为$\mathcal{O}(n)$，其线性增长为$n$。因此，当模型训练数据集较大时，每次迭代的梯度下降代价会很高。

随机梯度下降（SGD）减少了每次迭代的计算量。在随机梯度下降的每次迭代中，我们均匀地随机抽取一个索引$i\in\{1,\ldots, n\}$作为数据示例，并计算梯度$\nabla f_i(\mathbf{x})$以更新$\mathbf{x}$：

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}).$$

这里，$\eta$是学习率。我们可以看到，每次迭代的计算成本从梯度下降的$\mathcal{O}(n)$下降到常数$\mathcal{O}(1)$。我们应该提到，随机梯度$\nabla f_i(\mathbf{x})$是梯度$\nabla f(\mathbf{x})$的无偏估计。

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

这意味着，平均而言，随机梯度是一个很好的梯度估计。

现在，我们将其与梯度下降法进行比较，通过向梯度中添加均值为0、方差为1的随机噪声来模拟SGD。

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

如我们所见，SGD中变量的轨迹比我们在上一节梯度下降中观察到的轨迹噪声要大得多。这是由于梯度的随机性。也就是说，即使我们接近最小值，我们仍然受到瞬时梯度通过$\eta \nabla f_i(\mathbf{x})$注入的不确定性的影响。即使走了50步，质量还是不太好。更糟糕的是，在额外的步骤之后，它不会得到改进（我们鼓励读者自己尝试更多的步骤来证实这一点）。这给我们留下了唯一的选择——改变学习率$\eta$。然而，如果我们选择的太小，我们将不会取得任何有意义的进展最初。另一方面，如果我们选择太大，我们将无法得到一个很好的解决方案，如上所示。解决这些相互冲突的目标的唯一方法是随着优化的进行动态地降低学习率。

这也是将学习速率函数`lr`添加到`sgd`步进函数中的原因。在上面的示例中，任何用于学习速率调度的功能都处于休眠状态，因为我们将相关的`lr`函数设置为常量，即`lr = (lambda: 1)`。

## 动态学习率

用依赖时间的学习率$\eta(t)$代替$\eta$增加了控制优化算法收敛的复杂性。尤其需要弄清楚$\eta$的衰变速度。如果太快，我们将停止过早优化。如果我们减少得太慢，我们在优化上浪费了太多的时间。随着时间的推移，在调整$\eta$时使用了一些基本策略（我们将在后面的章节中讨论更高级的策略）：

$$
\begin{aligned}
    \eta(t) & = \eta_i \text{ if } t_i \leq t \leq t_{i+1}  && \mathrm{piecewise~constant} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \mathrm{exponential} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \mathrm{polynomial}
\end{aligned}
$$

在第一种情况下，我们会降低学习率，例如，当优化过程停滞时。这是训练深层网络的常用策略。或者我们可以通过指数衰减来更积极地降低它。不幸的是，这会导致在算法收敛之前过早停止。一个流行的选择是多项式衰减与$\alpha = 0.5$。在凸优化的情况下，有许多证明表明这个速率是良好的。让我们看看这在实践中是什么样子。

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

正如预期的那样，参数的方差显著减小。然而，这是以未能收敛到最优解$\mathbf{x} = (0, 0)$为代价的。即使经过1000个步骤，我们仍然离最优解很远。实际上，该算法根本无法收敛。另一方面，如果我们使用多项式衰减，其中学习率衰减与步数平方根倒数收敛是好的。

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

对于如何设置学习率，还有很多选择。例如，我们可以从一个小的速率开始，然后快速地上升，然后再下降，尽管速度较慢。我们甚至可以在更小和更大的学习率之间交替。这种时间表有很多种。现在，让我们集中讨论学习率计划，对其进行全面的理论分析是可能的，即在凸环境中的学习率。对于一般的非凸问题，很难得到有意义的收敛性保证，因为一般的最小化非线性非凸问题是NP困难的。如需调查，请参阅优秀[课堂讲稿](https://www.stat.cmu.edu/~ryantibs/converxopt-F15/teachments/26-nonconvex.pdf)Tibshirani 2015年。

## 凸目标的收敛性分析

以下是可选的，主要是为了传达更多关于问题的直觉。我们将自己局限于一个最简单的证明，如:cite:`Nesterov.Vial.2000`所描述的。更先进的证明技术明显存在，例如，当目标函数表现得特别好时。:cite:`Hazan.Rakhlin.Bartlett.2008`表明，对于强凸函数，即对于可以由$\mathbf{x}^\top \mathbf{Q} \mathbf{x}$从下面限定的函数，可以在减少学习率的同时，在少量步骤中最小化它们，如$\eta(t) = \eta_0/(\beta t + 1)$。不幸的是，这种情况从来没有真正发生在深度学习和我们留下了一个更缓慢的下降率在实践中。

以$\mathbf{w}_{t+1} = \mathbf{w}_{t} - \eta_t \partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w}).$美元为例$

特别地，假设$\mathbf{x}_t$是从一些分布$P(\mathbf{x})$中提取的，并且$l(\mathbf{x}, \mathbf{w})$是$\mathbf{w}$中所有$\mathbf{x}$的凸函数。最后用表示

$$R(\mathbf{w}) = E_{\mathbf{x} \sim P}[l(\mathbf{x}, \mathbf{w})]$$

预期风险和$R^*$的最小值。最后让$\mathbf{w}^*$是最小值（我们假设它存在于$\mathbf{w}$定义的域中）。在这种情况下，我们可以跟踪当前参数$\mathbf{w}_t$和风险最小化值$\mathbf{w}^*$之间的距离，并查看它是否随着时间的推移而改善：

$$\begin{aligned}
    \|\mathbf{w}_{t+1} - \mathbf{w}^*\|^2 & = \|\mathbf{w}_{t} - \eta_t \partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w}) - \mathbf{w}^*\|^2 \\
    & = \|\mathbf{w}_{t} - \mathbf{w}^*\|^2 + \eta_t^2 \|\partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w})\|^2 - 2 \eta_t
    \left\langle \mathbf{w}_t - \mathbf{w}^*, \partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w})\right\rangle.
   \end{aligned}
$$

梯度$\partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w})$可以从上面被Lipschitz常数$L$限定，因此我们得到了

$$\eta_t^2 \|\partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w})\|^2 \leq \eta_t^2 L^2.$$

我们最感兴趣的是$\mathbf{w}_t$和$\mathbf{w}^*$之间的距离在预期中如何变化。事实上，对于任何特定的步骤序列，距离都可能增加，这取决于我们遇到的$\mathbf{x}_t$步。因此我们需要约束内积。通过凸性我们得到了

$$
l(\mathbf{x}_t, \mathbf{w}^*) \geq l(\mathbf{x}_t, \mathbf{w}_t) + \left\langle \mathbf{w}^* - \mathbf{w}_t, \partial_{\mathbf{w}} l(\mathbf{x}_t, \mathbf{w}_t) \right\rangle.
$$

利用这两个不等式并将其插入到上面，我们得到了$t+1$时刻参数之间距离的界，如下所示：

$$\|\mathbf{w}_{t} - \mathbf{w}^*\|^2 - \|\mathbf{w}_{t+1} - \mathbf{w}^*\|^2 \geq 2 \eta_t (l(\mathbf{x}_t, \mathbf{w}_t) - l(\mathbf{x}_t, \mathbf{w}^*)) - \eta_t^2 L^2.$$

这意味着只要当前损失和最佳损失之间的预期差异超过$\eta_t L^2$，我们就会取得进展。由于前者必然收敛到$0$，因此学习率$\eta_t$也需要消失。

接下来我们对这个表达式进行期望。这就产生了

$$E_{\mathbf{w}_t}\left[\|\mathbf{w}_{t} - \mathbf{w}^*\|^2\right] - E_{\mathbf{w}_{t+1}\mid \mathbf{w}_t}\left[\|\mathbf{w}_{t+1} - \mathbf{w}^*\|^2\right] \geq 2 \eta_t [E[R[\mathbf{w}_t]] - R^*] -  \eta_t^2 L^2.$$

最后一步是对$t \in \{t, \ldots, T\}$的不等式求和。由于和望远镜和下降的较低的项，我们得到

$$\|\mathbf{w}_{0} - \mathbf{w}^*\|^2 \geq 2 \sum_{t=1}^T \eta_t [E[R[\mathbf{w}_t]] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$

注意，我们发现$\mathbf{w}_0$是给定的，因此可以放弃期望。上次定义

$$\bar{\mathbf{w}} := \frac{\sum_{t=1}^T \eta_t \mathbf{w}_t}{\sum_{t=1}^T \eta_t}.$$

然后通过凸性得出

$$\sum_t \eta_t E[R[\mathbf{w}_t]] \geq \sum \eta_t \cdot \left[E[\bar{\mathbf{w}}]\right].$$

把这个代入上面的不等式，就得到了界

$$
\left[E[\bar{\mathbf{w}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t}.
$$

这里$r^2 := \|\mathbf{w}_0 - \mathbf{w}^*\|^2$是参数初始选择和最终结果之间距离的界。简言之，收敛速度取决于损失函数通过Lipschitz常数$L$变化的速度，以及初始值$r$离最优性有多远。注意，边界是以$\bar{\mathbf{w}}$而不是$\mathbf{w}_T$为单位的。这是因为$\bar{\mathbf{w}}$是优化路径的平滑版本。现在让我们分析一下$\eta_t$的一些选择。

* **已知时间范围**。只要知道$r, L$和$T$，我们就可以选择$\eta = r/L \sqrt{T}$。这就产生了上限$r L (1 + 1/T)/2\sqrt{T} < rL/\sqrt{T}$。也就是说，我们以$\mathcal{O}(1/\sqrt{T})$的速度收敛到最优解。
* **未知时间范围**。每当我们想有一个好的解决方案的*任何时间$T$我们可以选择$\eta = \mathcal{O}(1/\sqrt{T})$。这就需要一个额外的对数因子，从而得到$\mathcal{O}(\log T / \sqrt{T})$的上界。

注意，对于强凸损失$l(\mathbf{x}, \mathbf{w}') \geq l(\mathbf{x}, \mathbf{w}) + \langle \mathbf{w}'-\mathbf{w}, \partial_\mathbf{w} l(\mathbf{x}, \mathbf{w}) \rangle + \frac{\lambda}{2} \|\mathbf{w}-\mathbf{w}'\|^2$，我们可以设计更快速收敛的优化调度。事实上，$\eta$中的指数衰减导致$\mathcal{O}(\log T / T)$形式的界。

## 随机梯度与有限样本

到目前为止，当谈到随机梯度下降时，我们玩得有点快和松。我们假设我们绘制实例$x_i$，通常带有来自某些分布$p(x, y)$的标签$y_i$，并且我们使用它以某种方式更新权重$w$。特别是，对于一个有限的样本量，我们只是认为离散分布$p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$允许我们对其进行SGD。

然而，这并不是我们所做的。在本节中的玩具示例中，我们只是将噪声添加到非随机梯度中，即，我们假装有$(x_i, y_i)$对。事实证明，这在这里是合理的（有关详细讨论，请参阅练习）。更令人不安的是，在以前的所有讨论中，我们显然没有这样做。相反，我们只对所有实例迭代了一次。要了解为什么这样做更可取，请考虑相反的情况，即我们从离散分布中抽样$n$个观测值，并进行替换。随机选择一个元素$i$的概率是$N^{-1}$。因此，选择它至少一次是必要的

$$P(\mathrm{choose~} i) = 1 - P(\mathrm{omit~} i) = 1 - (1-N^{-1})^N \approx 1-e^{-1} \approx 0.63.$$

类似的推理表明，${N \choose 1} N^{-1} (1-N^{-1})^{N-1} = \frac{N-1}{N} (1-N^{-1})^{N} \approx e^{-1} \approx 0.37$给出了精确选取一个样本一次的概率。这导致了方差的增加和数据效率的下降。因此，在实践中我们执行后者（这是本书中的默认选择）。最后一点注意，重复通过数据集以*不同*的随机顺序遍历它。

## 摘要

* 对于凸问题，我们可以证明，对于学习率的广泛选择，随机梯度下降将收敛到最优解。
* 对于深度学习来说，通常情况并非如此。然而，通过对凸问题的分析，我们可以深入了解如何进行优化，即逐步降低学习率，尽管不会太快。
* 当学习率太小或太大时就会出现问题。在实践中，只有经过多次实验才能找到合适的学习率。
* 当训练数据集中有更多的例子时，计算梯度下降的每次迭代的成本更高，因此在这些情况下，SGD是首选。
* SGD的最优性保证通常在非凸情况下不可用，因为需要检查的局部极小值的数量很可能是指数的。

## 练习

1. 对SGD进行不同学习速率和不同迭代次数的实验。特别地，绘制与最优解$(0, 0)$的距离作为迭代次数的函数。
1. 证明了对于函数$f(x_1, x_2) = x_1^2 + 2 x_2^2$，向梯度中加入正态噪声等价于最小化损失函数$l(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$，其中$x$来自正态分布。
    * 导出$\mathbf{x}$的分布均值和方差。
    * 证明这个性质对于$Q \succeq 0$的目标函数$f(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{\mu})^\top Q (\mathbf{x} - \mathbf{\mu})$一般成立。
1. 比较$\{(x_1, y_1), \ldots, (x_m, y_m)\}$有替代样本和无替代样本时SGD的收敛性。
1. 如果某个渐变（或者与其相关的某个坐标）始终大于所有其他渐变，您将如何更改SGD解算器？
1. 假设是$f(x) = x^2 (1 + \sin x)$。$f$有多少个本地最小值？你能改变$f$这样的方式，以尽量减少它需要评估所有的局部极小值？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/352)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/497)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1067)
:end_tab:
