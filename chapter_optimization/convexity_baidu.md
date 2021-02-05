# 凸性
:label:`sec_convexity`

凸性在优化算法的设计中起着至关重要的作用。这主要是因为在这种情况下，分析和测试算法要容易得多。换言之，如果算法即使在凸环境中也表现不佳，我们就不希望看到很好的结果。此外，即使深度学习中的优化问题一般都是非凸的，但在局部极小值附近往往表现出凸问题的一些性质。这会导致令人兴奋的新优化变体，如:cite:`Izmailov.Podoprikhin.Garipov.ea.2018`。

## 基础

让我们从基础开始。

### 套

集合是凸性的基础。简单地说，向量空间中的集合$X$是凸的，如果对于任何$a, b \in X$，连接$a$和$b$的线段也在$X$中。用数学术语来说，这意味着我们有$\lambda \in [0, 1]$个

$$\lambda \cdot a + (1-\lambda) \cdot b \in X \text{ whenever } a, b \in X.$$

这听起来有点抽象。考虑一下:numref:`fig_pacman`的图片。第一个集合不是凸的，因为其中有不包含的线段。另外两组没有这样的问题。

![Three shapes, the left one is nonconvex, the others are convex](../img/pacman.svg)
:label:`fig_pacman`

定义本身并不是特别有用，除非你能用它们做些什么。在本例中，我们可以查看:numref:`fig_convex_intersect`中所示的并集和交点。假设$X$和$Y$是凸集。$X \cap Y$也是凸的。要看到这一点，请考虑$a, b \in X \cap Y$。由于$X$和$Y$是凸的，连接$a$和$b$的线段包含在$X$和$Y$中。鉴于此，它们也需要包含在$X \cap Y$中，从而证明了我们的第一个定理。

![The intersection between two convex sets is convex](../img/convex-intersect.svg)
:label:`fig_convex_intersect`

我们可以用很少的努力来加强这个结果：给定凸集$X_i$，它们的交集$\cap_{i} X_i$是凸的。要知道相反的情况是不正确的，请考虑两个不相交的集合$X \cap Y = \emptyset$。现在选择$a \in X$和$b \in Y$。:numref:`fig_nonconvex`中连接$a$和$b$的线段需要包含一些既不在$X$中也不在$Y$中的部分，因为我们假设$X \cap Y = \emptyset$。因此，线段也不在$X \cup Y$中，从而证明了在一般情况下，凸集的并集不必是凸的。

![The union of two convex sets need not be convex](../img/nonconvex.svg)
:label:`fig_nonconvex`

深度学习中的问题通常定义在凸域上。例如，$\mathbb{R}^d$是一个凸集（毕竟，$\mathbb{R}^d$中任意两点之间的直线仍保留在$\mathbb{R}^d$中）。在某些情况下，我们处理长度有界的变量，如$\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\|_2 \leq r\}$定义的半径为$r$的球。

### 功能

现在我们有了凸集，我们可以引入凸函数$f$。给定一个凸集$X$，定义在它上面的一个函数$f: X \to \mathbb{R}$是凸的，如果对于所有$x, x' \in X$和$\lambda \in [0, 1]$我们都有

$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

为了说明这一点，让我们绘制一些函数，并检查哪些函数满足要求。我们需要导入一些库。

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

让我们定义几个函数，包括凸函数和非凸函数。

```{.python .input}
#@tab all
f = lambda x: 0.5 * x**2  # Convex
g = lambda x: d2l.cos(np.pi * x)  # Nonconvex
h = lambda x: d2l.exp(0.5 * x)  # Convex

x, segment = d2l.arange(-2, 2, 0.01), d2l.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
```

正如所料，余弦函数是非凸的，而抛物线和指数函数是非凸的。请注意，$X$是凸集的要求是有意义的条件所必需的。否则$f(\lambda x + (1-\lambda) x')$的结果可能无法很好地界定。凸函数有许多理想的性质。

### 詹森不等式

最有用的工具之一是詹森不等式。它相当于凸性定义的推广：

$$\begin{aligned}
    \sum_i \alpha_i f(x_i) & \geq f\left(\sum_i \alpha_i x_i\right)
    \text{ and }
    E_x[f(x)] & \geq f\left(E_x[x]\right),
\end{aligned}$$

其中$\alpha_i$是非负实数，例如$\sum_i \alpha_i = 1$。换句话说，凸函数的期望值大于期望值的凸函数。为了证明第一个不等式，我们一次重复地将凸性的定义应用于求和中的一项。这个期望可以用有限段上的极限来证明。

Jensen不等式的一个常见应用是关于部分观测随机变量的对数似然。也就是说，我们使用

$$E_{y \sim P(y)}[-\log P(x \mid y)] \geq -\log P(x).$$

以下为$\int P(y) P(x \mid y) dy = P(x)$起。这是用于变分方法。这里$y$是典型的未观察到的随机变量，$P(y)$是它可能如何分布的最佳猜测，$P(x)$是$y$整合出来的分布。例如，在集群中，$y$可能是集群标签，$P(x \mid y)$是应用集群标签时的生成模型。

## 属性

凸函数有一些有用的性质。我们将其描述如下。

### 局部极小就是全局极小

特别地，凸函数的局部极小也是全局极小。让我们假设相反的情况并证明它是错误的。如果$x^{\ast} \in X$是一个局部最小值，使得有一个小的正值$p$，那么对于满足$0 < |x - x^{\ast}| \leq p$的$x \in X$，有$f(x^{\ast}) < f(x)$。假设存在$x' \in X$，其中$f(x') < f(x^{\ast})$。根据凸性的性质，

$$\begin{aligned}
    f(\lambda x^{\ast} + (1-\lambda) x') &\leq \lambda f(x^{\ast}) + (1-\lambda) f(x') \\
    &< \lambda f(x^{\ast}) + (1-\lambda) f(x^{\ast}) \\
    &< f(x^{\ast}) \\
\end{aligned}$$

例如，存在$\lambda \in [0, 1)$、$\lambda = 1 - \frac{p}{|x^{\ast} - x'|}$，因此$0 < |\lambda x^{\ast} + (1-\lambda) x' - x^{\ast}| \leq p$。但是，由于$f(\lambda x^{\ast} + (1-\lambda) x') < f(x^{\ast})$，这违反了我们的本地最小声明。因此，不存在$x' \in X$，其中$f(x') < f(x^{\ast})$。局部最小值$x^{\ast}$也是全局最小值。

例如，函数$f(x) = (x-1)^2$具有$x=1$的局部最小值，它也是全局最小值。

```{.python .input}
#@tab all
f = lambda x: (x-1)**2
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```

凸函数的局部极小也是全局极小这一事实非常方便。这意味着，如果我们最小化函数，我们就不能“陷入困境”。不过，请注意，这并不意味着不能有多个全局最小值，或者甚至可能存在一个全局最小值。例如，函数$f(x) = \mathrm{max}(|x|-1, 0)$在间隔$[-1, 1]$上达到其最小值。相反，函数$f(x) = \exp(x)$没有在$\mathbb{R}$上获得最小值。对于$x \to -\infty$，它逐渐接近$0$，但是没有$x$是$f(x) = 0$。

### 凸函数与集

凸函数将凸集定义为*下集*。它们被定义为

$$S_b := \{x | x \in X \text{ and } f(x) \leq b\}.$$

这类集合是凸的。让我们尽快证明这一点。记住，对于任何$x, x' \in S_b$，我们需要显示$\lambda x + (1-\lambda) x' \in S_b$和$\lambda \in [0, 1]$一样长。但这直接源于$f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b$年以来凸性的定义。

请看下面的$f(x, y) = 0.5 x^2 + \cos(2 \pi y)$函数。它显然是非凸的。相应地，水平集是非凸的。事实上，它们通常由不相交的集合组成。

```{.python .input}
#@tab all
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 + 0.5 * d2l.cos(2 * np.pi * y)
# Plot the 3D surface
d2l.set_figsize((6, 4))
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.contour(x, y, z, offset=-1)
ax.set_zlim(-1, 1.5)
# Adjust labels
for func in [d2l.plt.xticks, d2l.plt.yticks, ax.set_zticks]:
    func([-1, 0, 1])
```

### 导数与凸性

只要函数的二阶导数存在，就很容易检查凸性。我们要做的就是检查$\partial_x^2 f(x) \succeq 0$，即它的所有特征值是否都是非负的。例如，函数$f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2_2$是凸的，因为$\partial_{\mathbf{x}}^2 f = \mathbf{1}$，即它的导数是单位矩阵。

首先要认识到的是，我们只需要证明一维函数的这个性质。毕竟，通常我们可以定义一些函数$g(z) = f(\mathbf{x} + z \cdot \mathbf{v})$。此函数具有一阶导数$g' = (\partial_{\mathbf{x}} f)^\top \mathbf{v}$和二阶导数$g'' = \mathbf{v}^\top (\partial^2_{\mathbf{x}} f) \mathbf{v}$。特别地，当$f$的Hessian是半正定的，即当它的所有特征值都大于零时，$\mathbf{v}$对所有$\mathbf{v}$。因此回到标量情况。

要知道$f''(x) \geq 0$对于凸函数，我们使用

$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$

由于二阶导数是由有限差分的极限给出的，因此

$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$

为了证明相反的结果是正确的，我们使用了$f'' \geq 0$这个事实，这意味着$f'$是一个单调递增的函数。$a < x < b$是$\mathbb{R}$的三个点。我们用中值定理来表示

$$\begin{aligned}
f(x) - f(a) & = (x-a) f'(\alpha) \text{ for some } \alpha \in [a, x] \text{ and } \\
f(b) - f(x) & = (b-x) f'(\beta) \text{ for some } \beta \in [x, b].
\end{aligned}$$

由单调性$f'(\beta) \geq f'(\alpha)$，因此

$$\begin{aligned}
    f(b) - f(a) & = f(b) - f(x) + f(x) - f(a) \\
    & = (b-x) f'(\beta) + (x-a) f'(\alpha) \\
    & \geq (b-a) f'(\alpha).
\end{aligned}$$

从几何学上看，$f(x)$位于连接$f(a)$和$f(b)$的线之下，因此证明了凸性。我们省略了一个更正式的推导，取而代之的是下面的图表。

```{.python .input}
#@tab all
f = lambda x: 0.5 * x**2
x = d2l.arange(-2, 2, 0.01)
axb, ab = d2l.tensor([-1.5, -0.5, 1]), d2l.tensor([-1.5, 1])
d2l.set_figsize()
d2l.plot([x, axb, ab], [f(x) for x in [x, axb, ab]], 'x', 'f(x)')
d2l.annotate('a', (-1.5, f(-1.5)), (-1.5, 1.5))
d2l.annotate('b', (1, f(1)), (1, 1.5))
d2l.annotate('x', (-0.5, f(-0.5)), (-1.5, f(-0.5)))
```

## 约束条件

凸优化的一个很好的特性是它允许我们有效地处理约束。也就是说，它允许我们解决形式上的问题：

$$\begin{aligned} \mathop{\mathrm{minimize~}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \text{ subject to } & c_i(\mathbf{x}) \leq 0 \text{ for all } i \in \{1, \ldots, N\}.
\end{aligned}$$

这里$f$是目标函数，$c_i$是约束函数。看看这是什么情况下考虑$c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$。在这种情况下，参数$\mathbf{x}$被约束到单位球。如果第二个约束是$c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$，则它对应于半空间上的所有$\mathbf{x}$。同时满足这两个约束相当于选择一个球片作为约束集。

### 拉格朗日函数

一般来说，求解约束优化问题是困难的。解决这个问题的一种方法来自物理学，它有一个相当简单的直觉。想象一个盒子里的球。球将滚动到最低的位置，重力将与箱子侧面施加在球上的力相平衡。简言之，目标函数的梯度（即重力）将被约束函数的梯度抵消（由于墙“推回”，需要保持在盒子内部）。请注意，任何未激活的约束（即，球未接触到墙）将无法对球施加任何力。

跳过拉格朗日函数$L$的推导（详见Boyd和Vandenberghe的书:cite:`Boyd.Vandenberghe.2004`），上述推理可通过以下鞍点优化问题表示：

$$L(\mathbf{x},\alpha) = f(\mathbf{x}) + \sum_i \alpha_i c_i(\mathbf{x}) \text{ where } \alpha_i \geq 0.$$

这里的变量$\alpha_i$是所谓的*拉格朗日乘子*，用于确保约束得到正确执行。它们的选择刚好足够大，以确保$c_i(\mathbf{x}) \leq 0$适用于所有$i$。例如，对于任何$\mathbf{x}$，对于$c_i(\mathbf{x}) < 0$，我们最终选择$\alpha_i = 0$。此外，这是一个*鞍点*优化问题，其中需要*最大化*$L$相对于$\alpha$，同时*最小化*它相对于$\mathbf{x}$。有大量的文献解释如何达到$L(\mathbf{x}, \alpha)$的功能。我们只需知道$L$的鞍点是原始约束优化问题最优解的位置。

### 处罚

至少近似满足约束优化问题的一种方法是采用拉格朗日函数$L$。我们只需将$\alpha_i c_i(\mathbf{x})$添加到目标函数$f(x)$，而不是满足$c_i(\mathbf{x}) \leq 0$。这样可以确保不会严重违反约束。

事实上，我们一直在使用这个技巧。考虑:numref:`sec_weight_decay`中的重量衰减。其中我们将$\frac{\lambda}{2} \|\mathbf{w}\|^2$添加到目标函数中，以确保$\mathbf{w}$不会增长过大。利用约束优化的观点我们可以看到，这将确保$\|\mathbf{w}\|^2 - r^2 \leq 0$对于一些半径$r$。调整$\lambda$的值可以改变$\mathbf{w}$的大小。

一般来说，添加惩罚是确保近似约束满足的好方法。在实践中，这被证明比精确的满足感更强大。此外，对于非凸问题，许多使精确方法在凸情况下如此吸引人的性质（如最优性）不再成立。

### 投影

另一种满足约束的策略是投影。同样，我们以前遇到过它们，例如，在处理:numref:`sec_rnn_scratch`中的渐变剪裁时。在这里，我们确保梯度的长度以$c$为界

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, c/\|\mathbf{g}\|).$$

这是$g$在半径为$c$的球上的投影。更一般地，在（凸）集$X$上的投影定义为

$$\mathrm{Proj}_X(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in X} \|\mathbf{x} - \mathbf{x}'\|_2.$$

因此，它是$X$到$\mathbf{x}$的最近点。这听起来有点抽象。:numref:`fig_projections`解释得更清楚一些。其中有两个凸集，一个圆和一个菱形。集合内的点（黄色）保持不变。集合外的点（黑色）映射到集合内最近的点（红色）。而对于$L_2$球，这使方向保持不变，这不必是一般情况下，因为可以看到在钻石的情况。

![Convex Projections](../img/projections.svg)
:label:`fig_projections`

凸投影的一个用途是计算稀疏权重向量。在本例中，我们将$\mathbf{w}$投影到$L_1$球上（后者是上图中菱形的通用版本）。

## 摘要

在深入学习的背景下，凸函数的主要目的是激励优化算法并帮助我们详细理解它们。下面我们将看到如何相应地导出梯度下降和随机梯度下降。

* 凸集的交集是凸的。工会不是。
* 凸函数的期望大于期望的凸函数（Jensen不等式）。
* 二次可微函数是凸的当且仅当其二阶导数始终只有非负特征值。
* 通过拉格朗日函数可以增加凸约束。实际上，只需在目标函数中加上一个惩罚。
* 投影映射到（凸）集中最靠近原点的点。

## 练习

1. 假设我们要通过在集合内的点之间绘制所有直线并检查这些直线是否包含来验证集合的凸性。
    * 证明只检查边界上的点就足够了。
    * 证明只检查集合的顶点就足够了。
1. 用$B_p[r] := \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\|_p \leq r\}$表示半径为$r$的球，使用$p$标准。证明$B_p[r]$对于所有$p \geq 1$都是凸的。
1. 给定的凸函数$f$和$g$也表明$\mathrm{max}(f, g)$是凸的。证明$\mathrm{min}(f, g)$不是凸的。
1. 证明了softmax函数的正规化是凸的。更具体地证明了$f(x) = \log \sum_i \exp(x_i)$的凸性。
1. 证明了线性子空间是凸集，即$X = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$。
1. 证明了在线性子空间为$\mathbf{b} = 0$的情况下，对于某些矩阵$\mathbf{M}$，投影$\mathrm{Proj}_X$可以写成$\mathbf{M} \mathbf{x}$。
1. 证明了对于凸二次可微函数$f$，对于某些$\xi \in [0, \epsilon]$，我们可以写出$f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$。
1. 给定向量$\mathbf{w} \in \mathbb{R}^d$和$\|\mathbf{w}\|_1 > 1$，计算$\ell_1$单位球上的投影。
    * 作为中间步骤，写出惩罚目标$\|\mathbf{w} - \mathbf{w}'\|_2^2 + \lambda \|\mathbf{w}'\|_1$并计算给定$\lambda > 0$的解。
    * 你能找到$\lambda$的“正确”值而不需要大量的尝试和错误吗？
1. 给定一个凸集$X$和两个向量$\mathbf{x}$和$\mathbf{y}$，证明投影永远不会增加距离，即$\|\mathbf{x} - \mathbf{y}\| \geq \|\mathrm{Proj}_X(\mathbf{x}) - \mathrm{Proj}_X(\mathbf{y})\|$。

[Discussions](https://discuss.d2l.ai/t/350)
