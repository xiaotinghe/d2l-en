# 凸性
:label:`sec_convexity`

凸性在优化算法的设计中起着至关重要的作用。这在很大程度上是因为在这种情况下分析和测试算法要容易得多。换句话说，如果算法即使在凸设置下也表现不佳，我们就不应该希望在其他情况下看到很好的结果。此外，尽管深度学习中的优化问题一般是非凸的，但它们往往在局部极小值附近表现出凸优化问题的一些性质。这可能会产生令人兴奋的新优化变体，如:cite:`Izmailov.Podoprikhin.Garipov.ea.2018`。

## 基础知识

让我们从基础开始吧。

### 集合

集合是凸性的基础。简单地说，向量空间中的集合$X$是凸的，如果对于任何$a, b \in X$，连接$a$和$b$的线段也在$X$中。用数学术语来说，这意味着我们总共有$\lambda \in [0, 1]$个

$$\lambda \cdot a + (1-\lambda) \cdot b \in X \text{ whenever } a, b \in X.$$

这听起来有点抽象。考虑一下图片:numref:`fig_pacman`。第一个集合不是凸集，因为其中有不包含的线段。另外两套没有遇到这样的问题。

![Three shapes, the left one is nonconvex, the others are convex](../img/pacman.svg)
:label:`fig_pacman`

定义本身并不是特别有用，除非您可以使用它们做些什么。在本例中，我们可以查看并集和交集，如:numref:`fig_convex_intersect`中所示。假设$X$和$Y$是凸集。那么$X \cap Y$也是凸的。要了解这一点，请考虑任何$a, b \in X \cap Y$。因为$X$和$Y$是凸的，所以连接$a$和$b$的线段都包含在$X$和$Y$中。考虑到这一点，它们也需要包含在$X \cap Y$中，从而证明了我们的第一个定理。

![The intersection between two convex sets is convex](../img/convex-intersect.svg)
:label:`fig_convex_intersect`

我们可以不费吹灰之力就加强这一结果：给定凸集$X_i$，它们的交集$\cap_{i} X_i$是凸的。为了看到相反的情况不是真的，考虑两个不相交的集合$X \cap Y = \emptyset$。现在选择$a \in X$和$b \in Y$。:numref:`fig_nonconvex`中连接$a$和$b$的线段需要包含既不在$X$中也不在$Y$中的某个部分，因为我们假设$X \cap Y = \emptyset$。因此，线段也不在$X \cup Y$，从而证明凸集的并一般不需要是凸的。

![The union of two convex sets need not be convex](../img/nonconvex.svg)
:label:`fig_nonconvex`

深度学习中的问题通常定义在凸域上。例如，$\mathbb{R}^d$是凸集(毕竟，$\mathbb{R}^d$中的任意两点之间的线保持在$\mathbb{R}^d$中)。在某些情况下，我们使用有限长度的变量，例如半径为$r$的球(由$\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\|_2 \leq r\}$定义)。

### 功能

现在我们有了凸集，我们可以引入凸函数$f$。给定凸集$X$，定义在其上的函数$f: X \to \mathbb{R}$是凸的，如果对所有$x, x' \in X$和对所有$\lambda \in [0, 1]$都有

$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

为了说明这一点，让我们绘制几个函数，并检查哪些函数满足要求。我们需要导入几个库。

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

正如预期的那样，余弦函数是非凸的，而抛物线和指数函数是非凸的。请注意，$X$是凸集的要求是该条件有意义所必需的。否则，$f(\lambda x + (1-\lambda) x')$的结果可能不会被很好地定义。凸函数具有许多理想的性质。

### 詹森不等式

最有用的工具之一是詹森不等式。它相当于凸度定义的推广：

$$\begin{aligned}
    \sum_i \alpha_i f(x_i) & \geq f\left(\sum_i \alpha_i x_i\right)
    \text{ and }
    E_x[f(x)] & \geq f\left(E_x[x]\right),
\end{aligned}$$

其中$\alpha_i$是非负实数，使得$\sum_i \alpha_i = 1$。换句话说，凸函数的期望值大于期望值的凸函数。为了证明第一个不等式，我们一次将凸性的定义重复应用于和中的一项。这个期望值可以通过在有限段上取极限来证明。

Jensen不等式的一个常见应用是关于部分观测随机变量的对数似然。也就是说，我们使用

$$E_{y \sim P(y)}[-\log P(x \mid y)] \geq -\log P(x).$$

这是从$\int P(y) P(x \mid y) dy = P(x)$开始的。这在变分方法中使用。这里，$y$通常是未观察到的随机变量，$P(y)$是它可能如何分布的最佳猜测，$P(x)$是$y$积分出来的分布。例如，在群集中，$y$可以是群集标签，而$P(x \mid y)$是应用群集标签时的生成模型。

## 属性

凸函数有一些有用的性质。我们对它们的描述如下。

### 局部最小化就是全局最小化

特别地，凸函数的局部极小值也是全局极小值。让我们假设相反，并证明它是错误的。如果$x^{\ast} \in X$是局部最小值，使得存在小的正值$p$，则对于满足$0 < |x - x^{\ast}| \leq p$的$x \in X$，存在$f(x^{\ast}) < f(x)$。假设存在$x' \in X$，其中$f(x') < f(x^{\ast})$。根据凸性的性质，

$$\begin{aligned}
    f(\lambda x^{\ast} + (1-\lambda) x') &\leq \lambda f(x^{\ast}) + (1-\lambda) f(x') \\
    &< \lambda f(x^{\ast}) + (1-\lambda) f(x^{\ast}) \\
    &< f(x^{\ast}) \\
\end{aligned}$$

举个例子，有$\lambda \in [0, 1)$,$\lambda = 1 - \frac{p}{|x^{\ast} - x'|}$个，所以有$0 < |\lambda x^{\ast} + (1-\lambda) x' - x^{\ast}| \leq p$个。然而，因为$f(\lambda x^{\ast} + (1-\lambda) x') < f(x^{\ast})$，这违反了我们当地的最低声明。因此，并不存在$x' \in X$为$f(x') < f(x^{\ast})$的情况。本地最小值$x^{\ast}$也是全局最小值。

例如，函数$f(x) = (x-1)^2$具有$x=1$的局部最小值，它也是全局最小值。

```{.python .input}
#@tab all
f = lambda x: (x-1)**2
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```

凸函数的局部极小值也是全局极小值的事实是非常方便的。这意味着如果我们最小化功能，我们就不会“卡住”。但请注意，这并不意味着不能有多个全局最小值，或者甚至可能存在一个全局最小值。例如，函数$f(x) = \mathrm{max}(|x|-1, 0)$在间隔$[-1, 1]$上达到其最小值。相反，函数$f(x) = \exp(x)$在$\mathbb{R}$上没有达到最小值。对于$x \to -\infty$，它渐近到$0$，但是对于$x$，没有哪个是$f(x) = 0$。

### 凸函数与凸集

凸函数将凸集定义为*下-集*。它们被定义为

$$S_b := \{x | x \in X \text{ and } f(x) \leq b\}.$$

这样的集合是凸集。让我们快速证明这一点。请记住，对于任何$x, x' \in S_b$，我们都需要显示$\lambda x + (1-\lambda) x' \in S_b$等于$\lambda \in [0, 1]$。但这是从$f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b$开始的凸性定义直接引申出来的。

请看下面的功能$f(x, y) = 0.5 x^2 + \cos(2 \pi y)$。它显然是非凸的。相应地，水平集是非凸的。事实上，它们通常由不相交的集合组成。

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

当函数存在二阶导数时，很容易检查其凸性。我们所要做的就是检查是否$\partial_x^2 f(x) \succeq 0$，也就是它的所有特征值是否都是非负的。例如，函数$f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2_2$是凸的，因为$\partial_{\mathbf{x}}^2 f = \mathbf{1}$，即，它的导数是单位矩阵。

首先要认识到的是，我们只需要证明一维函数的这个性质。毕竟，一般来说，我们总是可以定义一些函数$g(z) = f(\mathbf{x} + z \cdot \mathbf{v})$。该函数的一阶和二阶导数分别为$g' = (\partial_{\mathbf{x}} f)^\top \mathbf{v}$和$g'' = \mathbf{v}^\top (\partial^2_{\mathbf{x}} f) \mathbf{v}$。特别地，当$\mathbf{v}$的黑森函数为正半定时，即当其所有特征值都大于零时，所有$\mathbf{v}$的Hessian为$g'' \geq 0$。因此，回到标量情况。

要了解凸函数的$f''(x) \geq 0$，我们使用的事实是

$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$

由于二阶导数是由有限差分上的极限给出的，因此得出

$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$

为了证明反之亦然，我们使用$f'' \geq 0$意味着$f'$是单调递增函数这一事实。让$a < x < b$等于$\mathbb{R}$中的3个点。我们用中值定理来表示

$$\begin{aligned}
f(x) - f(a) & = (x-a) f'(\alpha) \text{ for some } \alpha \in [a, x] \text{ and } \\
f(b) - f(x) & = (b-x) f'(\beta) \text{ for some } \beta \in [x, b].
\end{aligned}$$

单调性$f'(\beta) \geq f'(\alpha)$，因此

$$\begin{aligned}
    f(b) - f(a) & = f(b) - f(x) + f(x) - f(a) \\
    & = (b-x) f'(\beta) + (x-a) f'(\alpha) \\
    & \geq (b-a) f'(\alpha).
\end{aligned}$$

从几何上看，$f(x)$低于连接$f(a)$和$f(b)$的线，从而证明了凸性。我们省略了一个更正式的推导，而采用下面的图表。

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

凸优化的一个很好的性质是它允许我们有效地处理约束。也就是说，它允许我们解决以下形式的问题：

$$\begin{aligned} \mathop{\mathrm{minimize~}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \text{ subject to } & c_i(\mathbf{x}) \leq 0 \text{ for all } i \in \{1, \ldots, N\}.
\end{aligned}$$

这里，$f$是目标，并且函数$c_i$是约束函数。要了解这一点，请考虑$c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$的情况。在这种情况下，参数$\mathbf{x}$被约束到单位球。如果第二个约束是$c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$，则这对应于位于半空间上的所有$\mathbf{x}$。同时满足两个约束相当于选择一片球作为约束集。

### 拉格朗日函数

一般来说，求解约束优化问题是困难的。解决这个问题的一种方法源于物理学，它有一个相当简单的直觉。想象一下盒子里有一个球。球将滚动到最低的地方，重力将与盒子侧面可以施加在球上的力相平衡。简而言之，目标函数的梯度(即重力)将被约束函数的梯度抵消(由于墙的“向后推”，需要保持在盒子内)。请注意，任何不活动的约束(即球不接触墙)将不能对球施加任何力。

跳过拉格朗日函数$L$的推导(有关细节:cite:`Boyd.Vandenberghe.2004`，例如参见博伊德和范登伯格的书)，上述推理可以通过以下鞍点优化问题来表达：

$$L(\mathbf{x},\alpha) = f(\mathbf{x}) + \sum_i \alpha_i c_i(\mathbf{x}) \text{ where } \alpha_i \geq 0.$$

这里，变量$\alpha_i$是确保适当实施约束的所谓的*拉格朗日乘数*。他们被选得足够大，以确保$c_i(\mathbf{x}) \leq 0$代表全部$i$人。例如，对于任何$\mathbf{x}$，如果自然是$c_i(\mathbf{x}) < 0$，我们最终会选择$\alpha_i = 0$。此外，这是一个“鞍点”优化问题，其中一个人希望相对于$\alpha$“最大化”$L$，同时相对于$\mathbf{x}$“最小化”它。有大量文献解释如何达到函数$L(\mathbf{x}, \alpha)$。就我们的目的而言，只要知道鞍点$L$是原始约束优化问题的最优解就足够了。

### 罚则

至少近似满足约束优化问题的一种方式是适配拉格朗日函数$L$。我们不满足$c_i(\mathbf{x}) \leq 0$，而是简单地将$\alpha_i c_i(\mathbf{x})$添加到目标函数$f(x)$。这确保了约束不会被严重违反。

事实上，我们一直都在使用这个伎俩。考虑一下:numref:`sec_weight_decay`中的重量衰减。在它中，我们将$\frac{\lambda}{2} \|\mathbf{w}\|^2$加到目标函数中，以确保$\mathbf{w}$不会增长得太大。使用约束优化的观点，我们可以看到，这将确保$\|\mathbf{w}\|^2 - r^2 \leq 0$对于某个半径为$r$。调整值$\lambda$允许我们改变$\mathbf{w}$的大小。

一般而言，增加罚金是确保近似约束满足的好方法。在实践中，事实证明，这比确切的满足感要稳健得多。此外，对于非凸问题，使精确方法在凸情况下如此吸引人的许多性质(例如，最优性)不再成立。

### 投影

满足约束的另一种策略是投影。同样，我们以前遇到过它们，例如，在处理:numref:`sec_rnn_scratch`中的渐变裁剪时。在那里，我们确保渐变的长度以$c$孔为界

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, c/\|\mathbf{g}\|).$$

结果是$g$的“投影”到半径为$c$的球上。更一般地，(凸)集$X$上的投影被定义为

$$\mathrm{Proj}_X(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in X} \|\mathbf{x} - \mathbf{x}'\|_2.$$

因此，它是$X$到$\mathbf{x}$之间距离最近的点。这听起来有点抽象。:numref:`fig_projections`的解释稍微更清楚一些。它有两个凸集，一个圆和一个菱形。集内的点(黄色)保持不变。集外的点(黑色)映射到集内最近的点(红色)。虽然对于$L_2$个球来说，这使得方向保持不变，但通常情况下不需要这样，就像在钻石的情况下可以看到的那样。

![Convex Projections](../img/projections.svg)
:label:`fig_projections`

凸投影的用途之一是计算稀疏权向量。在本例中，我们将$\mathbf{w}$投影到一个$L_1$球上(后者是上图中菱形的一般版本)。

## 摘要

在深度学习的背景下，凸函数的主要目的是激励优化算法并帮助我们更详细地理解它们。在下面，我们将看到如何相应地推导出梯度下降和随机梯度下降。

* 凸集的交集是凸的。工会不是这样的。
* 凸函数的期望值大于期望值的凸函数(Jensen不等式)。
* 一个二次可微函数是凸的当且仅当它的二阶导数始终只有非负特征值。
* 可以通过拉格朗日函数添加凸约束。在实践中，只需将它们与惩罚一起添加到目标函数中。
* 投影映射到(凸集)中最接近原始点的点。

## 练习

1. 假设我们想要通过在集合内的点之间绘制所有直线并检查这些直线是否包含来验证集合的凸性。
    * 证明只检查边界上的点就足够了。
    * 证明只检查集合的顶点就足够了。
1. 用$B_p[r] := \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\|_p \leq r\}$表示半径为$r$的球，使用$p$范数。证明$B_p[r]$对所有$p \geq 1$都是凸的。
1. 给定的凸函数$f$和$g$表示$\mathrm{max}(f, g)$也是凸的。证明$\mathrm{min}(f, g)$不是凸的。
1. 证明了Softmax函数的正规化是凸的。更具体地说，证明了$f(x) = \log \sum_i \exp(x_i)$的凸性。
1. 证明线性子空间是凸集，即$X = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$。
1. 证明在具有$\mathbf{b} = 0$的线性子空间的情况下，投影$\mathrm{Proj}_X$对于某些矩阵$\mathbf{M}$可以写成$\mathbf{M} \mathbf{x}$。
1. 证明了对于凸二次可微函数$f$，我们可以将$f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$写成$\xi \in [0, \epsilon]$。
1. 给定向量$\mathbf{w} \in \mathbb{R}^d$与$\|\mathbf{w}\|_1 > 1$，计算$\ell_1$单位球上的投影。
    * 作为中间步骤，写出惩罚目标$\|\mathbf{w} - \mathbf{w}'\|_2^2 + \lambda \|\mathbf{w}'\|_1$并计算给定$\lambda > 0$的解。
    * 你能不能不用反复试验就能找到$\lambda$的“正确”值？
1. 给定凸集$X$和两个向量$\mathbf{x}$和$\mathbf{y}$证明投影从不增加距离，即$\|\mathbf{x} - \mathbf{y}\| \geq \|\mathrm{Proj}_X(\mathbf{x}) - \mathrm{Proj}_X(\mathbf{y})\|$。

[Discussions](https://discuss.d2l.ai/t/350)
