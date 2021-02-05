# 阿达尔
:label:`sec_adagrad`

让我们首先考虑一些不常出现的学习问题。

## 稀疏特征和学习率

假设我们正在训练一个语言模型。为了获得良好的准确性，我们通常希望在继续训练时降低学习速度，通常是$\mathcal{O}(t^{-\frac{1}{2}})$或更低的速度。现在考虑一个稀疏特征的模型训练，也就是说，很少出现的特征。这在自然语言中很常见，例如，我们看到“预处理”这个词的可能性要比“学习”小得多。然而，它在其他领域也很常见，如计算广告和个性化协同过滤。毕竟，有很多事情只对少数人感兴趣。

与不常见特征相关联的参数仅在这些特征出现时接收有意义的更新。在学习率下降的情况下，常见特征的参数很快收敛到其最优值，而对于不常见的特征，在确定其最优值之前，我们仍然缺乏足够频繁的观察。换句话说，对于频繁的特征，学习率下降得太慢，对于不频繁的特征，学习率下降得太快。

解决这个问题的一个可能的方法是计算我们看到某个特性的次数，并将其用作调整学习速率的时钟。也就是说，我们可以使用$\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$，而不是选择$\eta = \frac{\eta_0}{\sqrt{t + c}}$的学习率。这里$s(i, t)$统计到$t$为止我们观察到的特性$i$的非零数。这实际上很容易实现，没有任何有意义的开销。然而，当我们没有非常稀疏的数据，而只是梯度通常很小，很少很大的数据时，它就失败了。毕竟，我们还不清楚在什么地方可以划出一条线来区分某个东西是否符合观察到的特征。

Adagrad by :cite:`Duchi.Hazan.Singer.2011`通过将相当粗糙的计数器$s(i, t)$替换为先前观察到的梯度的平方的集合来解决这个问题。特别是，它使用$s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$作为调整学习率的手段。这有两个好处：首先，我们不再需要仅仅决定梯度何时足够大。其次，它会随着梯度的大小自动缩放。通常对应于大梯度的坐标会显著缩小，而其他具有小梯度的坐标则会得到更温和的处理。在实践中，这导致了一个非常有效的优化程序计算广告和相关的问题。但这隐藏了Adagrad固有的一些额外的好处，这些好处最好在预处理的背景下理解。

## 预处理

凸优化问题有利于分析算法的特点。毕竟，对于大多数非凸问题来说，很难得到有意义的理论保证，但“直觉”和“洞察”往往会延续下去。让我们看看最小化$f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$的问题。

正如我们在:numref:`sec_momentum`中所看到的，可以根据特征分解$\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$重写这个问题，从而得到一个非常简化的问题，其中每个坐标可以单独求解：

$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$

在这里，我们使用$\mathbf{x} = \mathbf{U} \mathbf{x}$和$\mathbf{c} = \mathbf{U} \mathbf{c}$。修改后的问题的最小值为$\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$，最小值为$-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$。由于$\boldsymbol{\Lambda}$是一个包含$\mathbf{Q}$特征值的对角矩阵，因此计算起来容易得多。

如果我们稍微扰动$\mathbf{c}$，我们希望在$f$的最小值中只发现微小的变化。不幸的是，情况并非如此。虽然$\mathbf{c}$的微小变化导致$\bar{\mathbf{c}}$的微小变化，但$f$的最小值（和$\bar{f}$的最小值）并非如此。当特征值$\boldsymbol{\Lambda}_i$较大时，我们只会看到$\bar{x}_i$和$\bar{f}$的微小变化。相反，对于小型$\boldsymbol{\Lambda}_i$，$\bar{x}_i$的变化可能是巨大的。最大特征值与最小特征值之比称为优化问题的条件数。

$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$

如果条件数$\kappa$较大，则很难精确求解优化问题。我们需要确保我们在获得一个大的动态范围的价值观是正确的谨慎。我们的分析导致了一个明显的问题，尽管有点幼稚：我们不能简单地通过扭曲空间来“修复”这个问题，使得所有的特征值都是$1$。理论上这很容易：我们只需要$\mathbf{Q}$的特征值和特征向量就可以将问题从$\mathbf{x}$重缩放到$\mathbf{z} := \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$中的一个。在新的坐标系中，$\mathbf{x}^\top \mathbf{Q} \mathbf{x}$可以简化为$\|\mathbf{z}\|^2$。唉，这是一个相当不切实际的建议。计算特征值和特征向量通常比解决实际问题要贵得多。

虽然精确计算特征值可能会很昂贵，但猜测它们并计算它们甚至是近似值可能已经比什么都不做要好得多。特别是，我们可以使用对角线条目$\mathbf{Q}$并相应地重新缩放它。这比计算特征值要便宜得多。

$$\tilde{\mathbf{Q}} = \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$

在本例中，我们有$\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$，特别是$i$。在大多数情况下，这大大简化了条件数。例如，在我们前面讨论的案例中，这将完全消除手头的问题，因为问题是轴对齐的。

不幸的是，我们还面临着另一个问题：在深度学习中，我们通常甚至无法获得目标函数的二阶导数：对于$\mathbf{x} \in \mathbb{R}^d$，即使是在小批量上的二阶导数也可能需要$\mathcal{O}(d^2)$的空间和工作来计算，从而使其实际上不可行。Adagrad的独创性想法是用一个代理来表示Hessian曲线的那条难以捉摸的对角线，它计算起来既便宜又有效——梯度本身的大小。

为了了解为什么这样做，让我们看看$\bar{f}(\bar{\mathbf{x}})$。我们有这个

$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$

其中$\bar{\mathbf{x}}_0$是$\bar{f}$的最小值。因此，梯度的大小既取决于$\boldsymbol{\Lambda}$，也取决于距最优性的距离。如果$\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$没有改变，这就足够了。毕竟，在这种情况下，梯度$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$的大小就足够了。由于AdaGrad是一个随机梯度下降算法，我们将看到梯度与非零方差即使在最优性。因此，我们可以安全地使用梯度的方差作为海森尺度的廉价代理。彻底的分析超出了本节的范围（需要几页）。详情请读者参阅:cite:`Duchi.Hazan.Singer.2011`。

## 算法

让我们把上面的讨论正式化。我们使用变量$\mathbf{s}_t$来累积过去的梯度方差，如下所示。

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

这里的操作是按坐标应用的。即，$\mathbf{v}^2$具有条目$v_i^2$。同样，$\frac{1}{\sqrt{v}}$有条目$\frac{1}{\sqrt{v_i}}$，$\mathbf{u} \cdot \mathbf{v}$有条目$u_i v_i$。和前面一样，$\eta$是学习率，$\epsilon$是一个加法常数，确保我们不被$0$除。最后，我们初始化$\mathbf{s}_0 = \mathbf{0}$。

就像在动量的情况下，我们需要跟踪一个辅助变量，在这种情况下，允许每个坐标有一个单独的学习率。相对于新加坡元，这不会显著增加Adagrad的成本，因为主要成本通常是计算$l(y_t, f(\mathbf{x}_t, \mathbf{w}))$及其衍生产品。

请注意，在$\mathbf{s}_t$中累积平方梯度意味着$\mathbf{s}_t$基本上是以线性速率增长的（实际上比线性增长慢一些，因为梯度最初会减小）。这导致了$\mathcal{O}(t^{-\frac{1}{2}})$的学习率，尽管在每个坐标的基础上进行了调整。对于凸问题，这是完全足够的。不过，在深度学习中，我们可能希望更缓慢地降低学习速度。这导致了一些Adagrad变体，我们将在后面的章节中讨论。现在让我们看看它在二次凸问题中的表现。我们使用与以前相同的问题：

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

我们将使用之前相同的学习速率实现Adagrad，即$\eta = 0.4$。我们可以看到，自变量的迭代轨迹更平滑。然而，由于$\boldsymbol{s}_t$的累积效应，学习率不断衰减，因此自变量在迭代的后期不会移动太多。

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

```{.python .input}
#@tab all
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

当我们将学习率提高到$2$时，我们看到了更好的行为。这已经表明，即使在无噪声的情况下，学习率的降低也可能相当严重，我们需要确保参数适当收敛。

```{.python .input}
#@tab all
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

## 从头开始实施

就像动量法一样，Adagrad需要保持一个与参数形状相同的状态变量。

```{.python .input}
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def init_adagrad_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)

def adagrad(params, grads, states, hyperparams):
    eps = 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(s + tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

与:numref:`sec_minibatch_sgd`的实验相比，我们采用了更大的学习率来训练模型。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

## 简明实现

使用`adagrad`算法的`Trainer`实例，我们可以在胶子中调用Adagrad算法。

```{.python .input}
d2l.train_concise_ch11('adagrad', {'learning_rate': 0.1}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adagrad
d2l.train_concise_ch11(trainer, {'learning_rate' : 0.1}, data_iter)
```

## 摘要

* Adagrad在每个坐标的基础上动态地降低学习速率。
* 它使用梯度的大小作为一种调整进度的手段-具有较大梯度的坐标以较小的学习率进行补偿。
* 在深度学习问题中，由于记忆和计算的限制，计算精确的二阶导数通常是不可行的。梯度可以是一个有用的代理。
* 如果优化问题有一个相当不均匀的结构，Adagrad可以帮助减轻失真。
* Adagrad对于稀疏特征尤其有效，在稀疏特征中，对于不经常出现的项，学习率需要更缓慢地降低。
* 在深度学习问题上，阿达格拉德有时会过于激进地降低学习率。我们将在:numref:`sec_adam`的背景下讨论缓解这种情况的策略。

## 练习

1. 证明了对于一个正交矩阵$\mathbf{U}$和一个向量$\mathbf{c}$，以下公式成立：$\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$。为什么这意味着变量正交变化后扰动的大小不会改变？
1. 为$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$试用Adagrad，也为目标函数旋转45度，即$f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$。它有不同的表现吗？
1. 证明[Gerschgorin圆定理](https://en.wikipedia.org/wiki/Gershgorin\u circle\u定理)表示矩阵$\mathbf{M}$的特征值$\lambda_i$满足$j$中至少一个选择的$|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$。
1. 关于对角预条件矩阵$\mathrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{M})$的特征值，Gerschgorin定理告诉了我们什么？
1. 尝试使用Adagrad来获得一个合适的深层网络，比如:numref:`sec_lenet`，当它应用于时尚MNIST时。
1. 您需要如何修改Adagrad以降低学习率的衰减？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/355)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1072)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1073)
:end_tab:
