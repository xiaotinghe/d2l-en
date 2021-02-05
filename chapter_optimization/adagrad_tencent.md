# 阿达格勒
:label:`sec_adagrad`

让我们首先考虑使用不经常出现的功能的学习问题。

## 稀疏特征与学习率

假设我们正在训练一个语言模型。为了获得良好的准确性，我们通常希望在保持训练的同时降低学习率，通常是以$\mathcal{O}(t^{-\frac{1}{2}})$或更慢的速度进行训练。现在考虑关于稀疏特征(即，仅很少出现的特征)的模型训练。这在自然语言中很常见，例如，我们看到“预处理”这个词的可能性要比“学习”小得多。然而，它在其他领域也很常见，如计算广告和个性化协作过滤。毕竟，有很多东西只有一小部分人感兴趣。

与不常用功能关联的参数仅在这些功能出现时接收有意义的更新。给定一个递减的学习率，我们可能最终会遇到这样一种情况，即公共特征的参数相当快地收敛到它们的最佳值，而对于不频繁的特征，我们在确定它们的最佳值之前仍然没有足够频繁地观察它们。换言之，对于频繁的特征，学习速率要么下降得太慢，要么对于不频繁的特征下降得太快。

解决这个问题的一个可能的办法是计算我们看到特定功能的次数，并将其用作调整学习率的时钟。也就是说，我们可以使用$\eta = \frac{\eta_0}{\sqrt{t + c}}$的学习率，而不是选择$\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$的学习率。这里$s(i, t)$对我们观察到的特征$i$的非零数进行计数，直到时间$t$。这实际上很容易实现，不需要任何有意义的开销。然而，当我们不是很稀疏，而是只有梯度通常非常小且很少很大的数据时，它就会失败。毕竟，人们不清楚在符合观察到的特征和不符合观察到的特征之间应该在哪里划清界限。

Adagrad by :cite:`Duchi.Hazan.Singer.2011`通过将相当粗糙的计数器$s(i, t)$替换为先前观察到的梯度的平方的集合来解决这一问题。特别是用$s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$作为调整学习率的手段。这有两个好处：第一，我们不再需要决定梯度何时足够大。其次，它会随着渐变的大小自动缩放。通常对应于较大渐变的坐标会显著缩小，而其他渐变较小的坐标则会得到更温和的处理。在实践中，这为计算广告和相关问题提供了一个非常有效的优化过程。但这隐藏了Adagrad固有的一些额外好处，这些好处在预处理的背景下得到了最好的理解。

## 预处理

凸优化问题有利于分析算法的特点。毕竟，对于大多数非凸问题来说，很难推导出有意义的理论保证，但“直觉”和“洞察力”往往会延续下去。让我们看看最小化$f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$的问题。

正如我们在:numref:`sec_momentum`中看到的，可以根据它的特征分解$\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$来重写这个问题，以得到一个非常简单的问题，其中每个坐标都可以单独求解：

$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$

这里我们使用了$\mathbf{x} = \mathbf{U} \mathbf{x}$，因此使用了$\mathbf{c} = \mathbf{U} \mathbf{c}$。修改后的问题具有最小值$\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$和最小值$-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$。这很容易计算，因为$\boldsymbol{\Lambda}$是一个包含$\mathbf{Q}$的特征值的对角矩阵。

如果我们稍微扰动$\mathbf{c}$，我们希望在$f$的最小值中只会有轻微的变化。不幸的是，情况并非如此。虽然$\mathbf{c}$中的细微变化会导致$\bar{\mathbf{c}}$中同样的细微变化，但对于$f$的最小化(和$\bar{f}$的最小化)来说，情况并非如此。当本征值$\boldsymbol{\Lambda}_i$很大时，我们在$\bar{x}_i$中只会看到很小的变化，最小的变化是$\bar{f}$。相反，对于小的$\boldsymbol{\Lambda}_i$,$\bar{x}_i$的变化可能是巨大的。最大和最小特征值之比称为优化问题的条件数。

$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$

如果条件数$\kappa$较大，则很难精确求解优化问题。我们需要确保在一个大的动态范围内获得正确的值时要小心。我们的分析引出了一个显而易见的问题，尽管有些天真：难道我们不能简单地通过扭曲空间来“修复”这个问题，使所有的特征值都是$1$吗？从理论上讲，这相当简单：我们只需要$\mathbf{Q}$的特征值和特征向量，就可以将问题从$\mathbf{x}$重新缩放到$\mathbf{z} := \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$中的1。在新坐标系中，$\mathbf{x}^\top \mathbf{Q} \mathbf{x}$可以简化为$\|\mathbf{z}\|^2$。唉，这是一个相当不切实际的建议。一般来说，计算特征值和特征向量比解决实际问题要昂贵得多。

虽然精确地计算特征值可能很昂贵，但猜测它们并计算它们，甚至稍微近似地计算，可能已经比什么都不做要好得多。特别地，我们可以使用$\mathbf{Q}$的对角线条目，并相应地重新调整其比例。这比计算特征值便宜得多。

$$\tilde{\mathbf{Q}} = \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$

在这种情况下，我们有$\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$，具体地说，$\tilde{\mathbf{Q}}_{ii} = 1$对应于全部$i$。在大多数情况下，这大大简化了条件数。例如，我们前面讨论的情况，这将完全消除手头的问题，因为问题是轴对齐的。

不幸的是，我们还面临着另一个问题：在深度学习中，我们通常甚至无法访问目标函数的二阶导数：对于$\mathbf{x} \in \mathbb{R}^d$，即使是小批量的二阶导数也可能需要$\mathcal{O}(d^2)$的空间和工作来计算，因此实际上是不可行的。Adagrad的巧妙想法是使用一个代表黑森的难以捉摸的对角线，它既相对便宜又有效-梯度本身的大小。

为了了解为什么这是可行的，让我们看一下$\bar{f}(\bar{\mathbf{x}})$。我们有那个

$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$

其中$\bar{\mathbf{x}}_0$是$\bar{f}$的最小值。因此，梯度的大小既取决于$\boldsymbol{\Lambda}$，也取决于离最佳值的距离。如果$\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$没有改变，这就是我们所需要的。毕竟，在这种情况下，梯度$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$的大小就足够了。由于AdaGrad是一种随机梯度下降算法，即使在最优的情况下，我们也会看到非零方差的梯度。因此，我们可以安全地使用梯度的方差作为黑森尺度的廉价替代。透彻的分析超出了本节的范围(可能有几页)。我们请读者查阅:cite:`Duchi.Hazan.Singer.2011`以了解详细情况。

## 该算法

让我们把上面的讨论正式化。我们使用变量$\mathbf{s}_t$来累加过去的梯度方差，如下所示。

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

在这里，操作是以坐标方式应用的。也就是说，$\mathbf{v}^2$具有条目$v_i^2$。同样，$\frac{1}{\sqrt{v}}$具有条目$\frac{1}{\sqrt{v_i}}$，并且$\mathbf{u} \cdot \mathbf{v}$具有条目$u_i v_i$。如前所述，$\eta$是学习率，$\epsilon$是确保不除以$0$的加法常数。最后，我们初始化$\mathbf{s}_0 = \mathbf{0}$。

就像在动量的情况下，我们需要跟踪辅助变量，在这种情况下，允许每个坐标的单个学习率。与sgd相比，这不会显著增加adagrad的成本，因为主要成本通常是计算$l(y_t, f(\mathbf{x}_t, \mathbf{w}))$及其导数。

请注意，在$\mathbf{s}_t$中累积平方渐变意味着$\mathbf{s}_t$基本上以线性速率增长(实际上比线性增长稍慢，因为渐变最初会减小)。这导致$\mathcal{O}(t^{-\frac{1}{2}})$的学习率，尽管在每个坐标的基础上进行了调整。对于凸问题，这是完全足够的。然而，在深度学习中，我们可能想要把学习速度降低得相当慢。这导致了一些Adagrad变体，我们将在后续章节中讨论这些变体。现在让我们看看它在二次凸问题中的表现。我们使用与前面相同的问题：

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

我们将使用之前相同的学习率(即$\eta = 0.4$)来实现Adagrad。由此可见，自变量的迭代轨迹更加平滑。然而，由于$\boldsymbol{s}_t$的累积效应，学习率不断衰减，因此自变量在迭代后期不会移动那么多。

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

当我们将学习率提高到$2$时，我们会看到更好的行为。这已经表明，即使在无噪音的情况下，学习率的下降也可能相当剧烈，我们需要确保参数适当收敛。

```{.python .input}
#@tab all
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

## 从头开始实施

就像动量法一样，Adagrad需要维护一个与参数形状相同的状态变量。

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

与:numref:`sec_minibatch_sgd`的实验相比，我们使用了更大的学习率来训练模型。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

## 简明实施

使用算法`adagrad`的`Trainer`个实例，我们可以在胶子中调用Adagrad算法。

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

* Adagrad在每个坐标的基础上动态降低学习速率。
* 它使用梯度的大小作为调整进展速度的手段-梯度大的坐标用较小的学习率进行补偿。
* 在深度学习问题中，由于内存和计算的限制，精确的二阶导数的计算通常是不可行的。渐变可以是一个有用的代理。
* 如果优化问题的结构相当不均匀，Adagrad可以帮助减轻失真。
* Adagrad对于稀疏特征特别有效，在稀疏特征中，对于不经常出现的术语，学习率需要降低得更慢。
* 在深度学习问题上，Adagrad有时会过于激进地降低学习率。我们将在:numref:`sec_adam`的背景下讨论缓解这一问题的策略。

## 练习

1. 证明对于正交矩阵$\mathbf{U}$和向量$\mathbf{c}$，以下条件成立：$\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$。为什么这意味着变量正交改变后扰动的大小不变？
1. 尝试$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$的Adagrad，也可以将目标函数旋转45度，即$f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$。它的行为有什么不同吗？
1. 证明[Gerschgorin圆theorem](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem)，其表示矩阵$\mathbf{M}$的特征值$\lambda_i$对于$j$中的至少一个选择满足$|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$。
1. 关于对角预处理矩阵$\mathrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{M})$的特征值，格施戈林定理告诉我们什么？
1. 尝试使用Adagrad来建立一个合适的深层网络，比如应用于时尚MNIST的:numref:`sec_lenet`。
1. 您需要如何修改Adagrad才能降低学习率？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/355)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1072)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1073)
:end_tab:
