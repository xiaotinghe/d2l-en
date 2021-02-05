# RMSProp
:label:`sec_rmsprop`

:numref:`sec_adagrad`的一个关键问题是，学习率实际上是以$\mathcal{O}(t^{-\frac{1}{2}})$的预定义时间表降低的。虽然这通常适合于凸问题，但对于非凸问题，例如深度学习中遇到的问题，它可能不是理想的。然而，Adagrad的坐标自适应性作为预条件是非常可取的。

:cite:`Tieleman.Hinton.2012`提出了RMSProp算法，作为将速率调度与坐标自适应学习速率解耦的简单解决方案。问题是Adagrad将梯度$\mathbf{g}_t$的平方累加成状态向量$\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$。结果，由于缺乏规格化，$\mathbf{s}_t$继续无界限地增长，基本上随着算法的收敛而线性增长。

解决此问题的一种方法是使用$\mathbf{s}_t / t$。对于$\mathbf{g}_t$的合理分布，这将收敛。不幸的是，可能需要很长时间才能使限制行为变得重要，因为该过程会记住值的完整轨迹。另一种选择是以我们在动量法中使用的相同方式使用泄漏平均值，即对某些参数$\gamma > 0$使用$\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$。保持所有其他部分不变将生成RMSProp。

## 该算法

让我们把这些方程式详细地写出来。

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

常量$\epsilon > 0$通常设置为$10^{-6}$，以确保我们不会被零除或步长过大。给定该扩展，我们现在可以独立于在每个坐标基础上应用的缩放来自由地控制学习率$\eta$。在泄漏平均数方面，我们可以应用与以前应用于动量法的情况相同的推理。扩大$\mathbf{s}_t$收益率的定义

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$

和以前在:numref:`sec_momentum`中一样，我们使用$1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$。因此，权重总和归一化为$1$，观察的半衰期为$\gamma^{-1}$。让我们可视化一下过去40个时间步长对于$\gamma$的各种选择的权重。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

```{.python .input}
#@tab all
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```

## 从头开始实施

如前所述，我们使用二次函数$f(\mathbf{x})=0.1x_1^2+2x_2^2$来观察RMSProp的轨迹。回想一下，在:numref:`sec_adagrad`中，当我们使用学习率为0.4时的Adagrad时，由于学习率下降得太快，因此在算法的后期阶段，变量的移动非常缓慢。由于$\eta$是单独控制的，因此RMSProp不会出现这种情况。

```{.python .input}
#@tab all
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

接下来，我们实现了用于深度网络的RMSProp。这一点同样简单明了。

```{.python .input}
#@tab mxnet,pytorch
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

```{.python .input}
#@tab tensorflow
def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)
```

```{.python .input}
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

我们将初始学习率设置为$\gamma$，将加权项设置为0.9。也就是说，在过去的$\mathbf{s}$次方形梯度观测中，平均有$1/(1-\gamma) = 10$次聚合。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## 简明实施

由于RMSProp是一种相当流行的算法，因此在`Trainer`实例中也可以使用它。我们所需要做的就是使用名为`rmsprop`的算法实例化它，将$\gamma$赋给参数`gamma1`。

```{.python .input}
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

## 摘要

* RMSProp与Adagrad非常相似，因为它们都使用梯度的平方来缩放系数。
* RMSProp以强劲的势头分享了漏水的平均水平。然而，RMSProp使用该技术来调整系数式预处理器。
* 在实践中，学习速度需要实验者自行安排。
* 系数$\gamma$确定当调整每坐标比例时历史有多长。

## 练习

1. 如果我们设定为$\gamma = 1$，在实验上会发生什么？为什么？
1. 旋转优化问题以最小化$f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$。融合会发生什么情况？
1. 试试看RMSProp在一个真实的机器学习问题上发生了什么，比如关于Fashion-MNIST的培训。尝试使用不同的选项来调整学习速率。
1. 随着优化的进行，您是否希望调整$\gamma$？RMSProp对此有多敏感？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/356)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1074)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1075)
:end_tab:
