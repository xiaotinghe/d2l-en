# RMSProp公司
:label:`sec_rmsprop`

:numref:`sec_adagrad`中的一个关键问题是，学习率在预定义的$\mathcal{O}(t^{-\frac{1}{2}})$计划中降低。虽然这通常适用于凸问题，但对于非凸问题可能并不理想，例如在深度学习中遇到的问题。然而，Adagrad的坐标自适应性作为一个预条件是非常可取的。

:cite:`Tieleman.Hinton.2012`提出了RMSProp算法作为一个简单的解决方案，将速率调度与协调自适应学习速率解耦。问题是Adagrad将梯度$\mathbf{g}_t$的平方累加成一个状态向量$\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$。结果$\mathbf{s}_t$由于缺乏标准化而保持无边界增长，基本上随着算法的收敛呈线性增长。

解决这个问题的一种方法是使用$\mathbf{s}_t / t$。对于$\mathbf{g}_t$的合理分配，这将收敛。不幸的是，这可能需要很长时间，直到极限行为开始起作用，因为过程会记住值的完整轨迹。另一种方法是以我们在动量法中使用的相同方式使用泄漏平均值，即$\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$用于某些参数$\gamma > 0$。保持所有其他部件不变将产生RMSProp。

## 算法

让我们详细地写出方程式。

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

常数$\epsilon > 0$通常设置为$10^{-6}$，以确保我们不会遭受被零或过大步长除法的影响。考虑到这种扩展，我们现在可以自由地控制学习速率$\eta$，而不依赖于在每个坐标基础上应用的缩放。对于泄漏平均数，我们可以应用与动量法相同的推理。扩展$\mathbf{s}_t$的定义

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$

与:numref:`sec_momentum`之前一样，我们使用$1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$。因此，权重之和标准化为$1$，观察的半衰期为$\gamma^{-1}$。让我们想象一下$\gamma$的各种选择在过去40个时间步的权重。

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

如前所述，我们使用二次函数$f(\mathbf{x})=0.1x_1^2+2x_2^2$来观察RMSProp的轨迹。回想一下，在:numref:`sec_adagrad`中，当我们使用学习率为0.4的Adagrad时，由于学习率下降太快，变量在算法的后期移动非常缓慢。由于$\eta$是单独控制的，所以RMSProp不会发生这种情况。

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

接下来，我们将实现RMSProp，以便在深度网络中使用。这同样简单。

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

我们将初始学习率设置为0.01，将加权项$\gamma$设置为0.9。也就是说，$\mathbf{s}$在过去$1/(1-\gamma) = 10$次平方梯度观测中平均聚集。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## 简明实现

由于RMSProp是一种相当流行的算法，因此它也可以在`Trainer`实例中使用。我们需要做的就是使用名为`rmsprop`的算法实例化它，将$\gamma$分配给参数`gamma1`。

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

* RMSProp与Adagrad非常相似，因为两者都使用梯度的平方来缩放系数。
* RMSProp与动量平均法分享了漏动量。然而，RMSProp使用这种技术来调整按系数的预处理器。
* 学习速度需要实验者在实践中安排。
* 系数$\gamma$确定调整每个坐标比例时的历史长度。

## 练习

1. 如果我们设置$\gamma = 1$会发生什么？为什么？
1. 旋转优化问题以最小化$f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$。收敛会发生什么？
1. 在一个真正的机器学习问题上，比如时装设计师的培训，试试RMSProp会发生什么。尝试不同的选择来调整学习率。
1. 您想在优化过程中调整$\gamma$吗？RMSProp对此有多敏感？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/356)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1074)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1075)
:end_tab:
