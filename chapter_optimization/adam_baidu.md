# 亚当
:label:`sec_adam`

在本节之前的讨论中，我们遇到了许多有效优化的技术。让我们在这里详细回顾一下：

* 我们看到:numref:`sec_sgd`在解决优化问题时比梯度下降更有效，例如，由于其对冗余数据的固有弹性。
* 我们看到:numref:`sec_minibatch_sgd`通过矢量化提供了显著的额外效率，在一个小批量中使用更大的观测集。这是高效多机、多GPU和整体并行处理的关键。
* :numref:`sec_momentum`增加了一种机制，用于聚合过去梯度的历史，以加速收敛。
* :numref:`sec_adagrad`用于每个坐标缩放，以允许计算高效的预处理器。
* :numref:`sec_rmsprop`从学习率调整中解耦每个坐标缩放。

adam:cite:`Kingma.Ba.2014`将所有这些技术结合到一个有效的学习算法中。正如所料，这是一个算法，已成为相当流行的一个更强大和有效的优化算法，用于深度学习。不过，这并非没有问题。特别是，:cite:`Reddi.Kale.Kumar.2019`表明，有些情况下，亚当可以发散由于差方差控制。在后续工作中，:cite:`Zaheer.Reddi.Sachan.ea.2018`向亚当提出了一个热修复方案，称为Yogi，解决了这些问题。稍后再谈。现在让我们回顾一下Adam算法。

## 算法

Adam的一个关键组成部分是它使用指数加权移动平均（也称为泄漏平均）来获得动量和梯度的二阶矩的估计。也就是说，它使用状态变量

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

这里$\beta_1$和$\beta_2$是非负加权参数。他们的常见选择是$\beta_1 = 0.9$和$\beta_2 = 0.999$。也就是说，方差估计的移动速度比动量项慢得多。请注意，如果我们初始化$\mathbf{v}_0 = \mathbf{s}_0 = 0$，我们最初会有大量偏向较小值的偏差。这可以通过使用$\sum_{i=0}^t \beta^i = \frac{1 - \beta^t}{1 - \beta}$重新规范化术语来解决。相应地，归一化状态变量由下式给出

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \text{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

有了正确的估计，我们现在可以写出更新方程了。首先，我们以一种非常类似于RMSProp的方式重新缩放梯度以获得

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$

与RMSProp不同，我们的更新使用了动量$\hat{\mathbf{v}}_t$，而不是梯度本身。此外，由于使用$\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$而不是$\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$进行重缩放，因此在外观上略有不同。前者在实践中的效果可以说稍好一些，因此偏离了RMSProp。通常我们选择$\epsilon = 10^{-6}$是为了在数值稳定性和保真度之间进行良好的权衡。

现在我们已经准备好了计算更新的所有部分。这是有点虎头蛇尾，我们有一个简单的形式更新

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$

回顾亚当的设计，它的灵感是显而易见的。动量和尺度在状态变量中清晰可见。它们相当奇特的定义迫使我们使用debias术语（这可以通过稍微不同的初始化和更新条件来修复）。其次，考虑到RMSProp，这两个术语的组合非常简单。最后，显式学习速率$\eta$允许我们控制步长来解决收敛问题。

## 实施

从无到有地实现Adam并不十分令人畏惧。为方便起见，我们将时间步长计数器$t$存储在`hyperparams`字典中。除此之外，一切都很简单。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adam_states(feature_dim):
    v_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return ((v_w, s_w), (v_b, s_b))

def adam(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(beta2 * s + (1 - beta2) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
```

我们已经准备好用Adam来训练模型了。我们使用$\eta = 0.01$的学习率。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

由于`adam`是作为胶子`trainer`优化库的一部分提供的算法之一，因此更简洁的实现是直接的。因此，我们只需要在glion中传递实现的配置参数。

```{.python .input}
d2l.train_concise_ch11('adam', {'learning_rate': 0.01}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

## 瑜伽士

Adam的一个问题是，当$\mathbf{s}_t$的二阶矩估计崩溃时，它甚至在凸环境下也不能收敛。作为一个补丁，:cite:`Zaheer.Reddi.Sachan.ea.2018`为$\mathbf{s}_t$提出了一个改进的更新（和初始化）。为了理解发生了什么，让我们重写Adam更新如下：

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$

每当$\mathbf{g}_t^2$具有高方差或更新稀疏时，$\mathbf{s}_t$可能会过快忘记过去的值。一个可能的解决办法是用$\mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$替换$\mathbf{g}_t^2 - \mathbf{s}_{t-1}$。现在，更新的大小不再取决于偏差的大小。这将产生瑜伽更新

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$

作者进一步建议在更大的初始批次上初始化动量，而不仅仅是初始逐点估计。我们省略了细节，因为它们对讨论并不重要，而且即使没有这种趋同，仍然相当不错。

```{.python .input}
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = s + (1 - beta2) * np.sign(
            np.square(p.grad) - s) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab pytorch
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab tensorflow
def yogi(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(s + (1 - beta2) * tf.math.sign(
                   tf.math.square(grad) - s) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

## 摘要

* Adam将许多优化算法的特性结合到一个相当健壮的更新规则中。
* 在RMSProp的基础上创建的Adam还对minibatch随机梯度使用EWMA。
* 亚当在估计动量和第二个瞬间时，使用偏差修正来调整缓慢的启动。
* 对于具有显著方差的梯度，我们可能会遇到收敛问题。可以通过使用更大的小批量或切换到$\mathbf{s}_t$的改进估计值来修正。瑜伽士提供了这样一种选择。

## 练习

1. 调整学习率，观察分析实验结果。
1. 你能重写动量和第二时刻的更新，这样就不需要偏差修正了吗？
1. 为什么在我们趋同的过程中需要降低学习率$\eta$？
1. 试着构造一个亚当发散，瑜伽士收敛的例子？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/358)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1078)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1079)
:end_tab:
