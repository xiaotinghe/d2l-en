# 阿达德尔塔
:label:`sec_adadelta`

阿达德尔塔是阿达格拉德(:numref:`sec_adagrad`)的另一个变种。主要区别在于它减少了学习率适应坐标的量。此外，传统上它指的是没有学习率，因为它使用变化量本身作为未来变化的校准。该算法是在:cite:`Zeiler.2012`提出的。考虑到到目前为止对以前算法的讨论，这是相当简单的。

## 该算法

简而言之，阿达德尔塔使用两个状态变量，$\mathbf{s}_t$存储梯度的二阶矩的泄漏平均值，$\Delta\mathbf{x}_t$存储模型本身中参数变化的二阶矩的泄漏平均值。请注意，我们使用作者的原始符号和命名是为了与其他出版物和实现兼容(在Momentum、Adagrad、RMSProp和Adadelta中，没有其他实际原因需要使用不同的希腊变量来指示用于相同目的的参数)。

以下是Adadelta的技术细节。假设当前参数为$\rho$，我们将获得类似于:numref:`sec_rmsprop`的以下泄漏更新：

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

与:numref:`sec_rmsprop`的不同之处在于我们用重新缩放的梯度$\mathbf{g}_t'$执行更新，即，

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

那么什么是重新缩放的渐变$\mathbf{g}_t'$呢？我们可以这样计算：

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

其中$\Delta \mathbf{x}_{t-1}$是平方重新缩放梯度$\mathbf{g}_t'$的泄漏平均值。我们将$\Delta \mathbf{x}_{0}$初始化为$0$并在每个步骤用$\mathbf{g}_t'$更新它，即，

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

并且添加$\epsilon$(一个较小的值，例如$10^{-5}$)以保持数值稳定性。

## 实施

ADADELTA需要为每个变量维护两个状态变量$\mathbf{s}_t$和$\Delta\mathbf{x}_t$。这将产生以下实现。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # In-place updates via [:]
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-place updates via [:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

选择$\rho = 0.9$相当于每次参数更新的半衰期为10。这往往效果很好。我们得到以下行为。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

为了简化实现，我们只使用`adadelta`类中的`Trainer`算法。这将为更紧凑的调用生成以下一行代码。

```{.python .input}
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
# adadelta is not converging at default learning rate
# but it's converging at lr = 5.0
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate':5.0, 'rho': 0.9}, data_iter)
```

## 摘要

* Adadelta没有学习率参数。取而代之的是，它使用参数本身的变化率来调整学习率。
* Adadelta需要两个状态变量来存储梯度二阶矩和参数变化。
* Adadelta使用泄漏平均值来保持对适当统计数据的连续估计。

## 练习

1. 调整值$\rho$。会发生什么事？
1. 演示如何在不使用$\mathbf{g}_t'$的情况下实现算法。为什么这可能是个好主意？
1. 阿达德尔塔真的是免费学习吗？你能找到打破阿达德尔塔的优化问题吗？
1. 将Adadelta与Adagrad和RMS Prop进行比较，讨论它们的收敛行为。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/357)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1076)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1077)
:end_tab:
