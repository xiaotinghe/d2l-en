# 序列模型
:label:`sec_sequence`

想象一下你正在看Netflix上的电影。作为一个优秀的Netflix用户，你决定对每一部电影都给予虔诚的评价。毕竟，一部好电影就是一部好电影，你想看更多的，对吧？事实证明，事情并不是那么简单。随着时间的推移，人们对电影的看法会发生很大的变化。事实上，心理学家甚至对某些效应有了命名：

* 根据别人的意见，有“锚定”。例如，奥斯卡颁奖后，相应电影的收视率上升，尽管它仍然是同一部电影。这种影响持续几个月，直到奖项被遗忘。结果表明，这种效应使评级提高了半个百分点以上
:cite:`Wu.Ahmed.Beutel.ea.2017`.
* 有一种“享乐适应”，即人类迅速适应，接受一种改善或恶化的情况作为新的常态。例如，在看了很多好电影之后，人们对下一部电影同样好或更好的期望很高。因此，即使是一部普通的电影，在看过许多精彩的电影之后，也可能被认为是糟糕的。
* 有季节性。很少有观众喜欢在八月看圣诞老人的电影。
* 在某些情况下，由于导演或演员在制作中的不当行为，电影变得不受欢迎。
* 有些电影成为邪教电影，因为他们几乎滑稽坏。*来自外太空的计划9和巨魔2因此声名狼藉。

简而言之，电影收视率决不是固定不变的。因此，使用时间动力学可以得到更准确的电影推荐:cite:`Koren.2009`。当然，序列数据不仅仅是关于电影收视率的。下面给出了更多的插图。

* 许多用户在打开应用程序时都有非常特殊的行为。例如，社交媒体应用在学生放学后更受欢迎。股市交易应用程序在市场开放时更常用。
* 要预测明天的股价要比填补我们昨天错过的股价的空白困难得多，尽管两者都只是估计一个数字。毕竟，先见之明比事后诸葛亮难得多。在统计学中，前者（超出已知观测值的预测）称为*外推*，而后者（在现有观测值之间进行估计）称为*内插*。
* 音乐、语音、文本和视频在本质上都是连续的。如果我们把他们换掉，他们就没什么意义了。标题“狗咬人”远没有“人咬狗”那么令人惊讶，尽管两个词完全相同。
* 地震具有很强的相关性，即大地震发生后，很可能会有几次较小的余震，比没有强震的余震要大得多。事实上，地震是时空相关的，也就是说，余震通常发生在很短的时间跨度和很近的距离内。
* 人类之间的互动是连续的，这可以从Twitter上的争斗、舞蹈模式和辩论中看出。

## 统计工具

我们需要统计工具和新的深层神经网络结构来处理序列数据。为了简单起见，我们以:numref:`fig_ftse100`所示的股票价格（富时100指数）为例。

![FTSE 100 index over about 30 years.](../img/ftse100.png)
:width:`400px`
:label:`fig_ftse100`

让我们用$x_t$表示价格，即在*时间步*$t \in \mathbb{Z}^+$，我们观察价格$x_t$。请注意，对于本文中的序列，$t$通常是离散的，并随整数或其子集而变化。假设一个想在$t$日股市表现良好的交易员通过via预测$x_t$

$$x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1).$$

### 自回归模型

为了实现这一点，我们的交易员可以使用回归模型，比如我们在:numref:`sec_linear_concise`中训练的模型。只有一个主要问题：输入$x_{t-1}, \ldots, x_1$的数量因$t$而异。也就是说，这个数字随着我们遇到的数据量的增加而增加，我们需要一个近似值来使这个计算变得容易处理。本章后面的大部分内容将围绕如何有效估计$P(x_t \mid x_{t-1}, \ldots, x_1)$展开。简而言之，它归结为以下两种策略。

首先，假设可能相当长的序列$x_{t-1}, \ldots, x_1$不是真的必要。在这种情况下，我们可能满足于一些长度为$\tau$的时间跨度，而只使用$x_{t-1}, \ldots, x_{t-\tau}$个观测值。直接的好处是，现在参数的数量总是相同的，至少对于$t > \tau$。这使我们能够训练一个深层网络，如上所述。这种模型将被称为“自回归模型”，因为它们实际上是在自己身上执行回归。

第二种策略，如:numref:`fig_sequence-model`所示，是保留一些过去观测的总结$h_t$，同时在$\hat{x}_t$预测之外更新$h_t$。这就产生了估算$x_t$和$\hat{x}_t = P(x_t \mid h_{t})$的模型，并且更新了$h_t = g(h_{t-1}, x_{t-1})$表格。由于$h_t$从未被观察到，这些模型也被称为*潜在自回归模型*。

![A latent autoregressive model.](../img/sequence-model.svg)
:label:`fig_sequence-model`

这两种情况都提出了一个明显的问题，即如何生成训练数据。一个典型的方法是使用历史观察来预测下一次的观察。显然，我们不指望时间静止不动。然而，一个常见的假设是，虽然$x_t$的特定值可能会改变，但至少序列本身的动力学不会改变。这是合理的，因为新的动态只是，新的，因此不可预测的使用数据，我们迄今为止。统计学家把不变的动力学称为“平稳的”。不管我们做什么，我们都将通过

$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{t-1}, \ldots, x_1).$$

请注意，如果我们处理离散对象（如单词）而不是连续数字，上述考虑仍然有效。唯一的区别是，在这种情况下，我们需要使用分类器而不是回归模型来估计$P(x_t \mid  x_{t-1}, \ldots, x_1)$。

### 马尔可夫模型

回想一下，在自回归模型中，我们仅使用$x_{t-1}, \ldots, x_{t-\tau}$而不是$x_{t-1}, \ldots, x_1$来估计$x_t$。只要这个近似是精确的，我们就说序列满足一个马尔可夫条件。特别地，如果$\tau = 1$，我们有一个*一阶马尔可夫模型*并且$P(x)$由下式给出

$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{t-1}) \text{ where } P(x_1 \mid x_0) = P(x_1).$$

当$x_t$只假设一个离散值时，这样的模型特别好，因为在这种情况下，动态规划可以用来精确计算链上的值。例如，我们可以高效地计算$P(x_{t+1} \mid x_{t-1})$：

$$\begin{aligned}
P(x_{t+1} \mid x_{t-1})
&= \frac{\sum_{x_t} P(x_{t+1}, x_t, x_{t-1})}{P(x_{t-1})}\\
&= \frac{\sum_{x_t} P(x_{t+1} \mid x_t, x_{t-1}) P(x_t, x_{t-1})}{P(x_{t-1})}\\
&= \sum_{x_t} P(x_{t+1} \mid x_t) P(x_t \mid x_{t-1})
\end{aligned}
$$

通过使用我们只需要考虑过去观测的非常短的历史的事实：$P(x_{t+1} \mid x_t, x_{t-1}) = P(x_{t+1} \mid x_t)$。深入了解动态规划的细节超出了本节的范围。控制和强化学习算法广泛使用这些工具。

### 因果关系

原则上，以相反的顺序展开$P(x_1, \ldots, x_T)$没有错。毕竟，通过条件作用，我们总是可以通过

$$P(x_1, \ldots, x_T) = \prod_{t=T}^1 P(x_t \mid x_{t+1}, \ldots, x_T).$$

事实上，如果我们有一个马尔可夫模型，我们也可以得到一个反向的条件概率分布。然而，在许多情况下，数据存在一个自然的方向，即在时间上前进。很明显，未来的事件不能影响过去。因此，如果我们改变$x_t$，我们可能会影响$x_{t+1}$的未来，但不是相反。也就是说，如果我们改变$x_t$，过去事件的分布不会改变。因此，应该更容易解释$P(x_{t+1} \mid x_t)$而不是$P(x_t \mid x_{t+1})$。例如，它已经表明，在某些情况下，我们可以找到$x_{t+1} = f(x_t) + \epsilon$的一些加性噪声$\epsilon$，而反过来不是真的:cite:`Hoyer.Janzing.Mooij.ea.2009`。这是一个好消息，因为这通常是我们感兴趣估计的前进方向。彼得斯等人的书对这个话题作了更多的解释:cite:`Peters.Janzing.Scholkopf.2017`。我们几乎没有触及它的表面。

## 培训

在回顾了这么多统计工具之后，让我们在实践中尝试一下。我们首先生成一些数据。为了简单起见，我们使用正弦函数和一些加性噪声来生成序列数据，时间步长为$1, 2, \ldots, 1000$。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon, init
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab mxnet, pytorch
T = 1000  # Generate a total of 1000 points
time = d2l.arange(1, T + 1, dtype=d2l.float32)
x = d2l.sin(0.01 * time) + d2l.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

```{.python .input}
#@tab tensorflow
T = 1000  # Generate a total of 1000 points
time = d2l.arange(1, T + 1, dtype=d2l.float32)
x = d2l.sin(0.01 * time) + d2l.normal([T], 0, 0.2)
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

接下来，我们需要将这样一个序列转换为我们的模型可以训练的特性和标签。基于嵌入维数$\tau$，我们将数据映射为$y_t = x_t$和$\mathbf{x}_t = [x_{t-\tau}, \ldots, x_{t-1}]$对。精明的读者可能已经注意到，这给了我们$\tau$较少的数据例子，因为我们没有足够的历史为第一$\tau$。一个简单的解决方法，特别是如果序列很长，就是丢弃这几个术语。或者我们可以用零填充序列。这里我们只使用前600个特征标签对进行训练。

```{.python .input}
#@tab mxnet, pytorch
tau = 4
features = d2l.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = d2l.reshape(x[tau:], (-1, 1))
```

```{.python .input}
#@tab tensorflow
tau = 4
features = tf.Variable(d2l.zeros((T - tau, tau)))
for i in range(tau):
    features[:, i].assign(x[i: T - tau + i])
labels = d2l.reshape(x[tau:], (-1, 1))
```

```{.python .input}
#@tab all
batch_size, n_train = 16, 600
# Only the first `n_train` examples are used for training
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)
```

在这里，我们保持结构相当简单：只是一个MLP，它有两个完全连接的层，ReLU激活和平方损失。

```{.python .input}
# A simple MLP
def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(10, activation='relu'),
            nn.Dense(1))
    net.initialize(init.Xavier())
    return net

# Square loss
loss = gluon.loss.L2Loss()
```

```{.python .input}
#@tab pytorch
# Function for initializing the weights of the network
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# A simple MLP
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# Square loss
loss = nn.MSELoss()
```

```{.python .input}
#@tab tensorflow
# Vanilla MLP architecture
def get_net():
    net = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu'),
                              tf.keras.layers.Dense(1)])
    return net

# Least mean squares loss
# Note: L2 Loss = 1/2 * MSE Loss. TensorFlow has MSE Loss that is slightly
# different from MXNet's L2Loss by a factor of 2. Hence we halve the loss
# value to get L2Loss in TF
loss = tf.keras.losses.MeanSquaredError()
```

现在我们准备好训练模型了。下面的代码基本上与前面部分中的训练循环相同，例如:numref:`sec_linear_concise`。因此，我们将不深入探讨太多细节。

```{.python .input}
def train(net, train_iter, loss, epochs, lr):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    for epoch in range(epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
```

```{.python .input}
#@tab tensorflow
def train(net, train_iter, loss, epochs, lr):
    trainer = tf.keras.optimizers.Adam()
    for epoch in range(epochs):
        for X, y in train_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out) / 2
                params = net.trainable_variables
                grads = g.gradient(l, params)
            trainer.apply_gradients(zip(grads, params))
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
```

## 预测

由于训练损失很小，我们希望我们的模型能够很好地工作。让我们看看这在实践中意味着什么。首先要检查的是模型是否能够很好地预测下一个时间步发生的事情，即“提前一步预测”。

```{.python .input}
#@tab all
onestep_preds = net(features)
d2l.plot([time, time[tau:]], [d2l.numpy(x), d2l.numpy(onestep_preds)], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))
```

领先一步的预测看起来不错，正如我们所料。即使是预测也值得信赖。然而，这只是一个小问题：如果我们只观察序列数据直到时间步604，我们就不能希望接收到所有未来一步预测的输入。相反，我们需要一步一步地前进：

$$
\hat{x}_{605} = f(x_{601}, x_{602}, x_{603}, x_{604}), \\
\hat{x}_{606} = f(x_{602}, x_{603}, x_{604}, \hat{x}_{605}), \\
\hat{x}_{607} = f(x_{603}, x_{604}, \hat{x}_{605}, \hat{x}_{606}),\\
\hat{x}_{608} = f(x_{604}, \hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}),\\
\hat{x}_{609} = f(\hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}, \hat{x}_{608}),\\
\ldots
$$

通常，对于高达$x_t$的观测序列，其在时间步长$t+k$处的预测输出$\hat{x}_{t+k}$被称为*$k$步进预测*。因为我们观察到$x_{604}$，它的$k$阶跃预测是$\hat{x}_{604+k}$。换句话说，我们将不得不使用我们自己的预测来进行多步预测。让我们看看进展如何。

```{.python .input}
#@tab mxnet, pytorch
multistep_preds = d2l.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        d2l.reshape(multistep_preds[i - tau: i], (1, -1)))
```

```{.python .input}
#@tab tensorflow
multistep_preds = tf.Variable(d2l.zeros(T))
multistep_preds[:n_train + tau].assign(x[:n_train + tau])
for i in range(n_train + tau, T):
    multistep_preds[i].assign(d2l.reshape(net(
        d2l.reshape(multistep_preds[i - tau: i], (1, -1))), ()))
```

```{.python .input}
#@tab all
d2l.plot([time, time[tau:], time[n_train + tau:]],
         [d2l.numpy(x), d2l.numpy(onestep_preds),
          d2l.numpy(multistep_preds[n_train + tau:])], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
```

正如上面的例子所示，这是一个惊人的失败。经过几个预测步骤后，预测很快就衰减到一个常数。为什么这个算法效果这么差？这最终是由于错误的积累。假设在步骤1之后，我们有一些错误$\epsilon_1 = \bar\epsilon$。现在第2步的*输入*被$\epsilon_1$扰动，因此对于某些常数$c$，我们在$\epsilon_2 = \bar\epsilon + c \epsilon_1$的顺序中遭受一些错误，依此类推。这种误差很快就会偏离实际观测值。这是一个普遍现象。例如，未来24小时的天气预报往往相当准确，但除此之外，准确率迅速下降。我们将在本章及以后讨论改进这一点的方法。

让我们通过计算$k = 1, 4, 16, 64$的整个序列的预测来更仔细地了解$k$步进预测中的困难。

```{.python .input}
#@tab all
max_steps = 64
```

```{.python .input}
#@tab mxnet, pytorch
features = d2l.zeros((T - tau - max_steps + 1, tau + max_steps))
# Column `i` (`i` < `tau`) are observations from `x` for time steps from
# `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# Column `i` (`i` >= `tau`) are the (`i - tau + 1`)-step-ahead predictions for
# time steps from `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau, tau + max_steps):
    features[:, i] = d2l.reshape(net(features[:, i - tau: i]), -1)
```

```{.python .input}
#@tab tensorflow
features = tf.Variable(d2l.zeros((T - tau - max_steps + 1, tau + max_steps)))
# Column `i` (`i` < `tau`) are observations from `x` for time steps from
# `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau):
    features[:, i].assign(x[i: i + T - tau - max_steps + 1].numpy())

# Column `i` (`i` >= `tau`) are the (`i - tau + 1`)-step-ahead predictions for
# time steps from `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau, tau + max_steps):
    features[:, i].assign(d2l.reshape(net((features[:, i - tau: i])), -1))
```

```{.python .input}
#@tab all
steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [d2l.numpy(features[:, tau + i - 1]) for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
```

这清楚地说明了当我们试图预测未来时，预测的质量是如何变化的。尽管4步预测看起来仍然不错，但超出这一步的任何预测几乎都是无用的。

## 摘要

* 内插法和外推法在难度上有很大差别。因此，如果您有一个序列，则在训练时始终尊重数据的时间顺序，即永远不要训练未来的数据。
* 序列模型需要专门的统计工具进行估计。两种流行的选择是自回归模型和潜变量自回归模型。
* 对于因果模型（例如，时间向前），估计前进方向通常比估计相反方向容易得多。
* 对于时间步$t$之前的观测序列，其在时间步$t+k$处的预测输出是$k$*-步进预测*。当我们通过增加$k$进一步预测时，误差累积，预测的质量下降，通常是显著的。

## 练习

1. 在本部分的实验中对模型进行了改进。
    1. 是否包含超过过去4个观察结果？你真的需要多少？
    1. 如果没有噪音，你需要多少过去的观察？提示：你可以把$\sin$和$\cos$写成微分方程。
    1. 你能在保持特征总数不变的情况下合并旧的观察结果吗？这能提高精确度吗？为什么？
    1. 改变神经网络结构并评估其性能。
1. 投资者希望找到一个好的证券来购买。他观察过去的回报，以决定哪一个可能表现良好。这种策略可能会出什么问题？
1. 因果关系也适用于文本吗？到什么程度？
1. 举例说明何时可能需要一个潜在的自回归模型来捕捉数据的动态。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/113)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/114)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1048)
:end_tab:
