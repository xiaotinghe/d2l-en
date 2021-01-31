# 门控经常性单位(GRU)
:label:`sec_gru`

在:numref:`sec_bptt`中，我们讨论了如何在RNN中计算渐变。特别地，我们发现矩阵的长积会导致梯度的消失或爆炸。让我们简单地想一想，这种梯度异常在实践中意味着什么：

* 我们可能会遇到这样一种情况，即早期观测对于预测所有未来的观测具有非常重要的意义。考虑一下有些人为的情况，其中第一个观察包含校验和，目标是辨别序列末尾的校验和是否正确。在这种情况下，第一个令牌的影响至关重要。我们希望有一些机制来将重要的早期信息存储在*存储单元*中。如果没有这样的机制，我们将不得不给这个观测分配一个非常大的梯度，因为它影响到后来的所有观测。
* 我们可能会遇到一些令牌没有中肯观察的情况。例如，当解析网页时，可能存在与评估页面上传达的情感目的无关的辅助HTML代码。我们希望有某种机制来“跳过”潜在状态表示中的此类令牌。
* 我们可能会遇到在序列的各个部分之间存在逻辑中断的情况。例如，一本书的章节之间可能会有一个过渡，或者证券的熊市和牛市之间可能会有一个过渡。在这种情况下，如果有办法“重置”我们的内部状态表示，那就太好了。

已经提出了许多方法来解决这个问题。其中最早的一种是长短期记忆:cite:`Hochreiter.Schmidhuber.1997`，我们将在:numref:`sec_lstm`讨论。门控递归单元(Gru):cite:`Cho.Van-Merrienboer.Bahdanau.ea.2014`是一个稍微更精简的变体，它通常提供相当的性能，并且计算:cite:`Chung.Gulcehre.Cho.ea.2014`的速度要快得多。由于它的简单性，让我们从GRU开始。

## 门控隐藏态

普通RNN和GRU之间的关键区别在于后者支持隐藏状态的选通。这意味着我们有专门的机制来决定何时应该“更新”隐藏状态，以及什么时候应该“重置”隐藏状态。这些机制是学习的，它们解决了上面列出的问题。例如，如果第一个令牌非常重要，我们将学会在第一次观察之后不更新隐藏状态。同样，我们将学会跳过不相关的临时观察。最后，我们将学会在需要的时候重置潜伏状态。我们将在下面详细讨论这一点。

### 重置闸门和更新闸门

我们首先需要介绍的是*复位门*和*更新门*。我们将它们设计成具有$(0, 1)$中条目的向量，这样我们就可以执行凸组合。例如，复位门将允许我们控制我们可能仍然想要记住的先前状态的多少。同样，更新门将允许我们控制新状态中有多少只是旧状态的副本。

我们从设计这些大门开始。:numref:`fig_gru_1`示出了在给定当前时间步长的输入和前一时间步长的隐藏状态的情况下，用于GRU中的复位门和更新门的输入。两个门的输出由具有S形激活函数的两个全连接层给出。

![Computing the reset gate and the update gate in a GRU model.](../img/gru-1.svg)
:label:`fig_gru_1`

在数学上，对于给定的时间步长$t$，假设输入是小批量$\mathbf{X}_t \in \mathbb{R}^{n \times d}$(示例数量：$n$，输入数量：$d$)，并且前一时间步长的隐藏状态是$\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$(隐藏单元数量：$h$)。然后，如下计算复位门$\mathbf{R}_t \in \mathbb{R}^{n \times h}$和更新门$\mathbf{Z}_t \in \mathbb{R}^{n \times h}$：

$$
\begin{aligned}
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r),\\
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z),
\end{aligned}
$$

其中$\mathbf{W}_{xr}, \mathbf{W}_{xz} \in \mathbb{R}^{d \times h}$和$\mathbf{W}_{hr}, \mathbf{W}_{hz} \in \mathbb{R}^{h \times h}$是权重参数，$\mathbf{b}_r, \mathbf{b}_z \in \mathbb{R}^{1 \times h}$是偏差。注意，广播(见:numref:`subsec_broadcasting`)在求和过程中被触发。我们使用Sigmoid函数(如:numref:`sec_mlp`中介绍的)将输入值转换为区间$(0, 1)$。

### 候选隐藏状态

接下来，让我们将复位门$\mathbf{R}_t$与:eqref:`rnn_h_with_state`中的常规潜在状态更新机制集成。这会导致以下情况
*候选隐藏状态*
$\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$在时间步骤$t$：

$$\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{hh} + \mathbf{b}_h),$$
:eqlabel:`gru_tilde_H`

其中$\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$和$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$是权重参数，$\mathbf{b}_h \in \mathbb{R}^{1 \times h}$是偏差，符号$\odot$是阿达玛(基本)乘积运算符。这里，我们使用tanh形式的非线性来确保候选隐藏状态中的值保持在间隔$(-1, 1)$中。

结果是“候选”，因为我们仍然需要合并更新门的操作。与:eqref:`rnn_h_with_state`相比，现在可以通过$\mathbf{R}_t$和$\mathbf{H}_{t-1}$的元素乘法来减少前几个状态的影响，在:eqref:`gru_tilde_H`。每当复位门$\mathbf{R}_t$中的条目接近1时，我们恢复诸如在:eqref:`rnn_h_with_state`中的普通RNN。对于复位门$\mathbf{R}_t$的所有接近于0的条目，候选隐藏状态是以$\mathbf{X}_t$作为输入的最大似然处理的结果。因此，任何预先存在的隐藏状态都会被“重置”为默认值。

:numref:`fig_gru_2`示出了应用复位门之后的计算流程。

![Computing the candidate hidden state in a GRU model.](../img/gru-2.svg)
:label:`fig_gru_2`

### 隐藏状态

最后，我们需要结合更新门$\mathbf{Z}_t$的效果。这确定了新隐藏状态$\mathbf{H}_t \in \mathbb{R}^{n \times h}$仅是旧状态$\mathbf{H}_{t-1}$的程度以及使用了多少新候选状态$\tilde{\mathbf{H}}_t$。更新门$\mathbf{Z}_t$可以简单地通过在$\mathbf{H}_{t-1}$和$\tilde{\mathbf{H}}_t$之间进行元素式凸组合来用于此目的。这将导致GRU的最终更新公式：

$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.$$

每当更新门$\mathbf{Z}_t$接近1时，我们简单地保留旧状态。在这种情况下，来自$\mathbf{X}_t$的信息基本上被忽略，有效地跳过了依存关系链中的时间步骤$t$。相反，每当$\mathbf{Z}_t$接近0时，新的潜在状态$\mathbf{H}_t$接近候选潜在状态$\tilde{\mathbf{H}}_t$。这些设计可以帮助我们解决RNN中的消失梯度问题，并更好地捕获时间步长较大的序列的依赖关系。例如，如果更新门在整个子序列的所有时间步长都接近于1，则不管子序列的长度如何，在其开始的时间步长处的旧隐藏状态将被容易地保留并传递到其末尾。

:numref:`fig_gru_3`示出了在更新门起作用之后的计算流程。

![Computing the hidden state in a GRU model.](../img/gru-3.svg)
:label:`fig_gru_3`

总而言之，GRU具有以下两个显著特征：

* 重置门有助于捕获序列中的短期依赖关系。
* 更新门有助于捕获序列中的长期依赖关系。

## 从头开始实施

为了更好地理解GRU模型，让我们从头开始实现它。我们首先阅读我们在:numref:`sec_rnn_scratch`中使用的时光机数据集。下面给出了读取数据集的代码。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

### 正在初始化模型参数

下一步是初始化模型参数。我们从标准差为0.01的高斯分布中提取权重，并将偏差设置为0。超参数`num_hiddens`定义隐藏单元的数量。我们实例化与更新门、复位门、候选隐藏状态和输出层相关的所有权重和偏差。

```{.python .input}
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                np.zeros(num_hiddens, ctx=device))

    W_xz, W_hz, b_z = three()  # Update gate parameters
    W_xr, W_hr, b_r = three()  # Reset gate parameters
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = np.zeros(num_outputs, ctx=device)
    # Attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.python .input}
#@tab pytorch
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                d2l.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # Update gate parameters
    W_xr, W_hr, b_r = three()  # Reset gate parameters
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

### 定义模型

现在我们将定义隐藏状态初始化函数`init_gru_state`。就像`init_rnn_state`中定义的:numref:`sec_rnn_scratch`函数一样，该函数返回一个具有值全为零的形状(批次大小、隐藏单元数)的张量。

```{.python .input}
def init_gru_state(batch_size, num_hiddens, device):
    return (np.zeros(shape=(batch_size, num_hiddens), ctx=device), )
```

```{.python .input}
#@tab pytorch
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )
```

现在我们准备好定义GRU模型。其结构与基本RNN单元相同，只是更新方程更为复杂。

```{.python .input}
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = npx.sigmoid(np.dot(X, W_xz) + np.dot(H, W_hz) + b_z)
        R = npx.sigmoid(np.dot(X, W_xr) + np.dot(H, W_hr) + b_r)
        H_tilda = np.tanh(np.dot(X, W_xh) + np.dot(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)
```

```{.python .input}
#@tab pytorch
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

### 训练和预测

培训和预测工作的方式与:numref:`sec_rnn_scratch`中完全相同。训练结束后，我们打印出训练集和预测序列上的困惑，分别跟在提供的前缀“时间旅行者”和“旅行者”之后。

```{.python .input}
#@tab all
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

## 简明实施

在高级API中，我们可以直接实例化GPU模型。这封装了我们上面明确说明的所有配置细节。代码的速度要快得多，因为它使用编译的运算符，而不是我们前面详细说明的Python。

```{.python .input}
gru_layer = rnn.GRU(num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

## 摘要

* 门控RNN可以更好地捕获具有大时间步长距离的序列的依赖关系。
* 重置门有助于捕获序列中的短期依赖关系。
* 更新门有助于捕获序列中的长期依赖关系。
* 无论何时开启复位门，GRU都包含基本的RNN作为它们的极端情况。它们还可以通过打开更新门来跳过子序列。

## 练习

1. 假设我们只想使用在时间步骤$t'$的输入来预测在时间步骤$t > t'$的输出。每个时间步长的重置和更新门的最佳值是什么？
1. 调整超参数，分析它们对运行时间、复杂性和输出顺序的影响。
1. 比较`rnn.RNN`和`rnn.GRU`实现的运行时、困惑和输出字符串。
1. 如果您只实现GRU的一部分，例如，只有一个复位门或只有一个更新门，会发生什么情况？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/342)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1056)
:end_tab:
