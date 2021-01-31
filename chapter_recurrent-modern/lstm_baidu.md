# 长短时记忆
:label:`sec_lstm`

长期以来，潜变量模型存在着长期信息保存和短期输入跳跃的问题。解决这个问题的最早方法之一是长-短期记忆（LSTM）:cite:`Hochreiter.Schmidhuber.1997`。它共享GRU的许多属性。有趣的是，LSTMs的设计比GRUs稍微复杂一些，但比GRUs早了近20年。

## 门控存储器

可以说LSTM的设计灵感来自计算机的逻辑门。LSTM引入了一个形状与隐藏状态相同的*存储单元*（简称*单元*）（一些文献认为存储单元是隐藏状态的一种特殊类型），用于记录附加信息。为了控制存储单元，我们需要许多门。需要一个门来读取单元格中的条目。我们称之为
*输出门*。
需要第二个门来决定何时将数据读入单元。我们称之为*输入门*。最后，我们需要一种机制来重置单元格的内容，由一个*忘记门*控制。这种设计的动机与GRUs的动机是一样的，即能够通过一种专门的机制来决定何时记住和何时忽略隐藏状态中的输入。让我们看看这在实践中是如何运作的。

### 输入门、忘记门和输出门

与GRUs一样，输入LSTM门的数据是当前时间步的输入和前一时间步的隐藏状态，如:numref:`lstm_0`所示。它们由三个完全连接的层处理，用一个sigmoid激活函数来计算输入的值。和输出门。因此，三个门的值在$(0, 1)$的范围内。

![Computing the input gate, the forget gate, and the output gate in an LSTM model.](../img/lstm-0.svg)
:label:`lstm_0`

在数学上，假设有$h$个隐藏单元，批量大小是$n$，输入的数量是$d$。因此，输入是$\mathbf{X}_t \in \mathbb{R}^{n \times d}$，上一时间步的隐藏状态是$\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$。相应地，时间步$t$处的门定义如下：输入门是$\mathbf{I}_t \in \mathbb{R}^{n \times h}$，忘记门是$\mathbf{F}_t \in \mathbb{R}^{n \times h}$，输出门是$\mathbf{O}_t \in \mathbb{R}^{n \times h}$。计算如下：

$$
\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xi} + \mathbf{H}_{t-1} \mathbf{W}_{hi} + \mathbf{b}_i),\\
\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xf} + \mathbf{H}_{t-1} \mathbf{W}_{hf} + \mathbf{b}_f),\\
\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xo} + \mathbf{H}_{t-1} \mathbf{W}_{ho} + \mathbf{b}_o),
\end{aligned}
$$

其中$\mathbf{W}_{xi}, \mathbf{W}_{xf}, \mathbf{W}_{xo} \in \mathbb{R}^{d \times h}$和$\mathbf{W}_{hi}, \mathbf{W}_{hf}, \mathbf{W}_{ho} \in \mathbb{R}^{h \times h}$是权重参数，$\mathbf{b}_i, \mathbf{b}_f, \mathbf{b}_o \in \mathbb{R}^{1 \times h}$是偏差参数。

### 候选存储单元

接下来我们设计了存储单元。由于我们还没有指定各种门的操作，我们首先介绍*候选*存储单元$\tilde{\mathbf{C}}_t \in \mathbb{R}^{n \times h}$。其计算类似于上述三个门的计算，但使用值范围为$(-1, 1)$的$\tanh$函数作为激活函数。这导致在时间步骤$t$处的以下等式：

$$\tilde{\mathbf{C}}_t = \text{tanh}(\mathbf{X}_t \mathbf{W}_{xc} + \mathbf{H}_{t-1} \mathbf{W}_{hc} + \mathbf{b}_c),$$

其中$\mathbf{W}_{xc} \in \mathbb{R}^{d \times h}$和$\mathbf{W}_{hc} \in \mathbb{R}^{h \times h}$是权重参数，$\mathbf{b}_c \in \mathbb{R}^{1 \times h}$是偏差参数。

候选存储单元的快速图示如:numref:`lstm_1`所示。

![Computing the candidate memory cell in an LSTM model.](../img/lstm-1.svg)
:label:`lstm_1`

### 存储单元

在GRUs中，我们有一种机制来控制输入和遗忘（或跳过）。类似地，在LSTMs中，我们有两个专用门用于这样的目的：输入门$\mathbf{I}_t$控制我们通过$\tilde{\mathbf{C}}_t$考虑新数据的多少，忘记门$\mathbf{F}_t$处理我们保留的旧存储单元内容$\mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}$的多少。使用与之前相同的逐点乘法技巧，我们得到以下更新公式：

$$\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t.$$

如果忘记门总是大约为1，而输入门总是大约为0，则过去的存储单元$\mathbf{C}_{t-1}$将随着时间的推移而保存并传递到当前时间步长。这种设计是为了缓解消失梯度问题，并更好地捕捉序列中的长程依赖。

因此，我们得出:numref:`lstm_2`中的流程图。

![Computing the memory cell in an LSTM model.](../img/lstm-2.svg)

:label:`lstm_2`

### 隐藏状态

最后，我们需要定义如何计算隐藏状态$\mathbf{H}_t \in \mathbb{R}^{n \times h}$。这就是输出门发挥作用的地方。在LSTM中，它只是$\tanh$内存单元的门控版本。这确保$\mathbf{H}_t$的值始终在$(-1, 1)$的间隔内。

$$\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t).$$

当输出门接近1时，我们有效地将所有内存信息传递给预测器，而对于接近0的输出门，我们只将所有信息保留在内存单元中，不执行进一步的处理。

:numref:`lstm_3`提供了数据流的图形说明。

![Computing the hidden state in an LSTM model.](../img/lstm-3.svg)
:label:`lstm_3`

## 从头开始实施

现在让我们从头开始实现一个LSTM。与:numref:`sec_rnn_scratch`中的实验相同，我们首先加载时间机器数据集。

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

### 初始化模型参数

接下来我们需要定义并初始化模型参数。如前所述，超参数`num_hiddens`定义了隐藏单元的数量。我们按照标准差为0.01的高斯分布初始化权重，并将偏差设置为0。

```{.python .input}
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                np.zeros(num_hiddens, ctx=device))

    W_xi, W_hi, b_i = three()  # Input gate parameters
    W_xf, W_hf, b_f = three()  # Forget gate parameters
    W_xo, W_ho, b_o = three()  # Output gate parameters
    W_xc, W_hc, b_c = three()  # Candidate memory cell parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = np.zeros(num_outputs, ctx=device)
    # Attach gradients
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.python .input}
#@tab pytorch
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                d2l.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # Input gate parameters
    W_xf, W_hf, b_f = three()  # Forget gate parameters
    W_xo, W_ho, b_o = three()  # Output gate parameters
    W_xc, W_hc, b_c = three()  # Candidate memory cell parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

### 定义模型

在初始化函数中，LSTM的隐藏状态需要返回一个*附加*内存单元，其值为0，形状为（批大小，隐藏单元数）。因此，我们得到以下状态初始化。

```{.python .input}
def init_lstm_state(batch_size, num_hiddens, device):
    return (np.zeros((batch_size, num_hiddens), ctx=device),
            np.zeros((batch_size, num_hiddens), ctx=device))
```

```{.python .input}
#@tab pytorch
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))
```

实际模型的定义和我们之前讨论的一样：提供三个门和一个辅助存储单元。注意，只有隐藏状态被传递到输出层。存储器单元$\mathbf{C}_t$不直接参与输出计算。

```{.python .input}
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = npx.sigmoid(np.dot(X, W_xi) + np.dot(H, W_hi) + b_i)
        F = npx.sigmoid(np.dot(X, W_xf) + np.dot(H, W_hf) + b_f)
        O = npx.sigmoid(np.dot(X, W_xo) + np.dot(H, W_ho) + b_o)
        C_tilda = np.tanh(np.dot(X, W_xc) + np.dot(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * np.tanh(C)
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H, C)
```

```{.python .input}
#@tab pytorch
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)
```

### 训练与预测

让我们通过实例化:numref:`sec_rnn_scratch`中引入的`RNNModelScratch`类来训练LSTM，就像我们在:numref:`sec_gru`中所做的一样。

```{.python .input}
#@tab all
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                            init_lstm_state, lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

## 简明实现

使用高级API，我们可以直接实例化`LSTM`模型。这封装了我们在上面明确说明的所有配置细节。这段代码的速度要快得多，因为它使用编译运算符而不是Python来处理我们之前详细阐述的许多细节。

```{.python .input}
lstm_layer = rnn.LSTM(num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab))
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

LSTMs是具有非平凡状态控制的典型潜变量自回归模型。多年来已经提出了其许多变体，例如，多层、残余连接、不同类型的正则化。然而，由于序列的长程依赖性，训练LSTMs和其他序列模型（如gru）的成本相当高。稍后，我们将遇到替代模型，如变压器，可以在某些情况下使用。

## 摘要

* lstm有三种类型的门：输入门、忘记门和控制信息流的输出门。
* LSTM的隐层输出包括隐态和存储单元。只有隐藏状态被传递到输出层。记忆细胞完全是内部的。
* LSTM可以缓解消失和爆炸梯度。

## 练习

1. 调整超参数并分析它们对运行时间、复杂度和输出序列的影响。
1. 您需要如何更改模型以生成正确的单词，而不是字符序列？
1. 比较GRU、LSTM和常规RNN在给定隐藏维度下的计算开销。特别注意培训和推理成本。
1. 既然候选存储单元通过使用$\tanh$函数确保值范围在$-1$和$1$之间，为什么隐藏状态需要再次使用$\tanh$函数来确保输出值范围在$-1$和$1$之间？
1. 实现时间序列预测的LSTM模型，而不是字符序列预测。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/343)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1057)
:end_tab:
