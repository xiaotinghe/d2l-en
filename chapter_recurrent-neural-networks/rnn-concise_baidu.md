# 递归神经网络的简明实现
:label:`sec_rnn-concise`

虽然:numref:`sec_rnn_scratch`对了解RNN是如何实现的很有指导意义，但这既不方便也不快速。本节将展示如何使用深度学习框架的高级api提供的函数更有效地实现相同的语言模型。我们从读取时间机器数据集开始。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

## 定义模型

高级api提供了递归神经网络的实现。我们构造了一个单隐层256隐单元的递归神经网络层`rnn_layer`。事实上，我们甚至还没有讨论多层的含义——这将发生在:numref:`sec_deep_rnn`。现在，只需说多个层相当于一层RNN的输出，作为下一层RNN的输入。

```{.python .input}
num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()
```

```{.python .input}
#@tab pytorch
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)
```

:begin_tab:`mxnet`
初始化隐藏状态很简单。我们调用成员函数`begin_state`。这将返回一个列表（`state`），其中包含minibatch中每个示例的初始隐藏状态，其形状为（隐藏层数、批大小、隐藏单元数）。对于以后要介绍的一些模型（例如长-短期记忆），这样的列表还包含其他信息。
:end_tab:

:begin_tab:`pytorch`
我们使用张量来初始化隐藏状态，它的形状是（隐藏层的数量，批量大小，隐藏单元的数量）。
:end_tab:

```{.python .input}
state = rnn_layer.begin_state(batch_size=batch_size)
len(state), state[0].shape
```

```{.python .input}
#@tab pytorch
state = torch.zeros((1, batch_size, num_hiddens))
state.shape
```

通过一个隐藏状态和一个输入，我们可以用更新后的隐藏状态计算输出。需要强调的是，`rnn_layer`的“输出”（`Y`）不*涉及*输出层的计算：它指*每个*时间步的隐藏状态，它们可以用作后续输出层的输入。

:begin_tab:`mxnet`
此外，`rnn_layer`返回的更新后的隐藏状态（`state_new`）是指minibatch的*last*时间步的隐藏状态。它可以用来初始化顺序分区中一个历元内下一个小批量的隐藏状态。对于多个隐藏层，每个层的隐藏状态将存储在此变量（`state_new`）中。对于稍后要介绍的某些模型（例如，长-短期记忆），此变量还包含其他信息。
:end_tab:

```{.python .input}
X = np.random.uniform(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, state_new.shape
```

与:numref:`sec_rnn_scratch`类似，我们为一个完整的RNN模型定义了一个`RNNModel`类。注意`rnn_layer`只包含隐藏的递归层，我们需要创建一个单独的输出层。

```{.python .input}
#@save
class RNNModel(nn.Block):
    """The RNN model."""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = npx.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # The fully-connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `vocab_size`).
        output = self.dense(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

```{.python .input}
#@tab pytorch
#@save
class RNNModel(nn.Module):
    """The RNN model."""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # If the RNN is bidirectional (to be introduced later),
        # `num_directions` should be 2, else it should be 1.
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `vocab_size`).
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), 
                                device=device)
        else:
            # `nn.LSTM` takes a tuple of hidden states
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))
```

## 训练与预测

在训练模型之前，让我们用一个具有随机权重的模型进行预测。

```{.python .input}
device = d2l.try_gpu()
net = RNNModel(rnn_layer, len(vocab))
net.initialize(force_reinit=True, ctx=device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
```

```{.python .input}
#@tab pytorch
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
```

很明显，这种模式根本不起作用。接下来，我们使用:numref:`sec_rnn_scratch`中定义的超参数调用`train_ch8`，并使用高级API训练模型。

```{.python .input}
#@tab all
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

与上一节相比，由于深度学习框架的高级api对代码进行了更多的优化，该模型在较短的时间内实现了类似的复杂性。

## 摘要

* 深度学习框架的高级api提供了RNN层的实现。
* 高级api的RNN层返回输出和更新的隐藏状态，其中输出不涉及输出层计算。
* 与从头开始使用其实现相比，使用高级api会导致更快的RNN训练。

## 练习

1. 您能使用高级api使RNN模型过拟合吗？
1. 如果在RNN模型中增加隐藏层的数量会发生什么？你能使模型工作吗？
1. 使用RNN实现:numref:`sec_sequence`的自回归模型。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/335)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1053)
:end_tab:
