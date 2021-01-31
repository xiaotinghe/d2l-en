# 递归神经网络的简明实现
:label:`sec_rnn-concise`

虽然:numref:`sec_rnn_scratch`版对了解如何实现RNN很有指导意义，但这并不方便或快捷。本节将展示如何使用深度学习框架的高级API提供的函数更有效地实现相同的语言模型。我们一如既往地从读取时光机数据集开始。

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

高级API提供递归神经网络的实现。我们构造了具有单个隐含层和256个隐含单元的递归神经网络层`rnn_layer`。事实上，我们甚至还没有讨论拥有多层意味着什么-这将在:numref:`sec_deep_rnn`发生。现在，只要说多层仅仅相当于一层RNN的输出被用作下一层RNN的输入就足够了。

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
初始化隐藏状态很简单。我们调用成员函数`begin_state`。这将返回一个列表(`state`)，该列表包含小批量中每个示例的初始隐藏状态，其形状为(隐藏层数、批大小、隐藏单元数)。对于稍后要介绍的一些型号(例如，长时间、短期记忆)，这样的列表还包含其他信息。
:end_tab:

:begin_tab:`pytorch`
我们使用一个张量来初始化隐藏状态，其形状为(隐藏层数、批量大小、隐藏单元数)。
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

有了隐藏状态和输入，我们就可以用更新后的隐藏状态计算输出。应该强调的是，`rnn_layer`的“输出”(`Y`)不涉及输出层的计算：它指的是在*每个*时间步长的隐藏状态，并且它们可以用作后续输出层的输入。

:begin_tab:`mxnet`
此外，由`state_new`返回的更新的隐藏状态(`rnn_layer`)是指在小批量的最后*个时间步长的隐藏状态。它可用于在顺序分区中初始化一个时期内的下一小批的隐藏状态。对于多个隐藏层，每个层的隐藏状态将存储在该变量中(`state_new`)。对于稍后要介绍的一些模型(例如，长期短期记忆)，此变量还包含其他信息。
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

与:numref:`sec_rnn_scratch`类似，我们为完整的rnn模型定义了一个`RNNModel`类。请注意，`rnn_layer`只包含隐藏的重复层，我们需要创建一个单独的输出层。

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

## 训练和预测

在训练模型之前，让我们用具有随机权重的模型进行预测。

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

很明显，这种模式根本行不通。接下来，我们使用`train_ch8`中定义的相同超参数调用:numref:`sec_rnn_scratch`，并使用高级API训练我们的模型。

```{.python .input}
#@tab all
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

与上一节相比，这个模型实现了类似的困惑，尽管在更短的时间内，因为深度学习框架的高级API对代码进行了更多的优化。

## 摘要

* 深度学习框架的高级API提供了RNN层的实现。
* 高级API的RNN层返回输出和更新后的隐藏状态，其中输出不涉及输出层计算。
* 与从头开始使用RNN实现相比，使用高级API可以更快地进行RNN培训。

## 练习

1. 您能使用高级API使RNN模型更加适合吗？
1. 如果增加RNN模型中的隐藏层数量，会发生什么情况？你能使这个模型工作吗？
1. 使用随机神经网络实现:numref:`sec_sequence`的自回归模型。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/335)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1053)
:end_tab:
