# 深度递归神经网络

:label:`sec_deep_rnn`

到目前为止，我们只讨论了单一单向隐层的RNN。其中，潜变量和观测值如何相互作用的具体函数形式是相当任意的。这不是一个大问题，只要我们有足够的灵活性来建模不同类型的交互。然而，对于一个单层来说，这可能是相当具有挑战性的。在线性模型的情况下，我们通过添加更多的层来解决这个问题。在RNNs中，这有点棘手，因为我们首先需要决定如何以及在何处添加额外的非线性。

事实上，我们可以将多层RNN堆叠在一起。由于几个简单的层的组合，这就产生了一个灵活的机制。特别是，数据可能与堆栈的不同级别有关。例如，我们可能希望保持有关金融市场状况（熊市或牛市）的高水平数据可用，而在较低水平上，我们只记录短期的时间动态。

除了以上所有的抽象讨论之外，通过回顾:numref:`fig_deep_rnn`可能最容易理解我们感兴趣的模型系列。它描述了一个具有$L$个隐藏层的深RNN。每个隐藏状态都连续传递到当前层的下一个时间步和下一层的当前时间步。

![Architecture of a deep RNN.](../img/deep-rnn.svg)
:label:`fig_deep_rnn`

## 函数依赖关系

我们可以在:numref:`fig_deep_rnn`中描述的$L$隐藏层的深层架构中形式化功能依赖关系。我们下面的讨论主要集中在vanillarnn模型上，但它也适用于其他序列模型。

假设我们在时间步骤$t$有一个小批量输入$\mathbf{X}_t \in \mathbb{R}^{n \times d}$（示例数：$n$，每个示例中的输入数：$d$）。同时，将$l^\mathrm{th}$隐藏层（$l=1,\ldots,L$）的隐藏状态设为$\mathbf{H}_t^{(l)}  \in \mathbb{R}^{n \times h}$（隐藏单元数：$h$），输出层变量设为$\mathbf{O}_t \in \mathbb{R}^{n \times q}$（输出数：$q$）。设置$\mathbf{H}_t^{(0)} = \mathbf{X}_t$，使用激活功能$\phi_l$的$l^\mathrm{th}$隐藏层的隐藏状态表示如下：

$$\mathbf{H}_t^{(l)} = \phi_l(\mathbf{H}_t^{(l-1)} \mathbf{W}_{xh}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{hh}^{(l)}  + \mathbf{b}_h^{(l)}),$$
:eqlabel:`eq_deep_rnn_H`

其中，权重$\mathbf{W}_{xh}^{(l)} \in \mathbb{R}^{h \times h}$和$\mathbf{W}_{hh}^{(l)} \in \mathbb{R}^{h \times h}$以及偏移$\mathbf{b}_h^{(l)} \in \mathbb{R}^{1 \times h}$是$l^\mathrm{th}$隐藏层的模型参数。

最后，输出层的计算仅基于最终$L^\mathrm{th}$隐藏层的隐藏状态：

$$\mathbf{O}_t = \mathbf{H}_t^{(L)} \mathbf{W}_{hq} + \mathbf{b}_q,$$

其中，权重$\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$和偏置$\mathbf{b}_q \in \mathbb{R}^{1 \times q}$是输出层的模型参数。

与mlp一样，隐藏层的数目$L$和隐藏单元的数目$h$是超参数。换句话说，它们可以由我们调整或指定。另外，用GRU或LSTM的隐态计算代替:eqref:`eq_deep_rnn_H`的隐态计算，可以很容易地得到深选通RNN。

## 简明实现

幸运的是，实现多层RNN所需的许多后勤细节在高级API中都是现成的。为了简单起见，我们仅说明使用此类内置功能的实现。让我们以LSTM模型为例。该代码与我们之前在:numref:`sec_lstm`中使用的代码非常相似。实际上，唯一的区别是我们显式地指定了层的数量，而不是选择单个层的默认值。像往常一样，我们从加载数据集开始。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import npx
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

选择超参数等体系结构决策与:numref:`sec_lstm`的决策非常相似。我们选择相同数量的输入和输出，因为我们有不同的标记，即`vocab_size`。隐藏单元的数量仍然是256。唯一的区别是，我们现在通过指定`num_layers`的值来选择大量的隐藏层。

```{.python .input}
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
device = d2l.try_gpu()
lstm_layer = rnn.LSTM(num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
```

```{.python .input}
#@tab pytorch
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
```

## 训练与预测

因为现在我们用LSTM模型实例化了两个层，这个相当复杂的体系结构大大降低了训练速度。

```{.python .input}
#@tab all
num_epochs, lr = 500, 2
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

## 摘要

* 在深度RNN中，隐藏的状态信息被传递到当前层的下一时间步和下一层的当前时间步。
* 有许多不同口味的深RNN，如LSTMs，GRUs，或香草RNN。方便的是，这些模型都可以作为深度学习框架的高级api的一部分。
* 初始化模型需要小心。总的来说，深层RNN需要大量的工作（例如学习速度和裁剪）来确保适当的收敛。

## 练习

1. 尝试使用我们在:numref:`sec_rnn_scratch`中讨论的单层实现从头开始实现两层RNN。
2. 用GRU替换LSTM，比较精确度和训练速度。
3. 增加训练数据以包含多本书。你的困惑程度能降到多低？
4. 在为文本建模时，是否要合并不同作者的源代码？为什么这是个好主意？会出什么问题？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/340)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1058)
:end_tab:
