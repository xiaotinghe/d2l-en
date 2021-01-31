# 深度递归神经网络

:label:`sec_deep_rnn`

到目前为止，我们只讨论了具有单一单向隐层的RNN。在它中，潜变量和观测值如何相互作用的具体函数形式是相当随意的。只要我们有足够的灵活性来建模不同类型的交互，这就不是一个大问题。然而，对于单层，这可能是相当具有挑战性的。在线性模型的情况下，我们通过添加更多层修复了此问题。在RNN中，这有点棘手，因为我们首先需要决定如何以及在哪里添加额外的非线性。

事实上，我们可以将多层RNN堆叠在一起。由于几个简单层的组合，这导致了灵活的机构。具体地说，数据可能与堆栈的不同级别相关。例如，我们可能希望保持有关金融市场状况(熊市或牛市)的高级数据可用，而在较低级别，我们只记录较短期的时间动态。

除了上面所有的抽象讨论之外，通过回顾:numref:`fig_deep_rnn`可能是理解我们感兴趣的模型家族最容易的方法。它描述了一个具有$L$个隐层的深层随机神经网络。每个隐藏状态被连续传递到当前层的下一个时间步长和下一层的当前时间步长。

![Architecture of a deep RNN.](../img/deep-rnn.svg)
:label:`fig_deep_rnn`

## 函数依赖项

我们可以将:numref:`fig_deep_rnn`中描述的$L$个隐藏层的深层体系结构中的函数依赖关系形式化。我们下面的讨论主要集中在普通的RNN模型上，但它也适用于其他序列模型。

假设我们具有小批量输入$\mathbf{X}_t \in \mathbb{R}^{n \times d}$(示例数目：$n$，每个示例中的输入数目：$d$)在时间步骤$t$。同时，设$l^\mathrm{th}$个隐藏层($l=1,\ldots,L$)的隐藏状态为$\mathbf{H}_t^{(l)}  \in \mathbb{R}^{n \times h}$(隐藏单元数：$h$)，输出层变量为$\mathbf{O}_t \in \mathbb{R}^{n \times q}$(输出数：$q$)。设置$\mathbf{H}_t^{(0)} = \mathbf{X}_t$时，使用激活功能$\phi_l$的$l^\mathrm{th}$隐藏层的隐藏状态表示如下：

$$\mathbf{H}_t^{(l)} = \phi_l(\mathbf{H}_t^{(l-1)} \mathbf{W}_{xh}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{hh}^{(l)}  + \mathbf{b}_h^{(l)}),$$
:eqlabel:`eq_deep_rnn_H`

其中权重$\mathbf{W}_{xh}^{(l)} \in \mathbb{R}^{h \times h}$和$\mathbf{W}_{hh}^{(l)} \in \mathbb{R}^{h \times h}$连同偏差$\mathbf{b}_h^{(l)} \in \mathbb{R}^{1 \times h}$是$l^\mathrm{th}$隐藏层的模型参数。

最终，输出层的计算仅基于最终$L^\mathrm{th}$隐藏层的隐藏状态：

$$\mathbf{O}_t = \mathbf{H}_t^{(L)} \mathbf{W}_{hq} + \mathbf{b}_q,$$

其中权重$\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$和偏差$\mathbf{b}_q \in \mathbb{R}^{1 \times q}$是输出层的模型参数。

与MLP一样，隐藏层的数目$L$和隐藏单元的数目$h$是超参数。换句话说，它们可以由我们调整或指定。另外，将:eqref:`eq_deep_rnn_H`中的隐态计算替换为广义随机数单元或线性扫描隧道显微镜的隐态计算，可以很容易地得到深门极随机神经网络。

## 简明实施

幸运的是，实现RNN的多层所需的许多后勤细节在高级API中都很容易获得。为简单起见，我们仅说明使用此类内置功能的实现。让我们以LSTM模型为例。该代码与我们之前在:numref:`sec_lstm`中使用的代码非常相似。事实上，唯一的区别是我们显式地指定了层的数量，而不是选择单个层的默认值。像往常一样，我们从加载数据集开始。

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

架构决策(如选择超参数)与:numref:`sec_lstm`中的决策非常相似。我们选择相同数量的输入和输出，因为我们有不同的标记，即`vocab_size`。隐藏单元的数量仍为256个。唯一的区别是，我们现在通过指定值`num_layers`来选择非平凡数量的隐藏层。

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

## 训练和预测

因为现在我们使用LSTM模型实例化了两层，所以这个相当复杂的体系结构大大降低了培训的速度。

```{.python .input}
#@tab all
num_epochs, lr = 500, 2
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

## 摘要

* 在深度RNN中，隐藏状态信息被传递到当前层的下一个时间步长和下一层的当前时间步长。
* 存在许多不同风格的深层RNN，例如LSTM、GRU或普通RNN。方便的是，这些模型都可以作为深度学习框架的高级API的一部分提供。
* 模型的初始化需要小心。总体而言，深度RNN需要大量的工作(如学习率和修剪)来确保适当的收敛。

## 练习

1. 尝试使用我们在:numref:`sec_rnn_scratch`中讨论的单层实现从头开始实现两层RNN。
2. 将LSTM替换为GRU，并比较准确性和训练速度。
3. 增加培训数据以包括多本书。在困惑程度量表上，你能打到多低？
4. 在对文本建模时，您是否希望组合不同作者的来源？为什么这是个好主意？会出什么问题呢？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/340)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1058)
:end_tab:
