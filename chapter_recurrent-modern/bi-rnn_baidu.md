# 双向递归神经网络
:label:`sec_bi_rnn`

在序列学习中，到目前为止，我们假设我们的目标是根据我们所看到的，例如，在时间序列的上下文或语言模型的上下文中，对下一个输出进行建模。虽然这是一个典型的情况，但这并不是我们可能遇到的唯一情况。为了说明这个问题，考虑以下三个任务：在文本序列中填空：

* 我是`___`。
* 我饿了`___`。
* 我饿了`___`，我能吃半只猪。

根据可获得的信息量，我们可以用非常不同的词填空，如“高兴”、“不”和“非常”。很明显，短语的结尾（如果有的话）传达了关于选择哪个词的重要信息。不能利用这一点的序列模型将在相关任务上表现不佳。例如，要做好命名实体识别（例如，识别“绿色”指的是“格林先生”还是“颜色”）更长的范围上下文同样重要。为了获得解决这个问题的一些灵感，让我们绕道到概率图形模型。

## 隐马尔可夫模型中的动态规划

本小节用于说明动态规划问题。具体的技术细节对于理解深度学习模型并不重要，但它们有助于激发人们为什么可以使用深度学习以及为什么可以选择特定的体系结构。

如果我们想用概率图形模型来解决这个问题，我们可以设计一个潜变量模型，如下所示。在任何时间步骤$t$，我们假设存在一些潜在变量$h_t$，通过$P(x_t \mid h_t)$控制我们观察到的排放$x_t$。此外，任何转移$h_t \to h_{t+1}$由某个状态转移概率$P(h_{t+1} \mid h_{t})$给出。这个概率图形模型就是一个隐马尔可夫模型，如:numref:`fig_hmm`所示。

![A hidden Markov model.](../img/hmm.svg)
:label:`fig_hmm`

因此，对于$T$个观测序列，我们在观测和隐藏状态上具有以下联合概率分布：

$$P(x_1, \ldots, x_T, h_1, \ldots, h_T) = \prod_{t=1}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t), \text{ where } P(h_1 \mid h_0) = P(h_1).$$
:eqlabel:`eq_hmm_jointP`

现在假设我们观察所有$x_i$，除了一些$x_j$，我们的目标是计算$P(x_j \mid x_{-j})$，其中$x_{-j} = (x_1, \ldots, x_{j-1}, x_{j+1}, \ldots, x_{T})$。由于$P(x_j \mid x_{-j})$中没有潜在变量，我们考虑对$h_1, \ldots, h_T$的所有可能选择组合求和。如果任何$h_i$可以接受$k$个不同的值（有限的状态数），这意味着我们需要求$k^T$项的和——通常是不可能的！幸运的是，有一个优雅的解决方案：*动态规划*。

要了解它是如何工作的，请考虑依次对潜在变量$h_1, \ldots, h_T$求和。根据:eqref:`eq_hmm_jointP`，得出：

$$\begin{aligned}
    &P(x_1, \ldots, x_T) \\
    =& \sum_{h_1, \ldots, h_T} P(x_1, \ldots, x_T, h_1, \ldots, h_T) \\
    =& \sum_{h_1, \ldots, h_T} \prod_{t=1}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t) \\
    =& \sum_{h_2, \ldots, h_T} \underbrace{\left[\sum_{h_1} P(h_1) P(x_1 \mid h_1) P(h_2 \mid h_1)\right]}_{\pi_2(h_2) \stackrel{\mathrm{def}}{=}}
    P(x_2 \mid h_2) \prod_{t=3}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t) \\
    =& \sum_{h_3, \ldots, h_T} \underbrace{\left[\sum_{h_2} \pi_2(h_2) P(x_2 \mid h_2) P(h_3 \mid h_2)\right]}_{\pi_3(h_3)\stackrel{\mathrm{def}}{=}}
    P(x_3 \mid h_3) \prod_{t=4}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t)\\
    =& \dots \\
    =& \sum_{h_T} \pi_T(h_T) P(x_T \mid h_T).
\end{aligned}$$

一般来说，我们把*前向递归*作为

$$\pi_{t+1}(h_{t+1}) = \sum_{h_t} \pi_t(h_t) P(x_t \mid h_t) P(h_{t+1} \mid h_t).$$

递归初始化为$\pi_1(h_1) = P(h_1)$。抽象地说，这可以写成$\pi_{t+1} = f(\pi_t, x_t)$，其中$f$是一些可学习的函数。这看起来很像我们到目前为止在RNNs上下文中讨论的潜变量模型中的更新方程！

完全类似于前向递归，我们也可以用后向递归对同一组潜在变量求和。这将产生：

$$\begin{aligned}
    & P(x_1, \ldots, x_T) \\
     =& \sum_{h_1, \ldots, h_T} P(x_1, \ldots, x_T, h_1, \ldots, h_T) \\
    =& \sum_{h_1, \ldots, h_T} \prod_{t=1}^{T-1} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot P(h_T \mid h_{T-1}) P(x_T \mid h_T) \\
    =& \sum_{h_1, \ldots, h_{T-1}} \prod_{t=1}^{T-1} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot
    \underbrace{\left[\sum_{h_T} P(h_T \mid h_{T-1}) P(x_T \mid h_T)\right]}_{\rho_{T-1}(h_{T-1})\stackrel{\mathrm{def}}{=}} \\
    =& \sum_{h_1, \ldots, h_{T-2}} \prod_{t=1}^{T-2} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot
    \underbrace{\left[\sum_{h_{T-1}} P(h_{T-1} \mid h_{T-2}) P(x_{T-1} \mid h_{T-1}) \rho_{T-1}(h_{T-1}) \right]}_{\rho_{T-2}(h_{T-2})\stackrel{\mathrm{def}}{=}} \\
    =& \ldots \\
    =& \sum_{h_1} P(h_1) P(x_1 \mid h_1)\rho_{1}(h_{1}).
\end{aligned}$$

因此，我们可以将*向后递归*写成

$$\rho_{t-1}(h_{t-1})= \sum_{h_{t}} P(h_{t} \mid h_{t-1}) P(x_{t} \mid h_{t}) \rho_{t}(h_{t}),$$

初始化为$\rho_T(h_T) = 1$。向前和向后递归都允许我们在$\mathcal{O}(kT)$（线性）时间内对$(h_1, \ldots, h_T)$的所有值（而不是指数时间）求和$T$个潜在变量。这是使用图形模型进行概率推理的最大好处之一。它也是通用消息传递算法:cite:`Aji.McEliece.2000`的一个非常特殊的实例。结合正向和反向递归，我们能够计算

$$P(x_j \mid x_{-j}) \propto \sum_{h_j} \pi_j(h_j) \rho_j(h_j) P(x_j \mid h_j).$$

作为一个抽象，可以写在递归中。同样，这看起来非常像一个更新方程，只是向后运行，不像我们在RNNs中看到的那样。实际上，隐马尔可夫模型受益于知道未来数据何时可用。信号处理科学家将知道和不知道未来观测的两种情况区分为插值和外推。有关更多详细信息，请参阅本书的序贯蒙特卡罗算法的介绍章节:cite:`Doucet.De-Freitas.Gordon.2001`。

## 双向模型

如果我们想在RNN中有一种机制，提供与隐马尔可夫模型类似的前瞻能力，我们需要修改我们迄今为止看到的RNN设计。幸运的是，这在概念上很容易。我们从最后一个令牌开始从后向前运行RNN，而不是只在前向模式下从第一个令牌开始运行RNN。
*双向RNN*添加一个隐藏层，该层向后传递信息，以更灵活地处理此类信息。:numref:`fig_birnn`说明了具有单个隐藏层的双向RNN的体系结构。

![Architecture of a bidirectional RNN.](../img/birnn.svg)
:label:`fig_birnn`

事实上，这与隐马尔可夫模型动态规划中的前向和后向递归没有太大区别。主要区别在于，在前一种情况下，这些方程具有特定的统计意义。现在它们没有这样容易理解的解释，我们只能把它们当作通用的、可学习的函数。这种转变集中体现了许多指导现代深度网络设计的原则：首先，使用经典统计模型的函数依赖类型，然后以通用形式对它们进行参数化。

### 定义

双向RNN由:cite:`Schuster.Paliwal.1997`引入。有关各种体系结构的详细讨论，请参阅:cite:`Graves.Schmidhuber.2005`。让我们看看这样一个网络的具体情况。

对于任何时间步骤$t$，给定一个小批量输入$\mathbf{X}_t \in \mathbb{R}^{n \times d}$（示例数：$n$，每个示例中的输入数：$d$），并让隐层激活函数为$\phi$。在双向架构中，我们假设该时间步的前向和后向隐藏状态分别为$\overrightarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$和$\overleftarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$，其中$h$是隐藏单元的数目。前向和后向隐藏状态更新如下：

$$
\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{hh}^{(f)}  + \mathbf{b}_h^{(f)}),\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{hh}^{(b)}  + \mathbf{b}_h^{(b)}),
\end{aligned}
$$

其中，权重$\mathbf{W}_{xh}^{(f)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh}^{(f)} \in \mathbb{R}^{h \times h}, \mathbf{W}_{xh}^{(b)} \in \mathbb{R}^{d \times h}, \text{ and } \mathbf{W}_{hh}^{(b)} \in \mathbb{R}^{h \times h}$和偏差$\mathbf{b}_h^{(f)} \in \mathbb{R}^{1 \times h} \text{ and } \mathbf{b}_h^{(b)} \in \mathbb{R}^{1 \times h}$都是模型参数。

接下来，我们串联前向和后向隐藏状态$\overrightarrow{\mathbf{H}}_t$和$\overleftarrow{\mathbf{H}}_t$以获得要馈入输出层的隐藏状态$\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$。在具有多个隐藏层的深层双向RNN中，此类信息作为*输入*传递到下一个双向层。最后，输出层计算输出$\mathbf{O}_t \in \mathbb{R}^{n \times q}$（输出数：$q$）：

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

这里，权重矩阵$\mathbf{W}_{hq} \in \mathbb{R}^{2h \times q}$和偏置$\mathbf{b}_q \in \mathbb{R}^{1 \times q}$是输出层的模型参数。实际上，这两个方向可以有不同数量的隐藏单元。

### 计算成本及其应用

双向RNN的一个关键特性是，使用序列两端的信息来估计输出。也就是说，我们使用来自未来和过去观测的信息来预测当前的观测。在下一个令牌预测的情况下，这并不是我们想要的。毕竟，在预测下一个标记时，我们没有知道下一个标记的奢侈。因此，如果我们天真地使用双向RNN，我们将不会得到很好的准确性：在训练期间，我们有过去和未来的数据来估计现在。在测试期间，我们只有过去的数据，因此准确性较差。我们将在下面的实验中说明这一点。

雪上加霜的是，双向RNN也非常慢。其主要原因是前向传播需要在双向层中进行前向和后向递归，并且后向传播依赖于前向传播的结果。因此，渐变将有一个非常长的依赖链。

实际上，双向层的使用非常少，并且仅用于一组狭窄的应用程序，例如填充缺失的单词、注释标记（例如，用于命名实体识别）以及作为序列处理管道中的一个步骤对序列进行批发编码（例如，用于机器翻译）。在:numref:`sec_bert`和:numref:`sec_sentiment_rnn`中，我们将介绍如何使用双向RNN编码文本序列。

## 为错误的应用程序训练双向RNN

如果我们忽略了所有关于双向RNN使用过去和未来数据的建议，而只是将其应用于语言模型，我们将得到具有可接受复杂性的估计。尽管如此，该模型预测未来代币的能力仍受到严重影响，如下面的实验所示。尽管存在合理的困惑，但即使经过多次迭代，它也只会产生乱码。我们将下面的代码作为警告示例，以防在错误的上下文中使用它们。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import npx
from mxnet.gluon import rnn
npx.set_np()

# Load data
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# Define the bidirectional LSTM model by setting `bidirectional=True`
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
lstm_layer = rnn.LSTM(num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
# Train the model
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# Load data
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# Define the bidirectional LSTM model by setting `bidirectional=True`
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
# Train the model
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

由于上述原因，产出显然不令人满意。有关更有效地使用双向RNN的讨论，请参阅:numref:`sec_sentiment_rnn`中的情绪分析应用程序。

## 摘要

* 在双向RNN中，每个时间步长的隐藏状态由当前时间步长前后的数据同时确定。
* 双向RNN与概率图形模型中的前向-后向算法有着惊人的相似性。
* 双向RNN主要用于序列编码和给定双向上下文的观测值估计。
* 由于长梯度链，双向RNN的训练成本非常高。

## 练习

1. 如果不同方向使用不同数量的隐藏单元，$\mathbf{H}_t$的外形会有怎样的变化？
1. 设计了一个具有多个隐藏层的双向RNN。
1. 一词多义在自然语言中很常见。例如，“银行”一词在“我去银行存现金”和“我去银行坐下”的上下文中有不同的含义。我们如何设计一个神经网络模型，使得给定一个上下文序列和一个单词，该单词在上下文中的向量表示将被返回？哪种类型的神经结构更适合处理多义词？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/339)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1059)
:end_tab:
