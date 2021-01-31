# 双向递归神经网络
:label:`sec_bi_rnn`

在序列学习中，到目前为止，我们假设我们的目标是在给定到目前为止所见的情况下对下一个输出进行建模，例如，在时间序列的上下文中或在语言模型的上下文中。虽然这是一个典型的场景，但它不是我们可能遇到的唯一场景。要说明此问题，请考虑以下三项填充文本序列中空白处的任务：

* 我`___`岁。
* 我`___`饿了。
* 我`___`饿了，我能吃下半头猪。

根据可用信息量的不同，我们可能会用“高兴”、“不”和“非常”等截然不同的词来填空。显然，短语末尾(如果可用)传达了有关选择哪个词的重要信息。不能利用这一点的序列模型在相关任务上的性能将很差。例如，要做好命名实体识别(例如，识别“Green”是指“Green先生”还是指颜色)，较长范围的上下文同样至关重要。为了获得一些解决问题的灵感，让我们绕道看看概率图形模型。

## 隐马尔可夫模型中的动态规划

这一小节用来说明动态规划问题。具体的技术细节对于理解深度学习模型并不重要，但它们有助于激励人们为什么要使用深度学习，以及为什么要选择特定的体系结构。

如果我们想要使用概率图形模型来解决问题，例如，我们可以设计一个潜变量模型，如下所示。在任何时间步骤$t$，我们假设存在某个潜在变量$h_t$，其控制我们观测到的发射$x_t$通过$P(x_t \mid h_t)$。此外，任何转移$h_t \to h_{t+1}$由某个状态转移概率$P(h_{t+1} \mid h_{t})$给出。这样，该概率图形模型就是如:numref:`fig_hmm`中所示的“隐马尔可夫模型”。

![A hidden Markov model.](../img/hmm.svg)
:label:`fig_hmm`

因此，对于$T$个观测值的序列，我们在观测状态和隐藏状态上具有以下联合概率分布：

$$P(x_1, \ldots, x_T, h_1, \ldots, h_T) = \prod_{t=1}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t), \text{ where } P(h_1 \mid h_0) = P(h_1).$$
:eqlabel:`eq_hmm_jointP`

现在假设我们观察所有的$x_i$，除了大约$x_j$之外，我们的目标是计算$P(x_j \mid x_{-j})$，其中$x_{-j} = (x_1, \ldots, x_{j-1}, x_{j+1}, \ldots, x_{T})$。由于在$P(x_j \mid x_{-j})$中没有潜在变量，我们考虑对$h_1, \ldots, h_T$中所有可能的选择组合进行求和。如果任何$h_i$个可以采用$k$个不同的值(有限数量的状态)，这意味着我们需要超过$k^T$个项的总和-通常是不可能完成的任务！幸运的是，对此有一个很好的解决方案：*动态编程*。

要了解它是如何工作的，请依次考虑对潜在变量$h_1, \ldots, h_T$求和。根据:eqref:`eq_hmm_jointP`的数据，这将产生以下收益：

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

一般来说，我们使用“前向递归”作为

$$\pi_{t+1}(h_{t+1}) = \sum_{h_t} \pi_t(h_t) P(x_t \mid h_t) P(h_{t+1} \mid h_t).$$

递归被初始化为$\pi_1(h_1) = P(h_1)$。抽象地说，这可以写成$\pi_{t+1} = f(\pi_t, x_t)$，其中$f$是一些可学习的函数。这看起来非常像我们到目前为止在RNNs上下文中讨论的潜变量模型中的更新方程！

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

因此，我们可以将“向后递归”编写为

$$\rho_{t-1}(h_{t-1})= \sum_{h_{t}} P(h_{t} \mid h_{t-1}) P(x_{t} \mid h_{t}) \rho_{t}(h_{t}),$$

具有初始化$\rho_T(h_T) = 1$。前向和后向递归都允许我们在$\mathcal{O}(kT)$(线性)时间内对所有值$(h_1, \ldots, h_T)$的超过$(h_1, \ldots, h_T)$个潜在变量求和，而不是在指数时间内求和。这是用图形模型进行概率推理的最大好处之一。它也是通用消息传递算法:cite:`Aji.McEliece.2000`的非常特殊的实例。结合前向和后向递归，我们能够计算

$$P(x_j \mid x_{-j}) \propto \sum_{h_j} \pi_j(h_j) \rho_j(h_j) P(x_j \mid h_j).$$

注意，抽象地说，后向递归可以写成$\rho_{t-1} = g(\rho_t, x_t)$，其中$g$是可学习的函数。同样，这看起来非常像一个更新方程式，只是向后运行，不像我们在RNN中看到的那样。事实上，隐马尔可夫模型受益于知道未来数据何时可用。信号处理科学家将已知和不知道未来观测的两种情况区分为插值V.S.外推。有关更多详细信息，请参阅本书关于顺序蒙特卡罗算法的介绍性章节:cite:`Doucet.De-Freitas.Gordon.2001`。

## 双向模型

如果我们想要在RNN中有一种机制，能够提供与隐马尔可夫模型类似的前瞻能力，我们需要修改到目前为止我们已经看到的RNN设计。幸运的是，这在概念上很简单。我们不是仅在转发模式下从第一个令牌开始运行RNN，而是从从后到前运行的最后一个令牌开始另一个RNN。
*双向RNN*添加了在反向传递信息的隐藏层，以更灵活地处理此类信息。:numref:`fig_birnn`示出了具有单个隐藏层的双向RNN的体系结构。

![Architecture of a bidirectional RNN.](../img/birnn.svg)
:label:`fig_birnn`

实际上，这与隐马尔可夫模型动态编程中的前向和后向递归没有太大区别。主要区别在于，在前一种情况下，这些方程具有特定的统计意义。现在它们缺乏这种易于访问的解释，我们只能将其视为泛型和可学习的函数。这一转变集中体现了指导现代深度网络设计的许多原则：首先，使用经典统计模型的函数依赖类型，然后将其参数化为通用形式。

### 定义

:cite:`Schuster.Paliwal.1997`引入了双向RNN。有关各种体系结构的详细讨论，另请参阅文件:cite:`Graves.Schmidhuber.2005`。让我们来看看这样一个网络的具体情况。

对于任何时间步骤$t$，给定小批量输入$\mathbf{X}_t \in \mathbb{R}^{n \times d}$(示例数目：$n$，每个示例中的输入数目：$d$)，并且使隐藏层激活函数为$\phi$。在双向体系结构中，我们假设该时间步长的前向和后向隐藏状态分别为$\overrightarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$和$\overleftarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$，其中$h$是隐藏单元的数量。向前和向后隐藏状态更新如下：

$$
\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{hh}^{(f)}  + \mathbf{b}_h^{(f)}),\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{hh}^{(b)}  + \mathbf{b}_h^{(b)}),
\end{aligned}
$$

其中权重$\mathbf{W}_{xh}^{(f)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh}^{(f)} \in \mathbb{R}^{h \times h}, \mathbf{W}_{xh}^{(b)} \in \mathbb{R}^{d \times h}, \text{ and } \mathbf{W}_{hh}^{(b)} \in \mathbb{R}^{h \times h}$和偏差$\mathbf{b}_h^{(f)} \in \mathbb{R}^{1 \times h} \text{ and } \mathbf{b}_h^{(b)} \in \mathbb{R}^{1 \times h}$都是模型参数。

接下来，我们将前向和后向隐藏状态$\overrightarrow{\mathbf{H}}_t$和$\overleftarrow{\mathbf{H}}_t$连接以获得要馈送到输出层的隐藏状态$\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$。在具有多个隐藏层的深双向RNN中，这样的信息作为*输入*传递到下一个双向层。最后，输出层计算输出$\mathbf{O}_t \in \mathbb{R}^{n \times q}$(输出数：$q$)：

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

这里，权重矩阵$\mathbf{W}_{hq} \in \mathbb{R}^{2h \times q}$和偏差$\mathbf{b}_q \in \mathbb{R}^{1 \times q}$是输出层的模型参数。事实上，这两个方向可以有不同数量的隐藏单元。

### 计算成本及其应用

双向RNN的关键特征之一是使用来自序列两端的信息来估计输出。也就是说，我们使用来自未来和过去观测的信息来预测当前的观测。在下一个令牌预测的情况下，这并不完全是我们想要的。毕竟，在预测下一个令牌时，我们不能奢侈地知道下一个令牌。因此，如果我们天真地使用双向RNN，我们将不会获得非常好的准确性：在训练期间，我们有过去和未来的数据来估计现在。在测试期间，我们只有过去的数据，因此准确性很差。我们将在下面的实验中说明这一点。

雪上加霜的是，双向RNN也非常慢。这主要是因为前向传播需要双向层中的前向和后向递归，并且后向传播取决于前向传播的结果。因此，渐变将具有非常长的依赖链。

在实践中，双向层被非常少地使用，并且仅用于一组狭窄的应用，诸如填充遗漏的单词、注释令牌(例如，用于命名实体识别)、以及作为序列处理流水线中的步骤(例如，用于机器翻译)对序列进行批量编码。在:numref:`sec_bert`和:numref:`sec_sentiment_rnn`中，我们将介绍如何使用双向RNN对文本序列进行编码。

## 为错误的应用训练双向RNN

如果我们忽略所有关于双向RNN使用过去和未来数据的建议，并简单地将其应用于语言模型，我们将获得具有可接受的困惑的估计。尽管如此，如下面的实验所示，该模型预测未来令牌的能力受到严重影响。尽管有合理的困惑，但即使在多次迭代之后，它也只会生成胡言乱语。我们包含下面的代码作为警告示例，避免在错误的上下文中使用它们。

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

由于上述原因，产出显然不能令人满意。有关更有效地使用双向RNN的讨论，请参阅:numref:`sec_sentiment_rnn`中的情感分析应用程序。

## 摘要

* 在双向RNN中，每个时间步的隐藏状态由当前时间步之前和之后的数据同时确定。
* 双向RNN与概率图模型中的前向-后向算法有着惊人的相似之处。
* 在给定双向上下文的情况下，双向RNN最适用于序列编码和观测估计。
* 由于长梯度链，双向RNN的训练成本非常高。

## 练习

1. 如果不同方向使用不同数量的隐藏单位，$\mathbf{H}_t$的形状会发生怎样的变化？
1. 设计一个具有多个隐藏层的双向RNN。
1. 一词多义在自然语言中很常见。例如，“银行”一词在上下文中有不同的意思，“我去银行存现金”和“我去银行坐下来”。我们如何设计一个神经网络模型，使其在给定上下文序列和单词的情况下，返回该单词在上下文中的向量表示？哪种类型的神经结构更适合处理一词多义？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/339)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1059)
:end_tab:
