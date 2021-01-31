# 递归神经网络
:label:`sec_rnn`

在:numref:`sec_language_model`中，我们引入了$n$-gram模型，其中单词$x_t$在时间步$t$的条件概率仅取决于之前的$n-1$个单词。如果我们想纳入早于时间步长$t-(n-1)$的词对$x_t$的可能影响，我们需要增加$n$。然而，模型参数的数量也将随之指数增加，因为我们需要为词汇表集合$\mathcal{V}$存储$|\mathcal{V}|^n$个数字。因此，优选使用潜变量模型，而不是建模$P(x_t \mid x_{t-1}, \ldots, x_{t-n+1})$：

$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),$$

其中$h_{t-1}$是存储到时间步骤$t-1$的序列信息的*隐藏状态*(也称为隐藏变量)。通常，任何时间步骤$t$的隐藏状态都可以基于当前输入$x_{t}$和先前的隐藏状态$h_{t-1}$来计算：

$$h_t = f(x_{t}, h_{t-1}).$$
:eqlabel:`eq_ht_xt`

对于足够强大的函数$f$(:eqref:`eq_ht_xt`)，潜变量模型不是近似值。毕竟，$h_t$可能只是存储到目前为止观察到的所有数据。然而，它可能会使计算和存储都变得昂贵。

回想一下，我们在:numref:`chap_perceptrons`中讨论过具有隐藏单元的隐藏层。值得注意的是，隐藏层和隐藏状态指的是两个截然不同的概念。如上所述，隐藏层是在从输入到输出的路径上对视图隐藏的层。从技术上讲，隐藏状态是我们在给定步骤所做的任何事情的“输入”，它们只能通过查看先前时间点的数据来计算。

*递归神经网络(RNNs)是具有隐藏状态的神经网络。在介绍RNN模型之前，我们首先回顾:numref:`sec_mlp`中引入的MLP模型。

## 无隐态神经网络

让我们来看一看只有一个隐藏层的MLP。设隐藏层的激活函数为$\phi$。给定示例$\mathbf{X} \in \mathbb{R}^{n \times d}$的小批次，其中批次大小为$n$和$d$的输入，隐藏层的输出$\mathbf{H} \in \mathbb{R}^{n \times h}$被计算为

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{xh} + \mathbf{b}_h).$$
:eqlabel:`rnn_h_without_state`

在:eqref:`rnn_h_without_state`中，我们具有用于隐藏层的权重参数$\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$、偏置参数$\mathbf{b}_h \in \mathbb{R}^{1 \times h}$和隐藏单元的数目$h$。因此，在求和期间应用广播(见:numref:`subsec_broadcasting`)。接下来，将隐藏变量$\mathbf{H}$用作输出层的输入。输出图层由

$$\mathbf{O} = \mathbf{H} \mathbf{W}_{hq} + \mathbf{b}_q,$$

其中，$\mathbf{O} \in \mathbb{R}^{n \times q}$是输出变量，$\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$是权重参数，$\mathbf{b}_q \in \mathbb{R}^{1 \times q}$是输出层的偏移参数。如果是分类问题，我们可以用$\text{softmax}(\mathbf{O})$来计算输出类别的概率分布。

这完全类似于我们之前在:numref:`sec_sequence`中解决的回归问题，因此我们省略了细节。可以说，我们可以随机选择特征-标签对，并通过自动微分和随机梯度下降来学习网络参数。

## 具有隐状态的递归神经网络
:label:`subsec_rnn_w_hidden_states`

当我们有隐藏状态时，情况就完全不同了。让我们更详细地看看这个结构。

假设我们在时间步骤$\mathbf{X}_t \in \mathbb{R}^{n \times d}$具有一小批输入$t$。换言之，对于$n$个序列示例的小批量，$\mathbf{X}_t$的每行对应于来自该序列的时间步骤$t$处的一个示例。接下来，用$\mathbf{H}_t  \in \mathbb{R}^{n \times h}$表示时间步长$t$的隐藏变量。与最大似然算法不同的是，这里我们保存了前一个时间步的隐藏变量$\mathbf{H}_{t-1}$，并引入了一个新的权重参数$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$来描述如何在当前时间步中使用前一个时间步的隐藏变量。具体地，当前时间步长的隐藏变量的计算由当前时间步长的输入与前一个时间步长的隐藏变量一起确定：

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h).$$
:eqlabel:`rnn_h_with_state`

与:eqref:`rnn_h_without_state`相比，:eqref:`rnn_h_with_state`多添加了一项$\mathbf{H}_{t-1} \mathbf{W}_{hh}$，从而实例化了:eqref:`eq_ht_xt`。从相邻时间步长的隐藏变量$\mathbf{H}_t$和$\mathbf{H}_{t-1}$之间的关系可知，这些变量捕获并保留了序列直到其当前时间步长的历史信息，就像神经网络的当前时间步长的状态或记忆一样。因此，这样的隐藏变量被称为“隐藏状态”。由于隐藏状态使用与当前时间步长中的前一个时间步长相同的定义，因此:eqref:`rnn_h_with_state`的计算是*递归的*。因此，基于递归计算的隐状态神经网络被命名为
*递归神经网络*。
在RNN中执行:eqref:`rnn_h_with_state`计算的层称为“循环层”。

构建RNN有许多不同的方法。具有由:eqref:`rnn_h_with_state`定义的隐藏状态的RNN非常常见。对于时间步长$t$，输出层的输出类似于MLP中的计算：

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

随机神经网络的参数包括隐藏层的权重$\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$和偏差$\mathbf{b}_h \in \mathbb{R}^{1 \times h}$，以及输出层的权重$\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$和偏差$\mathbf{b}_q \in \mathbb{R}^{1 \times q}$。值得一提的是，即使在不同的时间步长，RNN也总是使用这些模型参数。因此，RNN的参数化成本不会随着时间步数的增加而增加。

:numref:`fig_rnn`示出了在三个相邻时间步长的rnn的计算逻辑。在任何时间步骤$t$，隐藏状态的计算可以被处理为：i)将当前时间步骤$t$的输入$\mathbf{X}_t$和前一时间步骤$t-1$的隐藏状态$t-1$连接；ii)将连接结果馈送到具有激活功能$\phi$的全连接层。这种完全连接层的输出是当前时间步长$\mathbf{H}_t$的隐藏状态$t$。在本例中，模型参数是$\mathbf{W}_{xh}$和$\mathbf{W}_{hh}$的串联，以及$\mathbf{b}_h$的偏移，所有这些参数都是从:eqref:`rnn_h_with_state`开始的。当前时间步$t$、$\mathbf{H}_t$的隐藏状态将参与计算下一时间步$\mathbf{H}_{t+1}$的隐藏状态$t+1$。此外，还将$\mathbf{H}_t$馈入全连接输出层，以计算当前时间步长$t$的输出$\mathbf{O}_t$。

![An RNN with a hidden state.](../img/rnn.svg)
:label:`fig_rnn`

我们刚才提到，隐藏态$\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}$的计算相当于$\mathbf{X}_t$和$\mathbf{H}_{t-1}$的级联矩阵乘法，以及$\mathbf{W}_{xh}$和$\mathbf{W}_{hh}$的级联矩阵乘法。虽然可以在数学上证明这一点，但在下面我们只使用一个简单的代码片段来说明这一点。首先，我们定义矩阵`X`、`W_xh`、`H`和`W_hh`，它们的形状分别为(3，1)、(1，4)、(3，4)和(4，4)。分别将`X`乘以`W_xh`，将`H`乘以`W_hh`，然后将这两个乘法相加，我们得到一个形状为(3，4)的矩阵。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab mxnet, pytorch
X, W_xh = d2l.normal(0, 1, (3, 1)), d2l.normal(0, 1, (1, 4))
H, W_hh = d2l.normal(0, 1, (3, 4)), d2l.normal(0, 1, (4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
#@tab tensorflow
X, W_xh = d2l.normal((3, 1), 0, 1), d2l.normal((1, 4), 0, 1)
H, W_hh = d2l.normal((3, 4), 0, 1), d2l.normal((4, 4), 0, 1)
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

现在，我们沿列(轴1)连接矩阵`X`和`H`，沿行(轴0)连接矩阵`W_xh`和`W_hh`。这两个连接分别产生形状(3，5)和形状(5，4)的矩阵。将这两个级联的矩阵相乘，我们得到与上面相同的形状(3，4)的输出矩阵。

```{.python .input}
#@tab all
d2l.matmul(d2l.concat((X, H), 1), d2l.concat((W_xh, W_hh), 0))
```

## 基于RNN的字符级语言模型

回想一下，对于:numref:`sec_language_model`中的语言建模，我们的目标是根据当前和过去的标记预测下一个标记，因此我们将原始序列移位一个标记作为标签。Bengio等人。首先提出使用神经网络进行语言建模:cite:`Bengio.Ducharme.Vincent.ea.2003`。在下面，我们将说明如何使用RNN来构建语言模型。设小批量大小为1，文本顺序为“机器”。为了简化后续部分的培训，我们将文本标记化为字符而不是单词，并考虑使用*字符级语言模型*。:numref:`fig_rnn_train`演示了如何通过用于字符级语言建模的RNN基于当前字符和先前字符预测下一个字符。

![A character-level language model based on the RNN. The input and label sequences are "machin" and "achine", respectively.](../img/rnn-train.svg)
:label:`fig_rnn_train`

在训练过程中，针对每个时间步对输出层的输出进行Softmax运算，然后使用交叉熵损失来计算模型输出与标签之间的误差。由于隐藏层中隐藏状态的递归计算，:numref:`fig_rnn_train`、$\mathbf{O}_3$中的时间步骤3的输出由文本序列“m”、“a”和“c”确定。由于训练数据中的序列的下一个字符是“h”，因此时间步长3的损失将取决于基于该时间步长的特征序列“m”、“a”、“c”和标签“h”生成的下一个字符的概率分布。

实际上，每个令牌都由一个$d$维向量表示，我们使用批大小$n>1$。因此，输入$\mathbf X_t$在时间步$t$将是$n\times d$矩阵，这与我们在:numref:`subsec_rnn_w_hidden_states`中讨论的相同。

## 困惑
:label:`subsec_perplexity`

最后，让我们讨论如何度量语言模型质量，这将在后续部分中用于评估我们基于RNN的模型。一种方法是检查文本有多令人惊讶。一个好的语言模型能够用高精度的记号来预测我们接下来会看到什么。考虑一下不同语言模型提出的短语“正在下雨”的以下延续：

1. “外面在下雨”
1. “香蕉树正下着雨呢”
1. “天下着雨，KCJ Pwepoiut”

就质量而言，示例1显然是最好的。这些词是明智的，逻辑上是连贯的。虽然它可能不能很准确地反映出语义上跟在哪个单词后面(“在旧金山”和“在冬天”会是非常合理的扩展)，但该模型能够捕捉到跟在后面的是哪种单词。示例2产生了一个无意义的扩展，这要糟糕得多。尽管如此，至少该模型已经学会了如何拼写单词以及单词之间的某种程度的相关性。最后，示例3指出训练不足的模型不能很好地拟合数据。

我们可以通过计算序列的似然率来衡量模型的质量。不幸的是，这是一个很难理解和难以比较的数字。毕竟，较短的序列比较长的序列更有可能出现，因此在托尔斯泰的巨著上对该模型进行了评估
*“战争与和平”不可避免地会比圣埃克苏佩里的中篇小说“小王子”产生的可能性要小得多。缺少的是相当于平均数的数字。

信息论在这里派上了用场。在引入软最大回归(:numref:`subsec_info_theory_basics`)时，我们定义了熵、奇异熵和交叉熵，并在[online appendix on information theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html)讨论了更多的信息论。如果我们想压缩文本，我们可以询问在给定当前令牌集的情况下预测下一个令牌。一个更好的语言模型应该能让我们更准确地预测下一个令牌。因此，它应该允许我们在压缩序列时花费更少的比特。因此，我们可以通过一个序列的所有$n$个令牌的平均交叉熵损失来衡量它：

$$\frac{1}{n} \sum_{t=1}^n -\log P(x_t \mid x_{t-1}, \ldots, x_1),$$
:eqlabel:`eq_avg_ce_for_lm`

其中$P$由语言模型给出，$x_t$是在时间步骤$t$从该序列观察到的实际令牌。这使得在不同长度的文档上的性能具有可比性。由于历史原因，自然语言处理的科学家更喜欢使用一个叫做“困惑”的量。简而言之，它是:eqref:`eq_avg_ce_for_lm`的指数：

$$\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right).$$

困惑可以最好地理解为我们在决定下一步选择哪个令牌时所拥有的真实选择数量的调和平均数。让我们看看几个例子：

* 在最好的情况下，模型总是完美地估计标签令牌的概率为1。在这种情况下，模型的困惑程度为1。
* 在最坏的情况下，模型总是预测标签令牌的概率为0。在这种情况下，困惑是正无穷大。
* 在基线上，该模型预测词汇表的所有可用标记上的均匀分布。在这种情况下，困惑程度等于词汇表的唯一标记的数量。事实上，如果我们在没有任何压缩的情况下存储序列，这将是我们能做的最好的编码。因此，这提供了一个任何有用的模型都必须超越的不平凡的上限。

在接下来的几节中，我们将为字符级语言模型实现RNN，并使用Pplexity来评估这些模型。

## 摘要

* 对隐藏状态使用递归计算的神经网络称为递归神经网络(RNN)。
* RNN的隐藏状态可以捕获直到当前时间步的序列的历史信息。
* RNN模型参数的数量不会随着时间步长的增加而增加。
* 我们可以使用RNN创建字符级语言模型。
* 我们可以用困惑来评价语言模型的质量。

## 练习

1. 如果我们使用RNN来预测文本序列中的下一个字符，那么任何输出所需的维度是什么？
1. 为什么RNN可以基于文本序列中所有先前的令牌在某个时间步长表示令牌的条件概率？
1. 如果你反向传播一个长序列，梯度会发生什么？
1. 与本节中描述的语言模型相关的问题有哪些？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/337)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1050)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1051)
:end_tab:
