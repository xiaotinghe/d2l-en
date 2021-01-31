# 循环神经网络
:label:`sec_rnn`

在:numref:`sec_language_model`中，我们引入了$n$ gram模型，其中单词$x_t$在时间步$t$的条件概率仅取决于$n-1$前面的单词。如果我们想将时间步长$t-(n-1)$之前的单词的可能影响合并到$x_t$上，我们需要增加$n$。但是，模型参数的数量也会随之呈指数增长，因为我们需要为词汇集$\mathcal{V}$存储$|\mathcal{V}|^n$个数字。因此，与其建模$P(x_t \mid x_{t-1}, \ldots, x_{t-n+1})$，不如使用潜变量模型：

$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),$$

其中$h_{t-1}$是*隐藏状态*（也称为隐藏变量），其存储到时间步骤$t-1$的序列信息。通常，可以基于当前输入$x_{t}$和先前隐藏状态$h_{t-1}$来计算步骤$t$处的任何时间的隐藏状态：

$$h_t = f(x_{t}, h_{t-1}).$$
:eqlabel:`eq_ht_xt`

对于:eqref:`eq_ht_xt`中足够强大的函数$f$，潜变量模型不是一个近似值。毕竟，$h_t$可以简单地存储它迄今为止观察到的所有数据。然而，它可能会使计算和存储都变得昂贵。

回想一下，我们在:numref:`chap_perceptrons`中讨论了隐藏层和隐藏单元。值得注意的是，隐藏层和隐藏状态指的是两个截然不同的概念。如前所述，隐藏层是在从输入到输出的路径上从视图中隐藏的层。从技术上讲，隐藏状态是我们在给定步骤中所做的任何事情的*输入*，它们只能通过查看以前时间步骤中的数据来计算。

*递归神经网络是一种具有隐状态的神经网络。在介绍RNN模型之前，我们首先回顾:numref:`sec_mlp`中介绍的MLP模型。

## 无隐态神经网络

让我们看一看具有单个隐藏层的MLP。设隐层的激活函数为$\phi$。给定批量大小为$n$和$d$输入的示例$\mathbf{X} \in \mathbb{R}^{n \times d}$的小批量，隐层的输出$\mathbf{H} \in \mathbb{R}^{n \times h}$计算如下

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{xh} + \mathbf{b}_h).$$
:eqlabel:`rnn_h_without_state`

在:eqref:`rnn_h_without_state`中，对于隐藏层，我们有权参数$\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$、偏置参数$\mathbf{b}_h \in \mathbb{R}^{1 \times h}$和隐藏单元数$h$。因此，在求和期间应用广播（参见:numref:`subsec_broadcasting`）。接下来，将隐藏变量$\mathbf{H}$用作输出层的输入。输出层由

$$\mathbf{O} = \mathbf{H} \mathbf{W}_{hq} + \mathbf{b}_q,$$

其中$\mathbf{O} \in \mathbb{R}^{n \times q}$是输出变量，$\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$是权重参数，$\mathbf{b}_q \in \mathbb{R}^{1 \times q}$是输出层的偏置参数。如果是分类问题，我们可以用$\text{softmax}(\mathbf{O})$来计算输出类别的概率分布。

这完全类似于我们之前在:numref:`sec_sequence`中解决的回归问题，因此我们省略了细节。可以说，我们可以随机选取特征标签对，通过自动微分和随机梯度下降来学习网络的参数。

## 隐状态递归神经网络
:label:`subsec_rnn_w_hidden_states`

当我们有隐藏状态时，情况就完全不同了。让我们更详细地看一下结构。

假设我们在时间步$t$有一小批输入$\mathbf{X}_t \in \mathbb{R}^{n \times d}$。换句话说，对于$n$序列示例的小批量，$\mathbf{X}_t$的每一行对应于来自该序列的时间步骤$t$的一个示例。接下来，用$\mathbf{H}_t  \in \mathbb{R}^{n \times h}$表示时间步长$t$的隐藏变量。与MLP不同的是，这里我们保存了前一时间步的隐藏变量$\mathbf{H}_{t-1}$，并引入了一个新的权重参数$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$来描述如何在当前时间步中使用前一时间步的隐藏变量。具体地说，当前时间步的隐藏变量的计算是由当前时间步的输入和前一时间步的隐藏变量一起确定的：

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h).$$
:eqlabel:`rnn_h_with_state`

与:eqref:`rnn_h_without_state`相比，:eqref:`rnn_h_with_state`增加了一个术语$\mathbf{H}_{t-1} \mathbf{W}_{hh}$，从而实例化了:eqref:`eq_ht_xt`。从相邻时间步的隐藏变量$\mathbf{H}_t$和$\mathbf{H}_{t-1}$之间的关系可知，这些变量捕捉并保留了序列直到其当前时间步的历史信息，就像神经网络当前时间步的状态或记忆一样。因此，这种隐藏变量称为*隐藏状态*。由于隐藏状态使用当前时间步长中的上一时间步长的相同定义，因此:eqref:`rnn_h_with_state`的计算是*递归的*。因此，基于递归计算的隐状态神经网络被称为隐状态神经网络
*递归神经网络*。
在RNN中执行:eqref:`rnn_h_with_state`计算的层称为*递归层*。

构建RNN有许多不同的方法。具有:eqref:`rnn_h_with_state`定义的隐藏状态的RNN非常常见。对于时间步长$t$，输出层的输出类似于MLP中的计算：

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

RNN的参数包括隐藏层的权重$\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$和偏置$\mathbf{b}_h \in \mathbb{R}^{1 \times h}$，以及输出层的权重$\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$和偏置$\mathbf{b}_q \in \mathbb{R}^{1 \times q}$。值得一提的是，即使在不同的时间步，RNN总是使用这些模型参数。因此，RNN的参数化成本不会随着时间步数的增加而增加。

:numref:`fig_rnn`说明了RNN在三个相邻时间步的计算逻辑。在任何时间步骤$t$处，隐藏状态的计算可被视为：i）将当前时间步骤$t$处的输入$\mathbf{X}_t$与上一时间步骤$t-1$处的隐藏状态$\mathbf{H}_{t-1}$串联；ii）将串联结果馈送到具有激活函数$\phi$的完全连接层。这种完全连接层的输出是当前时间步$t$的隐藏状态$\mathbf{H}_t$。在这种情况下，模型参数是$\mathbf{W}_{xh}$和$\mathbf{W}_{hh}$的串联，以及$\mathbf{b}_h$的偏差，全部来自:eqref:`rnn_h_with_state`。当前时间步骤$t$、$\mathbf{H}_t$的隐藏状态将参与计算下一时间步骤$t+1$的隐藏状态$\mathbf{H}_{t+1}$。此外，$\mathbf{H}_t$还将被馈入完全连接的输出层以计算当前时间步长$t$的输出$\mathbf{O}_t$。

![An RNN with a hidden state.](../img/rnn.svg)
:label:`fig_rnn`

我们刚才提到，$\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}$对隐藏状态的计算相当于$\mathbf{X}_t$和$\mathbf{H}_{t-1}$的级联和$\mathbf{W}_{xh}$和$\mathbf{W}_{hh}$的级联的矩阵乘法。虽然这可以用数学证明，但在下面我们只使用一个简单的代码片段来说明这一点。首先，我们定义了矩阵`X`、`W_xh`、`H`和`W_hh`，它们的形状分别是（3，1）、（1，4）、（3，4）和（4，4）。分别将`X`乘以`W_xh`，`H`乘以`W_hh`，然后将这两个乘法相加，得到形状矩阵（3，4）。

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

现在我们沿着列（轴1）连接矩阵`X`和`H`，沿着行（轴0）连接矩阵`W_xh`和`W_hh`。这两个串联分别产生形状（3，5）和形状（5，4）的矩阵。将这两个串联矩阵相乘，我们得到与上述形状（3，4）相同的输出矩阵。

```{.python .input}
#@tab all
d2l.matmul(d2l.concat((X, H), 1), d2l.concat((W_xh, W_hh), 0))
```

## 基于RNN的字符级语言模型

回想一下，对于:numref:`sec_language_model`中的语言建模，我们的目标是基于当前和过去的标记预测下一个标记，因此我们将原始序列移动一个标记作为标签。Bengio等人首先提出使用神经网络进行语言建模:cite:`Bengio.Ducharme.Vincent.ea.2003`。下面我们将说明如何使用RNN构建语言模型。让minibatch大小为1，文本的顺序为“machine”。为了简化后续章节中的训练，我们将文本标记为字符而不是单词，并考虑一个*字符级语言模型*。:numref:`fig_rnn_train`演示了如何通过用于字符级语言建模的RNN基于当前和以前的字符预测下一个字符。

![A character-level language model based on the RNN. The input and label sequences are "machin" and "achine", respectively.](../img/rnn-train.svg)
:label:`fig_rnn_train`

在训练过程中，我们对每个时间步长的输出层的输出进行softmax操作，然后利用交叉熵损失计算模型输出和标签之间的误差。由于隐藏层中的隐藏状态的循环计算，:numref:`fig_rnn_train`、$\mathbf{O}_3$中的时间步长3的输出由文本序列“m”、“a”和“c”确定。由于训练数据中的序列的下一个字符是“h”，因此时间步长3的丢失将取决于基于该时间步长的特征序列“m”、“a”、“c”和标签“h”生成的下一个字符的概率分布。

在实践中，每个令牌都由一个$d$维向量表示，我们使用一个批大小$n>1$。因此，在时间步$t$处的输入$\mathbf X_t$将是$n\times d$矩阵，这与我们在:numref:`subsec_rnn_w_hidden_states`中讨论的相同。

## 困惑
:label:`subsec_perplexity`

最后，让我们讨论如何度量语言模型的质量，这将在后面的小节中用于评估基于RNN的模型。一种方法是检查文本是否令人惊讶。一个好的语言模型能够用高精度的标记预测我们接下来将看到的内容。考虑以下由不同语言模式提出的短语“It is raining”的延续：

1. “外面在下雨”
1. “香蕉树在下雨”
1. “正在下雨”

就质量而言，示例1显然是最好的。这些词很有道理，逻辑上也很连贯。虽然这个模型可能不能很准确地反映出哪个词在语义上跟在后面（“在旧金山”和“在冬天”可能是完全合理的扩展），但它能够捕捉到哪个词跟在后面。例2由于产生了一个无意义的扩展而变得相当糟糕。尽管如此，至少模型已经学会了如何拼写单词以及单词之间的某种程度的相关性。最后，示例3指出了一个训练不好的模型，它不能正确地拟合数据。

我们可以通过计算序列的可能性来衡量模型的质量。不幸的是，这是一个很难理解和比较的数字。毕竟，较短的序列比较长的序列更有可能发生，因此，对托尔斯泰巨著的模型进行了评估
*战争与和平必然会产生比圣埃克苏佩里的中篇小说《小王子》小得多的可能性。所缺的相当于平均数。

信息论在这里很有用。我们在引入softmax回归（:numref:`subsec_info_theory_basics`）时定义了熵、超熵和交叉熵，[online appendix on information theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html)中讨论了更多的信息论。如果我们想压缩文本，我们可以询问关于预测下一个标记给定的当前标记集。一个更好的语言模型应该能让我们更准确地预测下一个标记。因此，它应该允许我们在压缩序列时花费更少的位。所以我们可以通过一个序列中所有$n$个标记的平均交叉熵损失来衡量：

$$\frac{1}{n} \sum_{t=1}^n -\log P(x_t \mid x_{t-1}, \ldots, x_1),$$
:eqlabel:`eq_avg_ce_for_lm`

其中$P$由语言模型给出，$x_t$是从序列中在时间步骤$t$处观察到的实际令牌。这使得不同长度的文档的性能具有可比性。由于历史的原因，自然语言处理领域的科学家更喜欢使用一个称为“困惑”的量。简而言之，它是:eqref:`eq_avg_ce_for_lm`的指数：

$$\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right).$$

困惑可以最好地理解为当我们决定下一个选择哪个标记时，实际选择数的调和平均数。让我们看看一些案例：

* 在最佳情况下，模型总是完美地估计标签令牌的概率为1。在这种情况下，模型的困惑是1。
* 在最坏的情况下，模型总是预测标签令牌的概率为0。在这种情况下，困惑就是正无穷大。
* 在基线时，该模型预测词汇表中所有可用标记的均匀分布。在这种情况下，困惑等于词汇表中唯一标记的数量。事实上，如果我们要存储序列而不进行任何压缩，这将是我们所能做的最好的编码。因此，这提供了一个重要的上限，任何有用的模型都必须超越这个上限。

在下面的部分中，我们将为字符级语言模型实现RNN，并使用Convertible来评估这些模型。

## 摘要

* 使用隐状态递归计算的神经网络称为递归神经网络（RNN）。
* RNN的隐藏状态可以捕获序列到当前时间步长的历史信息。
* RNN模型参数的数量不会随着时间步长的增加而增加。
* 我们可以使用RNN创建字符级语言模型。
* 我们可以用困惑度来评价语言模型的质量。

## 练习

1. 如果我们使用RNN来预测文本序列中的下一个字符，那么输出所需的维数是多少？
1. 为什么RNN可以基于文本序列中所有先前的令牌来表示某个时间步的令牌的条件概率？
1. 如果通过一个长序列反向传播，梯度会发生什么变化？
1. 与本节描述的语言模型相关的一些问题是什么？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/337)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1050)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1051)
:end_tab:
