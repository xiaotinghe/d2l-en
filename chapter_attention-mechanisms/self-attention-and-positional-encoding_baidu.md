# 自我注意与位置编码
:label:`sec_self-attention-and-positional-encoding`

在深度学习中，我们经常使用CNNs或RNNs来编码序列。现在有了注意力机制。想象一下，我们将一系列令牌输入到注意池中，以便同一组令牌充当查询、键和值。具体来说，每个查询关注所有键值对，并生成一个注意输出。由于查询、键和值来自同一个位置，因此执行
*自我注意*:cite:`Lin.Feng.Santos.ea.2017,Vaswani.Shazeer.Parmar.ea.2017`，也称为内部注意*:cite:`Cheng.Dong.Lapata.2016,Parikh.Tackstrom.Das.ea.2016,Paulus.Xiong.Socher.2017`。
在本节中，我们将讨论使用自我注意的序列编码，包括使用序列顺序的附加信息。

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn
```

## 自我关注

给定一个输入令牌序列$\mathbf{x}_1, \ldots, \mathbf{x}_n$，其中任何$\mathbf{x}_i \in \mathbb{R}^d$（$1 \leq i \leq n$），其自身注意力输出一个相同长度的序列$\mathbf{y}_1, \ldots, \mathbf{y}_n$，其中

$$\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d$$

根据:eqref:`eq_attn-pooling`中$f$注意池的定义。下面的代码片段使用多头部注意来计算形状张量的自我注意（批量大小、时间步数或标记中的序列长度，$d$）。输出张量具有相同的形状。

```{.python .input}
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
#@tab pytorch
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
attention.eval()
```

```{.python .input}
#@tab all
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
attention(X, X, X, valid_lens).shape
```

## 比较CNNs、RNNs和自我注意
:label:`subsec_cnn-rnn-self-attention`

让我们比较用于将$n$令牌序列映射到另一个等长序列的架构，其中每个输入或输出令牌由$d$维向量表示。具体来说，我们将考虑CNNs、RNNs和自我关注。我们将比较它们的计算复杂度、顺序操作和最大路径长度。注意，序列操作防止并行计算，而序列位置的任何组合之间的较短路径使得在序列:cite:`Hochreiter.Bengio.Frasconi.ea.2001`内更容易学习长程依赖性。

![Comparing CNN (padding tokens are omitted), RNN, and self-attention architectures.](../img/cnn-rnn-self-attention.svg)
:label:`fig_cnn-rnn-self-attention`

考虑一个内核大小为$k$的卷积层。在后面的章节中，我们将提供有关使用CNNs进行序列处理的更多细节。目前，我们只需要知道，由于序列长度是$n$，输入和输出通道数都是$d$，卷积层的计算复杂度是$\mathcal{O}(knd^2)$。如:numref:`fig_cnn-rnn-self-attention`所示，cnn是分层的，因此有$\mathcal{O}(1)$个顺序操作，最大路径长度是$\mathcal{O}(n/k)$。例如，$\mathbf{x}_1$和$\mathbf{x}_5$位于:numref:`fig_cnn-rnn-self-attention`中内核大小为3的两层CNN的感受野内。

当更新RNN的隐藏状态时，$d \times d$权重矩阵和$d$维隐藏状态的乘法具有$\mathcal{O}(d^2)$的计算复杂度。由于序列长度是$n$，所以递归层的计算复杂度是$\mathcal{O}(nd^2)$。根据:numref:`fig_cnn-rnn-self-attention`，有$\mathcal{O}(n)$个顺序操作不能并行化，最大路径长度也是$\mathcal{O}(n)$。

在自我注意中，查询、键和值都是$n \times d$个矩阵。考虑:eqref:`eq_softmax_QK_V`中的缩放点积注意，其中$n \times d$矩阵乘以$d \times n$矩阵，然后输出$n \times n$矩阵乘以$n \times d$矩阵。因此，自我注意的计算复杂度为$\mathcal{O}(n^2d)$。正如我们在:numref:`fig_cnn-rnn-self-attention`中看到的，每个令牌通过自我注意直接连接到任何其他令牌。因此，计算可以与$\mathcal{O}(1)$顺序操作并行，并且最大路径长度也是$\mathcal{O}(1)$。

总之，CNNs和self-attention都具有并行计算能力，self-attention具有最短的最大路径长度。然而，与序列长度相关的二次计算复杂度使得对于很长的序列，自我注意的速度非常慢。

## 位置编码
:label:`subsec_positional-encoding`

与递归地逐个处理序列令牌的RNN不同，自关注抛弃了序列操作，转而支持并行计算。为了使用序列顺序信息，我们可以通过向输入表示添加*位置编码*来注入绝对或相对位置信息。位置编码可以是学习的，也可以是固定的。下面，我们将描述一种基于正弦和余弦函数:cite:`Vaswani.Shazeer.Parmar.ea.2017`的固定位置编码。

假设输入表示$\mathbf{X} \in \mathbb{R}^{n \times d}$包含$n$序列标记的$d$维嵌入。位置编码使用相同形状的位置嵌入矩阵$\mathbf{P} \in \mathbb{R}^{n \times d}$输出$\mathbf{X} + \mathbf{P}$，其在$i^\mathrm{th}$行和$(2j)^\mathrm{th}$或$(2j + 1)^\mathrm{th}$列上的元素是相同的

$$\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right),\\p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right).\end{aligned}$$
:eqlabel:`eq_positional-encoding-def`

乍一看，这个三角函数设计看起来很奇怪。在解释这个设计之前，让我们先在下面的`PositionalEncoding`类中实现它。

```{.python .input}
#@save
class PositionalEncoding(nn.Block):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_ctx(X.ctx)
        return self.dropout(X)
```

```{.python .input}
#@tab pytorch
#@save
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
```

在位置嵌入矩阵$\mathbf{P}$中，行对应于序列内的位置，列表示不同的位置编码维度。在下面的示例中，我们可以看到位置嵌入矩阵的$6^{\mathrm{th}}$和$7^{\mathrm{th}}$列的频率高于$8^{\mathrm{th}}$和$9^{\mathrm{th}}$列。$6^{\mathrm{th}}$和$7^{\mathrm{th}}$（$8^{\mathrm{th}}$和$9^{\mathrm{th}}$相同）列之间的偏移是由于正弦和余弦函数的交替引起的。

```{.python .input}
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.initialize()
X = pos_encoding(np.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
#@tab pytorch
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(d2l.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

### 绝对位置信息

为了了解编码维度上单调减少的频率与绝对位置信息的关系，让我们打印出$0, 1, \ldots, 7$的二进制表示。我们可以看到，最低位、第二最低位和第三最低位分别在每个数字、每两个数字和每四个数字上交替出现。

```{.python .input}
#@tab all
for i in range(8):
    print(f'{i} in binary is {i:>03b}')
```

在二进制表示中，高位比低位的频率低。类似地，如下面的热图所示，位置编码通过使用三角函数沿编码维度降低频率。由于输出是浮点数，这样的连续表示比二进制表示更节省空间。

```{.python .input}
P = np.expand_dims(np.expand_dims(P[0, :, :], 0), 0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
#@tab pytorch
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

### 相对位置信息

除了捕获绝对位置信息外，上述位置编码还允许模型通过相对位置轻松学习参与。这是因为对于任何固定位置偏移$\delta$，位置$i + \delta$处的位置编码可以由位置$i$处的位置编码的线性投影来表示。

这个投影可以用数学来解释。表示$\omega_j = 1/10000^{2j/d}$，:eqref:`eq_positional-encoding-def`中的任何$(p_{i, 2j}, p_{i, 2j+1})$对可以线性投影到$(p_{i+\delta, 2j}, p_{i+\delta, 2j+1})$，用于任何固定偏移$\delta$：

$$\begin{aligned}
&\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}\\
=&\begin{bmatrix} \cos(\delta \omega_j) \sin(i \omega_j) + \sin(\delta \omega_j) \cos(i \omega_j) \\  -\sin(\delta \omega_j) \sin(i \omega_j) + \cos(\delta \omega_j) \cos(i \omega_j) \\ \end{bmatrix}\\
=&\begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix}\\
=& 
\begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix},
\end{aligned}$$

其中$2\times 2$投影矩阵不依赖于任何位置索引$i$。

## 摘要

* 在自我关注中，查询、键和值都来自同一个地方。
* CNNs和self-attention都具有并行计算能力，self-attention具有最短的最大路径长度。然而，与序列长度相关的二次计算复杂度使得对于很长的序列，自我注意的速度非常慢。
* 为了使用序列顺序信息，我们可以通过向输入表示添加位置编码来注入绝对或相对位置信息。

## 练习

1. 假设我们设计了一个深层架构来表示一个序列，通过使用位置编码堆叠自我注意层。可能是什么问题？
1. 你能设计一个可学习的位置编码方法吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1651)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1652)
:end_tab:
