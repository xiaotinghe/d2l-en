# 注意力评分功能
:label:`sec_attention-scoring-functions`

在:numref:`sec_nadaraya-waston`中，我们使用高斯核来模拟查询和密钥之间的交互。将:eqref:`eq_nadaraya-waston-gaussian`中高斯核的指数视为*注意评分函数*（简称*评分函数*），该函数的结果基本上被输入到softmax操作中。结果，我们得到了与键配对的值的概率分布（注意权重）。最后，注意池的输出只是基于这些注意权重的值的加权和。

在较高的层次上，我们可以使用上述算法来实例化:numref:`fig_qkv`中的注意机制框架。用$a$表示注意评分函数，:numref:`fig_attention_output`说明了如何将注意池的输出计算为值的加权和。由于注意力权重是一个概率分布，加权和本质上是一个加权平均数。

![Computing the output of attention pooling as a weighted average of values.](../img/attention-output.svg)
:label:`fig_attention_output`

在数学上，假设我们有一个查询$\mathbf{q} \in \mathbb{R}^q$和$m$键值对$(\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)$，其中任意$\mathbf{k}_i \in \mathbb{R}^k$和任意$\mathbf{v}_i \in \mathbb{R}^v$。注意池$f$被实例化为值的加权和：

$$f(\mathbf{q}, (\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)) = \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i \in \mathbb{R}^v,$$
:eqlabel:`eq_attn-pooling`

其中，查询$\mathbf{q}$和键$\mathbf{k}_i$的注意权重（标量）是通过将两个向量映射到标量的注意评分函数$a$的softmax操作来计算的：

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(a(\mathbf{q}, \mathbf{k}_i)) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_{j=1}^m \exp(a(\mathbf{q}, \mathbf{k}_j))} \in \mathbb{R}.$$
:eqlabel:`eq_attn-scoring-alpha`

我们可以看到，不同的注意评分函数$a$的选择导致了不同的注意池行为。在本节中，我们将介绍两个流行的评分函数，稍后我们将使用它们开发更复杂的注意机制。

```{.python .input}
import math
from d2l import mxnet as d2l
from mxnet import np, npx
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

## 屏蔽Softmax操作

正如我们刚才提到的，softmax操作用于输出概率分布作为注意权重。在某些情况下，并非所有的价值观都应该被纳入注意力池。例如，为了在:numref:`sec_machine_translation`中进行高效的小批量处理，一些文本序列被填充了没有意义的特殊标记。为了使注意力集中在有意义的令牌上作为值，我们可以指定一个有效的序列长度（以令牌的数量为单位），以便在计算softmax时过滤掉超出此指定范围的那些令牌。这样，我们就可以在下面的`masked_softmax`函数中实现这样一个*masked softmax操作*，其中超出有效长度的任何值都被屏蔽为零。

```{.python .input}
#@save
def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return npx.softmax(X)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = valid_lens.repeat(shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = npx.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, True,
                              value=-1e6, axis=1)
        return npx.softmax(X).reshape(shape)
```

```{.python .input}
#@tab pytorch
#@save
def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
```

为了演示此函数的工作原理，请考虑两个$2 \times 4$矩阵示例的小批量，其中这两个示例的有效长度分别为2和3。作为屏蔽softmax操作的结果，超出有效长度的值都被屏蔽为零。

```{.python .input}
masked_softmax(np.random.uniform(size=(2, 2, 4)), d2l.tensor([2, 3]))
```

```{.python .input}
#@tab pytorch
masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
```

类似地，我们也可以使用二维张量为每个矩阵示例中的每一行指定有效长度。

```{.python .input}
masked_softmax(np.random.uniform(size=(2, 2, 4)),
               d2l.tensor([[1, 3], [2, 4]]))
```

```{.python .input}
#@tab pytorch
masked_softmax(torch.rand(2, 2, 4), d2l.tensor([[1, 3], [2, 4]]))
```

## 附加注意
:label:`subsec_additive-attention`

一般来说，当查询和键是不同长度的向量时，我们可以使用附加注意作为评分函数。给定一个查询$\mathbf{q} \in \mathbb{R}^q$和一个键$\mathbf{k} \in \mathbb{R}^k$，*附加注意*评分函数

$$a(\mathbf q, \mathbf k) = \mathbf w_v^\top \text{tanh}(\mathbf W_q\mathbf q + \mathbf W_k \mathbf k) \in \mathbb{R},$$
:eqlabel:`eq_additive-attn`

其中可学习参数$\mathbf W_q\in\mathbb R^{h\times q}$、$\mathbf W_k\in\mathbb R^{h\times k}$和$\mathbf w_v\in\mathbb R^{h}$。相当于:eqref:`eq_additive-attn`，查询和密钥被连接并馈送到一个具有单个隐藏层的MLP中，该层的隐藏单元数为$h$，这是一个超参数。通过使用$\tanh$作为激活函数和禁用偏差项，我们在下面实现了附加注意。

```{.python .input}
#@save
class AdditiveAttention(nn.Block):
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # Use `flatten=False` to only transform the last axis so that the
        # shapes for the other axes are kept the same
        self.W_k = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.W_q = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.w_v = nn.Dense(1, use_bias=False, flatten=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of `queries`: (`batch_size`, no. of
        # queries, 1, `num_hiddens`) and shape of `keys`: (`batch_size`, 1,
        # no. of key-value pairs, `num_hiddens`). Sum them up with
        # broadcasting
        features = np.expand_dims(queries, axis=2) + np.expand_dims(
            keys, axis=1)
        features = np.tanh(features)
        # There is only one output of `self.w_v`, so we remove the last
        # one-dimensional entry from the shape. Shape of `scores`:
        # (`batch_size`, no. of queries, no. of key-value pairs)
        scores = np.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of `values`: (`batch_size`, no. of key-value pairs, value
        # dimension)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```{.python .input}
#@tab pytorch
#@save
class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of `queries`: (`batch_size`, no. of
        # queries, 1, `num_hiddens`) and shape of `keys`: (`batch_size`, 1,
        # no. of key-value pairs, `num_hiddens`). Sum them up with
        # broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of `self.w_v`, so we remove the last
        # one-dimensional entry from the shape. Shape of `scores`:
        # (`batch_size`, no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of `values`: (`batch_size`, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

让我们用一个玩具示例演示上述`AdditiveAttention`类，其中查询、键和值的形状（批量大小、令牌中的步数或序列长度、特征大小）分别为（$2$、$1$、$20$）、（$2$、$10$、$2$）和（$2$、$10$、$4$）。注意池输出的形状为（批量大小、查询步骤数、值的特征大小）。

```{.python .input}
queries, keys = d2l.normal(0, 1, (2, 1, 20)), d2l.ones((2, 10, 2))
# The two value matrices in the `values` minibatch are identical
values = np.arange(40).reshape(1, 10, 4).repeat(2, axis=0)
valid_lens = d2l.tensor([2, 6])

attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
attention.initialize()
attention(queries, keys, values, valid_lens)
```

```{.python .input}
#@tab pytorch
queries, keys = d2l.normal(0, 1, (2, 1, 20)), d2l.ones((2, 10, 2))
# The two value matrices in the `values` minibatch are identical
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
    2, 1, 1)
valid_lens = d2l.tensor([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)
attention.eval()
attention(queries, keys, values, valid_lens)
```

虽然附加注意包含可学习的参数，但由于在本例中每个键都是相同的，因此注意权重是一致的，由指定的有效长度确定。

```{.python .input}
#@tab all
d2l.show_heatmaps(d2l.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## 产品注意事项

计分函数的计算效率更高的设计可以是简单的点积。但是，点积运算要求查询和键具有相同的向量长度，例如$d$。假设查询和键的所有元素都是均值和单位方差为零的独立随机变量。两个向量的点积的均值为零，方差为$d$。为了确保无论向量长度如何，点积的方差仍然保持为1，*缩放点积注意*评分函数

$$a(\mathbf q, \mathbf k) = \mathbf{q}^\top \mathbf{k}  /\sqrt{d}$$

将点积除以$\sqrt{d}$。在实践中，为了提高效率，我们通常以小批量方式进行思考，例如计算$n$查询和$m$键值对的注意，其中查询和键的长度为$d$，值的长度为$v$。查询$\mathbf Q\in\mathbb R^{n\times d}$、键$\mathbf K\in\mathbb R^{m\times d}$和值$\mathbf V\in\mathbb R^{m\times v}$的按比例点积注意为

$$ \mathrm{softmax}\left(\frac{\mathbf Q \mathbf K^\top }{\sqrt{d}}\right) \mathbf V \in \mathbb{R}^{n\times v}.$$
:eqlabel:`eq_softmax_QK_V`

在缩放点积注意的后续实现中，我们使用dropout进行模型正则化。

```{.python .input}
#@save
class DotProductAttention(nn.Block):
    """Scaled dot product attention."""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = npx.batch_dot(queries, keys, transpose_b=True) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```{.python .input}
#@tab pytorch
#@save
class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

为了演示上述`DotProductAttention`类，我们使用与前面的玩具示例相同的键、值和有效长度来增加注意。对于点积运算，我们使查询的特征大小与键的特征大小相同。

```{.python .input}
queries = d2l.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.initialize()
attention(queries, keys, values, valid_lens)
```

```{.python .input}
#@tab pytorch
queries = d2l.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
attention(queries, keys, values, valid_lens)
```

与加法注意演示中相同，由于`keys`包含任何查询都无法区分的相同元素，因此获得了统一的注意权重。

```{.python .input}
#@tab all
d2l.show_heatmaps(d2l.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## 摘要

* 我们可以将注意力池的输出计算为一个加权平均值，其中不同的注意力评分函数选择会导致不同的注意力池行为。
* 当查询和键是不同长度的向量时，我们可以使用加法注意评分函数。当两者相同时，标度点积注意力评分函数的计算效率更高。

## 练习

1. 修改玩具示例中的关键点并将注意力权重可视化。加法注意和标度点积注意是否仍然输出相同的注意权重？为什么？为什么不？
1. 仅使用矩阵乘法，可以为具有不同向量长度的查询和键设计新的评分函数吗？
1. 当查询和键具有相同的向量长度时，向量求和是否比点积更好地设计评分函数？为什么？为什么不？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/346)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1064)
:end_tab:
