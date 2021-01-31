# 注意力评分函数
:label:`sec_attention-scoring-functions`

在:numref:`sec_nadaraya-waston`中，我们使用高斯内核对查询和键之间的交互进行建模。将:eqref:`eq_nadaraya-waston-gaussian`中高斯核的指数作为“注意力评分函数”(或简称“评分函数”)，该函数的结果实质上被输入到软最大运算中。结果，我们获得了与关键字配对的值的概率分布(关注权重)。最后，注意力集中的输出只是基于这些注意力权重的值的加权和。

在更高的层次上，我们可以使用上述算法来实例化:numref:`fig_qkv`中的注意机制框架。用$a$,:numref:`fig_attention_output`表示注意力评分函数示出了如何将注意力汇集的输出计算为值的加权和。由于注意力权重是概率分布，因此加权和实质上是加权平均值。

![Computing the output of attention pooling as a weighted average of values.](../img/attention-output.svg)
:label:`fig_attention_output`

在数学上，假设我们具有查询$\mathbf{q} \in \mathbb{R}^q$和$m$个键-值对$(\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)$，其中任意$\mathbf{k}_i \in \mathbb{R}^k$和任意$\mathbf{v}_i \in \mathbb{R}^v$。注意池$f$被实例化为值的加权和：

$$f(\mathbf{q}, (\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)) = \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i \in \mathbb{R}^v,$$
:eqlabel:`eq_attn-pooling`

其中查询$\mathbf{q}$和关键字$\mathbf{k}_i$的关注度权重(标量)是通过将两个向量映射到标量的关注度评分函数$a$的软最大运算来计算的：

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(a(\mathbf{q}, \mathbf{k}_i)) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_{j=1}^m \exp(a(\mathbf{q}, \mathbf{k}_j))} \in \mathbb{R}.$$
:eqlabel:`eq_attn-scoring-alpha`

如我们所见，注意力评分功能$a$的不同选择导致不同的注意力汇集行为。在本节中，我们将介绍两个流行的评分函数，稍后我们将使用它们来开发更复杂的注意力机制。

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

## 屏蔽软最大操作

正如我们刚才提到的，Softmax运算用于输出概率分布作为关注度权重。在某些情况下，并不是所有的价值都应该输入到注意力集中中。例如，为了在:numref:`sec_machine_translation`中进行高效的小批量处理，一些文本序列被填充了没有意义的特殊标记。要获得仅将有意义的令牌作为值的注意力集中，我们可以指定一个有效的序列长度(以令牌数表示)，以便在计算Softmax时过滤剔除超出此指定范围的那些。这样，我们就可以在下面的`masked_softmax`函数中实现这样的*屏蔽软最大操作*，其中任何超出有效长度的值都被屏蔽为零。

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

为了演示该函数是如何工作的，考虑一小批两个$2 \times 4$个矩阵示例，其中这两个示例的有效长度分别为2和3。作为掩码Softmax操作的结果，超出有效长度的值都被掩码为零。

```{.python .input}
masked_softmax(np.random.uniform(size=(2, 2, 4)), d2l.tensor([2, 3]))
```

```{.python .input}
#@tab pytorch
masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
```

类似地，我们还可以使用二维张量来指定每个矩阵示例中每行的有效长度。

```{.python .input}
masked_softmax(np.random.uniform(size=(2, 2, 4)),
               d2l.tensor([[1, 3], [2, 4]]))
```

```{.python .input}
#@tab pytorch
masked_softmax(torch.rand(2, 2, 4), d2l.tensor([[1, 3], [2, 4]]))
```

## 加性注意
:label:`subsec_additive-attention`

一般来说，当查询和关键字是不同长度的向量时，我们可以使用附加关注度作为评分函数。在给定查询$\mathbf{q} \in \mathbb{R}^q$和关键字$\mathbf{k} \in \mathbb{R}^k$的情况下，

$$a(\mathbf q, \mathbf k) = \mathbf w_v^\top \text{tanh}(\mathbf W_q\mathbf q + \mathbf W_k \mathbf k) \in \mathbb{R},$$
:eqlabel:`eq_additive-attn`

其中可学习参数$\mathbf W_q\in\mathbb R^{h\times q}$、$\mathbf W_k\in\mathbb R^{h\times k}$和$\mathbf w_v\in\mathbb R^{h}$。相当于:eqref:`eq_additive-attn`，查询和关键字被连接并馈送到具有单个隐藏层的mlp中，该隐藏层的隐藏单元数是$h$，这是一个超参数。通过使用$\tanh$作为激活函数并禁用偏向项，我们在下面实现了加性注意。

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

让我们用一个玩具示例来演示上面的`AdditiveAttention`类，其中查询、键和值的形状(批次大小、令牌中的步数或序列长度、特征大小)分别是($2$、$1$、$20$)、($2$、$10$、$2$)和($2$、$10$、$4$)。注意力集中输出的形状为(批次大小、查询步骤数、值的特征大小)。

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

虽然附加注意力包含可学习的参数，但由于在此示例中每个键都相同，因此注意力权重是统一的，由指定的有效长度确定。

```{.python .input}
#@tab all
d2l.show_heatmaps(d2l.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## 按比例调整网点-产品关注度

计算效率更高的计分函数设计可以是简单的点积。但是，点积操作要求查询和键都具有相同的向量长度，比如$d$。假设查询和键的所有元素都是均值和单位方差为零的独立随机变量。这两个向量的点积的均值和方差均为$d$。为了确保无论向量长度如何，点积的方差仍然为1，*缩放的点积关注度*评分函数

$$a(\mathbf q, \mathbf k) = \mathbf{q}^\top \mathbf{k}  /\sqrt{d}$$

将点积除以$\sqrt{d}$。在实践中，我们通常以小批的形式考虑效率，比如计算$n$个查询和$m$个键值对的关注度，其中查询和键的长度为$d$，值的长度为$v$。查询$\mathbf Q\in\mathbb R^{n\times d}$、键$\mathbf K\in\mathbb R^{m\times d}$和值$\mathbf V\in\mathbb R^{m\times v}$的按比例的点积关注度是

$$ \mathrm{softmax}\left(\frac{\mathbf Q \mathbf K^\top }{\sqrt{d}}\right) \mathbf V \in \mathbb{R}^{n\times v}.$$
:eqlabel:`eq_softmax_QK_V`

在下面的缩放点积注意的实现中，我们使用丢弃来进行模型正则化。

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

为了演示上面的`DotProductAttention`个类，我们使用与前面玩具示例相同的键、值和有效长度来增加注意力。对于点积操作，我们使查询的特征大小与键的特征大小相同。

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

与加性注意力演示中相同，由于`keys`包含不能通过任何查询区分的相同元素，因此获得了均匀的关注度权重。

```{.python .input}
#@tab all
d2l.show_heatmaps(d2l.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## 摘要

* 我们可以将注意力池的输出计算为值的加权平均值，其中不同的注意力评分函数的选择会导致不同的注意力池行为。
* 当查询和关键字是不同长度的向量时，我们可以使用加性注意力评分函数。当它们相同时，缩放的点积注意力评分函数的计算效率更高。

## 练习

1. 修改玩具示例中的关键点并可视化注意力权重。相加注意力和缩放的点积注意力仍然输出相同的注意力权重吗？为什么或者为什么不？
1. 仅使用矩阵乘法，是否可以为不同向量长度的查询和键设计新的评分函数？
1. 当查询和键具有相同的向量长度时，对于评分函数，向量求和比点积设计更好吗？为什么或者为什么不？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/346)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1064)
:end_tab:
