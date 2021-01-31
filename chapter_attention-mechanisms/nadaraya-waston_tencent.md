# 注意力集中：Nadaraya-Watson核回归
:label:`sec_nadaraya-waston`

现在您了解了:numref:`fig_qkv`框架下注意力机制的主要组成部分。简而言之，查询(意志线索)和按键(非意志线索)之间的交互会导致“注意力集中”。注意力集中选择性地聚合值(感觉输入)以产生输出。在本节中，我们将更详细地描述注意力集中，让您从更高的层面了解注意力机制在实践中是如何工作的。具体地说，1964年提出的Nadaraya-Watson核回归模型是一个简单但完整的例子，用于演示具有注意机制的机器学习。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## 生成数据集

为简单起见，让我们考虑以下回归问题：给定输入-输出对$\{(x_1, y_1), \ldots, (x_n, y_n)\}$的数据集，如何学习$f$以预测任何新输入$\hat{y} = f(x)$的输出$x$？

这里，我们根据具有噪声项$\epsilon$的以下非线性函数生成人工数据集：

$$y_i = 2\sin(x_i) + x_i^{0.8} + \epsilon,$$

其中$\epsilon$服从具有零均值和标准偏差0.5的正态分布。生成了50个训练用例和50个测试用例。为了更好地可视化稍后的注意力模式，对训练输入进行了排序。

```{.python .input}
n_train = 50  # No. of training examples
x_train = np.sort(d2l.rand(n_train) * 5)   # Training inputs
```

```{.python .input}
#@tab pytorch
n_train = 50  # No. of training examples
x_train, _ = torch.sort(d2l.rand(n_train) * 5)   # Training inputs
```

```{.python .input}
#@tab all
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal(0.0, 0.5, (n_train,))  # Training outputs
x_test = d2l.arange(0, 5, 0.1)  # Testing examples
y_truth = f(x_test)  # Ground-truth outputs for the testing examples
n_test = len(x_test)  # No. of testing examples
n_test
```

下面的函数绘制所有训练示例(由圆圈表示)、没有噪声项的地面真实数据生成函数`f`(标记为“真”)和学习的预测函数(标记为“PRED”)。

```{.python .input}
#@tab all
def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);
```

## 平均共用(Average Pooling)

我们从这个回归问题可能是世界上“最愚蠢的”估计器开始：使用平均汇集对所有训练输出进行平均：

$$f(x) = \frac{1}{n}\sum_{i=1}^n y_i,$$
:eqlabel:`eq_avg-pooling`

如下图所示。我们可以看到，这个估计器确实不是那么聪明。

```{.python .input}
y_hat = y_train.mean().repeat(n_test)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)
```

## 非参数注意汇集

显然，平均合并省略了$x_i$的投入。Nadaraya :cite:`Nadaraya.1964`和Waston :cite:`Watson.1964`提出了一个更好的想法，根据它们的输入位置对输出$y_i$进行加权：

$$f(x) = \sum_{i=1}^n \frac{K(x - x_i)}{\sum_{j=1}^n K(x - x_j)} y_i,$$
:eqlabel:`eq_nadaraya-waston`

其中$K$是*内核*。:eqref:`eq_nadaraya-waston`中的估计量称为*Nadaraya-Watson核回归*。在这里，我们不会深入讨论内核的细节。回想一下:numref:`fig_qkv`年的注意力机制框架。从注意力的角度看，我们可以用更广义的*注意力汇集*的形式重写:eqref:`eq_nadaraya-waston`：

$$f(x) = \sum_{i=1}^n \alpha(x, x_i) y_i,$$
:eqlabel:`eq_attn-pooling`

其中$x$是查询，$(x_i, y_i)$是键-值对。比较:eqref:`eq_attn-pooling`和:eqref:`eq_avg-pooling`，这里的注意力集中是值$y_i$的加权平均值。基于查询$y_i$和由$\alpha$建模的键$x_i$之间的交互，将$\alpha$中的*关注权重*$\alpha$分配给相应值$\alpha$。对于任何查询，其在所有键-值对上的关注度权重都是有效的概率分布：它们是非负的，并且总和为1。

要获得注意力集中的直觉，只需考虑如下定义的*高斯核*

$$
K(u) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{u^2}{2}).
$$

将高斯内核插入到:eqref:`eq_attn-pooling`和:eqref:`eq_nadaraya-waston`中可以得到

$$\begin{aligned} f(x) &=\sum_{i=1}^n \alpha(x, x_i) y_i\\ &= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}(x - x_i)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}(x - x_j)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}(x - x_i)^2\right) y_i. \end{aligned}$$
:eqlabel:`eq_nadaraya-waston-gaussian`

在:eqref:`eq_nadaraya-waston-gaussian`中，将获得更接近给定查询$x_i$的关键字$x$
*通过分配给键的相应值$y_i$的“更大的关注度”来获得更多的关注。

值得注意的是，纳达拉亚-沃森核回归是一个非参数模型；因此，:eqref:`eq_nadaraya-waston-gaussian`是*非参数注意力集中*的一个例子。在下文中，我们根据这个非参数注意力模型绘制预测图。预测的线条比平均合并产生的线条更平滑，更接近实际情况。

```{.python .input}
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the
# same testing inputs (i.e., same queries)
X_repeat = d2l.reshape(x_test.repeat(n_train), (-1, n_train))
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = npx.softmax(-(X_repeat - x_train)**2 / 2)
# Each element of `y_hat` is weighted average of values, where weights are
# attention weights
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the
# same testing inputs (i.e., same queries)
X_repeat = d2l.reshape(x_test.repeat_interleave(n_train), (-1, n_train))
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# Each element of `y_hat` is weighted average of values, where weights are
# attention weights
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

现在我们来看一下关注度权重。这里，测试输入是查询，而训练输入是关键。由于两个输入都已排序，我们可以看到查询-键对越接近，注意力池中的注意力权重就越高。

```{.python .input}
d2l.show_heatmaps(np.expand_dims(np.expand_dims(attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab pytorch
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

## 参数注意池

非参数Nadaraya-Watson核回归享有*一致性*好处：给定足够的数据，该模型收敛到最优解。尽管如此，我们可以很容易地将可学习的参数集成到注意力集中中。

作为示例，与:eqref:`eq_nadaraya-waston-gaussian`略有不同，以下查询$x$和关键字$x_i$之间的距离乘以可学习参数$w$：

$$\begin{aligned}f(x) &= \sum_{i=1}^n \alpha(x, x_i) y_i \\&= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}((x - x_i)w)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}((x - x_i)w)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}((x - x_i)w)^2\right) y_i.\end{aligned}$$
:eqlabel:`eq_nadaraya-waston-gaussian-para`

在睡觉部分，我们将通过学习:eqref:`eq_nadaraya-waston-gaussian-para`的注意力池参数来训练这个模型。

### 批量矩阵乘法
:label:`subsec_batch_dot`

为了更有效地计算小批量的关注度，我们可以利用深度学习框架提供的批处理矩阵乘法实用程序。

假设第一小批包含$n$个形状为$a\times b$的矩阵$\mathbf{X}_1, \ldots, \mathbf{X}_n$，第二小批包含$n$个形状为$b\times c$的矩阵$\mathbf{Y}_1, \ldots, \mathbf{Y}_n$。它们的批量矩阵相乘得到$n$个矩阵$\mathbf{X}_1\mathbf{Y}_1, \ldots, \mathbf{X}_n\mathbf{Y}_n$的形状为$a\times c$。因此，给定形状($n$、$a$、$b$)和($n$、$b$、$c$)的两个张量，它们的批矩阵乘法输出的形状为($n$、$a$、$c$)。

```{.python .input}
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
npx.batch_dot(X, Y).shape
```

```{.python .input}
#@tab pytorch
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
torch.bmm(X, Y).shape
```

在注意机制的背景下，我们可以使用小批量矩阵乘法来计算小批量中值的加权平均值。

```{.python .input}
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20), (2, 10))
npx.batch_dot(np.expand_dims(weights, 1), np.expand_dims(values, -1))
```

```{.python .input}
#@tab pytorch
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20.0), (2, 10))
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
```

### 定义模型

使用小批量矩阵乘法，下面我们定义基于:eqref:`eq_nadaraya-waston-gaussian-para`中的参数注意集中的参数版本的纳达拉亚-沃森核回归。

```{.python .input}
class NWKernelRegression(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = self.params.get('w', shape=(1,))

    def forward(self, queries, keys, values):
        # Shape of the output `queries` and `attention_weights`:
        # (no. of queries, no. of key-value pairs)
        queries = d2l.reshape(
            queries.repeat(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = npx.softmax(
            -((queries - keys) * self.w.data())**2 / 2)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return npx.batch_dot(np.expand_dims(self.attention_weights, 1),
                             np.expand_dims(values, -1)).reshape(-1)
```

```{.python .input}
#@tab pytorch
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # Shape of the output `queries` and `attention_weights`:
        # (no. of queries, no. of key-value pairs)
        queries = d2l.reshape(
            queries.repeat_interleave(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
```

### 培训

在下文中，我们将训练数据集转换为关键字和值，以训练注意力模型。在参数注意池中，任何训练输入都从除其自身之外的所有训练样本中获取键值对来预测其输出。

```{.python .input}
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = np.tile(x_train, (n_train, 1))
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = np.tile(y_train, (n_train, 1))
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).astype('bool')],
                   (n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).astype('bool')],
                     (n_train, -1))
```

```{.python .input}
#@tab pytorch
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = x_train.repeat((n_train, 1))
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = y_train.repeat((n_train, 1))
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                   (n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                     (n_train, -1))
```

利用平方损失和随机梯度下降对参数注意模型进行训练。

```{.python .input}
net = NWKernelRegression()
net.initialize()
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    with autograd.record():
        l = loss(net(x_train, keys, values), y_train)
    l.backward()
    trainer.step(1)
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

```{.python .input}
#@tab pytorch
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    # Note: L2 Loss = 1/2 * MSE Loss. PyTorch has MSE Loss which is slightly
    # different from MXNet's L2Loss by a factor of 2. Hence we halve the loss
    l = loss(net(x_train, keys, values), y_train) / 2
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

在对参数注意模型进行训练后，我们可以绘制出它的预测值。尝试用噪声拟合训练数据集时，预测线比之前绘制的非参数对应线更不平滑。

```{.python .input}
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = np.tile(x_train, (n_test, 1))
# Shape of `value`: (`n_test`, `n_train`)
values = np.tile(y_train, (n_test, 1))
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = x_train.repeat((n_test, 1))
# Shape of `value`: (`n_test`, `n_train`)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
```

与非参数注意池相比，在可学习和参数设置下，注意权重较大的区域变得更加清晰。

```{.python .input}
d2l.show_heatmaps(np.expand_dims(np.expand_dims(net.attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab pytorch
d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

## 摘要

* Nadaraya-Watson核回归是具有注意机制的机器学习的一个例子。
* Nadaraya-Watson核回归的注意力集中是训练输出的加权平均。从注意力的角度来看，基于查询的函数和与该值配对的关键字将注意力权重分配给一个值。
* 注意力集中可以是非参数的，也可以是参数的。

## 练习

1. 增加训练实例的数量。你能更好地学习非参数Nadaraya-Watson核回归吗？
1. 在参数注意集中实验中，我们所学到的$w$有什么价值？为什么在可视化注意力权重时，它会使加权区域变得更清晰？
1. 我们如何在非参数Nadaraya-Watson核回归中加入超参数来更好地预测呢？
1. 为这一部分的核心回归设计另一个参数注意池。训练这个新模型，并将其注意力权重可视化。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1598)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1599)
:end_tab:
