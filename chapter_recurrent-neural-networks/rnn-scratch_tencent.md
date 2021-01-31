# 递归神经网络的从头开始实现
:label:`sec_rnn_scratch`

在本节中，根据我们在:numref:`sec_rnn`中的描述，我们将从头开始为字符级语言模型实现一个RNN。这样的模型将在H.G.威尔斯的“时光机”中进行训练。与前面一样，我们首先阅读:numref:`sec_language_model`中引入的数据集。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab tensorflow
train_random_iter, vocab_random_iter = d2l.load_data_time_machine(
    batch_size, num_steps, use_random_iter=True)
```

## One-热编码

回想一下，在`train_iter`中，每个令牌都表示为数字索引。将这些指标直接输入神经网络可能会使学习变得困难。我们通常将每个标记表示为更具表现力的特征向量。最简单的表示称为“一次热编码”，这是在:numref:`subsec_classification-problem`中引入的。

简而言之，我们将每个索引映射到一个不同的单位向量：假设词汇表中不同的记号数量为$N$(`len(vocab)`)，记号索引的范围从0到$N-1$。如果令牌的索引是整数$i$，那么我们创建一个长度为$N$的全0向量，并将位置$i$处的元素设置为1。该向量是原始令牌的一个热点向量。索引为0和2的单热点向量如下所示。

```{.python .input}
npx.one_hot(np.array([0, 2]), len(vocab))
```

```{.python .input}
#@tab pytorch
F.one_hot(torch.tensor([0, 2]), len(vocab))
```

```{.python .input}
#@tab tensorflow
tf.one_hot(tf.constant([0, 2]), len(vocab))
```

我们每次抽样的小批次的形状是(批次大小，时间步数)。`one_hot`函数将这样的小批量转换为三维张量，最后一维等于词汇表大小(`len(vocab)`)。我们经常调换输入，以便获得Shape(时间步数、批大小、词汇表大小)的输出。这将允许我们更方便地循环通过用于更新小批量的隐藏状态的最外面的维度，逐个时间地。

```{.python .input}
X = d2l.reshape(d2l.arange(10), (2, 5))
npx.one_hot(X.T, 28).shape
```

```{.python .input}
#@tab pytorch
X = d2l.reshape(d2l.arange(10), (2, 5))
F.one_hot(X.T, 28).shape
```

```{.python .input}
#@tab tensorflow
X = d2l.reshape(d2l.arange(10), (2, 5))
tf.one_hot(tf.transpose(X), 28).shape
```

## 初始化模型参数

接下来，我们初始化RNN模型的模型参数。隐藏单元的数量`num_hiddens`是可调超参数。当训练语言模型时，输入和输出来自相同的词汇。因此，它们具有相同的维度，即等于词汇大小。

```{.python .input}
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = d2l.zeros(num_hiddens, ctx=device)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, ctx=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.python .input}
#@tab pytorch
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = d2l.zeros(num_hiddens, device=device)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

```{.python .input}
#@tab tensorflow
def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size
    
    def normal(shape):
        return d2l.normal(shape=shape,stddev=0.01,mean=0,dtype=tf.float32)

    # Hidden layer parameters
    W_xh = tf.Variable(normal((num_inputs, num_hiddens)), dtype=tf.float32)
    W_hh = tf.Variable(normal((num_hiddens, num_hiddens)), dtype=tf.float32)
    b_h = tf.Variable(d2l.zeros(num_hiddens), dtype=tf.float32)
    # Output layer parameters
    W_hq = tf.Variable(normal((num_hiddens, num_outputs)), dtype=tf.float32)
    b_q = tf.Variable(d2l.zeros(num_outputs), dtype=tf.float32)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return params
```

## RNN模型

要定义rnn模型，我们首先需要一个`init_rnn_state`函数来返回初始化时的隐藏状态。它返回一个以0填充且形状为(批次大小，隐藏单元数)的张量。使用元组可以更容易地处理隐藏状态包含多个变量的情况，我们将在后面的小节中遇到这些情况。

```{.python .input}
def init_rnn_state(batch_size, num_hiddens, device):
    return (d2l.zeros((batch_size, num_hiddens), ctx=device), )
```

```{.python .input}
#@tab pytorch
def init_rnn_state(batch_size, num_hiddens, device):
    return (d2l.zeros((batch_size, num_hiddens), device=device), )
```

```{.python .input}
#@tab tensorflow
def init_rnn_state(batch_size, num_hiddens):
    return (d2l.zeros((batch_size, num_hiddens)), )
```

下面的`rnn`函数定义了如何计算隐藏状态并在时间步长上输出。注意，RNN模型循环通过最外面的维度`inputs`，从而其逐个时间步地更新小批量的隐藏状态`H`。此外，这里的激活功能使用的是$\tanh$功能。如:numref:`sec_mlp`中所述，当元素均匀分布在实数上时，$\tanh$函数的平均值为0。

```{.python .input}
def rnn(inputs, state, params):
    # Shape of `inputs`: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        H = np.tanh(np.dot(X, W_xh) + np.dot(H, W_hh) + b_h)
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)
```

```{.python .input}
#@tab pytorch
def rnn(inputs, state, params):
    # Here `inputs` shape: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

```{.python .input}
#@tab tensorflow
def rnn(inputs, state, params):
    # Here `inputs` shape: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        X = tf.reshape(X,[-1,W_xh.shape[0]])
        H = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(H, W_hh) + b_h)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return d2l.concat(outputs, axis=0), (H,)
```

定义了所有需要的函数后，接下来我们创建一个类来包装这些函数并存储从头开始实现的RNN模型的参数。

```{.python .input}
class RNNModelScratch:  #@save
    """An RNN Model implemented from scratch."""
    def __init__(self, vocab_size, num_hiddens, device, get_params,
                 init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = npx.one_hot(X.T, self.vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, ctx):
        return self.init_state(batch_size, self.num_hiddens, ctx)
```

```{.python .input}
#@tab pytorch
class RNNModelScratch: #@save
    """A RNN Model implemented from scratch."""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
```

```{.python .input}
#@tab tensorflow
class RNNModelScratch: #@save
    """A RNN Model implemented from scratch."""
    def __init__(self, vocab_size, num_hiddens,
                 init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state, params):
        X = tf.one_hot(tf.transpose(X), self.vocab_size)
        X = tf.cast(X, tf.float32)
        return self.forward_fn(X, state, params)

    def begin_state(self, batch_size):
        return self.init_state(batch_size, self.num_hiddens)
```

让我们检查输出是否具有正确的形状，例如，以确保隐藏状态的维度保持不变。

```{.python .input}
#@tab mxnet
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.as_in_context(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
```

```{.python .input}
#@tab pytorch
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
```

```{.python .input}
#@tab tensorflow
# defining tensorflow training strategy
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)

num_hiddens = 512
with strategy.scope():
    net = RNNModelScratch(len(vocab), num_hiddens, init_rnn_state, rnn)
state = net.begin_state(X.shape[0])
params = get_params(len(vocab), num_hiddens)
Y, new_state = net(X, state, params)
Y.shape, len(new_state), new_state[0].shape
```

我们可以看到，输出形状是(时间步数$\times$、批大小、词汇大小)，而隐藏状态形状保持不变，即(批大小、隐藏单元数)。

## 预测

让我们首先定义预测函数以在用户提供的`prefix`之后生成新字符，该新字符是包含几个字符的字符串。在`prefix`中循环这些开始字符时，我们一直将隐藏状态传递到下一个时间步，而不生成任何输出。这称为“预热”阶段，在此期间模型自我更新(例如，更新隐藏状态)，但不进行预测。在预热期之后，隐藏状态通常比其开始时的初始值要好。所以我们生成预测的字符并发出它们。

```{.python .input}
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1, ctx=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(
        d2l.tensor([outputs[-1]], ctx=device), (1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
#@tab pytorch
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor(
        [outputs[-1]], device=device), (1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
#@tab tensorflow
def predict_ch8(prefix, num_preds, net, vocab, params):  #@save
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor([outputs[-1]]), (1, 1)).numpy()
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state, params)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state, params)
        outputs.append(int(y.numpy().argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

现在我们可以测试`predict_ch8`功能了。我们将前缀指定为`time traveller `，并让它生成10个额外的字符。考虑到我们没有训练过这个网络，它将产生无意义的预测。

```{.python .input}
#@tab mxnet,pytorch
predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
predict_ch8('time traveller ', 10, net, vocab, params)
```

## 渐变裁剪

对于长度为$T$的序列，我们在迭代中计算这$T$个时间步长上的梯度，这导致在反向传播期间产生长度为$\mathcal{O}(T)$的矩阵乘积链。如:numref:`sec_numerical_stability`中所述，它可能会导致数值不稳定性，例如，当$T$较大时，梯度可能会爆炸或消失。因此，RNN模型经常需要额外的帮助来稳定训练。

一般而言，当解决优化问题时，我们对模型参数采取更新步骤，例如在矢量形式$\mathbf{x}$中，在小批量上的负梯度$\mathbf{g}$的方向上。例如，使用$\eta > 0$作为学习率，我们在一次迭代中将$\mathbf{x}$更新为$\mathbf{x} - \eta \mathbf{g}$。让我们进一步假设目标函数$f$表现良好，比方说，*李普希茨连续*，常数$L$。也就是说，对于我们拥有的任何$\mathbf{x}$和$\mathbf{y}$

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

在这种情况下，我们可以安全地假设，如果我们将参数向量更新$\eta \mathbf{g}$，那么

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|,$$

这意味着我们观察到的变化不会超过$L \eta \|\mathbf{g}\|$。这既是一种诅咒，也是一种祝福。在诅咒的一面，它限制了进步的速度；而在祝福的一面，它限制了如果我们朝着错误的方向前进，事情可能出错的程度。

有时梯度可能非常大，优化算法可能无法收敛。我们可以通过降低$\eta$的学习率来解决这个问题。但是，如果我们只“很少”获得大的渐变会怎样呢？在这种情况下，这种方法似乎完全没有根据。一种流行的替代方案是通过将梯度$\mathbf{g}$投射回给定半径的球(例如$\theta$通过)来对其进行裁剪

$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$

通过这样做，我们知道梯度范数永远不会超过$\theta$，并且更新后的梯度与原始方向$\mathbf{g}$完全对齐。它还具有一个理想的副作用，即限制任何给定的小批次(以及其中的任何给定样本)对参数向量施加的影响。这赋予了模型一定程度的健壮性。渐变裁剪提供了渐变分解的快速修复。虽然它不能完全解决问题，但它是缓解问题的众多技术之一。

下面，我们定义一个函数来裁剪从头开始实现的模型或由高级API构建的模型的渐变。还要注意，我们计算所有模型参数的梯度范数。

```{.python .input}
def grad_clipping(net, theta):  #@save
    """Clip the gradient."""
    if isinstance(net, gluon.Block):
        params = [p.data() for p in net.collect_params().values()]
    else:
        params = net.params
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

```{.python .input}
#@tab pytorch
def grad_clipping(net, theta):  #@save
    """Clip the gradient."""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

```{.python .input}
#@tab tensorflow
def grad_clipping(grads, theta): #@save
    """Clip the gradient."""
    theta = tf.constant(theta, dtype=tf.float32)
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)).numpy()
                        for grad in grads))
    norm = tf.cast(norm, tf.float32)
    new_grad = []
    if tf.greater(norm, theta):
        for grad in grads:
            new_grad.append(grad * theta / norm)
    else:
        for grad in grads:
            new_grad.append(grad)
    return new_grad
```

## 培训

在训练模型之前，让我们定义一个函数来在一个时期内训练模型。它与我们如何在三个地方训练:numref:`sec_softmax_scratch`的模型是不同的：

1. 顺序数据的不同采样方法(随机采样和顺序分区)会导致隐藏状态的初始化不同。
1. 我们在更新模型参数之前裁剪梯度。这确保了即使在训练过程中的某个点梯度爆炸时，模型也不会发散。
1. 我们使用迷惑度对模型进行评价。如:numref:`subsec_perplexity`中所讨论的，这确保了不同长度的序列具有可比性。

具体地说，当使用顺序分区时，我们仅在每个时段开始时初始化隐藏状态。由于下一小批次中的$i^\mathrm{th}$子序列示例与当前$i^\mathrm{th}$子序列示例相邻，因此当前小批次末尾的隐藏状态将用于初始化下一小批次开始时的隐藏状态。以这种方式，存储在隐藏状态中的序列的历史信息可以流经一个历元内的相邻子序列。然而，任意点隐态的计算依赖于同一历元中所有先前的小批量，这使得梯度计算变得复杂。为了减少计算量，我们在处理任何小批量之前分离梯度，使得隐藏状态的梯度计算始终局限于一个小批量中的时间步长。

当使用随机采样时，我们需要重新初始化每个迭代的隐藏状态，因为每个示例都是用随机位置采样的。与:numref:`sec_softmax_scratch`中的`train_epoch_ch3`函数相同，`updater`是更新模型参数的通用函数。它可以是从头开始实现的`d2l.sgd`函数，也可以是深度学习框架中的内置优化函数。

```{.python .input}
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train a model within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], ctx=device)
        else:
            for s in state:
                s.detach()
        y = Y.T.reshape(-1)
        X, y = X.as_in_ctx(device), y.as_in_ctx(device)
        with autograd.record():
            y_hat, state = net(X, state)
            l = loss(y_hat, y).mean()
        l.backward()
        grad_clipping(net, 1)
        updater(batch_size=1)  # Since the `mean` function has been invoked
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

```{.python .input}
#@tab pytorch
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train a net within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()
            else:
                # `state` is a tuple of tensors for `nn.LSTM` and
                # for our custom scratch implementation 
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

```{.python .input}
#@tab tensorflow
#@save
def train_epoch_ch8(net, train_iter, loss, updater, params, use_random_iter):
    """Train a model within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0])
        with tf.GradientTape(persistent=True) as g:
            g.watch(params)
            y_hat, state= net(X, state, params)
            y = d2l.reshape(tf.transpose(Y), (-1))
            l = loss(y, y_hat)
        grads = g.gradient(l, params)
        grads = grad_clipping(grads, 1)
        updater.apply_gradients(zip(grads, params))
        
        # Keras loss by default returns the average loss in a batch
        # l_sum = l * float(d2l.size(y)) if isinstance(
        #     loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

训练功能支持从头开始或使用高级API实现的RNN模型。

```{.python .input}
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,  #@save
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, gluon.Block):
        net.initialize(ctx=device, force_reinit=True,
                         init=init.Normal(0.01))
        trainer = gluon.Trainer(net.collect_params(),
                                'sgd', {'learning_rate': lr})
        updater = lambda batch_size: trainer.step(batch_size)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

```{.python .input}
#@tab pytorch
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch8(net, train_iter, vocab, num_hiddens, lr, num_epochs, strategy,
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    with strategy.scope():
        params = get_params(len(vocab), num_hiddens)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        updater = tf.keras.optimizers.SGD(lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, params)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
             net, train_iter, loss, updater, params, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    device = d2l.try_gpu()._device_name
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

现在我们可以训练RNN模型了。由于我们在数据集中只使用了10000个标记，因此模型需要更多的历元才能更好地收敛。

```{.python .input}
#@tab mxnet,pytorch
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, num_hiddens, lr, num_epochs, strategy)
```

最后，让我们检查一下使用随机抽样法的结果。

```{.python .input}
#@tab mxnet,pytorch
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)
```

```{.python .input}
#@tab tensorflow
params = get_params(len(vocab_random_iter), num_hiddens)
train_ch8(net, train_random_iter, vocab_random_iter, num_hiddens, lr,
          num_epochs, strategy, use_random_iter=True)
```

虽然从头开始实现上述RNN模型很有指导意义，但并不方便。在下一节中，我们将了解如何改进RNN模型，例如如何使其更容易实现并使其运行得更快。

## 摘要

* 我们可以训练基于RNN的字符级语言模型来生成跟随用户提供的文本前缀的文本。
* 简单的RNN语言模型包括输入编码、RNN建模和输出生成。
* RNN模型需要状态初始化进行训练，尽管随机采样和顺序划分使用不同的方式。
* 在使用顺序划分时，需要分离梯度以降低计算量。
* 预热期允许模型在做出任何预测之前自我更新(例如，获得比其初始化值更好的隐藏状态)。
* 渐变剪裁可防止渐变爆炸，但不能修复渐变消失。

## 练习

1. 说明一次性编码等同于为每个对象选择不同的嵌入。
1. 调整超参数(例如，历元数、隐藏单元数、小批量中的时间步数和学习率)以改善困惑。
    * 你能降到多低？
    * 用可学习的嵌入替换一次性编码。这会带来更好的性能吗？
    * 它在H.G.Wells的其他书(例如，[*The War of the Worlds*](http://www.gutenberg.org/ebooks/36))上会有多好的效果？
1. 修改预测函数，例如使用采样，而不是选择最可能的下一个字符。
    * 会发生什么事？
    * 使模型偏向更可能的输出，例如，从$q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$到$\alpha > 1$抽样。
1. 在不裁剪渐变的情况下运行本节中的代码。会发生什么事？
1. 更改顺序分区，使其不会将隐藏状态与计算图分开。运行时间有变化吗？令人困惑的是什么呢？
1. 用RELU替换本节中使用的激活函数，并重复本节中的实验。我们还需要渐变裁剪吗？为什么？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/486)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1052)
:end_tab:
