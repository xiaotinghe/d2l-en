# 递归神经网络的从头开始实现
:label:`sec_rnn_scratch`

在本节中，我们将根据:numref:`sec_rnn`中的描述，从头开始为字符级语言模型实现RNN。这样的模型将在H.G.威尔斯的《时间机器》上训练。和前面一样，我们首先读取数据集，它在:numref:`sec_language_model`中介绍。

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

## 一个热编码

回想一下，在`train_iter`中，每个令牌都表示为一个数字索引。将这些指数直接输入神经网络可能会使学习变得困难。我们通常将每个标记表示为更具表现力的特征向量。最简单的表示称为“一个热编码”，它是在:numref:`subsec_classification-problem`中引入的。

简言之，我们将每个索引映射到一个不同的单位向量：假设词汇表中不同的令牌数为$N$（`len(vocab)`），令牌索引的范围为0到$N-1$。如果令牌的索引是整数$i$，那么我们创建一个长度为$N$的0向量，并将$i$处的元素设置为1。此向量是原始令牌的一个热向量。索引为0和2的一个热向量如下所示。

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

我们每次采样的小批量的形状是（批量大小、时间步数）。`one_hot`函数将这样一个小批量转换成三维张量，最后一个维度等于词汇表大小（`len(vocab)`）。我们经常变换输入，以便获得形状（时间步数、批量大小、词汇表大小）的输出。这将使我们能够更方便地通过最外层的维度，一步一步地更新minibatch的隐藏状态。

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

接下来，我们初始化RNN模型的模型参数。隐藏单元数`num_hiddens`是一个可调的超参数。当训练语言模型时，输入和输出来自相同的词汇表。因此，它们具有相同的维度，即等于词汇的大小。

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

为了定义RNN模型，我们首先需要一个`init_rnn_state`函数在初始化时返回隐藏状态。它返回一个张量，用0填充，形状为（批量大小，隐藏单元数）。使用元组可以更容易地处理隐藏状态包含多个变量的情况，我们将在后面的部分中遇到这些情况。

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

下面的`rnn`函数定义了如何在一个时间步计算隐藏状态和输出。请注意，RNN模型通过最外层维度`inputs`循环，以便逐时间步更新迷你批处理的隐藏状态`H`。此外，这里的激活函数使用$\tanh$函数。如:numref:`sec_mlp`所述，当元素均匀分布在实数上时，$\tanh$函数的平均值为0。

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

定义了所有需要的函数之后，接下来我们创建一个类来包装这些函数，并存储从头开始实现的RNN模型的参数。

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

让我们检查输出是否具有正确的形状，例如，以确保隐藏状态的维数保持不变。

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

我们可以看到输出形状是（时间步数$\times$批量大小，词汇表大小），而隐藏状态形状保持不变，即（批量大小，隐藏单元数）。

## 预测

让我们首先定义预测函数来生成用户提供的`prefix`之后的新字符，`prefix`是一个包含多个字符的字符串。在`prefix`中循环遍历这些开始字符时，我们不断地将隐藏状态传递到下一个时间步，而不生成任何输出。这被称为“预热”期，在此期间模型会自我更新（例如，更新隐藏状态），但不会进行预测。预热期过后，隐藏状态通常比开始时的初始值好。所以我们生成预测的角色并发射它们。

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

现在我们可以测试`predict_ch8`函数。我们将前缀指定为`time traveller `，并让它生成10个附加字符。鉴于我们没有训练网络，它会产生荒谬的预测。

```{.python .input}
#@tab mxnet,pytorch
predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
predict_ch8('time traveller ', 10, net, vocab, params)
```

## 渐变剪裁

对于长度为$T$的序列，我们在迭代中计算这些$T$时间步上的梯度，从而在反向传播过程中产生长度为$\mathcal{O}(T)$的矩阵积链。如:numref:`sec_numerical_stability`所述，当$T$较大时，它可能导致数值不稳定，例如梯度可能爆炸或消失。因此，RNN模型往往需要额外的帮助来稳定训练。

一般来说，在解决优化问题时，我们对模型参数采取更新步骤，例如在向量形式$\mathbf{x}$中，在小批量上的负梯度$\mathbf{g}$的方向上。例如，使用$\eta > 0$作为学习率，在一次迭代中，我们将$\mathbf{x}$更新为$\mathbf{x} - \eta \mathbf{g}$。让我们进一步假设目标函数$f$表现良好，例如，*Lipschitz连续*常数$L$。也就是说，对于任何$\mathbf{x}$和$\mathbf{y}$我们有

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

在这种情况下，我们可以安全地假设，如果我们将参数向量更新$\eta \mathbf{g}$，那么

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|,$$

这意味着我们不会观察到超过$L \eta \|\mathbf{g}\|$的变化。这既是诅咒也是祝福。在诅咒的一面，它限制了进步的速度；而在祝福的一面，它限制了如果我们朝着错误的方向前进，事情会出错的程度。

有时梯度可能很大，优化算法可能无法收敛。我们可以通过降低$\eta$的学习率来解决这个问题。但是如果我们很少得到大的梯度呢？在这种情况下，这种做法似乎完全没有根据。一个流行的替代方法是通过将渐变$\mathbf{g}$投影回给定半径的球（例如$\theta$）来剪裁渐变$\mathbf{g}$

$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$

通过这样做，我们知道梯度范数永远不会超过$\theta$，并且更新后的梯度完全与$\mathbf{g}$的原始方向对齐。它还有一个理想的副作用，即限制任何给定的小批量（以及其中任何给定的样本）对参数向量的影响。这赋予了模型一定程度的健壮性。渐变剪裁提供了一个快速修复渐变爆炸。虽然它并不能完全解决问题，但它是众多缓解问题的技术之一。

下面我们定义一个函数来剪裁从头开始实现的模型或由高级api构建的模型的渐变。还要注意，我们计算了所有模型参数的梯度范数。

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

在训练模型之前，让我们定义一个函数来训练一个历元中的模型。它与我们培训:numref:`sec_softmax_scratch`车型的方式有三个不同之处：

1. 顺序数据的不同采样方法（随机采样和顺序分区）将导致隐藏状态初始化的差异。
1. 我们在更新模型参数之前剪辑梯度。这确保了即使在训练过程中的某个点上梯度爆炸，模型也不会发散。
1. 我们用困惑来评价模型。如:numref:`subsec_perplexity`所述，这确保了不同长度的序列具有可比性。

具体地说，当使用顺序分区时，我们只在每个历元的开始处初始化隐藏状态。由于下一个minibatch中的$i^\mathrm{th}$子序列示例与当前$i^\mathrm{th}$子序列示例相邻，因此当前minibatch末尾的隐藏状态将用于初始化下一个minibatch开头的隐藏状态。这样，存储在隐藏状态中的序列的历史信息可以在一个历元内流过相邻的子序列。然而，任何一点隐态的计算都依赖于同一历元中所有的前一个小批量，这使得梯度计算变得复杂。为了降低计算量，我们在处理任何一个小批量之前先分离梯度，使得隐状态的梯度计算总是限制在一个小批量的时间步长内。

当使用随机抽样时，我们需要为每个迭代重新初始化隐藏状态，因为每个例子都是用一个随机位置抽样的。与:numref:`sec_softmax_scratch`中的`train_epoch_ch3`函数相同，`updater`是更新模型参数的常规函数。它既可以是从头开始实现的`d2l.sgd`函数，也可以是深度学习框架中的内置优化函数。

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

training函数支持从头开始或使用高级api实现的RNN模型。

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

现在我们可以训练RNN模型。因为我们在数据集中只使用10000个标记，所以模型需要更多的历元来更好地收敛。

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

最后，让我们检查一下使用随机抽样方法的结果。

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

虽然从头开始实现上述RNN模型是有指导意义的，但并不方便。在下一节中，我们将看到如何改进RNN模型，例如如何使其更易于实现并运行得更快。

## 摘要

* 我们可以训练一个基于RNN的字符级语言模型，根据用户提供的文本前缀生成文本。
* 一个简单的RNN语言模型包括输入编码、RNN建模和输出生成。
* RNN模型需要状态初始化来训练，尽管随机抽样和顺序划分使用不同的方法。
* 当使用顺序分割时，我们需要分离梯度以减少计算量。
* 预热期允许模型在进行任何预测之前进行自我更新（例如，获得比初始值更好的隐藏状态）。
* 渐变剪裁可以防止渐变爆炸，但不能修复消失的渐变。

## 练习

1. 显示一个热编码相当于为每个对象选择不同的嵌入。
1. 通过调整超参数（如历元数、隐单元数、小批量时间步数、学习率等）来改善复杂度。
    * 你能打多低？
    * 用可学习的嵌入替换一个热编码。这会带来更好的表现吗？
    * 它在H.G.威尔斯的其他书上的效果如何，例如[*The War of the Worlds*](http://www.gutenberg.org/ebooks/36)？
1. 修改预测函数，例如使用采样，而不是选择最有可能的下一个字符。
    * 会发生什么？
    * 将模型偏向更可能的输出，例如，从$q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$中抽取$\alpha > 1$。
1. 运行本节中的代码而不剪裁渐变。会发生什么？
1. 更改顺序分区，使其不会从计算图中分离隐藏状态。运行时间有变化吗？困惑呢？
1. 用ReLU替换本节中使用的激活函数，并重复本节中的实验。我们还需要渐变剪辑吗？为什么？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/486)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1052)
:end_tab:
