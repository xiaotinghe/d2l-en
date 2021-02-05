# 小批量随机梯度下降
:label:`sec_minibatch_sgd`

到目前为止，我们在基于梯度的学习方法中遇到了两个极端: :numref:`sec_gd`使用完整的数据集来计算梯度和更新参数，一次一个过程。相反，:numref:`sec_sgd`一次处理一个观察结果以取得进展。他们中的每一个都有自己的缺点。当数据非常相似时，梯度下降并不是特别“数据高效”。随机梯度下降并不是特别的“计算效率”，因为CPU和GPU不能充分利用矢量化的能力。这表明可能存在一个令人满意的中介，事实上，这就是我们到目前为止在我们讨论的示例中一直在使用的。

## 矢量化和缓存

决定使用小批量的核心是计算效率。在考虑并行到多个GPU和多个服务器时，这一点最容易理解。在这种情况下，我们需要向每个GPU发送至少一个图像。每台服务器有8个GPU和16个服务器，我们已经达到了128个的小批量大小。

当涉及到单GPU甚至CPU时，情况会稍微微妙一些。这些设备具有多种类型的存储器，通常是多种类型的计算单元，并且它们之间存在不同的带宽约束。例如，CPU具有少量的寄存器，然后是L1、L2，在某些情况下甚至L3高速缓存(在不同的处理器核心之间共享)。这些缓存的大小和延迟在增加(同时它们的带宽在减少)。可以说，处理器能够执行比主存储器接口能够提供的操作多得多的操作。

* 带有16核和AVX512型矢量化的2 GHz中央处理器每秒可以处理高达$2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$字节的数据。GPU的能力很容易超过这个数字的100倍。另一方面，中端服务器处理器的带宽可能不会超过100 Gb/s，即不到保持处理器正常工作所需带宽的十分之一。更糟糕的是，并非所有存储器访问都是相等的：首先，存储器接口通常是64位宽或更宽(例如，在高达384位的GPU上)，因此读取单个字节会产生更宽访问的成本。
* 第一次访问的开销很大，而顺序访问的开销相对较低(这通常称为突发读取)。要记住的事情还有很多，比如当我们有多个套接字、小芯片和其他结构时进行缓存。对此的详细讨论超出了本节的范围。例如，有关更深入的讨论，请参见本[Wikipedia article](https://en.wikipedia.org/wiki/Cache_hierarchy)。

缓解这些限制的方法是使用CPU缓存的层次结构，这些缓存实际上足够快，可以向处理器提供数据。这就是深度学习中批量学习背后的*推动力。为简单起见，考虑矩阵-矩阵乘法，比方说$\mathbf{A} = \mathbf{B}\mathbf{C}$。我们有许多计算$\mathbf{A}$的选项。例如，我们可以尝试以下操作：

1. 我们可以计算$\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}^\top$，也就是说，我们可以通过点积进行元素计算。
1. 我们可以计算$\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}^\top$，也就是说，我们可以一次计算一列。同样，我们可以计算$\mathbf{A}$，一次计算一行$\mathbf{A}_{i,:}$。
1. 我们可以简单地计算出$\mathbf{A} = \mathbf{B} \mathbf{C}$。
1. 我们可以将$\mathbf{B}$和$\mathbf{C}$分解成更小的挡路矩阵，然后一次一个挡路地计算$\mathbf{A}$。

如果我们遵循第一个选项，则每次要计算元素$\mathbf{A}_{ij}$时，都需要将一行和一列向量复制到cpu中。更糟糕的是，由于矩阵元素是顺序对齐的，因此当我们从存储器中读取两个矢量中的一个矢量时，需要访问它们中的许多不相交的位置。第二种选择要有利得多。在它中，我们能够将列向量$\mathbf{C}_{:,j}$保留在cpu高速缓存中，同时继续遍历$B$。这使得存储器带宽需求减半，访问速度相应加快。当然，方案3是最可取的。不幸的是，大多数矩阵可能不能完全放入缓存中(这毕竟是我们要讨论的问题)。然而，选项4提供了一个实用的替代方案：我们可以将矩阵的块移动到缓存中，并在本地将它们相乘。优化库为我们解决了这一问题。让我们看看这些操作在实践中的效率如何。

除了计算效率之外，Python和深度学习框架本身带来的开销也相当可观。回想一下，每次我们执行命令时，Python解释器都会向MXNet引擎发送一个命令，MXNet引擎需要将该命令插入到计算图中，并在调度期间处理该命令。这样的开销可能是非常有害的。简而言之，强烈建议尽可能使用向量化(和矩阵)。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

timer = d2l.Timer()
A = np.zeros((256, 256))
B = np.random.normal(0, 1, (256, 256))
C = np.random.normal(0, 1, (256, 256))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import numpy as np

timer = d2l.Timer()
A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np

timer = d2l.Timer()
A = tf.Variable(d2l.zeros((256, 256)))
B = tf.Variable(d2l.normal([256, 256], 0, 1))
C = tf.Variable(d2l.normal([256, 256], 0, 1))
```

按元素赋值只需分别迭代$\mathbf{B}$和$\mathbf{C}$的所有行和列，即可将值赋给$\mathbf{A}$。

```{.python .input}
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B[i, :], C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j].assign(tf.tensordot(B[i, :], C[:, j], axes=1))
timer.stop()
```

更快的策略是执行按列分配。

```{.python .input}
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(256):
    A[:, j].assign(tf.tensordot(B, C[:, j], axes=1))
timer.stop()
```

最后，最有效的方式是在一个挡路中完成整个操作。让我们看看各自的操作速度是多少。

```{.python .input}
# Compute A = BC in one go
timer.start()
A = np.dot(B, C)
A.wait_to_read()
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab pytorch
# Compute A = BC in one go
timer.start()
A = torch.mm(B, C)
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
A.assign(tf.tensordot(B, C, axes=1))
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

## 迷你批次

:label:`sec_minibatches`

在过去，我们理所当然地认为，我们会读取“小批量”数据，而不是单一的观测数据来更新参数。我们现在给出一个简短的理由。处理单个观测需要我们执行许多单个矩阵-向量(甚至是向量-向量)乘法，这是相当昂贵的，并且代表底层的深度学习框架产生了显著的开销。这既适用于评估应用于数据的网络(通常称为推断)，也适用于计算梯度以更新参数。也就是说，这适用于我们执行$\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$的时候，其中

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$

我们可以通过一次将其应用于一小批观测来提高该操作的*计算*效率。也就是说，我们将单个观测上的梯度$\mathbf{g}_t$替换为小批量上的一个

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$$

让我们看看这对$\mathbf{g}_t$的统计属性有什么影响：由于$\mathbf{x}_t$和小批量$\mathcal{B}_t$的所有元素都是从训练集中随机均匀地抽取的，所以梯度的期望保持不变。另一方面，方差显著减小。由于小批量梯度由平均的$b := |\mathcal{B}_t|$个独立梯度组成，其标准偏差减少了$b^{-\frac{1}{2}}$倍。这本身就是一件好事，因为它意味着更新更可靠地与完全渐变对齐。

天真地，这将表明选择大小批量$\mathcal{B}_t$将是普遍希望的。遗憾的是，在某种程度上，与计算成本的线性增加相比，标准偏差的额外减少是微乎其微的。在实践中，我们选择一个足够大的小批量来提供良好的计算效率，同时仍然适合GPU的内存。为了说明节省的成本，让我们看一下一些代码。在它中，我们执行相同的矩阵-矩阵乘法，但这一次被分成一次64列的“小批”。

```{.python .input}
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = np.dot(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab pytorch
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64].assign(tf.tensordot(B, C[:, j:j+64], axes=1))
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

正如我们所看到的，在小批量上的计算基本上与在全矩阵上的计算效率一样高。有必要提一句告诫。在:numref:`sec_batch_norm`中，我们使用了一种类型的正则化，这种正则化在很大程度上依赖于小批量中的方差大小。当我们增加后者时，方差会减小，并且由于批量归一化而带来的噪声注入带来的好处也会随之减少。有关如何重新缩放和计算适当条款的详细信息，请参见例如:cite:`Ioffe.2017`。

## 正在读取数据集

让我们看看如何从数据高效地生成小批量。在下文中，我们使用由美国宇航局开发的数据集来测试机翼[noise from different aircraft](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise)，以比较这些优化算法。为方便起见，我们只使用前$1,500$个示例。数据被白化以进行预处理，也就是说，我们去掉平均值，并将方差重新缩放到每个坐标$1$。

```{.python .input}
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab pytorch
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab tensorflow
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

## 从头开始实施

回想一下:numref:`sec_linear_scratch`中的小批量sgd实现。在以下内容中，我们将提供稍微更一般的实现。为方便起见，它与本章后面介绍的其他优化算法具有相同的调用签名。具体地说，我们添加状态输入`states`，并将超参数放置在字典`hyperparams`中。此外，我们将在训练函数中平均每个小批量样本的损失，因此优化算法中的梯度不需要除以批量大小。

```{.python .input}
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

```{.python .input}
#@tab pytorch
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, states, hyperparams):
    for param, grad in zip(params, grads):
        param.assign_sub(hyperparams['lr']*grad)
```

接下来，我们实现一个通用的训练函数，以便于使用本章后面介绍的其他优化算法。该算法对线性回归模型进行初始化，并可利用后续介绍的小批量SGD等算法对模型进行训练。

```{.python .input}
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab pytorch
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = tf.Variable(tf.random.normal(shape=(feature_dim, 1),
                                   mean=0, stddev=0.01),trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # Train
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
          with tf.GradientTape() as g:
            l = tf.math.reduce_mean(loss(net(X), y))

          dw, db = g.gradient(l, [w, b])
          trainer_fn([w, b], [dw, db], states, hyperparams)
          n += X.shape[0]
          if n % 200 == 0:
              timer.stop()
              p = n/X.shape[0]
              q = p/tf.data.experimental.cardinality(data_iter).numpy()
              r = (d2l.evaluate_loss(net, data_iter, loss),)
              animator.add(q, r)
              timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

让我们看看批量梯度下降的优化是如何进行的。这可以通过将小批量大小设置为1500(即设置为示例总数)来实现。因此，每个历元只更新一次模型参数。进展甚微。事实上，经过6个步骤之后，进度就停滞不前了。

```{.python .input}
#@tab all
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

当批量等于1时，我们使用SGD进行优化。为了实现简单，我们选择了一个恒定(虽然很小)的学习率。在SGD中，每当处理示例时都会更新模型参数。在我们的例子中，这相当于每个纪元1500个更新。我们可以看到，目标函数值的下降在一个时代之后会放缓。虽然这两个过程都在一个时期内处理了1500个示例，但在我们的实验中，SGD比梯度下降消耗了更多的时间。这是因为SGD更新参数的频率更高，而且一次处理一个观测值的效率较低。

```{.python .input}
#@tab all
sgd_res = train_sgd(0.005, 1)
```

最后，当批量等于100时，我们使用小批量SGD进行优化。每个历元所需的时间比SGD所需的时间和批量梯度下降所需的时间都要短。

```{.python .input}
#@tab all
mini1_res = train_sgd(.4, 100)
```

将批处理大小减少到10，则每个时期的时间会增加，因为每个批处理的工作负载执行效率较低。

```{.python .input}
#@tab all
mini2_res = train_sgd(.05, 10)
```

现在我们可以比较前四个实验的时间和损失。可以看出，虽然SGD在处理的样本数量方面比GD收敛得更快，但它比GD需要更多的时间才能达到相同的损失，因为通过示例计算梯度示例的效率不是那么高。MiniBatch SGD算法能够在收敛速度和计算效率之间进行折衷。小批量大小为10比SGD更有效；小批量大小为100甚至在运行时方面优于GD。

```{.python .input}
#@tab all
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
```

## 简明实施

在GLUON中，我们可以使用`Trainer`类来调用优化算法。这用于实现一般的训练功能。我们将在本章中使用这一点。

```{.python .input}
#@save
def train_concise_ch11(tr_name, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyperparams)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

```{.python .input}
#@tab pytorch
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # Initialization
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)

    loss = nn.MSELoss()
    # Note: L2 Loss = 1/2 * MSE Loss. PyTorch has MSE Loss which is slightly
    # different from MXNet's L2Loss by a factor of 2. Hence we halve the loss
    # value to get L2Loss in PyTorch
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)/2
            l.backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss)/2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

```{.python .input}
#@tab tensorflow
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    optimizer = trainer_fn(**hyperparams)
    loss = tf.keras.losses.MeanSquaredError()
    # Note: L2 Loss = 1/2 * MSE Loss. TensorFlow has MSE Loss which is
    # slightly different from MXNet's L2Loss by a factor of 2. Hence we halve
    # the loss value to get L2Loss in TensorFlow
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)/2
                params = net.trainable_variables
                grads = g.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                p = n/X.shape[0]
                q = p/tf.data.experimental.cardinality(data_iter).numpy()
                r = (d2l.evaluate_loss(net, data_iter, loss)/2,)
                animator.add(q, r)
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

用胶子重复上一个实验，表现出相同的行为。

```{.python .input}
data_iter, _ = get_data_ch11(10)
train_concise_ch11('sgd', {'learning_rate': 0.05}, data_iter)
```

```{.python .input}
#@tab pytorch
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.05}, data_iter)
```

```{.python .input}
#@tab tensorflow
data_iter, _ = get_data_ch11(10)
trainer = tf.keras.optimizers.SGD
train_concise_ch11(trainer, {'learning_rate': 0.05}, data_iter)
```

## 摘要

* 由于深度学习框架降低了开销，而且CPU和GPU上的内存局部性和缓存更好，矢量化可以提高代码的效率。
* 在SGD产生的统计效率和一次处理大批量数据产生的计算效率之间存在权衡。
* 小批量随机梯度下降提供了两个方面的最佳效果：计算效率和统计效率。
* 在小批量SGD中，我们处理通过训练数据的随机排列获得的成批数据(即，每个观测在每个历元仅被处理一次，尽管是随机顺序)。
* 在训练期间降低学习速度是明智的。
* 一般来说，小批量SGD比SGD和梯度下降更快，当以时钟时间衡量时，收敛到更小的风险。

## 练习

1. 修改批次大小和学习率，观察目标函数值和每个时期所消耗时间的下降率。
1. 阅读MXnet文档，并使用`Trainer`类`set_learning_rate`功能将小批量SGD的学习速率在每个时期之后降低到其前值的1/10。
1. 将迷你批次SGD与来自训练集的实际“样本替换”变体进行比较。会发生什么事？
1. 一个邪恶的精灵在没有通知您的情况下复制您的数据集(即，每次观察发生两次，并且您的数据集增长到原始大小的两倍，但是没有人告诉您)。SGD、小批量SGD和梯度下降的行为是如何改变的？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/353)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1068)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1069)
:end_tab:
