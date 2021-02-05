# 学习速率调度
:label:`sec_scheduler`

到目前为止，我们主要关注如何更新权重向量的优化算法，而不是更新它们的速度。尽管如此，调整学习速率通常和实际算法一样重要。有许多方面需要考虑：

* 最明显的是，学习速度的大小很重要。如果它太大，优化就会发散，如果它太小，训练时间太长，或者我们最终得到一个次优的结果。我们以前看到问题的条件号很重要（参见:numref:`sec_momentum`了解详细信息）。直观地说，它是最不敏感方向上的变化量与最敏感方向上的变化量之比。
* 其次，衰变率同样重要。如果学习率仍然很高，我们可能只会在最小值附近反弹，因此无法达到最优。:numref:`sec_minibatch_sgd`对此进行了详细讨论，并分析了:numref:`sec_sgd`中的性能保证。简而言之，我们希望速率衰减，但可能比$\mathcal{O}(t^{-\frac{1}{2}})$慢，这对于凸问题是一个很好的选择。
* 另一个同样重要的方面是*初始化*。这既涉及到参数最初是如何设置的（查看:numref:`sec_numerical_stability`了解详细信息），也涉及到参数最初是如何演变的。这被称为“预热”，也就是说，我们开始朝解决方案前进的速度有多快。开始时的大步骤可能没有好处，特别是因为初始参数集是随机的。最初的更新方向也可能毫无意义。
* 最后，有许多优化变量执行循环学习率调整。这超出了本章的范围。我们建议读者回顾:cite:`Izmailov.Podoprikhin.Garipov.ea.2018`中的细节，例如，如何通过对整个参数路径求平均来获得更好的解决方案。

考虑到管理学习率需要很多细节，大多数深度学习框架都有自动处理这些问题的工具。在本章中，我们将回顾不同的时间表对准确性的影响，并展示如何通过*学习率调度器*有效地进行管理。

## 玩具问题

我们从一个玩具问题开始，这个问题很便宜，可以很容易地计算，但也很重要，可以说明一些关键的方面。为此，我们选择了一个稍微现代化的LeNet版本（`relu`而不是`sigmoid`激活，MaxPooling而不是AveragePooling），用于时尚MNIST。此外，为了提高性能，我们对网络进行了混合。由于大多数代码都是标准的，所以我们只介绍基本内容，不做进一步的详细讨论。如需复习，请参阅:numref:`chap_cnn`。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, lr_scheduler, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the 
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device):
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.optim import lr_scheduler

def net_fn():
    class Reshape(nn.Module):
        def forward(self, x):
            return x.view(-1,1,28,28)
    
    model = torch.nn.Sequential(
        Reshape(),
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))
    
    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the 
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
          scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))
    
        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the 
# lenet section of chapter convolutional neural networks
def train(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu(), custom_callback = False):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    if custom_callback is False:
        net.fit(train_iter, epochs=num_epochs, verbose=0, 
                callbacks=[callback])
    else:
         net.fit(train_iter, epochs=num_epochs, verbose=0,
                 callbacks=[callback, custom_callback])
    return net
```

让我们看看如果我们用默认设置调用这个算法会发生什么，比如学习率$0.3$，训练$30$次迭代。注意，训练精度是如何不断提高的，而测试精度方面的进展却停滞不前。两条曲线之间的间隙表示过拟合。

```{.python .input}
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab tensorflow
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
```

## 调度者

调整学习速率的一种方法是在每一步都显式地设置它。这可以通过`set_learning_rate`方法方便地实现。我们可以在每个epoch之后（甚至在每个minibatch之后）向下调整它，例如，以动态方式响应优化的进展。

```{.python .input}
trainer.set_learning_rate(0.1)
print(f'learning rate is now {trainer.learning_rate:.2f}')
```

```{.python .input}
#@tab pytorch
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')
```

```{.python .input}
#@tab tensorflow
lr = 0.1
dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
dummy_model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
print(f'learning rate is now ,', dummy_model.optimizer.lr.numpy())
```

一般来说，我们需要定义一个调度器。当使用更新次数调用时，它返回适当的学习率值。让我们定义一个简单的方法，将学习速率设置为$\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$。

```{.python .input}
#@tab all
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

让我们在一系列值上绘制它的行为。

```{.python .input}
#@tab all
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

现在，让我们来看看这是如何发挥出对时尚MNIST培训。我们只是提供调度器作为训练算法的附加参数。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

这比以前好了很多。有两件事很突出：曲线比以前更加平滑。其次，过度装修的现象较少。不幸的是，为什么某些策略会减少理论上的过度拟合，这并不是一个很好解决的问题。有人认为，步长越小，参数越接近零，就越简单。然而，这并不能完全解释这一现象，因为我们并不是真的提前停止学习，而是轻轻地降低学习速度。

## 政策

虽然我们不可能涵盖所有种类的学习率调度程序，但我们尝试在下面简要概述流行的策略。常见的选择是多项式衰减和分段常数时间表。除此之外，余弦学习率时间表已被发现在一些问题的经验工作得很好。最后，在一些问题上，在使用大的学习率之前预热优化器是有益的。

### 因子调度程序

多项式衰减的一种替代方法是乘法衰减，即$\alpha \in (0, 1)$的$\eta_{t+1} \leftarrow \eta_t \cdot \alpha$。为了防止学习速率衰减超过合理的下限，更新方程经常被修改为$\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$。

```{.python .input}
#@tab all
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(d2l.arange(50), [scheduler(t) for t in range(50)])
```

这也可以通过MXNet中的内置调度器通过`lr_scheduler.FactorScheduler`对象来完成。它需要更多的参数，例如预热周期、预热模式（线性或常数）、所需更新的最大数量等；接下来我们将根据需要使用内置的调度程序，并仅在此处解释其功能。如图所示，如果需要，构建自己的调度器是相当简单的。

### 多因素调度程序

训练深度网络的一种常用策略是保持学习速率分段恒定，并每隔一定时间将其降低一个给定的量。也就是说，给定一组降低速率的时间，例如$s = \{5, 10, 20\}$每当$t \in s$时降低$\eta_{t+1} \leftarrow \eta_t \cdot \alpha$。假设每一步的值都减半，我们可以如下实现。

```{.python .input}
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(d2l.arange(num_epochs), [get_lr(trainer, scheduler) 
                                  for t in range(num_epochs)])
```

```{.python .input}
#@tab tensorflow
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr
  
    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

这种分段恒定学习速率调度背后的直觉是，可以让优化过程继续进行，直到根据权重向量的分布达到一个稳定点。然后（也只有那时）我们才降低速率，以便获得一个高质量的代理，使其达到一个好的局部最小值。下面的例子展示了如何产生更好的解决方案。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### 余弦调度器

:cite:`Loshchilov.Hutter.2016`提出了一个相当复杂的启发式方法。它依赖于这样一种观察，即我们可能不想在一开始就过快地降低学习率，而且，我们可能希望在最后使用非常小的学习率来“细化”解决方案。这将产生一个类似余弦的时间表，学习率的函数形式如下，范围为$t \in [0, T]$。

$$\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)$$

这里$\eta_0$是初始学习速率，$\eta_T$是时间$T$的目标速率。此外，对于$t > T$，我们只需将值固定到$\eta_T$，而不必再次增加它。在下面的示例中，我们设置最大更新步骤$T = 20$。

```{.python .input}
scheduler = lr_scheduler.CosineScheduler(max_update=20, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps
  
    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

在计算机视觉的背景下，这个时间表可以带来更好的结果。不过，请注意，这些改进并不能得到保证（如下所示）。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### 热身。

在某些情况下，初始化参数不足以保证一个好的解决方案。这对于一些先进的网络设计来说尤其是一个问题，可能会导致不稳定的优化问题。我们可以通过选择一个足够小的学习率来解决这个问题，以防止在开始时出现分歧。不幸的是，这意味着进展缓慢。相反，一个大的学习率最初会导致分歧。

解决这个难题的一个相当简单的方法是使用一个热身期，在此期间学习速率*增加*到其初始最大值，并将速率冷却到优化过程结束。为简单起见，通常使用线性增加。这就产生了如下所示的表格。

```{.python .input}
scheduler = lr_scheduler.CosineScheduler(20, warmup_steps=5, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

注意，网络最初收敛得更好（特别是观察前5个时期的性能）。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

预热可以应用于任何调度程序（不仅仅是余弦）。有关学习率计划和更多实验的更详细讨论，请参见:cite:`Gotmare.Keskar.Xiong.ea.2018`。特别是他们发现，在非常深的网络中，预热阶段限制了参数的发散量。这在直觉上是有意义的，因为我们预计由于网络中那些在开始时花费最多时间取得进展的部分的随机初始化，会出现显著的差异。

## 摘要

* 在训练过程中降低学习率可以提高准确度，并且（最令人困惑的是）减少模型的过度拟合。
* 在实践中，每当进展趋于平稳时，分段降低学习率是有效的。从本质上说，这确保了我们有效地收敛到一个合适的解决方案，只有这样才能通过降低学习率来减少参数的固有方差。
* 余弦调度器在一些计算机视觉问题中很流行。参见例如[GluonCV](http://gluon-cv.mxnet.io)以获取此类调度器的详细信息。
* 优化前的预热期可以防止发散。
* 优化在深度学习中有多种用途。除了最小化训练目标外，不同的优化算法和学习速率调度的选择会导致测试集上不同程度的泛化和过度拟合（对于相同数量的训练错误）。

## 练习

1. 在给定的固定学习率下进行优化行为实验。用这种方法你能得到的最好的模型是什么？
1. 如果你改变学习率下降的指数，收敛性会怎样变化？为了方便您在实验中使用`PolyScheduler`。
1. 将余弦调度器应用于大型计算机视觉问题，例如训练ImageNet。相对于其他调度器，它如何影响性能？
1. 热身应该持续多久？
1. 你能把优化和采样联系起来吗？首先使用:cite:`Welling.Teh.2011`随机梯度朗之万动力学的结果。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/359)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1080)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1081)
:end_tab:
