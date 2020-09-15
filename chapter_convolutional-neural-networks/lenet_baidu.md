# 卷积神经网络
:label:`sec_lenet`

我们现在有了组装一个功能齐全的CNN所需的所有成分。在我们之前遇到的图像数据中，我们将softmax回归模型（:numref:`sec_softmax_scratch`）和MLP模型（:numref:`sec_mlp_scratch`）应用于时装MNIST数据集中的服装图片。为了使这些数据适用于softmax回归和MLPs，我们首先将$28\times28$矩阵中的每个图像展平为一个固定长度的$784$维向量，然后用全连通层对其进行处理。现在我们已经掌握了卷积层的处理方法，我们可以在图像中保留空间结构。作为用卷积层代替完全连接层的另一个好处，我们将享受到需要更少参数的更简洁模型。

在本节中，我们将介绍*LeNet*，它是最早发布的cnn之一，因其在计算机视觉任务中的性能而受到广泛关注。这个模型是由当时at&T贝尔实验室的研究员Yann LeCun提出的（并以其命名），目的是识别图像:cite:`LeCun.Bottou.Bengio.ea.1998`中的手写数字。这项工作代表了十年来发展这项技术的研究的高潮。1989年，LeCun发表了第一篇通过反向传播成功训练CNN的研究。

当时LeNet取得了与支持向量机性能相匹配的优秀结果，成为有监督学习的主流方法。LeNet最终被用于识别ATM机中处理存款的数字。时至今日，一些自动取款机仍在运行Yann和他的同事Leon Bottou在上世纪90年代写的代码！

## 列奈

在高层次上，LeNet（LeNet-5）由两部分组成：（i）由两个卷积层组成的卷积编码器；和（ii）由三个完全连接的层组成的密集块；该体系结构在:numref:`img_lenet`中进行了总结。

![Data flow in LeNet. The input is a handwritten digit, the output a probability over 10 possible outcomes.](../img/lenet.svg)
:label:`img_lenet`

每个卷积块中的基本单元是一个卷积层、一个sigmoid激活函数和随后的平均池操作。请注意，虽然ReLUs和max pooling工作得更好，但这些发现在20世纪90年代还没有出现。每个卷积层使用$5\times 5$内核和sigmoid激活函数。这些层将空间排列的输入映射到多个二维特征映射，通常会增加通道的数量。第一卷积层有6个输出通道，而第二个卷积层有16个输出通道。每个$2\times2$池操作（步骤2）通过空间下采样将维数减少$4$倍。卷积块发出形状由（批大小、通道数、高度、宽度）给定的输出。

为了将输出从卷积块传递到稠密块，我们必须在小批量中展平每个示例。换言之，我们将这个四维输入转换成完全连接层所期望的二维输入：作为提醒，我们所希望的二维表示使用第一个维度索引小批量中的示例，第二个维度给出每个示例的平面向量表示。LeNet的密集块有三个完全连接的层，分别有120、84和10个输出。因为我们仍在执行分类，所以10维输出层对应于可能的输出类的数量。

虽然要真正理解LeNet内部的情况可能需要一些工作，但希望下面的代码片段能让您相信，用现代深度学习框架实现此类模型非常简单。我们只需要实例化一个`Sequential`块并将适当的层链接在一起。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        # `Dense` will transform an input of the shape (batch size, number of
        # channels, height, width) into an input of the shape (batch size,
        # number of channels * height * width) automatically by default
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1,1,28,28)

net = torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow.distribute import MirroredStrategy, OneDeviceStrategy

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='sigmoid'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])
```

我们对原始模型做了一点小小的改动，去掉了最后一层的高斯激活。除此之外，这个网络与最初的LeNet-5体系结构相匹配。

通过将一个单通道（黑白）$28 \times 28$图像通过网络并在每一层打印输出形状，我们可以检查模型，以确保其操作与我们期望的:numref:`img_lenet_vert`一致。

![Compressed notation for LeNet-5.](../img/lenet-vert.svg)
:label:`img_lenet_vert`

```{.python .input}
X = np.random.uniform(size=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 28, 28, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
```

注意，在整个卷积块中的每一层的表示的高度和宽度都减小了（与前一层相比）。第一卷积层使用2个像素的填充来补偿由于使用$5 \times 5$内核而导致的高度和宽度的减少。相反，第二卷积层放弃填充，因此高度和宽度都减少了4个像素。当我们往上一层的时候，通道的数量从输入的1层一层增加到第一层卷积层之后的6层和第二层卷积层之后的16个。但是，每个池层的高度和宽度都减半。最后，每一个完全连接的层都会减少维数，最终输出一个维数与类数相匹配的输出。

## 培训

现在我们已经实现了这个模型，让我们来做一个实验，看看LeNet在时尚MNIST上的表现。

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
```

虽然cnn的参数较少，但它们的计算成本仍可能高于类似的深度mlp，因为每个参数都参与更多的乘法运算。如果你有机会使用GPU，这可能是一个很好的时机将它付诸行动，以加快培训。

:begin_tab:`mxnet, pytorch`
为了进行评估，我们需要对:numref:`sec_softmax_scratch`中描述的`evaluate_accuracy`函数进行一点修改。因为完整的数据集在主内存中，所以在模型使用GPU计算数据集之前，我们需要将其复制到GPU内存中。
:end_tab:

```{.python .input}
def evaluate_accuracy_gpu(net, data_iter, device=None):  #@save
    """Compute the accuracy for a model on a dataset using a GPU."""
    if not device:  # Query the first device where the first parameter is on
        device = list(net.collect_params().values())[0].list_ctx()[0]
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        X, y = X.as_in_ctx(device), y.as_in_ctx(device)
        metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0]/metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """Compute the accuracy for a model on a dataset using a GPU."""
    net.eval()  # Set the model to evaluation mode
    if not device:
        device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

我们还需要更新我们的培训功能来处理GPU。与:numref:`sec_softmax_scratch`中定义的`train_epoch_ch3`不同，在进行前向和后向传播之前，我们需要将每一小批数据移动到我们指定的设备（希望是GPU）。

训练功能`train_ch6`也类似于:numref:`sec_softmax_scratch`中定义的`train_ch3`。由于我们将实现多层网络，因此我们将主要依赖于高级api。下面的训练函数假设从高级api创建的模型作为输入，并进行相应的优化。我们使用`device`中介绍的Xavier初始化初始化`device`参数指示的设备上的模型参数。与MLPs一样，我们的损失函数是交叉熵，我们通过小批量随机梯度下降使其最小化。因为每一个纪元都需要几十秒的时间，所以我们更频繁地看到训练的失败。

```{.python .input}
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu()):
    """Train a model with a GPU (defined in Chapter 6)."""
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            # Here is the major difference compared with `d2l.train_epoch_ch3`
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu()):
    """Train a model with a GPU (defined in Chapter 6)."""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            net.train()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_loss = metric[0]/metric[2]
            train_acc = metric[1]/metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))
    print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

```{.python .input}
#@tab tensorflow
class TrainCallback(tf.keras.callbacks.Callback):  #@save
    """A callback to visiualize the training progress."""
    def __init__(self, net, train_iter, test_iter, num_epochs, device_name):
        self.timer = d2l.Timer()
        self.animator = d2l.Animator(
            xlabel='epoch', xlim=[0, num_epochs], legend=[
                'train loss', 'train acc', 'test acc'])
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        self.device_name = device_name
    def on_epoch_begin(self, epoch, logs=None):
        self.timer.start()
    def on_epoch_end(self, epoch, logs):
        self.timer.stop()
        test_acc = self.net.evaluate(
            self.test_iter, verbose=0, return_dict=True)['accuracy']
        metrics = (logs['loss'], logs['accuracy'], test_acc)
        self.animator.add(epoch+1, metrics)
        if epoch == self.num_epochs - 1:
            batch_size = next(iter(self.train_iter))[0].shape[0]
            num_examples = batch_size * tf.data.experimental.cardinality(
                self.train_iter).numpy()
            print(f'loss {metrics[0]:.3f}, train acc {metrics[1]:.3f}, '
                  f'test acc {metrics[2]:.3f}')
            print(f'{num_examples / self.timer.avg():.1f} examples/sec on '
                  f'{str(self.device_name)}')

#@save
def train_ch6(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu()):
    """Train a model with a GPU (defined in Chapter 6)."""
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    net.fit(train_iter, epochs=num_epochs, verbose=0, callbacks=[callback])
    return net
```

现在让我们训练和评估LeNet-5模型。

```{.python .input}
#@tab all
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## 摘要

* CNN是一个使用卷积层的网络。
* 在CNN中，我们交织卷积、非线性和（通常）池操作。
* 在CNN中，卷积层通常被布置成这样，它们逐渐降低表示的空间分辨率，同时增加信道的数量。
* 在传统的cnn中，由卷积块编码的表示在发射输出之前由一个或多个完全连接的层处理。
* LeNet可以说是这类网络的第一次成功部署。

## 练习

1. 将平均池替换为最大池。会发生什么？
1. 尝试基于LeNet构造一个更复杂的网络来提高其精度。
    1. 调整卷积窗口大小。
    1. 调整输出通道的数量。
    1. 调整激活功能（如ReLU）。
    1. 调整卷积层数。
    1. 调整完全连接层的数量。
    1. 调整学习率和其他培训细节（例如初始化和时段数）
1. 在原始MNIST数据集上试用改进后的网络。
1. 显示不同输入（例如毛衣和外套）的第一层和第二层LeNet的激活。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/73)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/74)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/275)
:end_tab:
