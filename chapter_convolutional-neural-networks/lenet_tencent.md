# 卷积神经网络(LENet)
:label:`sec_lenet`

我们现在已经具备了组装一个功能齐全的CNN所需的所有材料。在我们之前遇到的图像数据中，我们将Softmax回归模型(:numref:`sec_softmax_scratch`)和mlp模型(:numref:`sec_mlp_scratch`)应用于Fashion-MNIST数据集中的服装图片。为了使这些数据服从Softmax回归和MLP，我们首先将$28\times28$矩阵中的每个图像展平成固定长度的$784$维向量，然后用完全连通的层对它们进行处理。既然我们已经掌握了卷积图层，我们就可以在图像中保留空间结构。作为用卷积层替换完全连通层的另一个好处，我们将享受到需要更少参数的更简约的模型。

在这一部分中，我们将介绍*LENet*，这是最早发布的CNN之一，它因其在计算机视觉任务中的性能而引起广泛关注。该模型是由当时在AT&T贝尔实验室担任研究员的严乐村(并以他的名字命名)提出的，目的是识别图像:cite:`LeCun.Bottou.Bengio.ea.1998`中的手写数字。这项工作代表了十年来开发这项技术的研究的顶峰。1989年，LeCun发表了第一个通过反向传播成功训练CNN的研究。

当时，LeNet取得了与支持向量机性能相当的突出结果，当时在监督学习中占据主导地位。LeNet最终被改造成识别用于处理ATM机存款的数字。时至今日，一些ATM机还在运行严恩和他的同事里昂·博图在20世纪90年代写的代码！

## 乐网

从高层次上讲，LENET(LENET-5)由两个部分组成：(I)由两个卷积层组成的卷积编码器；(Ii)由三个完全连接层组成的密集挡路；其体系结构在:numref:`img_lenet`中总结。

![Data flow in LeNet. The input is a handwritten digit, the output a probability over 10 possible outcomes.](../img/lenet.svg)
:label:`img_lenet`

每个卷积挡路中的基本单元是卷积层、S形激活函数和随后的平均合用操作。请注意，虽然RELU和MAX-Pooling工作得更好，但这些发现在20世纪90年代还没有完成。每个卷积层使用$5\times 5$核和Sigmoid激活函数。这些图层将空间排列的输入映射到多个二维要素地图，通常会增加通道数量。第一卷积层有6个输出通道，第二层有16个输出通道。每次$2\times2$的合并操作(步长2)通过空间下采样将维数降低$4$倍。卷积挡路发出具有(批次大小、通道数、高度、宽度)给定形状的输出。

为了将卷积挡路的输出传递给密集的挡路，我们必须将小批量中的每个示例都展平。换句话说，我们接受这个四维输入，并将其转换为完全连接层所期望的二维输入：提醒一下，我们所需的二维表示使用第一维索引小批量中的示例，第二维给出每个示例的平面向量表示。乐网的密集挡路有三个全连接层，分别有120、84和10个输出。因为我们仍在执行分类，所以10维输出层对应于可能的输出类的数量。

虽然要真正理解LeNet内部发生的事情可能需要做一些工作，但希望下面的代码片段能让您相信，使用现代深度学习框架实现这样的模型非常简单。我们只需要实例化一个`Sequential`的挡路，并将适当的层链接在一起。

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

我们对原始模型做了一个小改动，去掉了最后一层的高斯激活。除此之外，该网络与最初的LeNet-5架构相匹配。

通过通过网络传递单通道(黑白)$28 \times 28$图像并在每一层打印输出形状，我们可以检查模型以确保其操作符合我们对:numref:`img_lenet_vert`的预期。

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

注意，在整个卷积挡路的每一层处的表示的高度和宽度减小(与前一层相比)。第一卷积层使用2个像素的填充来补偿因使用$5 \times 5$内核而导致的高度和宽度的减小。相反，第二卷积层放弃填充，因此高度和宽度都减少了4个像素。当我们向上层叠时，通道数逐层增加，从输入中的1个增加到第一个卷积层之后的6个和第二个卷积层之后的16个。但是，每个池化层的高度和宽度都减半。最后，每个完全连接的层都会降低维数，最终输出其维数与类的数量相匹配的输出。

## 培训

现在我们已经实现了这个模型，让我们来做一个实验，看看LeNet在Fashion-MNIST上的表现如何。

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
```

虽然CNN的参数较少，但它们的计算成本仍可能比类似的深度MLP更高，因为每个参数参与的乘法要多得多。如果您可以使用GPU，这可能是将其付诸实施以加快培训的好时机。

:begin_tab:`mxnet, pytorch`
为了进行评估，我们需要对`evaluate_accuracy`中描述的:numref:`sec_softmax_scratch`函数稍作修改。由于完整的数据集在主内存中，因此在模型使用GPU计算数据集之前，我们需要将其复制到GPU内存中。
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

我们还需要更新我们的培训功能来处理GPU。与:numref:`sec_softmax_scratch`中定义的`train_epoch_ch3`不同，在进行向前和向后传播之前，我们现在需要将每个小批量数据移动到我们指定的设备(希望是图形处理器)。

训练函数`train_ch6`也类似于:numref:`sec_softmax_scratch`中定义的`train_ch3`。由于我们未来将实施具有多个层的网络，因此我们将主要依靠高级API。以下训练函数假定从高级API创建的模型作为输入，并相应地进行优化。我们使用`device`中介绍的哈维尔初始化来初始化`device`参数所指示的设备上的模型参数。就像MLP一样，我们的损失函数是交叉熵，我们通过小批量随机梯度下降来最小化它。因为每个纪元需要几十秒来运行，所以我们更频繁地将训练损失可视化。

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

* CNN是采用卷积层的网络。
* 在CNN中，我们交织卷积、非线性和(通常)合并操作。
* 在CNN中，卷积层通常被安排成使它们逐渐降低表示的空间分辨率，同时增加通道的数量。
* 在传统的CNN中，由卷积块编码的表示在输出之前由一个或多个全连接层处理。
* LeNet可以说是第一个成功部署这种网络的公司。

## 练习

1. 将平均池化替换为最大池化。会发生什么事？
1. 尝试在LENet的基础上构建更复杂的网络，以提高其准确性。
    1. 调整卷积窗口大小。
    1. 调整输出通道数。
    1. 调整激活功能(例如，RELU)。
    1. 调整卷积层数。
    1. 调整完全连接的层数。
    1. 调整学习速率和其他训练细节(例如，初始化和历元数)。
1. 在原始MNIST数据集上试用改进后的网络。
1. 显示不同输入(例如毛衣和外套)的第一层和第二层LENET的激活。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/73)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/74)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/275)
:end_tab:
