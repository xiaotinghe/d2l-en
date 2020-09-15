# 批量归一化
:label:`sec_batch_norm`

训练深度神经网络是困难的。而且，让它们在合理的时间内汇聚起来可能是一件棘手的事情。在这一部分中，我们将介绍“批量归一化”，这是一种流行且有效的技术，可持续加速深度网络:cite:`Ioffe.Szegedy.2015`的收敛。连同剩余块-稍后将在:numref:`sec_resnet`中介绍-批量标准化使从业者有可能例行训练百层以上的网络。

## 训练深度网络

为了促进批处理标准化，让我们回顾一下在训练机器学习模型和神经网络时出现的一些实际挑战。

首先，关于数据预处理的选择通常会对最终结果产生巨大的影响。回想一下我们应用MLP来预测房价(:numref:`sec_kaggle_house`)。在处理真实数据时，我们的第一步是将输入要素标准化，使每个要素的平均值为0，方差为1。直观地说，这个标准化与我们的优化器配合得很好，因为它将参数“先验地”置于相似的规模。

其次，对于典型的MLP或CNN，在我们训练时，中间层中的变量(例如，MLP中的仿射变换输出)可能会采用幅度变化很大的值：既沿着从输入到输出的层，跨同一层中的单元，也随着时间的推移，由于我们对模型参数的更新。批量归一化的发明者非正式地假设，这些变量分布中的这种漂移可能会阻碍网络的收敛。直观地说，我们可能会猜测，如果一层的变量值是另一层的100倍，这可能需要对学习率进行补偿性调整。

第三，更深的网络很复杂，很容易过度拟合。这意味着正规化变得更加关键。

批次归一化应用于各个层(可选地，应用于所有层)，其工作方式如下：在每次训练迭代中，我们首先通过减去它们的平均值并除以它们的标准差来归一化(批次归一化的)输入，其中两者都是基于当前小批次的统计来估计的。接下来，我们应用缩放系数和缩放偏移。正是由于这种基于*批量*统计的*归一化*，所以才有了*批量归一化*的名称。

请注意，如果我们尝试对大小为1的小批应用批标准化，我们将无法了解任何内容。这是因为减去平均值后，每个隐藏单元的值为0！正如您可能猜到的那样，由于我们花了整整一节来进行批次规范化，并且有足够大的小批次，因此该方法被证明是有效和稳定的。这里要说明的一点是，在应用批处理标准化时，批处理大小的选择可能比没有批处理标准化更重要。

形式上，用$\mathbf{x} \in \mathcal{B}$表示对批次规格化($\mathrm{BN}$)的输入，其来自小批次$\mathcal{B}$，批次规格化根据以下表达式转换$\mathbf{x}$：

$$\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

在:eqref:`eq_batchnorm`中，$\hat{\boldsymbol{\mu}}_\mathcal{B}$是样本平均值，$\hat{\boldsymbol{\sigma}}_\mathcal{B}$是小批量$\mathcal{B}$的样本标准偏差。在应用标准化之后，所得到的小批量的均值和单位方差为零。因为单位方差的选择(与其他一些幻数)是任意选择的，所以我们通常包括按元素
*比例参数*$\boldsymbol{\gamma}$和*移位参数*$\boldsymbol{\beta}$
它们的形状与$\mathbf{x}$相同。请注意，$\boldsymbol{\gamma}$和$\boldsymbol{\beta}$是需要与其他模型参数一起学习的参数。

因此，中间层的可变幅度在训练期间不能发散，因为批归一化主动地将它们居中并将它们重新缩放回给定的平均值和大小(通过$\hat{\boldsymbol{\mu}}_\mathcal{B}$和${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$)。实践者的一个直觉或智慧是，批量标准化似乎允许更积极的学习率。

形式上，我们:eqref:`eq_batchnorm`年度$\hat{\boldsymbol{\mu}}_\mathcal{B}$和${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$计算如下：

$$\begin{aligned} \hat{\boldsymbol{\mu}}_\mathcal{B} &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x},\\
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.\end{aligned}$$

请注意，我们在方差估计中添加了一个小常数$\epsilon > 0$，以确保即使在经验方差估计可能消失的情况下，我们也不会尝试除以零。估计$\hat{\boldsymbol{\mu}}_\mathcal{B}$和${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$通过使用均值和方差的噪声估计来抵消缩放问题。你可能认为这种嘈杂应该是个问题。事实证明，这实际上是有益的。

事实证明，这是深度学习中反复出现的主题。由于理论上还没有很好描述的原因，优化中的各种噪声源通常会导致更快的训练和更少的过度适应：这种变化似乎是正则化的一种形式。在一些初步研究中，:cite:`Teye.Azizpour.Smith.2018`和:cite:`Luo.Wang.Shao.ea.2018`分别将批次归一化的性质与贝叶斯先验和惩罚相关联。特别是，这揭示了为什么批次标准化对于$50 \sim 100$范围内的中等小批量效果最好的谜题。

固定一个经过训练的模型，您可能会认为我们更喜欢使用整个数据集来估计均值和方差。一旦训练完成，我们为什么要根据图像所在的批次，对相同的图像进行不同的分类呢？在训练过程中，这种精确的计算是不可行的，因为我们每次更新模型时，所有数据示例的中间变量都会发生变化。然而，一旦模型被训练，我们就可以基于整个数据集来计算每一层变量的均值和方差。事实上，这是采用批量归一化的模型的标准实践，因此批归一化层在*训练模式*(通过小批量统计进行归一化)和在*预测模式*(通过数据集统计进行归一化)中的功能不同。

现在我们准备看看批处理标准化在实践中是如何工作的。

## 批归一化图层

完全连接层和卷积层的批归一化实现略有不同。我们将在下面讨论这两种情况。回想一下，批次规格化与其他层之间的一个关键区别是，因为批次规格化一次在一个完整的小批次上操作，所以我们不能像以前在引入其他层时那样忽略批次维度。

### 完全连接层

当将批归一化应用于完全连接的层时，原始论文在仿射变换之后并且在非线性激活函数之前插入批归一化(稍后的应用可以恰好在激活函数之后插入批归一化):cite:`Ioffe.Szegedy.2015`。用$\mathbf{x}$表示对全连接层的输入，用$\mathbf{W}\mathbf{x} + \mathbf{b}$表示仿射变换(具有权重参数$\mathbf{W}$和偏置参数$\mathbf{b}$)，用$\phi$表示激活函数，我们可以将启用批归一化的全连接层输出$\mathbf{h}$的计算表示如下：

$$\mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

回想一下，均值和方差是在应用转换的“相同”小批次上计算的。

### 卷积层

类似地，对于卷积层，我们可以在卷积之后和非线性激活函数之前应用批归一化。当卷积有多个输出通道时，我们需要对这些通道的*每个*个输出进行批量归一化，并且每个通道都有自己的标量和移位参数，这两个参数都是标量。假设我们的小批包含$m$个示例，并且对于每个通道，卷积的输出具有高度$p$和宽度$q$。对于卷积层，我们同时对每个输出通道的$m \cdot p \cdot q$个元素执行每批归一化。因此，我们在计算均值和方差时收集所有空间位置上的值，并因此在给定通道内应用相同的均值和方差来归一化每个空间位置的值。

### 预测过程中的批次归一化

正如我们前面提到的，批处理标准化在训练模式和预测模式中的行为通常不同。首先，一旦我们训练了模型，样本均值中的噪声和小批量估计每个样本所产生的样本方差就不再是我们想要的了。其次，我们可能没有计算每批归一化统计数据的奢侈。例如，我们可能需要应用我们的模型来一次做出一个预测。

通常，在训练之后，我们使用整个数据集来计算变量统计的稳定估计，然后将它们固定在预测时间。因此，批处理标准化在训练期间和测试时的行为不同。回想一下，辍学也表现出这一特征。

## 从头开始实施

下面，我们从头开始实现一个带有张量的批归一化层。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `autograd` to determine whether the current mode is training mode or
    # prediction mode
    if not autograd.is_training():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully-connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `is_grad_enabled` to determine whether the current mode is training
    # mode or prediction mode
    if not torch.is_grad_enabled():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully-connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Compute reciprocal of square root of the moving variance element-wise
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Scale and shift
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

我们现在可以创建一个合适的`BatchNorm`层。我们的层将为Scale `gamma`和Shift `beta`维护适当的参数，这两个参数都将在培训过程中更新。此外，我们的层将维护均值和方差的移动平均值，以供随后在模型预测期间使用。

抛开算法细节不谈，请注意我们实现该层的设计模式。通常，我们在一个单独的函数中定义数学，比如`batch_norm`。然后，我们将此功能集成到一个自定义层中，该层的代码主要解决记账问题，例如将数据移动到正确的设备上下文、分配和初始化任何需要的变量、跟踪移动平均值(这里是平均值和方差)，等等。此模式支持将数学与样板代码完全分离。还要注意，为了方便起见，我们没有担心在这里自动推断输入形状，因此我们需要指定贯穿始终的特征数量。不要担心，深度学习框架中的高级批处理标准化API将为我们解决这一问题，我们稍后将演示这一点。

```{.python .input}
class BatchNorm(nn.Block):
    # `num_features`: the number of outputs for a fully-connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully-connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.zeros(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.9)
        return Y
```

```{.python .input}
#@tab pytorch
class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully-connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully-connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```

```{.python .input}
#@tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.9
        delta = variable * momentum + value * (1 - momentum)
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

## 批归一化在LENet中的应用

为了了解如何在上下文中应用`BatchNorm`，下面我们将其应用于传统的LENET模型(:numref:`sec_lenet`)。回想一下，批归一化是在卷积层或完全连接层之后但在相应的激活函数之前应用的。

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        BatchNorm(6, num_dims=4),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        BatchNorm(16, num_dims=4),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        BatchNorm(120, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        BatchNorm(84, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
# Recall that this has to be a function that will be passed to `d2l.train_ch6`
# so that model building or compiling need to be within `strategy.scope()` in
# order to utilize the CPU/GPU devices that we have
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10)]
    )
```

像以前一样，我们将根据Fashion-MNIST数据集对我们的网络进行培训。这一代码实际上与我们第一次训练LeNet(:numref:`sec_lenet`)时的代码相同。主要的不同之处在于学习速度要大得多。

```{.python .input}
#@tab mxnet, pytorch
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

```{.python .input}
#@tab tensorflow
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

让我们看一下从第一批归一化层学习的比例参数`gamma`和移位参数`beta`。

```{.python .input}
net[1].gamma.data().reshape(-1,), net[1].beta.data().reshape(-1,)
```

```{.python .input}
#@tab pytorch
net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,))
```

```{.python .input}
#@tab tensorflow
tf.reshape(net.layers[1].gamma, (-1,)), tf.reshape(net.layers[1].beta, (-1,))
```

## 简明实施

与我们刚刚定义的`BatchNorm`类相比，我们可以直接从深度学习框架中使用高级API中定义的`BatchNorm`类。代码看起来与我们上面实现的应用程序几乎完全相同。

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10),
    ])
```

下面，我们使用相同的超参数来训练我们的模型。请注意，与往常一样，高级API变体的运行速度要快得多，因为它的代码已编译为C++或CUDA，而我们的自定义实现必须由Python解释。

```{.python .input}
#@tab all
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## 争议

直观地说，批量归一化可以使优化后的景观更加平滑。然而，对于我们在训练深层模型时观察到的现象，我们必须小心区分投机性直觉和真实解释。回想一下，我们甚至不知道为什么更简单的深度神经网络(MLP和常规CNN)一开始就具有很好的泛化能力。即使在辍学和重量衰减的情况下，它们仍然非常灵活，以至于它们对看不见的数据进行概括的能力不能通过传统的学习理论概括保证来解释。

在最初提出批量归一化的论文中，作者除了介绍了一个强大而有用的工具外，还解释了它为什么有效：通过减少“内部协变量偏移”。据推测，作者所说的“内部协变量转移”的意思类似于上面表达的直觉-变量值的分布在训练过程中会发生变化的概念。然而，这种解释有两个问题：i)这种漂移与*协变量漂移*有很大的不同，使得这个名字用词不当。ii)解释提供了一种不明确的直觉，但留下了*这项技术究竟为什么有效*的问题，这是一个需要严格解释的悬而未决的问题。在这本书中，我们的目标是传达实践者用来指导他们发展深层神经网络的直觉。然而，我们认为，将这些指导性直觉与既定的科学事实分开是很重要的。最终，当你掌握了这份材料并开始写你自己的研究论文时，你会想要清楚地在技术主张和预感之间划清界限。

随着批量归一化的成功，其关于“内部协变量漂移”的解释一再出现在关于如何呈现机器学习研究的技术文献和更广泛的讨论中。在2017年NeurIPS大会上接受时间奖测试时，阿里·拉希米发表了一次令人难忘的演讲，他将*内部协变量转移*作为争论的焦点，将深度学习的现代实践比作炼金术。随后，在一份概述机器学习:cite:`Lipton.Steinhardt.2018`中令人不安的趋势的立场文件中，详细地重新讨论了这个例子。其他作者对批处理标准化的成功提出了另一种解释，一些人声称批处理标准化的成功，尽管在某些方面表现出与原始论文:cite:`Santurkar.Tsipras.Ilyas.ea.2018`中声称的行为相反的行为。

我们注意到，“内部协变量转移”并不比技术机器学习文献中每年数以千计的类似模糊主张更值得批评。很可能，作为这些辩论的焦点，它的共鸣归功于它对目标受众的广泛认知度。批量归一化已被证明是一种必不可少的方法，几乎应用于所有部署的图像分类器，赢得了介绍该技术的论文的数万条引用。

## 摘要

* 在模型训练过程中，批量归一化利用小批量的均值和标准差不断调整神经网络的中间输出，使整个神经网络各层的中间输出值更加稳定。
* 全连通层和卷积层的批量归一化方法略有不同。
* 批归一化层与退出层一样，在训练模式和预测模式下具有不同的计算结果。
* 批量标准化有许多有益的副作用，主要是正规化的副作用。另一方面，减少内部协变量转移的原始动机似乎不是一个有效的解释。

## 练习

1. 在批量归一化之前，我们可以从全连通层或卷积层中去掉偏置参数吗？为什么？
1. 比较批归一化和不批归一化的LENet的学习率。
    1. 绘制训练和测试精确度提高的图表。
    1. 你能把学习率调到多大？
1. 我们是否需要在每一层中进行批量规范化？用它做实验？
1. 你能用批量归一化来代替退学吗？这种行为是如何改变的？
1. 确定参数`beta`和`gamma`，观察分析结果。
1. 请从高级API查看`BatchNorm`的在线文档，以了解批处理标准化的其他应用程序。
1. 研究思路：想一想可以应用的其他规范化转换吗？你能应用概率积分变换吗？全秩协方差估计怎么样？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/330)
:end_tab:
