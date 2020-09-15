# 批次标准化
:label:`sec_batch_norm`

训练深层神经网络是困难的。而让它们在合理的时间内收敛可能会很棘手。在本节中，我们将介绍*批处理规范化*，这是一种流行而有效的技术，它可以持续加速深度网络:cite:`Ioffe.Szegedy.2015`的收敛。再加上剩余块（稍后将在:numref:`sec_resnet`中介绍），批处理规范化使从业者能够常规地训练超过100层的网络。

## 训练深层网络

为了激励批处理规范化，让我们回顾一下在训练机器学习模型和神经网络时出现的一些实际挑战。

首先，关于数据预处理的选择通常会对最终结果产生巨大的影响。回想一下我们应用MLPs预测房价（:numref:`sec_kaggle_house`）。处理真实数据时，我们的第一步是标准化输入特征，使其平均值为零，方差为1。直观地说，这种标准化可以很好地与我们的优化器配合使用，因为它将参数*一个先验的*放在一个相似的尺度上。

第二，对于典型的MLP或CNN，在我们训练的过程中，中间层中的变量（例如MLP中的仿射变换输出）可能具有广泛变化的大小：沿着从输入到输出的层，跨同一层中的单元，以及随着时间的推移，由于我们对模型参数的更新。批处理标准化的发明者非正式地假设，这些变量分布中的这种偏移可能会阻碍网络的收敛。直观地说，我们可能会猜想，如果一个层的可变值是另一层的100倍，这可能需要对学习率进行补偿调整。

第三，更深层次的网络很复杂，很容易过度拟合。这意味着正则化变得更加重要。

批处理规范化应用于各个层（可选地，适用于所有层），其工作原理如下：在每个训练迭代中，我们首先通过减去它们的平均值并除以它们的标准差来规范化输入（批处理规范化的输入），其中两者都是基于当前小批量的统计数据进行估计的。接下来，我们应用比例系数和比例偏移。正是由于基于*batch*统计的*normalization*，才有了它的名字。

请注意，如果我们尝试对大小为1的小批量应用批处理规范化，我们将无法了解任何内容。这是因为减去平均值后，每个隐藏单元的值都是0！正如您可能猜到的那样，由于我们花了一整节时间讨论批处理规范化，并且有足够大的小批量，因此该方法被证明是有效和稳定的。这里的一个要点是，当应用批处理规范化时，批大小的选择可能比没有批处理规范化更重要。

在形式上，通过$\mathbf{x} \in \mathcal{B}$表示批次标准化（$\mathrm{BN}$）的输入，即来自小批次$\mathcal{B}$的输入，批次标准化根据以下表达式转换$\mathbf{x}$：

$$\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

在:eqref:`eq_batchnorm`中，$\hat{\boldsymbol{\mu}}_\mathcal{B}$是样品平均值，$\hat{\boldsymbol{\sigma}}_\mathcal{B}$是小批量$\mathcal{B}$的样品标准差。应用标准化后，得到的小批量产品的平均值和单位方差为零。因为单位方差的选择（相对于其他一些幻数）是任意选择，我们通常包括元素
*刻度参数*$\boldsymbol{\gamma}$和*移位参数*$\boldsymbol{\beta}$
与$\mathbf{x}$形状相同。请注意，$\boldsymbol{\gamma}$和$\boldsymbol{\beta}$是需要与其他模型参数一起学习的参数。

因此，在训练期间，中间层的变量大小不能发散，因为批量标准化会主动地将其集中并重新调整到给定的平均值和大小（通过$\hat{\boldsymbol{\mu}}_\mathcal{B}$和${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$）。从业者的一个直觉或智慧是批量标准化似乎允许更积极的学习率。

形式上，我们计算$\hat{\boldsymbol{\mu}}_\mathcal{B}$和:eqref:`eq_batchnorm`中的$\hat{\boldsymbol{\mu}}_\mathcal{B}$，如下所示：

$$\begin{aligned} \hat{\boldsymbol{\mu}}_\mathcal{B} &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x},\\
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.\end{aligned}$$

请注意，我们在方差估计中添加了一个小常数$\epsilon > 0$，以确保我们从不尝试除以零，即使在经验方差估计可能会消失的情况下。估计值$\hat{\boldsymbol{\mu}}_\mathcal{B}$和${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$通过使用均值和方差的噪声估计来抵消标度问题。你可能认为这种噪音应该是个问题。事实证明，这实际上是有益的。

这在深度学习中是一个反复出现的主题。由于理论上还没有很好描述的原因，优化中的各种噪声源通常导致更快的训练和更少的过度拟合：这种变化似乎是一种正则化形式。在一些初步研究中，:cite:`Teye.Azizpour.Smith.2018`和:cite:`Luo.Wang.Shao.ea.2018`分别将批规范化的性质与贝叶斯先验和惩罚相关联。特别是，这揭示了为什么批量标准化最适合$50 \sim 100$范围内的中等小批量的难题。

修正一个经过训练的模型，您可能会认为我们更喜欢使用整个数据集来估计均值和方差。一旦训练完成，为什么我们要根据同一图像所处的批次对同一图像进行不同的分类？在训练过程中，这种精确的计算是不可行的，因为每次我们更新模型时，所有数据示例的中间变量都会发生变化。然而，一旦模型被训练，我们就可以根据整个数据集计算每一层变量的均值和方差。事实上，这是使用批处理规范化的模型的标准实践，因此批处理规范化层在*训练模式*（通过小批量统计进行规范化）和在*预测模式*（通过数据集统计进行规范化）中的功能不同。

我们现在准备看看批量标准化在实践中是如何工作的。

## 批处理规范化层

完全连接层和卷积层的批处理规范化实现略有不同。我们在下面讨论这两种情况。回想一下，批处理规范化与其他层之间的一个关键区别是，因为批处理规范化一次只能在一个完整的小批量上运行，所以我们不能像以前介绍其他层时那样忽略批处理维度。

### 全连接层

当对完全连接的层应用批处理规范化时，原始文件在仿射变换之后和非线性激活函数之前插入批处理规范化（以后的应用程序可能在激活函数之后插入批处理规范化）:cite:`Ioffe.Szegedy.2015`。用$\mathbf{x}$表示对全连通层的输入，用$\mathbf{x}$表示仿射变换（使用权参数$\mathbf{W}$和偏置参数$\phi$），用$\phi$表示激活函数，我们可以将启用批量标准化的全连接层输出$\mathbf{h}$的计算表示如下：

$$\mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

回想一下，均值和方差是在应用转换的*同一*小批量上计算的。

### 卷积层

类似地，对于卷积层，我们可以在卷积之后和非线性激活函数之前应用批标准化。当卷积有多个输出通道时，我们需要对这些通道的*每个*输出进行批量归一化，每个通道都有自己的尺度和移位参数，这两个参数都是标量。假设我们的小批量包含$m$个示例，对于每个通道，卷积的输出高度为$p$，宽度为$q$。对于卷积层，我们同时对每个输出通道的$m \cdot p \cdot q$个元素进行批处理归一化。因此，我们在计算平均值和方差时收集所有空间位置的值，然后在给定通道内应用相同的平均值和方差来规范化每个空间位置的值。

### 预测过程中的批量标准化

正如我们前面提到的，批处理规范化在训练模式和预测模式中的行为通常是不同的。首先，一旦我们训练了模型，样本均值和样本方差中的噪声就不再是理想的了。第二，我们可能没有计算每批标准化统计数据的奢侈。例如，我们可能需要应用我们的模型一次做出一个预测。

通常，在训练之后，我们使用整个数据集来计算变量统计的稳定估计，然后在预测时修正它们。因此，批量标准化在培训和测试期间的表现是不同的。回想一下，辍学也表现出这一特点。

## 从头开始实施

下面，我们从零开始使用张量实现一个批处理规范化层。

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

我们现在可以创建一个适当的`BatchNorm`层。我们的层将保持适当的参数为规模`gamma`和班次`beta`，这两个将在训练过程中更新。此外，我们的层将保持均值和方差的移动平均值，以便在模型预测期间使用。

抛开算法细节不谈，注意我们实现层的设计模式。通常，我们在一个单独的函数中定义数学，比如`batch_norm`。然后我们将这个功能集成到一个定制层中，该层的代码主要处理簿记事务，例如将数据移动到正确的设备上下文，分配和初始化任何必需的变量，跟踪移动平均值（这里是均值和方差），等等。这种模式可以将数学与样板代码完全分离。另外请注意，为了方便起见，我们不担心在这里自动推断输入形状，因此我们需要指定整个特征的数量。不用担心，深度学习框架中的高级批处理规范化api将为我们处理这些问题，我们稍后将对此进行演示。

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

## 批标准化在LeNet中的应用

为了了解如何在上下文中应用`BatchNorm`，下面我们将其应用于传统的LeNet模型（:numref:`sec_lenet`）。回想一下，批量规范化是在卷积层或完全连接层之后，但在相应的激活函数之前应用的。

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

我们会像以前一样在我们的网络上训练数据集。这个代码实际上与我们第一次训练LeNet（:numref:`sec_lenet`）时的代码完全相同。主要的区别是学习率大得多。

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

让我们看看从第一批处理规范化层学习到的scale参数`gamma`和shift参数`beta`。

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

与我们刚刚定义的`BatchNorm`类相比，我们可以直接从深度学习框架中使用高级api中定义的`BatchNorm`类。我们的应用程序与上面的代码几乎相同。

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

下面，我们使用相同的超参数来训练我们的模型。请注意，通常，高级API变型运行得更快，因为它的代码已经编译成C++或CUDA，而我们的自定义实现必须由Python来解释。

```{.python .input}
#@tab all
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## 争议

直观地说，批处理规范化被认为可以使优化环境更平滑。然而，在训练深层模型时，我们必须小心区分投机直觉和对我们观察到的现象的真实解释。回想一下，我们甚至不知道为什么更简单的深层神经网络（MLPs和传统cnn）一开始就能很好地推广。即使出现了辍学和权重衰减，它们仍然非常灵活，以至于无法通过传统的学习理论泛化保证来解释它们对未知数据的泛化能力。

在最初提出批处理标准化的论文中，作者除了介绍一个强大而有用的工具外，还解释了它的工作原理：通过减少*内部协变量偏移*。据推测，作者所说的“内部协变量转移”指的是类似于上述直觉的东西，即变量值的分布在训练过程中会发生变化。然而，这种解释有两个问题：i）这种漂移与协变量漂移非常不同，这使得这个名字用词不当。ii）这种解释提供了一种不明确的直觉，但却留下了一个有待严格解释的问题，即“为什么这项技术准确地起作用”。在这本书中，我们的目的是传达实践者用来指导他们发展深层神经网络的直觉。然而，我们认为，重要的是将这些指导性直觉与既定的科学事实区分开来。最终，当你掌握了这些材料并开始撰写自己的研究论文时，你会想清楚地区分技术主张和直觉。

在批处理标准化的成功之后，关于如何呈现机器学习研究的技术文献和更广泛的讨论中，关于内部协变量转移的解释一再出现。阿里·拉希米（Ali Rahimi）在接受2017年NeurIPS大会的时间测试奖（Test of Time Award）时发表了一篇令人难忘的演讲，他将“内部协变量转移”（internal covariate shift）作为焦点，将现代深度学习的实践比作炼金术。随后，在一份立场文件中对该示例进行了详细回顾，概述了机器学习中令人不安的趋势:cite:`Lipton.Steinhardt.2018`。另一些作者对批处理规范化的成功提出了另一种解释，一些人声称，尽管批处理规范化的成功表现出与原始论文:cite:`Santurkar.Tsipras.Ilyas.ea.2018`中声称的行为在某些方面是相反的。

我们注意到，*内部协变量变化*并不比每年在机器学习技术文献中提出的数千个类似模糊的说法更值得批评。很可能，它作为这些辩论焦点的共鸣，要归功于它对目标受众的广泛认可。批处理规范化已经被证明是一种不可或缺的方法，应用于几乎所有已部署的图像分类器中，这篇介绍该技术的论文引文数万篇。

## 摘要

* 在模型训练过程中，批量归一化利用小批量的均值和标准差不断调整神经网络的中间输出，使整个神经网络各层的中间输出值更加稳定。
* 完全连接层和卷积层的批处理规范化方法略有不同。
* 批处理规范化层和丢失层一样，在训练模式和预测模式下具有不同的计算结果。
* 批量规范化有许多有益的副作用，主要是正则化。另一方面，减少内部协变量变化的原始动机似乎不是一个有效的解释。

## 练习

1. 在批量标准化之前，是否可以从完全连接层或卷积层中删除偏差参数？为什么？
1. 比较LeNet在批处理和不使用批处理标准化的情况下的学习率。
    1. 描绘出训练和测试准确性的提高。
    1. 你的学习率有多高？
1. 我们需要在每一层进行批量标准化吗？尝试一下？
1. 你能用批处理规范化来代替丢失吗？行为如何改变？
1. 固定参数`beta`和`gamma`，观察分析结果。
1. 查看来自高级api的`BatchNorm`的在线文档，以查看用于批处理规范化的其他应用程序。
1. 研究思路：想想其他可以应用的规范化转换？你能应用概率积分变换吗？全秩协方差估计如何？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/330)
:end_tab:
