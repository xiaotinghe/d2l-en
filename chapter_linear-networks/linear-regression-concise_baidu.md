# 线性回归的简明实现
:label:`sec_linear_concise`

在过去的几年里，对深度学习的广泛而强烈的兴趣激发了公司、学者和业余爱好者开发各种成熟的开源框架，以自动化实现基于梯度的学习算法的重复工作。在:numref:`sec_linear_scratch`中，我们只依赖于（i）用于数据存储和线性代数的张量；以及（ii）用于计算梯度的自微分。在实践中，由于数据迭代器、损失函数、优化器和神经网络层非常常见，现代图书馆也为我们实现了这些组件。

在第614节中，我们将简要介绍如何在第614节中使用线性回归模型来实现高层次的回归。

## 生成数据集

首先，我们将生成与:numref:`sec_linear_scratch`中相同的数据集。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch.utils import data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

## 读取数据集

我们可以调用框架中现有的API来读取数据，而不是滚动我们自己的迭代器。我们传入`features`和`labels`作为参数，并在实例化数据迭代器对象时指定`batch_size`。此外，布尔值`is_train`表示是否希望数据迭代器对象在每个历元（通过数据集）上对数据进行洗牌。

```{.python .input}
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a Gluon data iterator."""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab pytorch
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab tensorflow
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a TensorFlow data iterator."""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset
```

```{.python .input}
#@tab all
batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

现在我们可以使用`data_iter`，其方式与我们在:numref:`sec_linear_scratch`中调用`data_iter`函数的方式大致相同。为了验证它是否有效，我们可以阅读并打印第一批小样本。与:numref:`sec_linear_scratch`相比，这里我们使用`iter`构造一个Python迭代器，并使用`next`从迭代器中获取第一项。

```{.python .input}
#@tab all
next(iter(data_iter))
```

## 定义模型

当我们在:numref:`sec_linear_scratch`中从头开始实现线性回归时，我们明确地定义了模型参数，并用基本的线性代数运算对计算进行了编码以产生输出。你应该知道怎么做。但是一旦你的模型变得更加复杂，一旦你几乎每天都要这样做，你会很高兴得到帮助的。这种情况类似于从头开始编写自己的博客。做一两次是有益的和有益的，但是如果每次你需要一个博客，你就得花一个月的时间来重新设计博客，那你就是一个糟糕的web开发人员。

对于标准操作，我们可以使用框架的预定义层，这使得我们可以特别关注用于构建模型的层，而不必关注实现。我们将首先定义一个模型变量`net`，它将引用`Sequential`类的一个实例。`Sequential`类为将链接在一起的多个层定义了一个容器。给定输入数据，`Sequential`实例将其通过第一层，然后将输出作为第二层的输入传递，依此类推。在下面的示例中，我们的模型只包含一个层，所以我们实际上不需要`Sequential`。但是，由于几乎所有未来的模型都将涉及多个层，所以我们无论如何都会使用它来熟悉最标准的工作流。

回想一下单层网络的体系结构，如:numref:`fig_single_neuron`所示。该层被称为“完全连接”，因为它的每个输入通过矩阵向量乘法连接到每个输出。

:begin_tab:`mxnet`
在Glon中，完全连接层定义在`Dense`类中。由于我们只想生成一个标量输出，所以我们将这个数字设置为1。

值得注意的是，为了方便起见，胶子不需要我们为每个层指定输入形状。所以在这里，我们不需要告诉胶子有多少输入进入这个线性层。当我们第一次尝试通过我们的模型传递数据时，例如，当我们稍后执行`net(X)`时，胶子将自动推断每个层的输入数量。稍后我们将更详细地描述这是如何工作的。
:end_tab:

:begin_tab:`pytorch`
在PyToch中，完全连接层定义为73229365类。注意，我们将两个参数传递到`nn.Linear`中。第一个标注指定输入特征标注，即2，第二个是输出特征标注，输出特征标注是单个标量，因此为1。
:end_tab:

:begin_tab:`tensorflow`
在Keras中，完全连接层在`Dense`类中定义。因为我们只想生成一个标量输出，所以我们将这个数字设置为1。

值得注意的是，为了方便起见，Keras不要求我们为每个层指定输入形状。所以在这里，我们不需要告诉Keras有多少输入进入这个线性层。当我们第一次尝试通过模型传递数据时，例如，当我们稍后执行`net(X)`时，Keras将自动推断每个层的输入数量。稍后我们将更详细地描述这是如何工作的。
:end_tab:

```{.python .input}
# `nn` is an abbreviation for neural networks
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))
```

```{.python .input}
#@tab pytorch
# `nn` is an abbreviation for neural networks
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
```

```{.python .input}
#@tab tensorflow
# `keras` is the high-level API for TensorFlow
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1))
```

## 初始化模型参数

在使用`net`之前，我们需要初始化模型参数，例如线性回归模型中的权重和偏差。深度学习框架通常有一种预定义的方法来初始化参数。在这里，我们指定每个权重参数应该从平均值为0，标准偏差为0.01的正态分布中随机抽样。bias参数将初始化为零。

:begin_tab:`mxnet`
我们将从MXNet导入`initializer`模块。本模块提供了多种模型参数初始化方法。Gluon使`init`成为访问`initializer`包的快捷方式（缩写）。我们只通过调用`init.Normal(sigma=0.01)`来指定如何初始化权重。默认情况下，偏移参数初始化为零。
:end_tab:

:begin_tab:`pytorch`
正如我们在构造`nn.Linear`时指定的输入和输出维度一样。现在我们直接访问参数来指定它们的初始值。我们首先通过`net[0]`（网络中的第一层）定位该层，然后使用`weight.data`和`bias.data`方法来获取参数。接下来，我们使用替换方法`normal_`和`fill_`覆盖参数值。
:end_tab:

:begin_tab:`tensorflow`
TensorFlow中的`initializers`模块提供了各种模型参数初始化方法。在Keras中指定初始化方法的最简单方法是通过指定`kernel_initializer`来创建层。在这里我们再次重现`net`。
:end_tab:

```{.python .input}
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```

```{.python .input}
#@tab tensorflow
initializer = tf.initializers.RandomNormal(stddev=0.01)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))
```

:begin_tab:`mxnet`
上面的代码看起来很简单，但是您应该注意到这里发生了一些奇怪的事情。我们正在初始化一个网络的参数，即使胶子还不知道输入有多少维！在我们的例子中可能是2，也可能是2000。胶子可以让我们摆脱这种情况，因为在场景后面，初始化实际上是延迟的。只有当我们第一次尝试通过网络传递数据时，才会进行真正的初始化。只需小心记住，由于参数尚未初始化，我们无法访问或操作它们。
:end_tab:

:begin_tab:`pytorch`

:end_tab:

:begin_tab:`tensorflow`
上面的代码看起来很简单，但是您应该注意到这里发生了一些奇怪的事情。我们正在初始化一个网络的参数，即使Keras还不知道输入有多少维！在我们的例子中可能是2，也可能是2000。Keras让我们摆脱了这个问题，因为在幕后，初始化实际上是*延迟*的。只有当我们第一次尝试通过网络传递数据时，才会进行真正的初始化。只需小心记住，由于参数尚未初始化，我们无法访问或操作它们。
:end_tab:

## 定义损失函数

:begin_tab:`mxnet`
在胶子中，`loss`模块定义了各种损耗函数。在这个例子中，我们将使用平方损失的胶子实现（`L2Loss`）。
:end_tab:

:begin_tab:`pytorch`
`MSELoss`类计算均方误差，也称为平方$L_2$范数。默认情况下，它返回示例的平均损失。
:end_tab:

:begin_tab:`tensorflow`
`MeanSquaredError`类计算均方误差，也称为平方$L_2$范数。默认情况下，它返回示例的平均损失。
:end_tab:

```{.python .input}
loss = gluon.loss.L2Loss()
```

```{.python .input}
#@tab pytorch
loss = nn.MSELoss()
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.MeanSquaredError()
```

## 定义优化算法

:begin_tab:`mxnet`
小批量随机梯度下降是优化神经网络的一个标准工具，因此胶子通过其`Trainer`类支持该算法的许多变体。当我们实例化`Trainer`时，我们将指定要优化的参数（可通过`net.collect_params()`从模型`net`获得）、我们希望使用的优化算法（`sgd`）和优化算法所需的超参数字典。小批量随机梯度下降只需要我们设置值`learning_rate`，这里设置为0.03。
:end_tab:

:begin_tab:`pytorch`
小批量随机梯度下降是优化神经网络的一个标准工具，因此Pythorch在`optim`模块中支持该算法和该算法的许多变体。当我们实例化一个`SGD`实例时，我们将指定要优化的参数（可通过`net.parameters()`从我们的网络获得），并提供优化算法所需的超参数字典。小批量随机梯度下降只需要我们设置值`lr`，这里设置为0.03。
:end_tab:

:begin_tab:`tensorflow`
Minibatch随机梯度下降是优化神经网络的标准工具，因此Keras在`optimizers`模块中支持该工具以及该算法的许多变体。小批量随机梯度下降只需要我们设置值`learning_rate`，这里设置为0.03。
:end_tab:

```{.python .input}
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=0.03)
```

## 培训

您可能已经注意到，通过深度学习框架的高级api来表达我们的模型需要相对较少的代码行。我们不需要单独分配参数，定义我们的损失函数，或实现小批量随机梯度下降。一旦我们开始使用更复杂的模型，高级api的优势将大大增加。然而，一旦我们把所有的基本部分都准备好了，训练循环本身就与我们从头开始实现所有东西时所做的非常相似。

为了刷新您的记忆：对于一些时代，我们将对数据集（`train_data`）进行一次完整的传递，迭代地获取一小批输入和相应的基本真理标签。对于每一个小批量，我们都要经过以下程序：

* 通过调用`net(X)`生成预测，并计算损失`l`（前向传播）。
* 通过运行反向传播计算梯度。
* 通过调用我们的优化器来更新模型参数。

为了更好地测量，我们计算每个历元之后的损耗并打印出来以监视进度。

```{.python .input}
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l.mean().asnumpy():f}')
```

```{.python .input}
#@tab pytorch
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

```{.python .input}
#@tab tensorflow
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with tf.GradientTape() as tape:
            l = loss(net(X, training=True), y)
        grads = tape.gradient(l, net.trainable_variables)
        trainer.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

下面，我们将在有限数据上训练得到的模型参数与生成数据集的实际参数进行比较。为了访问参数，我们首先从`net`访问我们需要的层，然后访问该层的权重和偏差。在我们从头开始的实现中，请注意，我们估计的参数与它们对应的基本真实值非常接近。

```{.python .input}
w = net[0].weight.data()
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
b = net[0].bias.data()
print(f'error in estimating b: {true_b - b}')
```

```{.python .input}
#@tab pytorch
w = net[0].weight.data
print('error in estimating w:', true_w - d2l.reshape(w, true_w.shape))
b = net[0].bias.data
print('error in estimating b:', true_b - b)
```

```{.python .input}
#@tab tensorflow
w = net.get_weights()[0]
print('error in estimating w', true_w - d2l.reshape(w, true_w.shape))
b = net.get_weights()[1]
print('error in estimating b', true_b - b)
```

## 摘要

:begin_tab:`mxnet`
* 使用胶子，我们可以更简洁地实现模型。
* 在胶子中，`data`模块提供数据处理工具，`nn`模块定义了大量的神经网络层，`loss`模块定义了许多常见的损耗函数。
* MXNet的模块`initializer`提供了各种模型参数初始化方法。
* 维度和存储是自动推断的，但在初始化参数之前，请小心不要尝试访问参数。
:end_tab:

:begin_tab:`pytorch`
* 使用PyTorch的高级api，我们可以更简洁地实现模型。
* 在Pythorch中，`data`模块提供数据处理工具，`nn`模块定义了大量的神经网络层和常见的损耗函数。
* 我们可以通过用以`_`结尾的方法替换参数来初始化参数。
:end_tab:

:begin_tab:`tensorflow`
* 使用TensorFlow的高级API，我们可以更简洁地实现模型。
* 在TensorFlow中，`data`模块提供数据处理工具，`keras`模块定义了大量的神经网络层和常见的损耗函数。
* TensorFlow的模块`initializers`提供了各种模型参数初始化方法。
* 维度和存储是自动推断出来的（但是要注意不要试图在初始化参数之前访问它们）。
:end_tab:

## 练习

:begin_tab:`mxnet`
1. 如果将`l = loss(output, y)`替换为`l = loss(output, y).mean()`，则需要将`trainer.step(batch_size)`更改为`trainer.step(1)`，以便代码的行为相同。为什么？
1. 查看MXNet文档，了解模块`gluon.loss`和`init`中提供了哪些丢失函数和初始化方法。用胡贝尔的损失来代替损失。
1. 你怎样才能到达`dense.weight`的坡度？

[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
1. 如果我们用`nn.MSELoss(reduction='sum')`替换`nn.MSELoss(reduction='sum')`，我们如何改变代码的学习速率，使其行为相同呢。为什么？
1. 查看Pythorch文档，了解提供了哪些丢失函数和初始化方法。用胡贝尔的损失来代替损失。
1. 你怎样才能到达`net[0].weight`的坡度？

[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
1. 查看TensorFlow文档，了解提供了哪些损失函数和初始化方法。用胡贝尔的损失来代替损失。

[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:
