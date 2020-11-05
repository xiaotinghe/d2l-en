# 线性回归的简明实现
:label:`sec_linear_concise`

在过去的几年里，对深度学习的广泛而强烈的兴趣激发了公司、学者和业余爱好者开发各种成熟的开源框架，以自动化实现基于梯度的学习算法的重复工作。在:numref:`sec_linear_scratch`中，我们仅依靠(I)张量来存储数据和线性代数；以及(Ii)自动微分来计算梯度。在实践中，由于数据迭代器、损失函数、优化器和神经网络层非常常见，现代库也为我们实现了这些组件。

在本节中，我们将向您展示如何使用深度学习框架的高级API简明扼要地实现从:numref:`sec_linear_scratch`开始的线性回归模型。

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

## 正在读取数据集

我们可以调用框架中的现有API来读取数据，而不是使用我们自己的迭代器。我们传入`features`和`labels`作为参数，并在实例化数据迭代器对象时指定`batch_size`。此外，布尔值`is_train`指示我们是否希望数据迭代器对象在每个纪元上混洗数据(通过数据集)。

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

现在我们可以使用`data_iter`，其方式与我们在:numref:`sec_linear_scratch`中调用`data_iter`函数的方式大致相同。为了验证它是否正常工作，我们可以阅读并打印第一小批示例。与:numref:`sec_linear_scratch`相比，这里我们使用`iter`构造Python迭代器，并使用`next`从迭代器获取第一项。

```{.python .input}
#@tab all
next(iter(data_iter))
```

## 定义模型

当我们在:numref:`sec_linear_scratch`从头开始实施线性回归时，我们显式定义了模型参数，并使用基本的线性代数运算对计算进行编码以生成输出。你“应该”知道如何做到这一点。但是，一旦您的模型变得更加复杂，一旦您几乎每天都要这样做，您就会很高兴得到帮助。这种情况类似于从零开始编写自己的博客。做一两次是有益的，也是有教育意义的，但如果每次你需要一个博客，你都要花一个月的时间重新发明轮子，那么你就是一个糟糕的网络开发人员。

对于标准操作，我们可以使用框架的预定义层，这允许我们特别关注用于构建模型的层，而不必关注实现。我们将首先定义一个模型变量`net`，它将引用`Sequential`类的一个实例。`Sequential`类为将链接在一起的多个层定义了一个容器。给定输入数据后，`Sequential`实例将其传递给第一层，然后将输出作为第二层的输入传递，依此类推。在下面的示例中，我们的模型只包含一层，因此我们实际上不需要`Sequential`。但是，由于我们未来的几乎所有模型都将涉及多个层，我们无论如何都会使用它，只是为了让您熟悉最标准的工作流。

回想一下单层网络的体系结构，如:numref:`fig_single_neuron`中所示。这一层之所以说是“完全连接的”，是因为它的每一个输入都通过矩阵-向量乘法连接到它的每一个输出。

:begin_tab:`mxnet`
在胶子中，完全连接层在`Dense`类中定义。由于我们只想生成单个标量输出，因此将该数字设置为1。

值得注意的是，为了方便起见，GLUON不需要我们指定每一层的输入形状。所以在这里，我们不需要告诉胶子，有多少输入进入了这个线性层。当我们第一次尝试通过我们的模型传递数据时，例如，当我们稍后执行`net(X)`时，胶子会自动推断每一层的输入数量。我们稍后将更详细地描述这是如何工作的。
:end_tab:

:begin_tab:`pytorch`
在PyTorch中，完全连接层在`Linear`类中定义。请注意，我们将两个参数传递给了`nn.Linear`。第一个指定输入特征尺寸，它是2，第二个是输出特征尺寸，它是单个标量，因此是1。
:end_tab:

:begin_tab:`tensorflow`
在KERAS中，完全连接层在`Dense`类中定义。由于我们只想生成单个标量输出，因此将该数字设置为1。

值得注意的是，为了方便起见，KERAS不需要我们指定每个图层的输入形状。所以在这里，我们不需要告诉凯拉斯有多少输入进入了这个线性层。当我们第一次尝试通过我们的模型传递数据时，例如，当我们稍后执行`net(X)`时，KERAS将自动推断每一层的输入数量。我们稍后将更详细地描述这是如何工作的。
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

## 正在初始化模型参数

在使用`net`之前，我们需要初始化模型参数，例如线性回归模型中的权重和偏差。深度学习框架通常具有初始化参数的预定义方式。在这里，我们指定每个权重参数应该从均值为0且标准差为0.01的正态分布中随机抽样。偏置参数将被初始化为零。

:begin_tab:`mxnet`
我们将从MXnet导入`initializer`模块。本模块提供多种模型参数初始化方法。Gluon使`init`成为访问`initializer`包的快捷方式(缩写)。我们只通过调用`init.Normal(sigma=0.01)`来指定如何初始化权重。默认情况下，偏置参数初始化为零。
:end_tab:

:begin_tab:`pytorch`
因为我们在构造`nn.Linear`时已经指定了输入和输出尺寸。现在，我们直接访问参数以指定它们的初始值。我们首先通过`net[0]`(网络中的第一层)定位该层，然后使用`weight.data`和`bias.data`方法访问参数。接下来，我们使用替换方法`normal_`和`fill_`来覆盖参数值。
:end_tab:

:begin_tab:`tensorflow`
TensorFlow中的`initializers`模块提供了多种模型参数初始化方法。在KERAS中指定初始化方法的最简单方法是通过指定`kernel_initializer`来创建层。在这里，我们再次重现了`net`。
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
上面的代码可能看起来很简单，但是您应该注意到这里发生了一些奇怪的事情。我们正在为网络初始化参数，即使胶子还不知道输入将有多少维！它可能是我们示例中的2，也可能是2000。GLUON让我们得以逃脱惩罚，因为在幕后，初始化实际上是“延迟的”。只有当我们第一次尝试通过网络传递数据时，才会进行真正的初始化。只需小心记住，由于参数尚未初始化，我们无法访问或操作它们。
:end_tab:

:begin_tab:`pytorch`

:end_tab:

:begin_tab:`tensorflow`
上面的代码可能看起来很简单，但是您应该注意到这里发生了一些奇怪的事情。我们正在为网络初始化参数，尽管Kera还不知道输入将有多少维！它可能是我们示例中的2，也可能是2000。Kera让我们得以逃脱惩罚，因为在幕后，初始化实际上是“延迟的”。只有当我们第一次尝试通过网络传递数据时，才会进行真正的初始化。只需小心记住，由于参数尚未初始化，我们无法访问或操作它们。
:end_tab:

## 定义损失函数

:begin_tab:`mxnet`
在胶子中，`loss`模块定义了各种损耗函数。在本例中，我们将使用平方损失的胶子实现(`L2Loss`)。
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
迷你批次随机梯度下降是优化神经网络的标准工具，因此Gluon通过其`Trainer`类与该算法的许多变体一起支持它。当我们实例化`Trainer`时，我们将指定要优化的参数(可从我们的模型`net`到`net.collect_params()`获得)、我们希望使用的优化算法(`sgd`)以及我们的优化算法所需的超参数字典。小批量随机梯度下降只需要我们设置值`learning_rate`，这里设置为0.03.
:end_tab:

:begin_tab:`pytorch`
小批量随机梯度下降是优化神经网络的标准工具，因此派火炬在`optim`模块中支持它以及该算法的许多变体。当我们实例化一个`SGD`实例时，我们将使用我们的优化算法所需的超参数字典来指定要优化的参数(可以从我们的网络通过`net.parameters()`获得)。小批量随机梯度下降只需要我们设置值`lr`，这里设置为0.03.
:end_tab:

:begin_tab:`tensorflow`
小批量随机梯度下降是优化神经网络的标准工具，因此KERAS在`optimizers`模块中支持它以及该算法的许多变体。小批量随机梯度下降只需要我们设置值`learning_rate`，这里设置为0.03.
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

您可能已经注意到，通过深度学习框架的高级API表达我们的模型只需要相对较少的代码行。我们不必单独分配参数、定义损失函数或实现小批量随机梯度下降。一旦我们开始使用更复杂的模型，高级API的优势将大大增强。然而，一旦我们有了所有的基本部分，培训循环本身就与我们从头开始实现一切时所做的惊人地相似。

提醒一下您的记忆：对于一定数量的纪元，我们将对数据集(`train_data`)进行一次完整的遍历，迭代地获取一小批输入和相应的地面事实标签。对于每个小批量，我们都要经历以下仪式：

* 通过呼叫`net(X)`生成预测，并计算损失`l`(前向传播)。
* 通过运行反向传播来计算梯度。
* 通过调用我们的优化器更新模型参数。

为了更好地衡量，我们计算每个时期之后的损失，并将其打印出来以监控进度。

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

下面，我们将比较通过在有限数据上训练而学习的模型参数与生成数据集的实际参数。要访问参数，我们首先访问`net`中需要的层，然后访问该层的权重和偏移。与我们从头开始的实现一样，请注意，我们估计的参数与其对应的实际参数很接近。

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
* 在GLUON中，`data`模块提供了数据处理工具，`nn`模块定义了大量的神经网络层，`loss`模块定义了许多常见的损失函数。
* MXnet的模块`initializer`提供用于模型参数初始化的各种方法。
* 维数和存储是自动推断的，但请注意，在参数初始化之前不要尝试访问它们。
:end_tab:

:begin_tab:`pytorch`
* 使用PyTorch的高级API，我们可以更简洁地实现模型。
* 在PyTorch中，`data`模块提供了数据处理工具，`nn`模块定义了大量的神经网络层和常见的损失函数。
* 我们可以通过将参数的值替换为以`_`结尾的方法来初始化参数。
:end_tab:

:begin_tab:`tensorflow`
* 使用TensorFlow的高级API，我们可以更简洁地实现模型。
* 在TensorFlow中，`data`模块提供了数据处理工具，`keras`模块定义了大量的神经网络层和常用的损失函数。
* TensorFlow模块`initializers`提供用于模型参数初始化的各种方法。
* 维数和存储是自动推断的(但请注意，在参数初始化之前不要尝试访问它们)。
:end_tab:

## 练习

:begin_tab:`mxnet`
1. 如果我们将`l = loss(output, y)`替换为`l = loss(output, y).mean()`，则需要将`trainer.step(batch_size)`更改为`trainer.step(1)`才能使代码的行为相同。为什么？
1. 查看mxnet文档，了解模块`gluon.loss`和`init`中提供了哪些损耗功能和初始化方法。用胡伯的损失来弥补损失。
1. 如何获得`dense.weight`的梯度？

[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
1. 如果我们将`nn.MSELoss(reduction='sum')`替换为`nn.MSELoss()`，我们如何才能更改代码的学习率以使其行为相同。为什么？
1. 查看PyTorch文档，了解提供了哪些损耗函数和初始化方法。用胡伯的损失来弥补损失。
1. 如何获得`net[0].weight`的梯度？

[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
1. 查看TensorFlow文档，了解提供了哪些损耗函数和初始化方法。用胡伯的损失来弥补损失。

[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:
