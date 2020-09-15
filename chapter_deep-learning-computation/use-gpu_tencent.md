# GPU
:label:`sec_use_gpu`

在:numref:`tab_intro_decade`，我们讨论了计算在过去二十年中的快速增长。简而言之，从2000年开始，GPU性能每十年提升1000倍。这提供了很好的机会，但也表明需要提供这样的性能。

在本节中，我们将开始讨论如何为您的研究利用这种计算性能。首先使用单个GPU，稍后介绍如何使用多个GPU和多个服务器(具有多个GPU)。

具体来说，我们将讨论如何使用单个NVIDIA GPU进行计算。首先，确保您至少安装了一个NVIDIA GPU。然后，下载[NVIDIA driver and CUDA](https://developer.nvidia.com/cuda-downloads)并按照提示设置适当的路径。一旦这些准备工作完成，就可以使用`nvidia-smi`命令查看图形卡信息。

```{.python .input}
#@tab all
!nvidia-smi
```

:begin_tab:`mxnet`
您可能已经注意到，MXnet张量看起来与Numpy `ndarray`几乎相同。但有几个关键的不同之处。MXNet与NumPy的主要区别之一是它支持不同的硬件设备。

在MXNet中，每个数组都有一个上下文。到目前为止，默认情况下，所有变量和相关计算都已分配给CPU。通常，其他环境可能是各种GPU。当我们在多台服务器上部署作业时，情况可能会变得更加复杂。通过智能地将数组分配给上下文，我们可以最大限度地减少在设备之间传输数据所花费的时间。例如，当在使用GPU的服务器上训练神经网络时，我们通常希望模型的参数驻留在GPU上。

接下来，我们需要确认是否安装了MXNet的GPU版本。如果已经安装了MXNet的CPU版本，我们需要先卸载它。例如，使用`pip uninstall mxnet`命令，然后根据您的CUDA版本安装相应的MXnet版本。假设您已经安装了CUDA10.0，您可以通过`pip install mxnet-cu100`安装支持CUDA10.0的MXnet版本。
:end_tab:

:begin_tab:`pytorch`
在PyTorch中，每个数组都有一个设备，我们通常将其称为上下文。到目前为止，默认情况下，所有变量和相关计算都已分配给CPU。通常，其他环境可能是各种GPU。当我们在多台服务器上部署作业时，情况可能会变得更加复杂。通过智能地将数组分配给上下文，我们可以最大限度地减少在设备之间传输数据所花费的时间。例如，当在使用GPU的服务器上训练神经网络时，我们通常希望模型的参数驻留在GPU上。

接下来，我们需要确认是否安装了PyTorch的GPU版本。如果已经安装了PyTorch的CPU版本，我们需要先卸载它。例如，使用`pip uninstall torch`命令，然后根据您的CUDA版本安装相应的PyTorch版本。假设您已经安装了CUDA10.0，您可以通过`pip install torch-cu100`安装支持CUDA10.0的PyTorch版本。
:end_tab:

要运行本节中的程序，您至少需要两个GPU。请注意，对于大多数台式计算机来说，这可能过于奢侈，但在云中很容易获得，例如，通过使用AWS EC2多GPU实例。几乎所有其他部分都不需要多个GPU。相反，这只是为了说明数据如何在不同设备之间流动。

## 计算设备

我们可以指定存储和计算的设备，如CPU和GPU。默认情况下，张量在主内存中创建，然后使用CPU进行计算。

:begin_tab:`mxnet`
在mxnet中，cpu和gpu可以用`cpu()`和`gpu()`表示。需要注意的是，`cpu()`(或括号中的任意整数)表示所有物理CPU和内存。这意味着MXNet的计算将尝试使用所有CPU核心。但是，`gpu()`仅代表一张卡和相应的内存。如果有多个GPU，我们使用`gpu(i)`表示$i^\mathrm{th}$个GPU($i$从0开始)。另外，`gpu(0)`和`gpu()`是等效的。
:end_tab:

:begin_tab:`pytorch`
在PyTorch中，cpu和gpu可以用`torch.device('cpu')`和`torch.cuda.device('cuda')`表示。需要注意的是，`cpu`设备指的是所有物理CPU和内存。这意味着PyTorch的计算将尝试使用所有CPU核心。但是，`gpu`设备仅代表一张卡和相应的内存。如果有多个GPU，我们使用`torch.cuda.device(f'cuda:{i}')`表示$i^\mathrm{th}$个GPU($i$从0开始)。另外，`gpu:0`和`gpu`是等效的。
:end_tab:

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

npx.cpu(), npx.gpu(), npx.gpu(1)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

torch.device('cpu'), torch.cuda.device('cuda'), torch.cuda.device('cuda:1')
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

tf.device('/CPU:0'), tf.device('/GPU:0'), tf.device('/GPU:1')
```

我们可以查询可用GPU的数量。

```{.python .input}
npx.num_gpus()
```

```{.python .input}
#@tab pytorch
torch.cuda.device_count()
```

```{.python .input}
#@tab tensorflow
len(tf.config.experimental.list_physical_devices('GPU'))
```

现在，我们定义了两个方便的函数，这两个函数允许我们在请求的GPU不存在的情况下运行代码。

```{.python .input}
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    return npx.gpu(i) if npx.num_gpus() >= i + 1 else npx.cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu()] if no GPU exists."""
    devices = [npx.gpu(i) for i in range(npx.num_gpus())]
    return devices if devices else [npx.cpu()]

try_gpu(), try_gpu(10), try_all_gpus()
```

```{.python .input}
#@tab pytorch
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()
```

```{.python .input}
#@tab tensorflow
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    devices = [tf.device(f'/GPU:{i}') for i in range(num_gpus)]
    return devices if devices else [tf.device('/CPU:0')]

try_gpu(), try_gpu(10), try_all_gpus()
```

## 张量和GPU

默认情况下，在CPU上创建张量。我们可以查询张量所在的设备。

```{.python .input}
x = np.array([1, 2, 3])
x.ctx
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1, 2, 3])
x.device
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1, 2, 3])
x.device
```

需要注意的是，每当我们想要在多个终端上运行时，它们都需要在同一设备上运行。例如，如果我们将两个张量相加，我们需要确保这两个参数位于同一设备上-否则框架将不知道将结果存储在哪里，甚至不知道如何决定在哪里执行计算。

### GPU上的存储

有几种方法可以在GPU上存储张量。例如，我们可以在创建张量时指定存储设备。接下来，我们在第一个`X`上创建张量变量`gpu`。在GPU上创建的张量仅消耗此GPU的内存。我们可以使用`nvidia-smi`命令查看gpu内存使用情况。通常，我们需要确保不会创建超过GPU内存限制的数据。

```{.python .input}
X = np.ones((2, 3), ctx=try_gpu())
X
```

```{.python .input}
#@tab pytorch
X = torch.ones(2, 3, device=try_gpu())
X
```

```{.python .input}
#@tab tensorflow
with try_gpu():
    X = tf.ones((2, 3))
X
```

假设您至少有两个GPU，下面的代码将在第二个GPU上创建一个随机张量。

```{.python .input}
Y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
Y
```

```{.python .input}
#@tab pytorch
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Y = tf.random.uniform((2, 3))
Y
```

### 正在复制

如果我们想要计算`X + Y`，我们需要决定在哪里执行此操作。例如，如:numref:`fig_copyto`所示，我们可以将`X`转移到第二个图形处理器并在那里执行操作。
*不要*简单地将`X`和`Y`相加，
因为这将导致异常。运行时引擎将不知道该做什么：它无法在同一设备上找到数据，因此会失败。由于`Y`在第二个图形处理器上运行，我们需要将`X`移到那里，然后才能将这两个GPU相加。

![Copy data to perform an operation on the same device.](../img/copyto.svg)
:label:`fig_copyto`

```{.python .input}
Z = X.copyto(try_gpu(1))
print(X)
print(Z)
```

```{.python .input}
#@tab pytorch
Z = X.cuda(1)
print(X)
print(Z)
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Z = X
print(X)
print(Z)
```

现在数据都在同一个图形处理器上(`Z`和`Y`都是)，我们可以把它们加起来了。

```{.python .input}
#@tab all
Y + Z
```

:begin_tab:`mxnet`
假设您的变量`Z`已经存在于您的第二个图形处理器上。如果我们还是打`Z.copyto(gpu(1))`怎么办？即使变量已经驻留在所需设备上，它也会复制并分配新内存。有时，根据我们的代码运行环境的不同，同一设备上可能已经存在两个变量。因此，我们只想在变量当前位于不同的设备中时进行复制。在这种情况下，我们可以拨打`as_in_ctx`。如果变量已经存在于指定的设备中，则这是一个无操作。除非您特别想复制一份，否则`as_in_ctx`是您的首选方法。
:end_tab:

:begin_tab:`pytorch`
假设您的变量`Z`已经存在于您的第二个图形处理器上。如果我们还是打`Z.cuda(1)`怎么办？它将返回`Z`，而不是复制并分配新内存。
:end_tab:

:begin_tab:`tensorflow`
假设您的变量`Z`已经存在于您的第二个图形处理器上。如果我们仍然在同一设备范围内呼叫`Z2 = Z`，会发生什么情况？它将返回`Z`，而不是复制并分配新内存。
:end_tab:

```{.python .input}
Z.as_in_ctx(try_gpu(1)) is Z
```

```{.python .input}
#@tab pytorch
Z.cuda(1) is Z
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Z2 = Z
Z2 is Z
```

### 附注

人们使用GPU进行机器学习，因为他们希望它们速度快。但在设备之间传输变量的速度很慢。所以在我们让你做之前，我们希望你百分之百确定你想做一些缓慢的事情。如果深度学习框架只是自动复制而不崩溃，那么您可能不会意识到您已经编写了一些速度较慢的代码。

此外，在设备(CPU、GPU和其他计算机)之间传输数据比计算慢得多。这也使得并行化变得更加困难，因为我们必须等待数据被发送(或者更确切地说，是被接收)，然后才能继续进行更多的操作。这就是为什么复制操作应该非常小心的原因。根据经验，许多小手术比一次大手术要糟糕得多。此外，除非您知道自己在做什么，否则一次执行几个操作要比代码中散布的许多单个操作要好得多。情况就是这样，因为如果一个设备必须等待另一个设备才能做其他事情，这样的操作可以挡路。这有点像排队订购咖啡，而不是通过电话预购，然后发现它在你准备好的时候已经准备好了。

最后，当我们打印张量或将张量转换为NumPy格式时，如果数据不在主内存中，框架将首先将其复制到主内存，从而导致额外的传输开销。更糟糕的是，它现在受到可怕的全局解释器锁的影响，这使得一切都要等待Python完成。

## 神经网络和GPU

同样，神经网络模型可以指定设备。下面的代码将模型参数放在GPU上。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
```

```{.python .input}
#@tab tensorflow
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)])
```

在接下来的章节中，我们将看到更多如何在GPU上运行模型的示例，因为它们将变得更加计算密集。

当输入是GPU上的张量时，模型将在同一GPU上计算结果。

```{.python .input}
#@tab all
net(X)
```

让我们确认模型参数存储在同一GPU上。

```{.python .input}
net[0].weight.data().ctx
```

```{.python .input}
#@tab pytorch
net[0].weight.data.device
```

```{.python .input}
#@tab tensorflow
net.layers[0].weights[0].device, net.layers[0].weights[1].device
```

简而言之，只要所有的数据和参数都在同一设备上，我们就可以有效地学习模型。在接下来的章节中，我们将看到几个这样的例子。

## 摘要

* 我们可以指定用于存储和计算的设备，例如CPU或GPU。默认情况下，数据在主内存中创建，然后使用CPU进行计算。
* 深度学习框架要求所有用于计算的输入数据都在同一设备上，无论是CPU还是相同的GPU。
* 不经意地移动数据可能会显著降低性能。一个典型的错误如下：计算GPU上每个小批量的损失，并在命令行上将其报告给用户(或将其记录在NumPy `ndarray`中)将触发全局解释器锁，从而使所有GPU停滞。最好是为GPU内部的日志记录分配内存，并且只移动较大的日志。

## 练习

1. 尝试执行更大的计算任务，例如大型矩阵的乘法，看看CPU和GPU之间的速度差异。如果是一个计算量很小的任务呢？
1. 我们应该如何在GPU上读写模型参数？
1. 测量计算$100 \times 100$个矩阵的1,000个矩阵-矩阵乘法所需的时间，并记录输出矩阵的弗罗贝尼乌斯范数，一次记录一个结果，而不是在图形处理器上保存日志，只传输最终结果。
1. 测量在两个GPU上同时执行两个矩阵-矩阵乘法运算所需的时间，而不是在一个GPU上按顺序执行两个矩阵-矩阵乘法运算所需的时间。提示：您应该看到近乎线性的缩放。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/62)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/63)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/270)
:end_tab:
