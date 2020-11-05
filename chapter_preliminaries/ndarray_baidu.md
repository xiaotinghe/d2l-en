# 数据操纵
:label:`sec_ndarray`

为了完成任何事情，我们需要某种方法来存储和操作数据。一般来说，我们需要对数据做两件重要的事情：（i）获取数据；和（ii）在数据进入计算机后对其进行处理。获取数据时没有某种存储方法是没有意义的，所以让我们先用合成数据来处理。首先，我们介绍$n$维数组，也称为*张量*。

如果您使用过Python中使用最广泛的科学计算包NumPy，那么您会发现这一部分很熟悉。无论使用哪种框架，它的*tensor类*（MXNet中的`ndarray`，PyTorch和TensorFlow中的`Tensor`）都与NumPy的`ndarray`相似，但有一些致命的特性。首先，GPU很好地支持加速计算，而NumPy只支持CPU计算。其次，tensor类支持自动微分。这些特性使得张量类适合于深度学习。在整本书中，当我们说张量时，我们指的是张量类的实例，除非另有说明。

## 入门

在这一节中，我们的目标是让你开始和运行，为你装备基本的数学和数值计算工具，你将建立在你的基础上，通过本书。不要担心，如果你努力摸索一些数学概念或库函数。下面几节将结合实际例子重温这一材料，它将下沉。另一方面，如果你已经有了一些背景知识并且想更深入地了解数学内容，那就跳过这一节。

:begin_tab:`mxnet`
首先，我们从MXNet导入`np`（`numpy`）和`npx`（`numpy_extension`）模块。在这里，`np`模块包含NumPy支持的功能，而`npx`模块包含一组扩展，用于在类似NumPy的环境中实现深度学习。使用张量时，我们几乎总是调用`set_np`函数：这是为了兼容MXNet的其他组件的张量处理。
:end_tab:

:begin_tab:`pytorch`
首先，我们进口`torch`。请注意，尽管它名为PyTorch，但我们应该导入`torch`而不是`pytorch`。
:end_tab:

:begin_tab:`tensorflow`
首先，我们进口`tensorflow`。由于名称有点长，我们通常使用短别名`tf`导入它。
:end_tab:

```{.python .input}
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
```

张量表示一组（可能是多维的）数值数组。对于一个轴，张量对应于一个*向量*。对于两个轴，张量对应于一个*矩阵*。超过两个轴的张量没有特殊的数学名称。

首先，我们可以使用`arange`创建一个行向量`x`，其中包含以0开头的前12个整数，尽管它们在默认情况下是作为浮点创建的。张量中的每个值都称为张量的*元素*。例如，张量`x`中有12个元素。除非另有规定，否则新的张量将存储在主存储器中，并指定用于基于CPU的计算。

```{.python .input}
x = np.arange(12)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(12)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(12)
x
```

我们可以通过检查张量的`shape`属性来访问它的*shape*（沿每个轴的长度）。

```{.python .input}
#@tab all
x.shape
```

如果我们只想知道张量中元素的总数，即所有形状元素的乘积，我们可以检查它的大小。因为我们在这里处理的是一个向量，它的`shape`的单个元素与其大小相同。

```{.python .input}
x.size
```

```{.python .input}
#@tab pytorch
x.numel()
```

```{.python .input}
#@tab tensorflow
tf.size(x)
```

要改变张量的形状而不改变元素的数量或它们的值，我们可以调用`reshape`函数。例如，我们可以将张量`x`从形状为（12，）的行向量转换为形状为（3，4）的矩阵。这个新的张量包含完全相同的值，但将它们视为3行4列的矩阵。重申一下，虽然形状改变了，但元素没有改变。请注意，大小不会因重塑而改变。

```{.python .input}
#@tab mxnet, pytorch
X = x.reshape(3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

不需要通过手动指定每个尺寸来重塑形状。如果我们的目标形状是一个有形状（高度，宽度）的矩阵，那么在我们知道宽度之后，高度是隐式给出的。为什么要我们自己来做这个划分呢？在上面的例子中，为了得到一个包含3行的矩阵，我们同时指定它应该有3行和4列。幸运的是，张量可以自动计算出给定其他维度的一维。我们通过放置`-1`作为我们希望张量自动推断的维度来调用此功能。在我们的例子中，我们可以等价地调用`x.reshape(-1, 4)`或`x.reshape(3, -1)`，而不是调用`x.reshape(3, 4)`。

通常情况下，我们希望我们的矩阵初始化为零、一、其他一些常量或从特定分布中随机抽样的数字。我们可以创建一个表示张量的张量，所有元素都设置为0，形状为（2，3，4），如下所示：

```{.python .input}
np.zeros((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.zeros((2, 3, 4))
```

类似地，我们可以创建张量，每个元素设置为1，如下所示：

```{.python .input}
np.ones((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.ones((2, 3, 4))
```

通常情况下，我们希望从某种概率分布中随机抽取张量中每个元素的值。例如，当我们构造数组作为神经网络的参数时，我们通常会随机初始化它们的值。下面的代码片段创建了一个具有shape（3，4）的张量。它的每个元素都是从标准高斯（正态）分布中随机抽样的，平均值为0，标准偏差为1。

```{.python .input}
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
#@tab pytorch
torch.randn(3, 4)
```

```{.python .input}
#@tab tensorflow
tf.random.normal(shape=[3, 4])
```

我们还可以通过提供包含数值的Python列表（或列表列表）来指定所需张量中每个元素的精确值。这里，最外层的列表对应于轴0，而内部列表对应于轴1。

```{.python .input}
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## 操作

这本书不是关于软件工程的。我们的兴趣不仅限于从数组读写数据。我们要对这些数组执行数学运算。一些最简单和最有用的操作是*elementwise*操作。它们对数组的每个元素应用标准标量操作。对于以两个数组作为输入的函数，elementwise操作对两个数组中的每对对应元素应用一些标准的二进制运算符。我们可以从任何从标量映射到标量的函数创建元素式函数。

在数学表示法中，我们将用签名$f: \mathbb{R} \rightarrow \mathbb{R}$来表示这样一个*一元*标量运算符（接受一个输入）。这意味着函数是从任意实数（$\mathbb{R}$）映射到另一个实数。同样，我们用签名$f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$表示一个*二进制*标量运算符（取两个实输入，产生一个输出）。给定任意两个形状相同的向量$\mathbf{u}$和$\mathbf{v}$*和一个二元运算符$f$，我们可以通过为所有$i$设置$c_i \gets f(u_i, v_i)$来生成一个向量$\mathbf{c} = F(\mathbf{u},\mathbf{v})$，其中$c_i, u_i$和$v_i$是向量$\mathbf{c}, \mathbf{u}$和$\mathbf{v}$的$i^\mathrm{th}$个元素。在这里，我们通过*提升*标量函数到一个元素向量运算来生成值为$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$的向量。

对于任意形状的同形张量，常用的标准算术运算符（`+`、`-`、`*`、`/`和`**`）都进行了元素运算。我们可以对任意两个形状相同的张量进行元素运算。在下面的示例中，我们使用逗号来表示一个5元素元组，其中每个元素都是元素操作的结果。

```{.python .input}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

更多的运算可以在元素上应用，包括一元运算符，如求幂。

```{.python .input}
np.exp(x)
```

```{.python .input}
#@tab pytorch
torch.exp(x)
```

```{.python .input}
#@tab tensorflow
tf.exp(x)
```

除了元素计算，我们还可以执行线性代数运算，包括向量点积和矩阵乘法。我们将在:numref:`sec_linear-algebra`中解释线性代数的关键位（没有假定的先验知识）。

我们也可以将多个张量串联在一起，端到端地堆叠起来形成一个更大的张量。我们只需要提供一个张量列表，告诉系统沿着哪个轴连接。下面的例子展示了当我们沿着行（轴0，形状的第一个元素）和列（轴1，形状的第二个元素）连接两个矩阵时会发生什么。我们可以看到，第一个输出张量的0轴长度（$6$）是两个输入张量的0轴长度之和（$3 + 3$）；而第二个输出张量的轴1长度（$8$）是两个输入张量的轴1长度之和（$4 + 4$）。

```{.python .input}
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
#@tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

有时，我们想通过*逻辑语句*构造一个二元张量。以`X == Y`为例。对于每个位置，如果`X`和`Y`在该位置相等，则新张量中的对应项取1，这意味着逻辑语句`X == Y`在该位置为真；否则该位置取0。

```{.python .input}
#@tab all
X == Y
```

将张量中的所有元素相加得到只有一个元素的张量。

```{.python .input}
#@tab mxnet, pytorch
X.sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(X)
```

## 广播机制
:label:`subsec_broadcasting`

在上一节中，我们看到了如何对两个形状相同的张量执行元素操作。在某些情况下，即使形状不同，我们仍然可以通过调用*广播机制*来执行元素操作。这种机制的工作方式如下：首先，通过适当地复制元素来展开一个或两个数组，以便在这个转换之后，两个张量具有相同的形状。其次，对生成的数组执行元素级操作。

在大多数情况下，我们沿着一个轴进行广播，其中一个数组最初的长度只有1，例如在下面的示例中：

```{.python .input}
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
#@tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
#@tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

由于`a`和`b`分别是$3\times1$和$1\times2$矩阵，所以如果我们要添加它们，它们的形状就不匹配。我们*广播*两个矩阵的条目到一个更大的$3\times2$矩阵中，如下所示：对于矩阵`a`，它复制列，而对于矩阵`b`，它在两个元素相加之前复制行。

```{.python .input}
#@tab all
a + b
```

## 索引和切片

就像在任何其他Python数组中一样，张量中的元素可以通过索引访问。与在任何Python数组中一样，第一个元素的索引为0，并指定范围以包括第一个但在*之前*最后一个元素。与标准Python列表一样，我们可以使用负索引根据元素到列表末尾的相对位置来访问元素。

因此，`[-1]`选择最后一个元素，`[1:3]`选择第二个和第三个元素，如下所示：

```{.python .input}
#@tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
除了阅读，我们还可以通过指定索引来编写矩阵的元素。
:end_tab:

:begin_tab:`tensorflow`
TensorFlow中的`Tensors`是不可变的，不能分配给。TensorFlow中的`Variables`是支持赋值的可变状态容器。请记住，TensorFlow中的渐变不会在`Variable`作业中向后流动。

除了给整个`Variable`赋值外，我们还可以通过指定索引来编写`Variable`的元素。
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X[1, 2] = 9
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

如果我们想给多个元素指定相同的值，我们只需索引所有元素，然后给它们赋值。例如，`[0:2, :]`访问第一行和第二行，其中`:`获取沿轴1（列）的所有元素。当我们讨论矩阵的索引时，这显然也适用于向量和超过2维的张量。

```{.python .input}
#@tab mxnet, pytorch
X[0:2, :] = 12
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[0:2, :].assign(tf.ones(X_var[0:2,:].shape, dtype = tf.float32) * 12)
X_var
```

## 节省内存

运行操作可能会导致新内存分配给主机结果。例如，如果我们写`Y = X + Y`，我们将取消引用`Y`用来指向的张量，而是指向新分配的内存`Y`。在下面的示例中，我们用Python的`id()`函数演示这一点，该函数为我们提供了内存中被引用对象的确切地址。运行`Y = Y + X`之后，我们将发现`id(Y)`指向另一个位置。这是因为Python首先计算`Y + X`，为结果分配新的内存，然后让`Y`指向内存中的这个新位置。

```{.python .input}
#@tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

这可能是不可取的，有两个原因。首先，我们不想总是不必要地到处分配内存。我们每秒钟可能会更新几百兆字节的参数。通常，我们希望执行这些更新*就地*。其次，我们可以从多个变量指向相同的参数。如果不进行就地更新，其他引用仍将指向旧的内存位置，从而使部分代码可能无意中引用过时的参数。

:begin_tab:`mxnet, pytorch`
幸运的是，执行就地操作很容易。我们可以将一个操作的结果赋给一个以前分配的数组，使用切片表示法，例如`Y[:] = <expression>`。为了说明这个概念，我们首先创建一个与另一个`Y`形状相同的新矩阵`Z`，使用`zeros_like`分配一个$0$条目的块。
:end_tab:

:begin_tab:`tensorflow`
`Variables`是TensorFlow中状态的可变容器。它们提供了一种存储模型参数的方法。我们可以用`assign`将运算结果分配给`Variable`。为了说明这个概念，我们创建了一个`Variable` `Z`，其形状与另一张量`Y`相同，使用`zeros_like`分配$0$个条目的块。
:end_tab:

```{.python .input}
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

:begin_tab:`mxnet, pytorch`
如果`X`的值在随后的计算中没有被重用，我们也可以使用`X[:] = X + Y`或`X += Y`来减少操作的内存开销。
:end_tab:

:begin_tab:`tensorflow`
即使在`Variable`中持久地存储状态，也可能希望通过避免对不是模型参数的张量进行过多分配来进一步减少内存使用量。

因为TensorFlow `Tensors`是不可变的，并且梯度不会流经`Variable`赋值，所以TensorFlow不能提供一种显式的方法来就地运行单个操作。

然而，TensorFlow提供了`tf.function`修饰符，将计算封装在TensorFlow图中，该图在运行前经过编译和优化。这允许TensorFlow删除未使用的值，并重新使用不再需要的先前分配。这将最小化TensorFlow计算的内存开销。
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
#@tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # This unused value will be pruned out
    A = X + Y  # Allocations will be re-used when no longer needed
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## 转换为其他Python对象

转换成NumPy张量，或者反过来，很容易。转换后的结果不共享内存。这个小小的不便实际上是相当重要的：当您在CPU或gpu上执行操作时，您不想停止计算，等待Python的NumPy包是否希望使用相同的内存块执行其他操作。

```{.python .input}
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
#@tab pytorch
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
```

```{.python .input}
#@tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

要将size-1张量转换为Python标量，我们可以调用`item`函数或Python的内置函数。

```{.python .input}
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

## 摘要

* 存储和操作用于深度学习的数据的主要接口是张量（$n$维数组）。它提供了各种功能，包括基本的数学运算、广播、索引、切片、内存节省以及到其他Python对象的转换。

## 练习

1. 运行此部分中的代码。将本节中的条件语句`X == Y`更改为`X < Y`或`X > Y`，然后查看可以得到什么样的张量。
1. 将广播机制中由元素操作的两个张量替换为其他形状，例如三维张量。结果和预期一样吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/187)
:end_tab:
