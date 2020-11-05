# 数据操作
:label:`sec_ndarray`

为了做任何事情，我们需要一些方法来存储和操作数据。一般来说，我们需要对数据做两件重要的事情：(I)获取它们；(Ii)一旦它们进入计算机，就对它们进行处理。如果没有某种方法来存储数据，那么获取数据是没有意义的，所以让我们先通过使用合成数据来弄脏我们的手吧。首先，我们引入$n$维数组，也称为*张量*。

如果您使用过NumPy，这是Python中使用最广泛的科学计算包，那么您会发现这一节很熟悉。无论你使用哪种框架，它的*张量类*(在MXnet中是`ndarray`，在PyTorch和TensorFlow中都是`Tensor`)与NumPy的`ndarray`相似，但有一些致命的特性。首先，GPU支持加速计算，而NumPy只支持CPU计算。其次，张量类支持自动微分。这些性质使得张量类适合于深度学习。在整本书中，当我们说张量时，我们指的是张量类的实例，除非另有说明。

## 快速入门

在本节中，我们的目标是让您上手并运行，为您配备基本的数学和数值计算工具，在您阅读本书的过程中，这些工具将成为您的基础。如果您很难理解一些数学概念或库函数，请不要担心。接下来的几节将在实际例子的背景下重新讨论这一材料，它将会沉没。另一方面，如果您已经有了一些背景知识，并且想要更深入地了解数学内容，只需跳过这一节即可。

:begin_tab:`mxnet`
首先，我们从MXnet导入`np`(`numpy`)和`npx`(`numpy_extension`)模块。这里，`np`模块包括NumPy支持的功能，而`npx`模块包含一组扩展，这些扩展是为了在类似NumPy的环境中支持深度学习而开发的。在使用张量时，我们几乎总是调用`set_np`函数：这是为了兼容MXnet的其他组件进行的张量处理。
:end_tab:

:begin_tab:`pytorch`
首先，我们进口`torch`。请注意，虽然它名为PyTorch，但我们应该导入`torch`，而不是`pytorch`。
:end_tab:

:begin_tab:`tensorflow`
首先，我们进口`tensorflow`。因为名字有点长，我们经常用简短的别名`tf`导入。
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

张量表示数值的(可能是多维的)数组。对于一个轴，张量(在数学上)对应于一个*矢量*。有两个轴，一个张量对应一个*矩阵*。具有两个以上轴的张量没有特殊的数学名称。

首先，我们可以使用`arange`创建一个行向量`x`，其中包含从0开始的前12个整数，尽管它们在默认情况下创建为浮点数。张量中的每个值都称为张量的*元素*。例如，在张量`x`中有12个元素。除非另有说明，否则新的张量将存储在主存储器中，并指定用于基于CPU的计算。

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

我们可以通过检查张量的`shape`性质来了解张量的*形状*(沿每个轴的长度)。

```{.python .input}
#@tab all
x.shape
```

如果我们只想知道张量中元素的总数，即所有形状元素的乘积，我们可以检查它的大小。因为我们在这里处理的是一个向量，所以它的`shape`的单个元素与它的大小相同。

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

要更改张量的形状而不更改元素的数量或它们的值，我们可以调用`reshape`函数。例如，我们可以将张量`x`从形状为(12，)的行向量转换为形状为(3，4)的矩阵。这个新张量包含完全相同的值，但将它们视为组织为3行4列的矩阵。我要重申的是，虽然形状已经改变，但元素并没有改变。请注意，大小不会因重塑而改变。

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

无需通过手动指定每个尺寸来重塑形状。如果我们的目标形状是一个具有形状(高度、宽度)的矩阵，那么在我们知道宽度之后，高度就是隐式给定的。为甚麽我们要自己进行分组表决呢？在上面的示例中，要获得一个有3行的矩阵，我们指定它应该有3行4列。幸运的是，给定睡觉，张量可以自动计算出一维。我们通过为希望张量自动推断的维度放置`-1`来调用此功能。在我们的例子中，我们可以等效地拨打`x.reshape(-1, 4)`或`x.reshape(3, -1)`，而不是拨打`x.reshape(3, 4)`。

通常，我们希望使用0、1、其他一些常量或从特定分布中随机抽样的数字来初始化矩阵。我们可以创建一个张量，该张量表示所有元素都设置为0且形状为(2，3，4)的张量，如下所示：

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

类似地，我们可以创建每个元素设置为1的张量，如下所示：

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

通常，我们希望从某个概率分布中随机抽样张量中每个元素的值。例如，当我们构造数组作为神经网络中的参数时，我们通常会随机初始化它们的值。下面的代码片断创建一个形状为(3，4)的张量。它的每个元素都是从标准高斯(正态)分布中随机抽样的，平均值为0，标准差为1。

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

我们还可以通过提供包含数值的Python列表(或列表列表)来指定所需张量中每个元素的确切值。这里，最外层的列表对应于轴0，内部列表对应于轴1。

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

## 运营

这本书不是关于软件工程的。我们的兴趣并不局限于简单地从数组读取数据和向数组写入数据。我们希望对这些数组执行数学运算。一些最简单、最有用的操作是“基于元素”的操作。它们将标准标量操作应用于数组的每个元素。对于将两个数组作为输入的函数，按元素操作会对两个数组中的每对对应元素应用一些标准二元运算符。我们可以从任何从标量映射到标量的函数中创建元素级函数。

在数学表示法中，我们将用签名$f: \mathbb{R} \rightarrow \mathbb{R}$表示这样的*一元*标量运算符(接受一个输入)。这仅仅意味着该函数正在从任何实数($\mathbb{R}$)映射到另一个实数。同样，我们用签名$f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$表示*二进制*标量运算符(取两个实数输入，并产生一个输出)。给定具有相同形状*的任意两个矢量$\mathbf{u}$和$\mathbf{v}$*以及二元运算符$f$，我们可以通过将矢量$\mathbf{c} = F(\mathbf{u},\mathbf{v})$设置为全部$i$来产生矢量$i$，其中$c_i, u_i$和$v_i$是矢量$\mathbf{c}, \mathbf{u}$和$\mathbf{v}$的$i^\mathrm{th}$个元素。在这里，我们通过将标量函数“提升”到一个元素向量运算来产生向量值$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$。

常用的标准算术运算符(`+`、`-`、`*`、`/`和`**`)都被*提升*为任意形状的同形张量的基本运算。我们可以对形状相同的任意两个张量调用单元化运算。在下面的示例中，我们使用逗号来表示一个由5个元素组成的元组，其中每个元素都是逐个元素操作的结果。

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

可以按元素应用更多的运算，包括像求幂这样的一元运算符。

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

除了元素计算之外，我们还可以执行线性代数运算，包括向量点积和矩阵乘法。我们将在:numref:`sec_linear-algebra`中解释线性代数的关键部分(没有假定的先验知识)。

我们还可以将多个张量“串联”在一起，将它们端到端地堆叠起来，形成一个更大的张量。我们只需要提供张量列表，并告诉系统沿哪个轴连接即可。下面的示例显示了沿行(轴0，形状的第一个元素)与列(轴1，形状的第二个元素)连接两个矩阵时发生的情况。我们可以看到，第一个输出张量的0轴长度($6$)是两个输入张量的0轴长度($3 + 3$)的总和；而第二个输出张量的1轴长度($8$)是两个输入张量的1轴长度($4 + 4$)的总和。

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

有时，我们想要通过*逻辑语句*来构造二元张量。以`X == Y`为例。对于每个位置，如果`X`和`Y`在该位置相等，则新张量中的对应条目取值1，这意味着逻辑语句`X == Y`在该位置为真；否则该位置为0。

```{.python .input}
#@tab all
X == Y
```

将张量中的所有元素相加得到一个只有一个元素的张量。

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

在上面的部分中，我们了解了如何对相同形状的两个张量执行元素级运算。在一定条件下，即使形状不同，我们仍然可以通过调用*广播机制*来执行元素级操作。该机制的工作方式如下：首先，通过适当复制元素来扩展一个或两个数组，以便在此转换之后，两个张量具有相同的形状。其次，对生成的数组执行元素级操作。

在大多数情况下，我们沿着数组初始长度仅为1的轴进行广播，如下例所示：

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

因为`a`和`b`分别是$3\times1$和$1\times2$个矩阵，所以如果我们要将它们相加，它们的形状是不匹配的。我们将两个矩阵的条目广播到更大的$3\times2$矩阵中，如下所示：对于矩阵`a`，它复制列，并且对于矩阵`b`，它在将这两个元素相加之前复制行。

```{.python .input}
#@tab all
a + b
```

## 索引和切片

就像在任何其他Python数组中一样，张量中的元素可以通过索引访问。与在任何Python数组中一样，第一个元素的索引为0，并且指定范围以包括第一个但*在*最后一个元素之前。与标准Python列表一样，我们可以使用负索引根据元素到列表末尾的相对位置来访问元素。

因此，`[-1]`选择最后一个元素，`[1:3]`选择第二个和第三个元素，如下所示：

```{.python .input}
#@tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
除了读取之外，我们还可以通过指定索引来写入矩阵的元素。
:end_tab:

:begin_tab:`tensorflow`
TensorFlow中的`Tensors`是不可变的，不能赋值给。TensorFlow中的`Variables`是支持赋值的状态的可变容器。请记住，TensorFlow中的渐变不会在`Variable`个指定中向后流动。

除了为整个`Variable`赋值之外，我们还可以通过指定索引来编写`Variable`的元素。
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

如果我们想给多个元素分配相同的值，我们只需为所有元素建立索引，然后为它们赋值。例如，`[0:2, :]`访问第一行和第二行，其中`:`获取沿轴1(列)的所有元素。虽然我们讨论了矩阵的索引，但这显然也适用于超过2维的向量和张量。

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

运行操作可能会导致将新内存分配给宿主结果。例如，如果我们编写`Y = X + Y`，我们将取消引用`Y`用来指向的张量，而是将`Y`指向新分配的内存。在下面的示例中，我们使用Python的`id()`函数演示了这一点，该函数给出了被引用对象在内存中的确切地址。运行`Y = Y + X`之后，我们会发现`id(Y)`指向不同的位置。这是因为Python首先计算`Y + X`，为结果分配新的内存，然后使`Y`指向内存中的这个新位置。

```{.python .input}
#@tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

这可能是不受欢迎的，原因有两个。首先，我们不想一直在不必要的情况下到处分配内存。在机器学习中，我们可能拥有数百兆字节的参数，并且每秒多次更新所有参数。通常，我们希望“就地”执行这些更新。其次，我们可以从多个变量指向相同的参数。如果我们不就地更新，其他引用仍将指向旧的内存位置，从而使我们的部分代码有可能无意中引用陈旧的参数。

:begin_tab:`mxnet, pytorch`
幸运的是，执行就地操作很容易。我们可以将操作结果分配给具有切片记数法(例如，`Y[:] = <expression>`)的先前分配的数组。为了说明这一概念，我们首先创建一个新矩阵`Z`，它具有与另一个`Y`相同的形状，使用`zeros_like`来分配$0$个条目的挡路。
:end_tab:

:begin_tab:`tensorflow`
`Variables`是TensorFlow中状态的可变容器。它们提供了一种存储模型参数的方法。我们可以将运算结果赋给`Variable`和`assign`。为了说明这个概念，我们创建了一个`Variable` `Z`，它与另一个张量`Y`具有相同的形状，使用`zeros_like`来分配$0$个条目的挡路。
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
如果值`X`在后续计算中没有重复使用，我们还可以使用`X[:] = X + Y`或`X += Y`来减少操作的内存开销。
:end_tab:

:begin_tab:`tensorflow`
即使您将状态持久地存储在`Variable`中，您也可能希望通过避免为不是模型参数的张量分配过多的空间来进一步减少内存使用量。

由于TensorFlow `Tensors`是不可变的，并且渐变不会流经`Variable`赋值，因此TensorFlow不提供就地运行单个操作的显式方法。

但是，TensorFlow提供了`tf.function`修饰器来将计算包装在TensorFlow图中，该图在运行前经过编译和优化。这允许TensorFlow修剪未使用的值，并重用不再需要的先前分配。这最大限度地减少了TensorFlow计算的内存开销。
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

转换为NumPy张量很容易，反之亦然。转换的结果不共享内存。这一微小的不便实际上相当重要：当您在CPU或GPU上执行操作时，您不想暂停计算，等待看Python的NumPy包是否想要使用相同的内存块执行其他操作。

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

要将大小为1的张量转换为Python标量，我们可以调用`item`函数或Python的内置函数。

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

* 存储和操作深度学习数据的主要界面是张量($n$维数组)。它提供了各种功能，包括基本数学运算、广播、索引、切片、内存节省和到其他Python对象的转换。

## 练习

1. 运行本节中的代码。将本节中的条件语句`X == Y`更改为`X < Y`或`X > Y`，然后看看可以获得哪种张量。
1. 将广播机构中按元素操作的两个张量替换为其他形状，例如，三维张量。结果和预期的一样吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/187)
:end_tab:
