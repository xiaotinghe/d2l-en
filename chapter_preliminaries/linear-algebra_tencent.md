# 线性代数
:label:`sec_linear-algebra`

现在您可以存储和操作数据了，让我们简要回顾一下理解和实现本书中涵盖的大多数模型所需的基本线性代数的子集。下面，我们将介绍线性代数中的基本数学对象、算术和运算，并用数学符号和相应的代码实现来表示它们。

## 标量

如果你从未学习过线性代数或机器学习，那么你过去的数学经历可能就是一次只想一个数字。而且，如果你曾经平衡过支票簿，甚至在餐厅付过晚餐的钱，那么你已经知道如何做一些基本的事情，比如把成对的数字相加和相乘。例如，帕洛阿尔托的温度是华氏$52$度。形式上，我们称只包含一个数值的值为标量。如果要将该值转换为摄氏度(公制系统更合理的温度刻度)，则可以计算表达式$c = \frac{5}{9}(f - 32)$，将$f$设置为$52$。在这个等式中，每一项-$5$、$9$和$32$-都是标量值。占位符$c$和$f$被称为*变量*，并且它们表示未知的标量值。

在本书中，我们采用数学表示法，其中标量变量用普通的小写字母表示(例如，$x$、$y$和$z$)。我们用$\mathbb{R}$表示所有(连续的)实值标量的空间。为方便起见，我们将严格定义“空间”到底是什么，但现在只需记住表达式$x \in \mathbb{R}$是表示$x$是实值标量的正式方式。符号$\in$可以发音为“in”，并且简单地表示集合中的成员。类似地，我们可以写成$x, y \in \{0, 1\}$来说明$x$和$y$是值只能是$0$或$1$的数字。

标量由只有一个元素的张量表示。在下一段代码中，我们实例化两个标量，并使用它们执行一些熟悉的算术运算，即加、乘、除和求幂。

```{.python .input}
from mxnet import np, npx
npx.set_np()

x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
#@tab pytorch
import torch

x = torch.tensor([3.0])
y = torch.tensor([2.0])

x + y, x * y, x / y, x**y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.constant([3.0])
y = tf.constant([2.0])

x + y, x * y, x / y, x**y
```

## 向量

您可以将向量视为标量值的简单列表。我们称这些值为向量的*元素*(*条目*或*组件*)。当我们的向量表示数据集中的示例时，它们的值具有一定的现实意义。例如，如果我们训练一个模型来预测贷款违约的风险，我们可能会将每个申请者与一个向量相关联，该向量的组成部分与他们的收入、雇佣年限、以前的违约次数和其他因素相对应。如果我们在研究医院患者可能面临的心脏病发作的风险，我们可能会用一个向量来表示每个患者，向量的组成部分反映了他们最近的生命体征、胆固醇水平、每天锻炼分钟等。在数学表示法中，我们通常会将向量表示为黑体、小写字母(例如，$\mathbf{x}$、$\mathbf{y}$和$\mathbf{z})$)。

我们通过一维张量来处理矢量。通常，张量可以有任意长度，受计算机内存限制的限制。

```{.python .input}
x = np.arange(4)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(4)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(4)
x
```

我们可以使用下标来引用向量的任何元素。例如，我们可以引用$\mathbf{x}$的$i^\mathrm{th}$个元素乘以$x_i$。请注意，元素$x_i$是标量，因此我们在引用它时不会将字体加粗。大量文献认为列向量是向量的默认方向，本书也是如此。在数学中，向量$\mathbf{x}$可以写为

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},$$
:eqlabel:`eq_vec_def`

其中$x_1, \ldots, x_n$是矢量的元素。在代码中，我们通过索引张量来访问任何元素。

```{.python .input}
x[3]
```

```{.python .input}
#@tab pytorch
x[3]
```

```{.python .input}
#@tab tensorflow
x[3]
```

### 长度、维度和形状

让我们回顾一下:numref:`sec_ndarray`中的一些概念。矢量只是一个数字数组。正如每个数组都有长度一样，每个向量也有长度。在数学表示法中，如果我们想说一个向量$\mathbf{x}$由$n$个实值标量组成，我们可以将其表示为$\mathbf{x} \in \mathbb{R}^n$。向量的长度通常称为向量的“维度”。

与普通Python数组一样，我们可以通过调用Python的内置`len()`函数来访问张量的长度。

```{.python .input}
len(x)
```

```{.python .input}
#@tab pytorch
len(x)
```

```{.python .input}
#@tab tensorflow
len(x)
```

当张量表示一个向量(恰好只有一个轴)时，我们还可以通过`.shape`属性访问它的长度。该形状是一个元组，它列出了沿张量每个轴的长度(维数)。对于只有一个轴的张量，形状只有一个元素。

```{.python .input}
x.shape
```

```{.python .input}
#@tab pytorch
x.shape
```

```{.python .input}
#@tab tensorflow
x.shape
```

请注意，在这些上下文中，“维度”这个词往往会过载，这往往会让人感到困惑。为了清楚起见，我们使用*向量*或*轴*的维数来表示其长度，即向量或轴的元素数量。然而，我们使用张量的维数来表示张量具有的轴数。从这个意义上说，张量的某个轴的维数就是该轴的长度。

## 矩阵

正如向量将标量从零阶推广到一阶一样，矩阵将向量从一阶推广到二阶。矩阵(我们通常用粗体大写字母(例如，$\mathbf{X}$、$\mathbf{Y}$和$\mathbf{Z}$)表示)在代码中表示为具有两个轴的张量。

在数学表示法中，我们使用$\mathbf{A} \in \mathbb{R}^{m \times n}$来表示矩阵$\mathbf{A}$由$m$行和$n$列实值标量组成。可视地，我们可以将任何矩阵$\mathbf{A} \in \mathbb{R}^{m \times n}$示为表格，其中每个元素$a_{ij}$属于$i^{\mathrm{th}}$行和$j^{\mathrm{th}}$列：

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`

对于任意$\mathbf{A} \in \mathbb{R}^{m \times n}$,$\mathbf{A}$的形状是($m$,$n$)或$m \times n$。具体地说，当一个矩阵具有相同的行数和列数时，它的形状就变成了正方形；因此，它被称为“正方形矩阵”。

当调用任何我们喜欢的用于实例化张量的函数时，我们可以通过指定具有两个组件$m \times n$和$n$的形状来创建一个$m$矩阵。

```{.python .input}
A = np.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab pytorch
A = torch.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20), (5, 4))
A
```

我们可以通过指定行($i$)和列($j$)的索引来访问矩阵$\mathbf{A}$在:eqref:`eq_matrix_def`中的标量元素$[\mathbf{A}]_{ij}$，例如$[\mathbf{A}]_{ij}$。当没有给出矩阵$\mathbf{A}$的标量元素时，例如在:eqref:`eq_matrix_def`中，我们可以简单地使用具有索引下标$a_{ij}$的矩阵$\mathbf{A}$的小写字母来引用$[\mathbf{A}]_{ij}$。为使表示法简单，仅在必要时(如$a_{2, 3j}$和$[\mathbf{A}]_{2i-1, 3}$)才在单独的索引中插入逗号。

有时，我们想要翻转斧头。当我们交换矩阵的行和列时，结果称为矩阵的“转置”。形式上，我们表示矩阵$\mathbf{A}$的转置为$\mathbf{A}^\top$，如果是$\mathbf{B} = \mathbf{A}^\top$，则表示任何$i$和$j$的转置为$b_{ij} = a_{ji}$。因此，:eqref:`eq_matrix_def`中$\mathbf{A}$的转置是一个$n \times m$矩阵：

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

现在我们用代码访问矩阵的转置。

```{.python .input}
A.T
```

```{.python .input}
#@tab pytorch
A.T
```

```{.python .input}
#@tab tensorflow
tf.transpose(A)
```

作为方阵的一种特殊类型，*对称矩阵*$\mathbf{A}$等于它的转置：$\mathbf{A} = \mathbf{A}^\top$。这里，我们定义对称矩阵`B`。

```{.python .input}
B = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab pytorch
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab tensorflow
B = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

现在我们将`B`与它的转座进行比较。

```{.python .input}
B == B.T
```

```{.python .input}
#@tab pytorch
B == B.T
```

```{.python .input}
#@tab tensorflow
B == tf.transpose(B)
```

矩阵是有用的数据结构：它们允许我们组织具有不同变化形式的数据。例如，矩阵中的行可能对应于不同的房子(数据示例)，而列可能对应于不同的属性。如果您曾经使用过电子表格软件或阅读过:numref:`sec_pandas`，这听起来应该很熟悉。因此，尽管单个向量的默认方向是列向量，但在表示表格数据集的矩阵中，将每个数据示例视为矩阵中的行向量是更传统的做法。而且，正如我们将在后面几章中看到的那样，这一惯例将使共同的深度学习实践成为可能。例如，沿着张量的最外面的轴，我们可以访问或枚举数据示例的小批次，或者如果不存在小批次，则仅访问或枚举数据示例。

## 张量

正如向量泛化标量，矩阵泛化向量一样，我们可以构建具有更多轴的数据结构。张量(本小节中的“张量”指的是代数对象)为我们提供了一种描述具有任意轴数的$n$维数组的通用方法。例如，向量是一阶张量，而矩阵是二阶张量。张量用特殊字体的大写字母表示(例如，$\mathsf{X}$、$\mathsf{Y}$和$\mathsf{Z}$)，并且它们的索引机制(例如，$x_{ijk}$和$[\mathsf{X}]_{1, 2i-1, 3}$)类似于矩阵的索引机制。

当我们开始处理图像时，张量将变得更加重要，图像以$n$维数组的形式出现，其中3个轴对应于高度、宽度和用于堆叠颜色通道(红色、绿色和蓝色)的*通道*轴。现在，我们将跳过高阶张量，专注于基础知识。

```{.python .input}
X = np.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab pytorch
X = torch.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(24), (2, 3, 4))
X
```

## 张量算法的基本性质

任意数量的轴的标量、向量、矩阵和张量(本小节中的“张量”指的是代数对象)都有一些很好的属性，这些属性经常会派上用场。例如，您可能已经从按元素操作的定义中注意到，任何按元素操作的一元运算都不会更改其操作数的形状。类似地，给定任意两个形状相同的张量，任何二元元素运算的结果都将是该形状的张量。例如，将两个形状相同的矩阵相加将对这两个矩阵执行元素相加。

```{.python .input}
A = np.arange(20).reshape(5, 4)
B = A.copy()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
#@tab pytorch
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20, dtype=tf.float32), (5, 4))
B = A  # No cloning of `A` to `B` by allocating new memory
A, A + B
```

具体地说，两个矩阵的元素相乘称为它们的*阿达玛乘积*(数学符号$\odot$)。考虑矩阵$\mathbf{B} \in \mathbb{R}^{m \times n}$，其行$i$和列$j$的元素是$b_{ij}$。矩阵$\mathbf{A}$(在:eqref:`eq_matrix_def`中定义)和$\mathbf{B}$的阿达玛乘积

$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
A * B
```

```{.python .input}
#@tab pytorch
A * B
```

```{.python .input}
#@tab tensorflow
A * B
```

将张量与标量相乘或相加也不会改变张量的形状，其中操作数张量的每个元素都将与标量相加或相乘。

```{.python .input}
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

## 减少
:label:`subseq_lin-alg-reduction`

我们可以对任意张量执行的一个有用的操作是计算它们元素的和。在数学记数法中，我们用$\sum$个符号来表示和。为了表示长度为$\mathbf{x}$的向量$d$中元素的总和，我们写为$\sum_{i=1}^d x_i$。在代码中，我们可以只调用计算总和的函数。

```{.python .input}
x = np.arange(4)
x, x.sum()
```

```{.python .input}
#@tab pytorch
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
#@tab tensorflow
x = tf.range(4, dtype=tf.float32)
x, tf.reduce_sum(x)
```

我们可以用任意形状的张量元素表示和。例如，$m \times n$矩阵$\mathbf{A}$的元素之和可以写为$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$。

```{.python .input}
A.shape, A.sum()
```

```{.python .input}
#@tab pytorch
A.shape, A.sum()
```

```{.python .input}
#@tab tensorflow
A.shape, tf.reduce_sum(A)
```

默认情况下，调用计算总和的函数
*沿张量的所有轴将*张量缩减为标量。
我们还可以指定通过求和来减小张量的轴。以矩阵为例。为了通过对所有行的元素求和来减少行维(轴0)，我们在调用函数时指定`axis=0`。由于输入矩阵沿轴0减小以生成输出向量，因此输入的轴0的维度在输出形状中丢失。

```{.python .input}
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis0 = tf.reduce_sum(A, axis=0)
A_sum_axis0, A_sum_axis0.shape
```

指定`axis=1`将通过对所有列的元素求和来降低列维(轴1)。因此，输入的轴1的尺寸在输出形状中丢失。

```{.python .input}
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis1 = tf.reduce_sum(A, axis=1)
A_sum_axis1, A_sum_axis1.shape
```

通过求和沿行和列减少矩阵相当于对矩阵的所有元素求和。

```{.python .input}
A.sum(axis=[0, 1])  # Same as `A.sum()`
```

```{.python .input}
#@tab pytorch
A.sum(axis=[0, 1])  # Same as `A.sum()`
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(A, axis=[0, 1])  # Same as `tf.reduce_sum(A)`
```

一个相关的数量是“平均值”，也称为“平均值”。我们用总和除以元素总数来计算平均值。在代码中，我们可以只调用函数来计算任意形状张量的平均值。

```{.python .input}
A.mean(), A.sum() / A.size
```

```{.python .input}
#@tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

同样，用于计算平均值的函数也可以沿指定的轴减少张量。

```{.python .input}
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab pytorch
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

### 非递减和
:label:`subseq_lin-alg-non-reduction`

但是，在调用用于计算总和或平均值的函数时，有时保持轴数不变可能很有用。

```{.python .input}
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab pytorch
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A
```

例如，由于`sum_A`在每行相加后仍然保持其两个轴，所以我们可以用广播将`A`除以`sum_A`。

```{.python .input}
A / sum_A
```

```{.python .input}
#@tab pytorch
A / sum_A
```

```{.python .input}
#@tab tensorflow
A / sum_A
```

如果我们想要计算沿着某个轴的`A`个元素的累积和，比如`axis=0`(逐行)，我们可以调用`cumsum`函数。此函数不会沿任何轴减少输入张量。

```{.python .input}
A.cumsum(axis=0)
```

```{.python .input}
#@tab pytorch
A.cumsum(axis=0)
```

```{.python .input}
#@tab tensorflow
tf.cumsum(A, axis=0)
```

## 点积

到目前为止，我们只执行了元素级运算、求和和求平均值。如果这就是我们所能做的一切，那么线性代数可能就不应该有自己的一部分了。但是，最基本的运算之一是点积。给定两个矢量$\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$，它们的*点积*$\mathbf{x}^\top \mathbf{y}$(或$\langle \mathbf{x}, \mathbf{y}  \rangle$)是相同位置的元素乘积的和：$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$。

```{.python .input}
y = np.ones(4)
x, y, np.dot(x, y)
```

```{.python .input}
#@tab pytorch
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
#@tab tensorflow
y = tf.ones(4, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

请注意，我们可以通过执行逐个元素的乘法，然后进行求和来等价地表示两个向量的点积：

```{.python .input}
np.sum(x * y)
```

```{.python .input}
#@tab pytorch
torch.sum(x * y)
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(x * y)
```

点积在很多情况下都很有用。例如，给定由向量$\mathbf{x}  \in \mathbb{R}^d$表示的某组值和由$\mathbf{w} \in \mathbb{R}^d$表示的一组权重，$\mathbf{x}$中的值根据权重$\mathbf{w}$的加权和可以表示为点积$\mathbf{x}^\top \mathbf{w}$。当权重非负且总和为1(即$\left(\sum_{i=1}^{d} {w_i} = 1\right)$)时，点积表示*加权平均*。将两个矢量归一化为单位长度后，点积表示它们之间的夹角的余弦。我们将在本节后面正式介绍“长度”这一概念。

## 矩阵向量积

既然我们知道了如何计算点积，我们就可以开始理解*矩阵向量积*。回想一下矩阵$\mathbf{A} \in \mathbb{R}^{m \times n}$和向量$\mathbf{x} \in \mathbb{R}^n$分别在:eqref:`eq_matrix_def`和:eqref:`eq_vec_def`中定义和可视化。让我们首先根据矩阵$\mathbf{A}$的行向量将其可视化

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

其中每个$\mathbf{a}^\top_{i} \in \mathbb{R}^n$是表示矩阵$\mathbf{A}$的$i^\mathrm{th}$行的行向量。矩阵向量乘积$\mathbf{A}\mathbf{x}$仅仅是长度为$m$的列向量，其$i^\mathrm{th}$个元素是点积$\mathbf{a}^\top_i \mathbf{x}$：

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

我们可以认为乘以矩阵$\mathbf{A}\in \mathbb{R}^{m \times n}$是将向量从$\mathbb{R}^{n}$投影到$\mathbb{R}^{m}$的变换。事实证明，这些转换非常有用。例如，我们可以将旋转表示为乘以一个方阵。正如我们将在后续章节中看到的，我们还可以使用矩阵向量积来描述在给定前一层的值的情况下计算神经网络中的每一层时所需的最密集的计算。

用张量用代码表示矩阵向量乘积，我们使用与点积相同的`dot`函数。当我们使用矩阵`A`和向量`x`调用`np.dot(A, x)`时，执行矩阵-向量乘积。请注意，柱尺寸`A`(其沿轴1的长度)必须与尺寸`x`(其长度)相同。

```{.python .input}
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
#@tab pytorch
A.shape, x.shape, torch.mv(A, x)
```

```{.python .input}
#@tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

## 矩阵-矩阵乘法

如果您已经掌握了点积和矩阵向量乘积的诀窍，那么*矩阵-矩阵乘法*应该是直截了当的。

假设我们有两个矩阵$\mathbf{A} \in \mathbb{R}^{n \times k}$和$\mathbf{B} \in \mathbb{R}^{k \times m}$：

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$

用$\mathbf{a}^\top_{i} \in \mathbb{R}^k$表示表示矩阵$\mathbf{A}$的$i^\mathrm{th}$行的行向量，并且设$\mathbf{b}_{j} \in \mathbb{R}^k$是来自矩阵$\mathbf{B}$的$j^\mathrm{th}$列的列向量。为了产生矩阵乘积$\mathbf{C} = \mathbf{A}\mathbf{B}$，最容易根据其行向量考虑$\mathbf{A}$，根据其列向量考虑$\mathbf{B}$：

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$

然后，矩阵乘积$\mathbf{C} \in \mathbb{R}^{n \times m}$被产生，因为我们简单地将每个元素$c_{ij}$计算为点积$\mathbf{a}^\top_i \mathbf{b}_j$：

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

我们可以认为矩阵-矩阵乘法$\mathbf{AB}$简单地执行$m$个矩阵-向量乘积，并将结果拼接在一起以形成$n \times m$个矩阵。在下面的代码片段中，我们对`A`和`B`执行矩阵乘法。这里，`A`是5行4列的矩阵，`B`是4行3列的矩阵。乘法后，我们得到一个5行3列的矩阵。

```{.python .input}
B = np.ones(shape=(4, 3))
np.dot(A, B)
```

```{.python .input}
#@tab pytorch
B = torch.ones(4, 3)
torch.mm(A, B)
```

```{.python .input}
#@tab tensorflow
B = tf.ones((4, 3), tf.float32)
tf.matmul(A, B)
```

矩阵-矩阵乘法可以简单地称为*矩阵乘法*，不应与Hadamard乘积混淆。

## 规范
:label:`subsec_lin-algebra-norms`

线性代数中一些最有用的运算符是*范数*。非正式地，向量的范数告诉我们向量有多大。这里考虑的*大小*的概念不是维度，而是组件的大小。

在线性代数中，向量范数是将向量映射到标量的函数$f$，满足几个性质。给定任何向量$\mathbf{x}$，第一个属性表示，如果按常数$\alpha$缩放向量的所有元素，则其范数也按相同常数因子的*绝对值*缩放：

$$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$

第二个属性是熟悉的三角形不等式：

$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$

第三个属性简单地说，规范必须是非负的：

$$f(\mathbf{x}) \geq 0.$$

这是有道理的，因为在大多数上下文中，任何东西的最小*大小*都是0。最后一个性质要求实现最小范数，并且仅通过由全零组成的向量来实现。

$$\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(\mathbf{x})=0.$$

你可能会注意到，规范听起来很像距离的度量。如果你记得欧几里德距离小学的距离(想想毕达哥拉斯定理)，那么非负性和三角形不等式的概念可能会耳熟能详。事实上，欧几里德距离是一种范数：具体地说，它是$L_2$范数。假设$n$维向量$\mathbf{x}$中的元素是$x_1, \ldots, x_n$。$L_2$*范数*的$\mathbf{x}$是矢量元素的平方和的平方根：

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$$

其中下标$2$在$L_2$规范中通常被省略，即$\|\mathbf{x}\|$等于$\|\mathbf{x}\|_2$。在代码中，我们可以按如下方式计算向量的$L_2$范数。

```{.python .input}
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
#@tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
#@tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

在深度学习中，我们更多地使用$L_2$的平方范数。您还会经常遇到$L_1$*范数*，它表示为向量元素的绝对值之和：

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$

与$L_2$常量相比，它受异常值的影响较小。为了计算$L_1$范数，我们用元素上的和组成绝对值函数。

```{.python .input}
np.abs(u).sum()
```

```{.python .input}
#@tab pytorch
torch.abs(u).sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(tf.abs(u))
```

$L_2$规范和$L_1$规范都是更一般的$L_p$*规范*的特例：

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

类似于向量的$L_2$范数，矩阵$\mathbf{X} \in \mathbb{R}^{m \times n}$的*弗罗贝尼乌斯范数*是矩阵元素的平方和的平方根：

$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$

Frobenius范数满足向量范数的所有性质。它的行为就像是矩阵形状向量的$L_2$范数。调用以下函数将计算矩阵的Frobenius范数。

```{.python .input}
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
#@tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
#@tab tensorflow
tf.norm(tf.ones((4, 9)))
```

### 规范和目标
:label:`subsec_norms_and_objectives`

虽然我们不想超前太多，但我们已经可以种植一些直觉来解释为什么这些概念是有用的。在深度学习中，我们经常尝试解决优化问题：
*最大化*分配给观测数据的概率；
**最小化预测之间的距离
以及地面实况观察。将矢量表示指定给项(如文字、产品或新闻文章)，以使相似项之间的距离最小化，而使不同项之间的距离最大化。通常，目标，可能是深度学习算法的最重要的组成部分(除了数据之外)，被表示为规范。

## 关于线性代数的更多内容

仅在这一节中，我们就已经教了你所有的线性代数，你需要这些知识才能理解现代深度学习中的一大块不同寻常的东西。线性代数还有很多东西，其中很多数学对机器学习很有用。例如，矩阵可以分解成因子，这些分解可以揭示现实世界数据集中的低维结构。机器学习的整个子领域都专注于使用矩阵分解及其对高阶张量的推广来发现数据集中的结构并解决预测问题。但这本书关注的是深度学习。我们相信，一旦您在实际数据集上部署了有用的机器学习模型，您将更愿意学习更多的数学知识。因此，虽然我们保留稍后介绍更多数学的权利，但我们将在这里结束这一节。

如果您渴望了解更多关于线性代数的知识，您可以参考[online appendix on linear algebraic operations](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.html)或其他优秀的资源:cite:`Strang.1993,Kolter.2008,Petersen.Pedersen.ea.2008`。

## 摘要

* 标量、向量、矩阵和张量是线性代数中的基本数学对象。
* 矢量泛化标量，矩阵泛化矢量。
* 标量、向量、矩阵和张量分别具有零、一、二和任意数量的轴。
* 张量可以沿指定的轴减`sum`和`mean`。
* 两个矩阵的元素相乘称为它们的Hadamard积。它不同于矩阵乘法。
* 在深度学习中，我们经常使用诸如$L_1$规范、$L_2$规范和弗罗贝尼乌斯规范这样的规范。
* 我们可以对标量、向量、矩阵和张量执行各种操作。

## 练习

1. 证明了矩阵$\mathbf{A}$的转置是$\mathbf{A}$：$(\mathbf{A}^\top)^\top = \mathbf{A}$。
1. 给定两个矩阵$\mathbf{A}$和$\mathbf{B}$，表明转置的和等于和的转置：$\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$。
1. 给定任何方阵$\mathbf{A}$,$\mathbf{A} + \mathbf{A}^\top$总是对称的吗？为什么？
1. 我们在本节中定义了形状(2，3，4)的张量`X`。`len(X)`的产量是多少？
1. 对于任意形状的张量`X`,`len(X)`是否总是对应于特定轴`X`的长度？那个轴是什么？
1. 运行`A / A.sum(axis=1)`，看看会发生什么。你能分析一下原因吗？
1. 当你在曼哈顿的两个地点之间旅行时，你需要在坐标上，也就是在大道和街道上，覆盖多远的距离？你能对角线旅行吗？
1. 考虑一个形状为(2，3，4)的张量。轴0、1和2上的求和输出是什么形状？
1. 将具有3个或更多轴的张量馈送到`linalg.norm`函数，并观察其输出。这个函数对任意形状的张量计算什么？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/196)
:end_tab:
