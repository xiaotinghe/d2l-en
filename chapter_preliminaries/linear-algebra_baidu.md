# 线性代数
:label:`sec_linear-algebra`

现在您可以存储和操作数据了，让我们简要回顾一下基本线性代数的子集，您将需要理解和实现本书中涉及的大多数模型。下面，我们将介绍线性代数中的基本数学对象、算法和运算，并用数学符号表示它们，以及相应的代码实现。

## 标量

如果你从未学习过线性代数或机器学习，那么你过去的数学经验可能是一次只考虑一个数字。而且，如果你曾经平衡过一张支票簿，甚至在餐馆吃过晚饭，那么你已经知道如何做一些基本的事情，比如数对的加法和乘法。例如，帕洛阿尔托的温度是$52$华氏度。从形式上讲，我们称之为只包含一个数字量*标量*的值。如果要将该值转换为摄氏度（公制更敏感的温度刻度），则需要计算表达式$c = \frac{5}{9}(f - 32)$，将$f$设置为$52$。在这个方程中，$5$、$9$和$32$中的每一项都是标量值。占位符$c$和$f$称为*变量*，它们表示未知标量值。

在这本书中，我们采用数学符号，标量变量用普通小写字母表示（例如$x$、$y$和$z$）。我们用$\mathbb{R}$表示所有（连续）*实值*标量的空间。为了方便起见，我们将严格定义空间，但现在请记住，表达式$x$是实值标量。符号$\in$可以读作“in”，并简单地表示集合中的成员。类似地，我们可以写$x, y \in \{0, 1\}$来声明$x$和$y$是数值，它们的值只能是$0$或$1$。

标量由一个只有一个元素的张量表示。在下一个片段中，我们实例化两个标量，并对它们执行一些熟悉的算术运算，即加法、乘法、除法和求幂。

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

你可以把向量看作是一个简单的标量值列表。我们将这些值称为向量的*elements*（*entries*或*components*）。当我们的向量表示数据集中的示例时，它们的值具有一定的现实意义。例如，如果我们正在训练一个预测贷款违约风险的模型，我们可能会将每个申请人与一个向量相关联，该向量的组成部分对应于他们的收入、工作年限、以前违约的次数和其他因素。如果我们正在研究医院患者可能面临的心脏病发作风险，我们可以用一个向量来表示每个患者，该向量的组成部分捕捉了他们最近的生命体征、胆固醇水平、每天的运动分钟数等。在数学记法中，我们通常将向量表示为粗体、小写字母（例如，$\mathbf{x}$、$\mathbf{y}$，和$\mathbf{z})$。

我们通过一维张量处理向量。一般情况下，张量可以有任意长度，这取决于机器的内存限制。

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

我们可以用下标来引用向量的任何元素。例如，我们可以通过$x_i$引用$i^\mathrm{th}$的$i^\mathrm{th}$元素。注意，元素$x_i$是一个标量，因此在引用它时，我们不会将字体加粗。大量文献认为列向量是向量的默认方向，本书也是如此。在数学中，向量$\mathbf{x}$可以写成

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},$$
:eqlabel:`eq_vec_def`

其中$x_1, \ldots, x_n$是向量的元素。在代码中，我们通过索引张量来访问任何元素。

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

### 长度、尺寸和形状

让我们重温:numref:`sec_ndarray`中的一些概念。向量就是一组数字。就像每个数组都有一个长度，每个向量也是一样。在数学表示法中，如果我们想说向量$\mathbf{x}$由$n$个实值标量组成，我们可以将其表示为$\mathbf{x} \in \mathbb{R}^n$。向量的长度通常称为向量的*维数*。

与普通Python数组一样，我们可以通过调用Python内置的`len()`函数来访问张量的长度。

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

当张量表示一个向量（正好有一个轴），我们也可以通过`.shape`属性来访问它的长度。形状是一个元组，它列出了张量每个轴的长度（维度）。只有一个元素的张量。

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

注意，“维度”一词在这些上下文中往往会被重载，这往往会使人们感到困惑。为了澄清，我们使用*向量*或*轴*的维数来引用其长度，即向量或轴的元素数。然而，我们使用张量的维数来指张量所拥有的轴数。从这个意义上说，张量的某些轴的尺寸将是该轴的长度。

## 矩阵

正如向量将标量从零阶推广到一阶，矩阵将向量从一阶推广到二阶。矩阵，我们通常用粗体大写字母表示（例如，$\mathbf{X}$、$\mathbf{Y}$和$\mathbf{Z}$）在代码中表示为带有两个轴的张量。

在数学表示法中，我们使用$\mathbf{A} \in \mathbb{R}^{m \times n}$来表示矩阵$\mathbf{A}$由$m$行和$n$列实值标量组成。我们可以直观地将任何矩阵$\mathbf{A} \in \mathbb{R}^{m \times n}$表示为一个表，其中每个元素$a_{ij}$属于$i^{\mathrm{th}}$行和$j^{\mathrm{th}}$列：

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`

对于任何$\mathbf{A} \in \mathbb{R}^{m \times n}$，$\mathbf{A}$的形状是（$m$，$n$）或$m \times n$。具体地说，当一个矩阵具有相同的行数和列数时，它的形状变成正方形；因此，它被称为*正方形矩阵*。

当调用我们最喜欢的函数实例化张量时，我们可以通过指定包含两个组件$m$和$n$的形状来创建$m \times n$矩阵。

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

通过指定行（732293612）和列（$j$）的索引，我们可以访问:eqref:`eq_matrix_def`中矩阵$a_{ij}$的标量元素$a_{ij}$。当矩阵$\mathbf{A}$的标量元素（如:eqref:`eq_matrix_def`）没有给出时，我们可以简单地使用带下标$a_{ij}$的矩阵$\mathbf{A}$的小写字母来表示$[\mathbf{A}]_{ij}$。为了保持符号的简单性，只有在必要时才在不同的索引中插入逗号，例如$a_{2, 3j}$和$[\mathbf{A}]_{2i-1, 3}$。

有时，我们想翻转轴。当我们交换矩阵的行和列时，结果称为矩阵的*转置*。形式上，我们表示矩阵$\mathbf{A}$被732293612转置，如果$\mathbf{B} = \mathbf{A}^\top$，则$b_{ij} = a_{ji}$用于任何$i$和$j$。因此，$\mathbf{A}$在:eqref:`eq_matrix_def`中的转置是$n \times m$矩阵：

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

现在我们在代码中访问矩阵的转置。

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

作为方阵的一种特殊类型，*对称矩阵*$\mathbf{A}$等于它的转置：$\mathbf{A} = \mathbf{A}^\top$。这里我们定义了一个对称矩阵`B`。

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

现在我们将`B`与其转座进行比较。

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

矩阵是有用的数据结构：它们允许我们组织具有不同变化模式的数据。例如，矩阵中的行可能对应于不同的房屋（数据示例），而列可能对应于不同的属性。如果您曾经使用过电子表格软件或阅读过:numref:`sec_pandas`，这听起来应该很熟悉。因此，虽然单个向量的默认方向是列向量，但在表示表格数据集的矩阵中，将每个数据示例视为矩阵中的行向量更为传统。而且，正如我们将在后面几章中看到的，这项公约将使共同的深入学习实践成为可能。例如，沿着张量的最外轴，我们可以访问或枚举数据示例的小批量，或者如果不存在小批量，则仅访问数据示例。

## 张量

就像向量概括标量，矩阵概括向量一样，我们可以用更多的轴来构建数据结构。张量（本小节中的“张量”指的是代数对象）为我们提供了一种描述具有任意数量轴的$n$维数组的通用方法。例如，向量是一阶张量，矩阵是二阶张量。张量用特殊字体的大写字母表示（例如$\mathsf{X}$、$\mathsf{Y}$和$\mathsf{Z}$），它们的索引机制（例如$x_{ijk}$和$[\mathsf{X}]_{1, 2i-1, 3}$）与矩阵相似。

当我们开始处理图像时，张量将变得更加重要，图像以$n$维数组的形式出现，其中3个轴对应于高度、宽度和一个*通道*轴，用于堆叠颜色通道（红色、绿色和蓝色）。现在，我们将跳过高阶张量，集中在基础上。

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

任意数量轴的标量、向量、矩阵和张量（本小节中的“张量”指的是代数对象）都有一些很好的特性，这些特性通常很有用。例如，您可能已经从元素操作的定义中注意到，任何元素一元操作都不会更改其操作数的形状。同样，给定任意两个具有相同形状的张量，任何二元元素运算的结果都将是相同形状的张量。例如，添加两个形状相同的矩阵对这两个矩阵执行元素加法。

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

具体地说，两个矩阵的元素相乘称为它们的*哈达玛积*（数学符号$\odot$）。考虑矩阵$\mathbf{B} \in \mathbb{R}^{m \times n}$，其行$i$和列$j$的元素是$b_{ij}$。矩阵$\mathbf{A}$（定义于$\mathbf{A}$）和$\mathbf{B}$的Hadamard积

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

将张量乘以或加上一个标量也不会改变张量的形状，其中操作数张量的每个元素都将与标量相加或相乘。

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

我们可以对任意张量执行的一个有用的操作是计算它们的元素之和。在数学记数法中，我们用$\sum$符号表示和。为了表示长度为$d$的向量$\mathbf{x}$中元素的和，我们写$\sum_{i=1}^d x_i$。在代码中，我们可以调用函数来计算和。

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

我们可以表示任意形状张量元素的和。例如，$m \times n$矩阵$\mathbf{A}$的元素之和可以写入$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$。

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

默认情况下，调用函数来计算总和
*把一个张量沿它的所有轴减少成一个标量。
我们也可以指定通过求和来减少张量的轴。以矩阵为例。为了通过汇总所有行的元素来减少行维度（轴0），我们在调用函数时指定`axis=0`。由于输入矩阵沿0轴减小以生成输出向量，因此输入的0轴维数在输出形状中丢失。

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

指定`axis=1`将通过汇总所有列的元素来减少列维度（轴1）。因此，输入轴1的尺寸在输出形状中丢失。

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

通过求和来减少一个矩阵的行和列，相当于将矩阵的所有元素相加。

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

一个相关的量是*平均数*，它也被称为*平均数*。我们用元素总数除以和来计算平均数。在代码中，我们可以调用函数来计算任意形状张量的平均值。

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

同样，用于计算平均值的函数也可以沿指定轴减少张量。

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

### 非还原总和
:label:`subseq_lin-alg-non-reduction`

但是，有时在调用计算和或平均值的函数时，保持轴的数量不变是有用的。

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

例如，由于`sum_A`在每行求和后仍然保留其两个轴，因此可以用广播将`A`除以`sum_A`。

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

如果我们想计算`A`元素沿某个轴的累计和，比如`axis=0`（逐行），我们可以调用`cumsum`函数。此函数不会减少沿任何轴的输入张量。

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

## Dot产品

到目前为止，我们只进行了元素运算、求和和和平均值。如果这就是我们所能做的，线性代数可能就不配有它自己的部分了。然而，最基本的运算之一是点积。给定两个向量$\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$，它们的*点积*$\mathbf{x}^\top \mathbf{y}$（或$\langle \mathbf{x}, \mathbf{y}  \rangle$）是同一位置上元素乘积的和：$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$。

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

注意，我们可以通过执行元素乘法，然后求和，来等价地表示两个向量的点积：

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

点积在很多情况下都很有用。例如，给定由向量$\mathbf{x}  \in \mathbb{R}^d$表示的某组值和由$\mathbf{w} \in \mathbb{R}^d$表示的一组权重，$\mathbf{x}$中根据权重$\mathbf{w}$的值的加权和可以表示为点积$\mathbf{x}^\top \mathbf{w}$。当权重为非负且和为1（即$\left(\sum_{i=1}^{d} {w_i} = 1\right)$）时，点积表示*加权平均值*。将两个矢量归一化为单位长度后，点积表示它们之间夹角的余弦。我们将在本节后面正式介绍*length*的概念。

## 矩阵向量积

既然我们知道了如何计算点积，我们就可以开始理解矩阵向量积了。回忆一下分别在:eqref:`eq_matrix_def`和:eqref:`eq_vec_def`中定义和可视化的矩阵$\mathbf{A} \in \mathbb{R}^{m \times n}$和向量$\mathbf{x} \in \mathbb{R}^n$。让我们从矩阵$\mathbf{A}$的行向量可视化开始

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

其中每个$\mathbf{a}^\top_{i} \in \mathbb{R}^n$是表示矩阵$\mathbf{A}$的$i^\mathrm{th}$行的行向量。矩阵向量积$\mathbf{A}\mathbf{x}$只是长度为$m$的列向量，其$i^\mathrm{th}$元素是点乘$\mathbf{a}^\top_i \mathbf{x}$：

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

我们可以把矩阵$\mathbf{A}\in \mathbb{R}^{m \times n}$的乘法看作是将向量从$\mathbb{R}^{n}$投影到$\mathbb{R}^{m}$的转换。这些转换结果非常有用。例如，我们可以用一个方阵来表示旋转。正如我们将在后面的章节中看到的，我们还可以使用矩阵向量积来描述在给定前一层的值的情况下计算神经网络中的每一层时所需的最密集的计算。

用张量表示矩阵向量积，我们使用与点积相同的`dot`函数。当我们使用矩阵`A`和向量`x`调用`np.dot(A, x)`时，将执行矩阵-向量乘积。请注意，`A`的列尺寸（其沿轴1的长度）必须与`x`（其长度）的尺寸相同。

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

## 矩阵矩阵乘法

如果你已经掌握了点积和矩阵向量积的诀窍，那么矩阵矩阵乘法应该很简单。

假设我们有两个矩阵$\mathbf{A} \in \mathbb{R}^{n \times k}$和73229365：

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

用$\mathbf{a}^\top_{i} \in \mathbb{R}^k$表示表示矩阵$\mathbf{A}$的$i^\mathrm{th}$行的行向量，并让$\mathbf{b}_{j} \in \mathbb{R}^k$是来自矩阵$\mathbf{B}$的$j^\mathrm{th}$列的列向量。为了得到矩阵乘积$\mathbf{C} = \mathbf{A}\mathbf{B}$，最容易想到的是$\mathbf{A}$的行向量和$\mathbf{B}$的列向量：

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

然后矩阵积$\mathbf{C} \in \mathbb{R}^{n \times m}$就产生了，因为我们简单地将每个元素$c_{ij}$计算为点积$\mathbf{a}^\top_i \mathbf{b}_j$：

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

我们可以把矩阵矩阵乘法$\mathbf{AB}$看作是简单地执行$m$矩阵向量积，然后将结果拼接在一起，形成一个$n \times m$矩阵。在下面的代码片段中，我们对`A`和`B`执行矩阵乘法。这里，`A`是5行4列的矩阵，`B`是4行3列的矩阵。乘法后得到一个5行3列的矩阵。

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

矩阵矩阵乘法可以简单地称为*矩阵乘法*，不应与Hadamard乘积混淆。

## 规范
:label:`subsec_lin-algebra-norms`

线性代数中一些最有用的算符是*范数*。非正式地说，向量的范数告诉我们向量有多大。这里考虑的*大小*的概念与维度无关，而是与组件的大小有关。

在线性代数中，向量范数是一个函数$f$，它将一个向量映射到一个标量，满足一些属性。给定任何向量$\mathbf{x}$，第一个属性表示，如果我们用常数因子$\alpha$来缩放向量的所有元素，那么它的范数也会按相同常数因子的*绝对值*来缩放：

$$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$

第二个性质是常见的三角形不等式：

$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$

第三个属性简单地说，规范必须是非负的：

$$f(\mathbf{x}) \geq 0.$$

这是有道理的，因为在大多数情况下，任何东西的最小*大小*都是0。最后一个性质要求得到最小范数，并且只能由一个由所有零组成的向量来实现。

$$\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(\mathbf{x})=0.$$

你可能会注意到规范听起来很像距离的度量。如果你还记得小学时的欧几里德距离（想想毕达哥拉斯定理），那么非负性和三角不等式的概念可能会敲响警钟。事实上，欧几里德距离是一个范数：特别是$L_2$范数。假设$n$维向量$\mathbf{x}$中的元素是$x_1, \ldots, x_n$。$\mathbf{x}$的$L_2$*范数*是向量元素平方和的平方根：

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$$

其中，在$L_2$规范中经常省略下标$2$，即$\|\mathbf{x}\|$相当于$\|\mathbf{x}\|_2$。在代码中，我们可以计算向量的$L_2$范数，如下所示。

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

在深度学习中，我们更多地使用$L_2$的平方范数。您还会经常遇到$L_1$*范数*，它表示为矢量元素的绝对值之和：

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$

与$L_2$常模相比，它受异常值的影响较小。为了计算$L_1$范数，我们用元素的和构成绝对值函数。

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

$L_2$标准和$L_1$标准都是更一般的$L_p$*标准*的特殊情况：

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

与$L_2$向量范数类似，矩阵$\mathbf{X} \in \mathbb{R}^{m \times n}$的*Frobenius范数*是矩阵元素平方和的平方根：

$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$

Frobenius范数满足向量范数的所有性质。它的行为就像是一个矩阵型向量的$L_2$范数。调用以下函数将计算矩阵的Frobenius范数。

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

虽然我们不想走得太远，但我们已经可以建立一些直觉来解释为什么这些概念是有用的。在深度学习中，我们经常试图解决优化问题：
*最大化*分配给观测数据的概率；
*最小化*预测之间的距离
以及地面真相观察。为项目（如单词、产品或新闻文章）指定向量表示，以便最小化相似项目之间的距离，并最大化不同项目之间的距离。通常，目标，也许是深度学习算法（除了数据之外）最重要的组成部分，被表示为规范。

## 再论线性代数

在这一节中，我们已经教了你所有的线性代数，你将需要了解现代深度学习的一个重要部分。线性代数还有很多，其中很多数学对机器学习很有用。例如，矩阵可以分解为因子，这些分解可以揭示现实世界数据集中的低维结构。机器学习的整个子领域都集中在使用矩阵分解及其对高阶张量的推广来发现数据集中的结构并解决预测问题。但这本书的重点是深度学习。我们相信，一旦在真实数据集上部署有用的机器学习模型，你会更倾向于学习更多的数学。因此，虽然我们保留以后引入更多数学的权利，但我们将在这里结束这一部分。

如果你想学习更多关于线性代数的知识，你可以参考[online appendix on linear algebraic operations](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.html)或其他优秀的资源:cite:`Strang.1993,Kolter.2008,Petersen.Pedersen.ea.2008`。

## 摘要

* 标量、向量、矩阵和张量是线性代数中的基本数学对象。
* 向量概括标量，矩阵概括向量。
* 零张量，两个向量，一个向量，两个标量。
* 张量可沿指定轴减少`sum`和`mean`。
* 两个矩阵的元素相乘称为它们的Hadamard积。它不同于矩阵乘法。
* 在深度学习中，我们经常使用$L_1$标准、$L_2$规范和Frobenius规范。
* 我们可以对标量、向量、矩阵和张量执行各种操作。

## 练习

1. 证明矩阵$\mathbf{A}$的转置是$\mathbf{A}$:$(\mathbf{A}^\top)^\top = \mathbf{A}$。
1. 给出两个矩阵$\mathbf{A}$和$\mathbf{B}$，证明转置之和等于一个和的转置：$\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$。
1. 给定任何一个方阵$\mathbf{A}$，$\mathbf{A} + \mathbf{A}^\top$是否总是对称的？为什么？
1. 我们在这一节中定义了形状（2，3，4）的张量`X`。`len(X)`的产量是多少？
1. 对于任意形状的张量`X`，`len(X)`是否总是对应于某个轴`X`的长度？那个轴是什么？
1. 运行`A / A.sum(axis=1)`看看会发生什么。你能分析一下原因吗？
1. 当你在曼哈顿的两点之间旅行时，你需要在坐标上走多远，也就是说，在大街和街道上？你能斜着走吗？
1. 考虑具有形状（2，3，4）的张量。沿轴0、1和2的求和输出的形状是什么？
1. 向`linalg.norm`功能输入带有3个或更多轴的张量并观察其输出。对于任意形状的张量，这个函数计算什么？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/196)
:end_tab:
