# 自动区分
:label:`sec_autograd`

正如我们在:numref:`sec_calculus`中所解释的，在几乎所有的深度学习优化算法中，微分是至关重要的一步。虽然求这些导数的计算很简单，只需要一些基本的微积分，但对于复杂的模型，手工计算更新可能是一件痛苦的事情(而且往往容易出错)。

深度学习框架通过自动计算导数，即*自动微分*来加速这项工作。在实践中，基于我们设计的模型，系统构建了一个*计算图*，跟踪哪些数据通过哪些操作组合起来产生输出。自动微分使系统能够随后反向传播梯度。在这里，*反向传播*只是指通过计算图形进行跟踪，填充关于每个参数的偏导数。

```{.python .input}
from mxnet import autograd, np, npx
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

## 一个简单的例子

作为一个玩具示例，假设我们对函数$y = 2\mathbf{x}^{\top}\mathbf{x}$相对于列向量$\mathbf{x}$的微分感兴趣。首先，让我们创建变量`x`并为其分配初始值。

```{.python .input}
x = np.arange(4.0)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(4.0)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(4, dtype=tf.float32)
x
```

在我们计算$y$相对于$\mathbf{x}$的梯度之前，我们需要一个地方来存储它。重要的是，我们不要在每次对参数求导时都分配新的内存，因为我们经常会数千次或数百万次更新相同的参数，并且可能很快就会耗尽内存。注意，标量函数相对于向量$\mathbf{x}$的梯度本身是向量值的，并且具有与$\mathbf{x}$相同的形状。

```{.python .input}
# We allocate memory for a tensor's gradient by invoking `attach_grad`
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
x.grad
```

```{.python .input}
#@tab pytorch
x.requires_grad_(True)  # Same as `x = torch.arange(4.0, requires_grad=True)`
x.grad  # The default value is None
```

```{.python .input}
#@tab tensorflow
x = tf.Variable(x)
```

现在让我们计算一下$y$。

```{.python .input}
# Place our code inside an `autograd.record` scope to build the computational
# graph
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input}
#@tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
#@tab tensorflow
# Record all computations onto a tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

因为`x`是长度为4的向量，所以执行`x`和`x`的内积，从而产生我们赋给`y`的标量输出。接下来，通过调用反向传播函数并打印梯度，我们可以自动计算`y`相对于`x`的每个分量的梯度。

```{.python .input}
y.backward()
x.grad
```

```{.python .input}
#@tab pytorch
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

函数$y = 2\mathbf{x}^{\top}\mathbf{x}$相对于$\mathbf{x}$的梯度应该是$4\mathbf{x}$。让我们快速验证是否正确计算了所需的梯度。

```{.python .input}
x.grad == 4 * x
```

```{.python .input}
#@tab pytorch
x.grad == 4 * x
```

```{.python .input}
#@tab tensorflow
x_grad == 4 * x
```

现在让我们计算`x`的另一个函数。

```{.python .input}
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Overwritten by the newly calculated gradient
```

```{.python .input}
#@tab pytorch
# PyTorch accumulates the gradient in default, we need to clear the previous 
# values
x.grad.zero_() 
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
```

## 向后显示非标量变量

从技术上讲，当`y`不是标量时，矢量`y`相对于矢量`x`的微分的最自然解释是矩阵。对于高阶和高维`y`和`x`，微分结果可以是高阶张量。

然而，虽然这些更奇特的对象确实出现在高级机器学习中(包括在深度学习中)，但当我们向后调用向量时，更经常的情况是，我们试图计算一批*训练示例中每个成分的损失函数的导数。这里，我们的目的不是计算微分矩阵，而是为批次中的每个示例单独计算的偏导数之和。

```{.python .input}
# When we invoke `backward` on a vector-valued variable `y` (function of `x`),
# a new scalar variable is created by summing the elements in `y`. Then the
# gradient of that scalar variable with respect to `x` is computed
with autograd.record():
    y = x * x  # `y` is a vector
y.backward()
x.grad  # Equals to y = sum(x * x)
```

```{.python .input}
#@tab pytorch
# Invoking `backward` on a non-scalar requires passing in a `gradient` argument
# which specifies the gradient of the differentiated function w.r.t `self`.
# In our case, we simply want to sum the partial derivatives, so passing
# in a gradient of ones is appropriate
x.grad.zero_()
y = x * x
# y.backward(torch.ones(len(x))) equivalent to the below
y.sum().backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as `y = tf.reduce_sum(x * x)`
```

## 分离计算

有时，我们希望将一些计算移到记录的计算图之外。例如，假设`y`被计算为`x`的函数，随后`z`被计算为`y`和`x`的函数。现在，假设我们想要计算`z`相对于`x`的梯度，但出于某种原因想要将`y`视为常量，并且只考虑`x`在计算`y`之后所起的作用。

这里，我们可以分离`y`以返回新的变量`u`，该变量与`y`具有相同的值，但是丢弃关于如何在计算图中计算`y`的任何信息。换句话说，渐变不会倒流通过`u`到`x`。因此，下面的反向传播函数计算`z = u * x`相对于`x`的偏导数，同时将`u`视为常数，而不是计算`z = x * x * x`相对于`x`的偏导数。

```{.python .input}
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
#@tab tensorflow
# Set `persistent=True` to run `t.gradient` more than once
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

由于记录了`y`的计算，因此我们可以随后在`y`上调用反向传播，以获得`y = x * x`相对于`x`的导数，即`2 * x`。

```{.python .input}
y.backward()
x.grad == 2 * x
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
#@tab tensorflow
t.gradient(y, x) == 2 * x
```

## 计算Python控制流的梯度

使用自动微分的一个好处是，即使构建需要通过Python控制流迷宫(例如，条件、循环和任意函数调用)的函数的计算图，我们仍然可以计算得到的变量的梯度。在下面的代码片段中，请注意`while`循环的迭代次数和`if`语句的计算都取决于输入`a`的值。

```{.python .input}
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

让我们计算一下梯度。

```{.python .input}
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
#@tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
#@tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

现在我们可以分析上面定义的`f`函数。注意，它在其输入`a`中是分段线性的。换言之，对于任何`a`，存在某个恒定标量`k`，使得`f(a) = k * a`，其中`k`的值取决于输入`a`。因此，`d / a`允许我们验证梯度是否正确。

```{.python .input}
a.grad == d / a
```

```{.python .input}
#@tab pytorch
a.grad == d / a
```

```{.python .input}
#@tab tensorflow
d_grad == d / a
```

## 摘要

* 深度学习框架可以自动计算导数。要使用它，我们首先将梯度附加到我们想要对其进行偏导数的变量上。然后，我们记录目标值的计算，执行其反向传播函数，并访问结果梯度。

## 练习

1. 为什么二阶导数比一阶导数的计算成本高得多？
1. 在运行用于反向传播的函数之后，立即再次运行它，看看会发生什么情况。
1. 在控制流示例中，我们计算`d`相对于`a`的导数，如果我们将变量`a`更改为随机向量或矩阵，会发生什么情况。此时，计算`f(a)`的结果不再是标量。结果会怎样呢？我们该如何分析这件事呢？
1. 重新设计一个查找控制流梯度的示例。运行并分析结果。
1. 就让$f(x) = \sin(x)$吧。图$f(x)$和$\frac{df(x)}{dx}$，其中后者是在不利用该$f'(x) = \cos(x)$的情况下计算的。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/200)
:end_tab:
