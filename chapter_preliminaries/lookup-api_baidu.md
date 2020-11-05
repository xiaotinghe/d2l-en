# 文档

由于本书篇幅的限制，我们不可能介绍每一个MXNet函数和类（您可能不希望我们这样做）。API文档和其他教程和示例提供了本书之外的大量文档。在本节中，我们将为您提供一些探索mxnetapi的指导。

## 查找模块中的所有函数和类

为了知道模块中可以调用哪些函数和类，我们调用`dir`函数。例如，我们可以查询模块中的所有属性来生成随机数：

```{.python .input  n=1}
from mxnet import np
print(dir(np.random))
```

```{.python .input  n=1}
#@tab pytorch
import torch
print(dir(torch.distributions))
```

```{.python .input  n=1}
#@tab tensorflow
import tensorflow as tf
print(dir(tf.random))
```

通常，我们可以忽略以`__`开头和结尾的函数（Python中的特殊对象）或以单个`_`开头的函数（通常是内部函数）。基于剩余的函数名或属性名，我们可能会猜测该模块提供了各种生成随机数的方法，包括从均匀分布（`uniform`）、正态分布（`normal`）和多项式分布（`multinomial`）中抽样。

## 查找特定函数和类的用法

对于如何使用给定函数或类的更具体的说明，我们可以调用`help`函数。作为一个例子，让我们研究一下张量的`ones`函数的用法说明。

```{.python .input}
help(np.ones)
```

```{.python .input}
#@tab pytorch
help(torch.ones)
```

```{.python .input}
#@tab tensorflow
help(tf.ones)
```

从文档中，我们可以看到`ones`函数创建了一个具有指定形状的新张量，并将所有元素的值设置为1。只要可能，您应该运行一个快速测试来确认您的解释：

```{.python .input}
np.ones(4)
```

```{.python .input}
#@tab pytorch
torch.ones(4)
```

```{.python .input}
#@tab tensorflow
tf.ones(4)
```

在Jupyter笔记本中，我们可以使用`？`在另一个窗口中显示文档。例如，list？`将创建与`help(list)`几乎相同的内容，并将其显示在新的浏览器窗口中。另外，如果我们使用两个问号，比如“list”？？`，也将显示实现该函数的Python代码。

## 摘要

* 官方文档提供了本书之外的大量描述和示例。
* 我们可以通过调用`dir`和`help`函数来查找API的用法文档，或者`？` and `？？`在Jupyter笔记本里。

## 练习

1. 查阅深度学习框架中任何函数或类的文档。你能在这个框架的官方网站上找到文件吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/199)
:end_tab:
