# 数据预处理
:label:`sec_pandas`

到目前为止，我们已经介绍了各种技术来处理已经存储在张量中的数据。为了将深度学习应用于解决实际问题，我们通常从预处理原始数据开始，而不是以张量格式精心准备的数据。在Python中流行的数据分析工具中，`pandas`包是常用的。与Python庞大生态系统中的许多其他扩展包一样，`pandas`可以与张量一起工作。因此，我们将简要介绍一下使用`pandas`对原始数据进行预处理并将其转换为张量格式的步骤。我们将在后面的章节中介绍更多的数据预处理技术。

## 读取数据集

例如，我们首先创建一个人工数据集，该数据集存储在csv（逗号分隔值）文件`../data/house_tiny.csv`中。以其他格式存储的数据可以用类似的方式处理。下面的`mkdir_if_not_exist`函数确保目录`../data`存在。请注意，注释“#@save”是一个特殊标记，其中以下函数、类或语句保存在`d2l`包中，以便以后可以直接调用它们（例如`d2l.mkdir_if_not_exist(path)`），而无需重新定义。

```{.python .input}
#@tab all
import os

def mkdir_if_not_exist(path):  #@save
    """Make a directory if it does not exist."""
    if not isinstance(path, str):
        path = os.path.join(*path)
    if not os.path.exists(path):
        os.makedirs(path)
```

下面我们将数据集逐行写入csv文件。

```{.python .input}
#@tab all
data_file = '../data/house_tiny.csv'
mkdir_if_not_exist('../data')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Column names
    f.write('NA,Pave,127500\n')  # Each row represents a data example
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

要从创建的csv文件加载原始数据集，我们导入`pandas`包并调用73229365函数。这个数据集有四行三列，每行描述一个房子的房间数量（“NUMROOM”）、小巷类型（“小巷”）和房子的价格（“价格”）。

```{.python .input}
#@tab all
# If pandas is not installed, just uncomment the following line:
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## 处理丢失的数据

注意，“NaN”条目缺少值。为了处理缺失的数据，典型的方法包括*imputation*和*deletion*，其中插补用替换的代替缺失的值，而删除则忽略缺失的值。这里我们将考虑插补。

通过基于整数位置的索引（`iloc`），我们将`data`分为`inputs`和`outputs`，前者占据前两列，后者只保留最后一列。对于`inputs`中丢失的数值，我们用同一列的平均值替换“NaN”条目。

```{.python .input}
#@tab all
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

对于`inputs`中的范畴或离散值，我们将“NaN”视为一个范畴。由于“Alley”列只接受“Pave”和“NaN”两种类型的分类值，`pandas`可以自动将此列转换为“Alley_Pave”和“Alley_NaN”两个列。巷类型为“Pave”的行将把“alley_Pave”和“alley_nan”的值设置为1和0。缺少alley类型的行将其值设置为0和1。

```{.python .input}
#@tab all
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

## 转换为张量格式

现在73229365和`outputs`中的所有条目都是数字的，它们可以转换为张量格式。一旦数据以这种格式存在，就可以进一步使用我们在:numref:`sec_ndarray`中引入的张量功能来进一步操纵它们。

```{.python .input}
from mxnet import np

X, y = np.array(inputs.values), np.array(outputs.values)
X, y
```

```{.python .input}
#@tab pytorch
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

X, y = tf.constant(inputs.values), tf.constant(outputs.values)
X, y
```

## 摘要

* 与Python庞大生态系统中的许多其他扩展包一样，`pandas`可以与张量一起工作。
* 插补和删除可用于处理缺失数据。

## 练习

创建包含更多行和列的原始数据集。

1. 删除缺少值最多的列。
2. 将预处理后的数据集转换为张量格式。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/195)
:end_tab:
