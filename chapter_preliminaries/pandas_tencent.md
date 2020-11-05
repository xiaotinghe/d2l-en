# 数据预处理
:label:`sec_pandas`

到目前为止，我们已经介绍了各种用于操作已经存储在张量中的数据的技术。为了将深度学习应用于解决现实世界的问题，我们通常从预处理原始数据开始，而不是从那些以张量格式精心准备的数据开始。在Python语言中流行的数据分析工具中，常用的是`pandas`包。与Python庞大生态系统中的许多其他扩展包一样，`pandas`可以与张量一起工作。因此，我们将简要介绍使用`pandas`预处理原始数据并将其转换为张量格式的步骤。我们将在后面的章节中介绍更多的数据预处理技术。

## 正在读取数据集

作为示例，我们从创建存储在csv(逗号分隔值)文件`../data/house_tiny.csv`中的人工数据集开始。可以以类似的方式处理以其他格式存储的数据。下面的`mkdir_if_not_exist`函数确保目录`../data`存在。请注意，注释`#@save`是一个特殊标记，其中将以下函数、类或语句保存在`d2l`包中，以便以后无需重新定义即可直接调用(例如，`d2l.mkdir_if_not_exist(path)`)。

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

下面，我们将数据集逐行写入CSV文件。

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

要从创建的csv文件加载原始数据集，我们导入`pandas`包并调用`read_csv`函数。该数据集有四行三列，每行描述房间的数量(“NumRooms”)、巷子类型(“Alley”)和房子的价格(“Price”)。

```{.python .input}
#@tab all
# If pandas is not installed, just uncomment the following line:
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## 处理丢失的数据

请注意，“NaN”条目缺少值。要处理丢失的数据，典型的方法包括*补偿*和*删除*，其中补偿用替换的值替换丢失的值，而删除忽略丢失的值。在这里，我们将考虑归责。

通过基于整数位置的索引(`iloc`)，我们将`data`分为`inputs`和`outputs`，其中前者占据前两列，而后者只保留最后一列。对于`inputs`中缺少的数值，我们用同一列的平均值替换“NaN”条目。

```{.python .input}
#@tab all
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

对于`inputs`中的定义值或离散值，我们将“NaN”视为一个范畴。因为“ALLEY”列只接受两种类型的分类值“PAVE”和“NAN”，所以`pandas`可以自动将该列转换为两列“ALLEY_PAVE”和“ALLEY_NAN”。巷道类型为“Pave”的行将“ALLEY_PAVE”和“ALLEY_NAN”的值设置为1和0。缺少巷道类型的行将其值设置为0和1。

```{.python .input}
#@tab all
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

## 转换为张量格式

现在`inputs`和`outputs`中的所有条目都是数字的，可以将它们转换为张量格式。一旦数据采用这种格式，就可以使用我们在:numref:`sec_ndarray`中引入的那些张量功能进一步处理它们。

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
* 可以使用补偿和删除来处理丢失的数据。

## 练习

创建具有更多行和列的原始数据集。

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
