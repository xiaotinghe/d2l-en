# 影像分类数据集
:label:`sec_fashion_mnist`

用于图像分类的广泛使用的数据集之一是MNIST数据集:cite:`LeCun.Bottou.Bengio.ea.1998`。虽然它作为基准数据集运行良好，但按照今天的标准，即使是简单的模型也能达到95%以上的分类准确率，这使得它不适合区分较强和较弱的模型。今天，MNIST更多的是作为一种理智的检查，而不是一种基准。为了加大赌注，我们将在接下来的几节中集中讨论2017年发布的质量相似但相对复杂的时尚-MNIST数据集:cite:`Xiao.Rasul.Vollgraf.2017`。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon
import sys

d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms
from torch.utils import data

d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

## 正在读取数据集

我们可以通过框架中的内置函数将Fashion-MNIST数据集下载并读取到内存中。

```{.python .input}
mnist_train = gluon.data.vision.FashionMNIST(train=True)
mnist_test = gluon.data.vision.FashionMNIST(train=False)
```

```{.python .input}
#@tab pytorch
# `ToTensor` converts the image data from PIL type to 32-bit floating point
# tensors. It divides all numbers by 255 so that all pixel values are between
# 0 and 1
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
```

```{.python .input}
#@tab tensorflow
mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
```

时尚-MNIST由来自10个类别的图像组成，每个类别由训练数据集中的6000个图像和测试数据集中的1000个图像表示。*测试数据集*(或*测试集*)用于评估模型性能，而不是用于训练。因此，训练集和测试集分别包含60000和10000个图像。

```{.python .input}
#@tab mxnet, pytorch
len(mnist_train), len(mnist_test)
```

```{.python .input}
#@tab tensorflow
len(mnist_train[0]), len(mnist_test[0])
```

每个输入图像的高度和宽度都是28像素。请注意，数据集由灰度图像组成，其通道数为1。为简洁起见，在本书中，我们将高度为$h$、宽度为$w$像素的任何图像的形状存储为$h \times w$或($h$,$w$)。

```{.python .input}
#@tab all
mnist_train[0][0].shape
```

Fashion-MNIST中的图像与以下类别相关：T恤、裤子、套头衫、连衣裙、外套、凉鞋、衬衫、运动鞋、包和踝靴。以下函数在文本中的数字标签索引及其名称之间进行转换。

```{.python .input}
#@tab all
def get_fashion_mnist_labels(labels):  #@save
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

现在我们可以创建一个函数来可视化这些示例。

```{.python .input}
#@tab all
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(d2l.numpy(img))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

以下是训练数据集中前几个示例的图像及其对应的标签(文本形式)。

```{.python .input}
X, y = mnist_train[:18]
show_images(X.squeeze(axis=-1), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab pytorch
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab tensorflow
X = tf.constant(mnist_train[0][:18])
y = tf.constant(mnist_train[1][:18])
show_images(X, 2, 9, titles=get_fashion_mnist_labels(y));
```

## 正在读取一个MiniBatch

为了使我们在从训练和测试集中读取数据时更加轻松，我们使用内置的数据迭代器，而不是从头开始创建数据迭代器。回想一下，在每次迭代中，数据加载器每次都读取大小为`batch_size`的小批量数据。我们还随机调整训练数据迭代器的示例。

```{.python .input}
batch_size = 256

def get_dataloader_workers():  #@save
    """Use 4 processes to read the data except for Windows."""
    return 0 if sys.platform.startswith('win') else 4

# `ToTensor` converts the image data from uint8 to 32-bit floating point. It
# divides all numbers by 255 so that all pixel values are between 0 and 1
transformer = gluon.data.vision.transforms.ToTensor()
train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                   batch_size, shuffle=True,
                                   num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
batch_size = 256

def get_dataloader_workers():  #@save
    """Use 4 processes to read the data."""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab tensorflow
batch_size = 256
train_iter = tf.data.Dataset.from_tensor_slices(
    mnist_train).batch(batch_size).shuffle(len(mnist_train[0]))
```

让我们看看读取培训数据所需的时间。

```{.python .input}
#@tab all
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
```

## 把所有的东西放在一起

现在我们定义获取和读取Fashion-MNIST数据集的`load_data_fashion_mnist`函数。它返回训练集和验证集的数据迭代器。此外，它还接受一个可选参数以将图像大小调整为另一个形状。

```{.python .input}
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    if resize:
        trans.insert(0, dataset.transforms.Resize(resize))
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab pytorch
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab tensorflow
def load_data_fashion_mnist(batch_size, resize=None):   #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    # Divide all numbers by 255 so that all pixel values are between
    # 0 and 1, add a batch dimension at the last. And cast label to int32
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (
        tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
    return (
        tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(
            batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
            batch_size).map(resize_fn))
```

下面，我们通过指定`load_data_fashion_mnist`参数来测试`resize`函数的图像大小调整功能。

```{.python .input}
#@tab all
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

现在，我们已经准备好在下面几节中使用Fashion-MNIST数据集。

## 摘要

* Fashion-MNIST是一个服装分类数据集，由代表10个类别的图像组成。在随后的章节和章节中，我们将使用此数据集来评估各种分类算法。
* 我们将高度为$h$、宽度为$w$像素的任何图像的形状存储为$h \times w$或($h$,$w$)。
* 数据迭代器是高效性能的关键组件。依靠利用高性能计算的实现良好的数据迭代器来避免减慢您的训练循环。

## 练习

1. 将`batch_size`(例如，减少到1)是否会影响阅读性能？
1. 数据迭代器的性能很重要。您认为目前的实施速度够快吗？探索各种选项来改进它。
1. 请查看该框架的在线API文档。还有哪些其他数据集可用？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/48)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/49)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/224)
:end_tab:
