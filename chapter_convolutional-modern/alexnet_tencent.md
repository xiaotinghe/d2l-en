# 深卷积神经网络(AlexNet)
:label:`sec_alexnet`

虽然在引入LeNet之后，CNN在计算机视觉和机器学习领域广为人知，但它们并没有立即在该领域占据主导地位。虽然LENet在早期的小数据集上取得了很好的效果，但在更大、更现实的数据集上训练CNN的性能和可行性尚未确定。事实上，在20世纪90年代初和2012年分水岭结果之间的大部分时间里，神经网络经常被其他机器学习方法超越，比如支持向量机。

对于计算机视觉来说，这种比较可能是不公平的。也就是说，尽管卷积网络的输入由原始或轻微处理(例如，通过居中)的像素值组成，但从业者永远不会将原始像素馈送到传统模型中。取而代之的是，典型的计算机视觉流水线由人工设计的特征提取流水线组成。这些功能不是“学习功能”，而是“精心制作”的。大部分的进步来自于对功能有了更聪明的想法，而学习算法往往被放在了事后的考虑中。

虽然在20世纪90年代已经有了一些神经网络加速器，但它们还不足以制造具有大量参数的深部多通道、多层CNN。此外，数据集仍然相对较小。除了这些障碍之外，训练神经网络的关键技巧，包括参数初始化启发式、随机梯度下降的巧妙变体、非挤压激活函数和有效的正则化技术，仍然缺乏。

因此，与训练“端到端”(像素到分类)系统不同，经典管道看起来更像这样：

1. 获取有趣的数据集。在早期，这些数据集需要昂贵的传感器(当时，100万像素的图像是最先进的)。
2. 根据光学、几何学、其他分析工具的一些知识，偶尔根据幸运的研究生的偶然发现，用手工制作的特征对数据集进行预处理。
3. 通过一组标准的特征提取器，例如SIFT(比例不变特征变换):cite:`Lowe.2004`、冲浪(加速稳健特征):cite:`Bay.Tuytelaars.Van-Gool.2006`或任何数量的其他手动调谐管道来馈送数据。
4. 将生成的表示形式转储到您最喜欢的分类器(可能是线性模型或内核方法)中，以训练分类器。

如果你与机器学习研究人员交谈，他们认为机器学习既重要又美好。精妙的理论证明了各种量词的性质。机器学习领域欣欣向荣，严谨，而且非常有用。然而，如果你与计算机视觉研究人员交谈，你会听到一个非常不同的故事。他们会告诉你，图像识别的肮脏真相是，推动进步的是特征，而不是学习算法。计算机视觉研究人员理所当然地认为，比起任何学习算法，稍微更大或更干净的数据集或稍微改进的特征提取管道对最终精度的影响要大得多。

## 学习表征

另一种预测现状的方法是，管道中最重要的部分是代表性。直到2012年，这个表示都是机械计算的。事实上，设计一组新的特征函数，改进结果，并撰写方法是一种突出的论文体裁。SIFT :cite:`Lowe.2004`、冲浪:cite:`Bay.Tuytelaars.Van-Gool.2006`、HOG(定向梯度直方图):cite:`Dalal.Triggs.2005`、[bags of visual words](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision)和类似的特征提取器占据了主导地位。

另一组研究人员，包括Yann LeCun，Geoff Hinton，Yoshua Bengio，Andrew Ng，Shun-ichi Amari和Juergen Schmidhuber，有不同的计划。他们认为特征本身应该被学习。此外，他们认为，要想变得相当复杂，这些特征应该由多个共同学习的层分层组成，每个层都有可学习的参数。在图像的情况下，最低层可能会检测边缘、颜色和纹理。事实上，Alex Krizhevsky，Ilya Sutskever和Geoff Hinton提出了CNN的新变体，
*AlexNet*，
在2012 ImageNet挑战赛中取得优异表现。Alexnet是以亚历克斯·克里日夫斯基(Alex Krizhevsky)的名字命名的，他是突破性的图像网分类论文:cite:`Krizhevsky.Sutskever.Hinton.2012`的第一作者。

有趣的是，在网络的最低层，该模型学习了类似于一些传统过滤器的特征提取器。:numref:`fig_filters`是从ALEXNET纸:cite:`Krizhevsky.Sutskever.Hinton.2012`再现的，并且描述了较低级别的图像描述符。

![Image filters learned by the first layer of AlexNet.](../img/filters.png)
:width:`400px`
:label:`fig_filters`

网络中的更高层可能建立在这些表示的基础上，以表示更大的结构，如眼睛、鼻子、草叶等。甚至更高层可能代表整个物体，如人、飞机、狗或飞盘。最终，最终的隐藏状态学习图像的紧凑表示，该图像汇总其内容，以便容易地分离属于不同类别的数据。

虽然多层CNN的最终突破出现在2012年，但一个核心的研究小组一直致力于这一想法，多年来一直试图学习视觉数据的分层表示法。2012年的最终突破可以归因于两个关键因素。

### 缺少配料：数据

具有多层的深层模型需要大量数据才能进入其性能明显优于基于凸优化的传统方法(例如，线性方法和核方法)的区域。然而，考虑到计算机的存储能力有限，传感器的相对费用，以及20世纪90年代相对紧缩的研究预算，大多数研究都依赖于微小的数据集。许多论文介绍了UCI数据集的收集，其中许多只包含数百或(几个)数千张在非自然环境下拍摄的低分辨率图像。

2009年，ImageNet数据集发布，挑战研究人员从100万个例子中学习模型，每个例子来自1000个不同类别的物体。由引入这一数据集的李飞飞(Fei-fei Li)领导的研究人员利用谷歌图像搜索(Google Image Search)对每个类别的大型候选集进行预过滤，并使用亚马逊机械土耳其(Amazon Mechanical Turk)众包管道来确认每张图像是否属于相关的类别。这个规模是史无前例的。这项名为ImageNet Challenges的相关比赛推动了计算机视觉和机器学习的研究，挑战研究人员找出哪些模型在比学者之前认为的规模更大的范围内表现最好。

### 缺少的成分：硬件

深度学习模型是计算周期的贪婪消费者。训练可能需要数百个时代，并且每次迭代都需要通过计算代价高昂的多层线性代数运算来传递数据。这就是为什么在20世纪90年代和21世纪初，基于更高效的优化凸面目标的简单算法被首选的主要原因之一。

*事实证明，图形处理器*(GPU)改变了游戏规则
使深度学习变得可行。这些芯片长期以来一直是为了加速图形处理而开发的，以使计算机游戏受益。特别是，它们针对高吞吐量的$4 \times 4$矩阵矢量产品进行了优化，这些产品是许多计算机图形任务所需的。幸运的是，这个数学与计算卷积层所需的数学惊人地相似。大约在那个时候，NVIDIA和ATI已经开始针对一般计算操作优化GPU，甚至将其作为“通用GPU”(GPGPU)进行市场推广。

为了提供一些直观的信息，考虑一下现代微处理器(CPU)的内核。每个内核都相当强大，以很高的时钟频率运行，并且具有大容量缓存(高达几兆字节的L3)。每个内核都非常适合执行范围广泛的指令，具有分支预测器、深度流水线和其他使其能够运行各种程序的功能。然而，这种明显的优势也是它的致命弱点：通用内核的建造成本非常高。它们需要大量的芯片面积、复杂的支持结构(内存接口、核心之间的缓存逻辑、高速互连等等)，并且它们在任何单一任务中都相对较差。现代笔记本电脑最多有4个核心，即使是高端服务器也很少超过64个核心，原因很简单，因为它不划算。

相比之下，图形处理器由$100 \sim 1000$个小的处理单元组成(细节在英伟达、ATI、ARM和其他芯片供应商之间略有不同)，通常被分成更大的组(英伟达称之为WARPS)。虽然每个内核都相对较弱，有时甚至以低于1 GHz的时钟频率运行，但正是这些内核的总数使GPU比CPU快了几个数量级。例如，NVIDIA最新一代的Volta为专用指令提供了每个芯片高达120TFlops的性能(对于更通用的指令提供了高达24TFlops)，而CPU的浮点性能到目前为止还没有超过1TFLOP。实现这一目标的原因其实很简单：首先，功耗往往会随着时钟频率呈“二次曲线”增长。因此，对于运行速度快4倍(一个典型数字)的cpu核心的功率预算，您可以使用速度为$1/4$的16个gpu核心，这将产生$16 \times 1/4 = 4$倍的性能。此外，GPU内核要简单得多(事实上，在很长一段时间内，它们甚至不能*执行通用代码)，这使得它们更节能。最后，深度学习中的许多操作都需要很高的存储带宽。同样，GPU的总线宽度至少是CPU的10倍，在这里大放异彩。

回到2012年。当Alex Krizhevsky和Ilya Sutskever实现了可以在GPU硬件上运行的深度CNN时，一个重大突破出现了。他们意识到，CNN中的计算瓶颈，卷积和矩阵乘法，都是可以在硬件上并行化的操作。使用两台内存为3 GB的NVIDIA GTX 580，他们实现了快速卷积。代码[cuda-convnet](https://code.google.com/archive/p/cuda-convnet/)足够好了，几年来它一直是行业标准，并在深度学习热潮的头几年提供了动力。

## AlexNet

AlexNet使用了一个8层的CNN，以惊人的优势赢得了ImageNet 2012大型视觉识别挑战赛(ImageNet Large Scale Visual Recognition Challest 2012)。这个网络首次表明，通过学习获得的特征可以超越人工设计的特征，打破了以前计算机视觉的范式。

如:numref:`fig_alexnet`所示，AlexNet和LeNet的架构非常相似。请注意，我们提供了略微简化的AlexNet版本，消除了2012年需要的一些设计怪癖，以使该模型适合两个小型GPU。

![From LeNet (left) to AlexNet (right).](../img/alexnet.svg)
:label:`fig_alexnet`

AlexNet和LeNet的设计理念非常相似，但也有重大差异。首先，AlexNet比相对较小的LeNet5要深得多。AlexNet由八个层组成：五个卷积层、两个完全连接的隐藏层和一个完全连接的输出层。第二，AlexNet使用RELU而不是Sigmoid作为其激活函数。让我们深入研究下面的细节。

### 架构

在Alexnet的第一层中，卷积窗口形状为$11\times11$。由于ImageNet中的大多数图像都比MNIST图像高和宽十倍以上，因此ImageNet数据中的对象往往占用更多的像素。因此，需要更大的卷积窗口来捕获对象。第二层中的卷积窗口形状减小到$5\times5$，然后是$3\times3$。此外，在第一卷积层、第二卷积层和第五卷积层之后，网络增加了最大汇聚层，窗口形状为$3\times3$，步长为2，而且ALEXNET的卷积信道比LENet多10倍。

在最后一个卷积层之后，有两个具有4096个输出的完全连接层。这两个巨大的全连接层产生了近1 GB的模型参数。由于早期GPU的内存有限，最初的AlexNet采用了双数据流设计，因此它们的两个GPU中的每一个都只能负责存储和计算模型的一半。幸运的是，GPU内存现在相对充足，所以我们现在很少需要在GPU之间拆分模型(我们的AlexNet模型在这方面与最初的论文有所不同)。

### 激活函数

此外，AlexNet将Sigmoid激活功能改为更简单的REU激活功能。一方面，REU激活函数的计算更简单。例如，它没有在Sigmoid激活函数中找到的求幂运算。另一方面，当使用不同的参数初始化方法时，RELU激活功能使模型训练变得更容易。这是因为，当Sigmoid激活函数的输出非常接近0或1时，这些区域的梯度几乎为0，因此反向传播不能继续更新一些模型参数。相反，REU激活函数在正区间的梯度始终为1，因此，如果模型参数初始化不当，则S型函数在正区间可能获得几乎为0的梯度，从而不能有效地训练模型。

### 容量控制和预处理

AlexNet通过丢弃(:numref:`sec_dropout`)来控制完全连接层的模型复杂性，而LeNet仅使用权重衰减。为了进一步增加数据，AlexNet的训练循环添加了大量的图像增强功能，如翻转、裁剪和颜色更改。这使得模型更加稳健，更大的样本量有效地减少了过拟合。我们将在:numref:`sec_image_augmentation`中更详细地讨论数据增强。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
# Here, we use a larger 11 x 11 window to capture objects. At the same time,
# we use a stride of 4 to greatly reduce the height and width of the output.
# Here, the number of output channels is much larger than that in LeNet
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Make the convolution window smaller, set padding to 2 for consistent
        # height and width across the input and output, and increase the
        # number of output channels
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Use three successive convolutional layers and a smaller convolution
        # window. Except for the final convolutional layer, the number of
        # output channels is further increased. Pooling layers are not used to
        # reduce the height and width of input after the first two
        # convolutional layers
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Here, the number of outputs of the fully-connected layer is several
        # times larger than that in LeNet. Use the dropout layer to mitigate
        # overfitting
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        # Output layer. Since we are using Fashion-MNIST, the number of
        # classes is 10, instead of 1000 as in the paper
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

net = nn.Sequential(
    # Here, we use a larger 11 x 11 window to capture objects. At the same
    # time, we use a stride of 4 to greatly reduce the height and width of the
    # output. Here, the number of output channels is much larger than that in
    # LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # Make the convolution window smaller, set padding to 2 for consistent
    # height and width across the input and output, and increase the number of
    # output channels
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # Use three successive convolutional layers and a smaller convolution
    # window. Except for the final convolutional layer, the number of output
    # channels is further increased. Pooling layers are not used to reduce the
    # height and width of input after the first two convolutional layers
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # Here, the number of outputs of the fully-connected layer is several
    # times larger than that in LeNet. Use the dropout layer to mitigate
    # overfitting
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # Output layer. Since we are using Fashion-MNIST, the number of classes is
    # 10, instead of 1000 as in the paper
    nn.Linear(4096, 10))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def net():
    return tf.keras.models.Sequential([
        # Here, we use a larger 11 x 11 window to capture objects. At the same
        # time, we use a stride of 4 to greatly reduce the height and width of
        # the output. Here, the number of output channels is much larger than
        # that in LeNet
        tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # Make the convolution window smaller, set padding to 2 for consistent
        # height and width across the input and output, and increase the
        # number of output channels
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # Use three successive convolutional layers and a smaller convolution
        # window. Except for the final convolutional layer, the number of
        # output channels is further increased. Pooling layers are not used to
        # reduce the height and width of input after the first two
        # convolutional layers
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        # Here, the number of outputs of the fully-connected layer is several
        # times larger than that in LeNet. Use the dropout layer to mitigate
        # overfitting
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # Output layer. Since we are using Fashion-MNIST, the number of
        # classes is 10, instead of 1000 as in the paper
        tf.keras.layers.Dense(10)
    ])
```

我们构造了一个高度和宽度均为224的单通道数据示例，以观察各层的输出形状。它与:numref:`fig_alexnet`中的Alexnet架构相匹配。

```{.python .input}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'Output shape:\t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'Output shape:\t', X.shape)
```

## 正在读取数据集

虽然本文中对AlexNet进行了ImageNet方面的培训，但我们在这里使用的是Fashion-MNIST，因为即使在现代GPU上训练ImageNet模型进行聚合也可能需要数小时或数天的时间。将Alexnet直接应用于Fashion-MNIST的问题之一是其图像的分辨率($28 \times 28$像素)低于ImageNet图像。为了使其正常工作，我们将其上采样到$224 \times 224$(通常不是一个明智的做法，但是我们在这里这样做是为了忠实于Alexnet架构)。我们使用`resize`函数中的`d2l.load_data_fashion_mnist`参数执行此大小调整。

```{.python .input}
#@tab all
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
```

## 培训

现在，我们可以开始训练AlexNet了。与:numref:`sec_lenet`的LENET相比，这里的主要变化是使用了更小的学习率和更慢的训练速度，这是因为网络更深更广，图像分辨率更高，卷积的成本更高。

```{.python .input}
#@tab all
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## 摘要

* AlexNet具有与LeNet相似的结构，但使用了更多的卷积层和更大的参数空间来适应大规模的ImageNet数据集。
* 今天，AlexNet已经被更有效的体系结构超越，但它是当今使用的从浅层网络到深层网络的关键一步。
* 虽然AlexNet的实现似乎只比LeNet多几行，但学术界花了很多年才接受这一概念变化，并利用其出色的实验结果。这也是因为缺乏有效的计算工具。
* 辍学、重修和预处理是在计算机视觉任务中取得优异性能的其他关键步骤。

## 练习

1. 尝试增加纪元数。与乐网相比，结果有何不同？为什么？
1. 对于Fashion-MNIST数据集来说，AlexNet可能太复杂了。
    1. 尝试简化模型以使训练更快，同时确保准确度不会显著下降。
    1. 设计一个更好的模型，可以直接处理$28 \times 28$张图像。
1. 修改批量大小，观察精度和GPU内存的变化。
1. 分析了AlexNet的计算性能。
    1. AlexNet内存占用的主要部分是什么？
    1. 在AlexNet中，计算的主要部分是什么？
    1. 计算结果时内存带宽如何？
1. 将辍学和重修应用到LENet-5。情况好转了吗？预处理怎么样？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/75)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/76)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/276)
:end_tab:
