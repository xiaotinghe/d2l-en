# 注意提示
:label:`sec_attention-cues`

谢谢你对这本书的关注。注意力是一种稀缺的资源：此刻你正在读这本书而忽略了其余的。因此，和金钱一样，你的注意力也会受到机会成本的影响。为了确保您现在投入的注意力是值得的，我们一直非常积极地仔细关注，以产生一本好书。注意力是生命拱门的基石，是任何作品例外论的关键。

由于经济学研究的是稀缺资源的分配，我们正处于注意力经济时代，人们的注意力被视为一种有限的、有价值的、稀缺的、可以交换的商品。已经开发了许多商业模式来利用它。在音乐或视频流服务上，我们要么关注他们的广告，要么花钱隐藏他们。对于网络游戏世界的增长，我们要么关注参与战斗，吸引新玩家，要么花钱瞬间变得强大。没有什么是免费的。

总而言之，信息在我们的环境中并不稀缺，注意力才是。当我们观察一个视觉场景时，我们的视神经接收信息的速度是每秒$10^8$比特，远远超过了我们大脑所能完全处理的速度。幸运的是，我们的祖先从经验（也被称为数据）中学到，并非所有的感官输入都是平等的。纵观人类历史，将注意力仅引导到感兴趣信息的一小部分的能力使我们的大脑能够更聪明地分配资源，以生存、成长和社交，例如探测掠食者、猎物和配偶。

## 生物学中的注意线索

为了解释我们的注意力是如何在视觉世界中展开的，一个由两个部分组成的框架已经出现并普遍存在。这个想法可以追溯到19世纪90年代的威廉·詹姆斯，他被认为是“美国心理学之父”:cite:`James.2007`。在这个框架中，受试者有选择地使用“非政治线索”和“意志线索”引导注意力的焦点。

非政治暗示是基于环境中物体的显著性和显著性。想象一下你面前有五样东西：一份报纸、一份研究报告、一杯咖啡、一本笔记本和一本书，比如:numref:`fig_eye-coffee`。所有的纸制品都是黑白印刷的，而咖啡杯是红色的。换言之，这种咖啡在这种视觉环境中具有内在的显著性和显著性，自动地、不自觉地引起人们的注意。所以你把中央凹（视力最高的黄斑中心）放在咖啡上，如:numref:`fig_eye-coffee`所示。

![Using the nonvolitional cue based on saliency (red cup, non-paper), attention is involuntarily directed to the coffee.](../img/eye-coffee.svg)
:width:`400px`
:label:`fig_eye-coffee`

喝了咖啡后，你会变得含咖啡因，想看书。所以你转动你的头，重新聚焦你的眼睛，看着:numref:`fig_eye-book`所描述的书。与:numref:`fig_eye-coffee`中的案例不同，咖啡使你倾向于根据显著性进行选择，在这个任务相关的案例中，你在认知和意志控制下选择书。使用基于可变选择标准的意志线索，这种形式的注意更为谨慎。在受试者的自愿努力下，它也更强大。

![Using the volitional cue (want to read a book) that is task-dependent, attention is directed to the book under volitional control.](../img/eye-book.svg)
:width:`400px`
:label:`fig_eye-book`

## 查询、键和值

受非政治性和意志性注意线索的启发，我们将在下面描述一个框架，通过整合这两种注意线索来设计注意机制。

首先，考虑一个更简单的情况，只有非政治线索可用。为了使选择偏向于感官输入，我们可以简单地使用参数化的完全连接层，甚至非参数化的最大或平均池。

因此，将注意力机制与那些完全连接的层或池层区分开来的是包含了意志线索。在注意机制的上下文中，我们将意志线索称为*查询*。给定任何查询，注意机制通过*注意池*使选择偏向于感官输入（例如，中间特征表示）。在注意机制的上下文中，这些感觉输入被称为“值”。更一般地说，每个值都与一个*键*配对，这可以被认为是感官输入的非政治暗示。如:numref:`fig_qkv`所示，我们可以设计注意池，以便给定的查询（意志线索）可以与键（非政治线索）交互，从而引导对值（感官输入）的偏见选择。

![Attention mechanisms bias selection over values (sensory inputs) via attention pooling, which incorporates queries (volitional cues) and keys (nonvolitional cues).](../img/qkv.svg)
:label:`fig_qkv`

注意注意注意机制的设计有很多选择。例如，我们可以设计一个不可微注意模型，该模型可以使用强化学习方法:cite:`Mnih.Heess.Graves.ea.2014`进行训练。鉴于:numref:`fig_qkv`框架的主导地位，本章将重点关注该框架下的模型。

## 注意力可视化

平均池可以被视为输入的加权平均，其中权重是一致的。实际上，注意力池使用加权平均来聚合值，其中权重是在给定的查询和不同的键之间计算的。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

为了可视化注意力权重，我们定义了`show_heatmaps`函数。它的输入`matrices`具有以下形状（要显示的行数、要显示的列数、查询数、键数）。

```{.python .input}
#@tab all
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
```

为了演示，我们考虑了一个简单的情况，其中注意权重只有在查询和键相同时才为1，否则为0。

```{.python .input}
#@tab all
attention_weights = d2l.reshape(d2l.eye(10), (1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
```

在接下来的部分中，我们将经常调用此函数来可视化注意权重。

## 摘要

* 人类的注意力是一种有限的、有价值的、稀缺的资源。
* 受试者利用非政治性和意志性线索选择性地引导注意力。前者基于显著性，后者依赖于任务。
* 注意机制不同于完全连接层或池层，因为包含了意志线索。
* 注意机制通过注意池使选择偏向于价值观（感觉输入），注意池包括询问（意志线索）和关键（非政治线索）。键和值是成对的。
* 我们可以可视化查询和键之间的注意权重。

## 练习

1. 在机器翻译中，一个标记一个标记地解码序列时，什么是意志线索？什么是非政治暗示和感官输入？
1. 随机生成一个$10 \times 10$矩阵，并使用softmax操作确保每一行都是有效的概率分布。可视化输出注意权重。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1596)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1592)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1710)
:end_tab:
