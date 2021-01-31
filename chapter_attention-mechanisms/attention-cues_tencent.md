# 注意提示
:label:`sec_attention-cues`

感谢您对这本书的关注。注意力是一种稀缺资源：此刻你正在读这本书，而忽略了睡觉。因此，与金钱相似，你的注意力是以机会成本支付的。为了确保您现在的注意力投入是值得的，我们非常有动力仔细地关注，以出版一本好书。注意力是生命之门的基石，是任何作品例外的关键。

自从经济学研究稀缺资源的配置以来，我们正处于注意力经济时代，在这个时代，人类的注意力被视为一种有限的、有价值的、稀缺的可以交换的商品。为了利用它，已经开发了许多商业模式。在音乐或视频流媒体服务上，我们要么关注他们的广告，要么花钱隐藏他们。对于网络游戏世界的增长，我们要么注重参与战斗，吸引新的游戏玩家，要么花钱瞬间变得强大。没有什么是免费的。

总而言之，我们环境中的信息并不稀缺，而是注意力稀缺。当观察视觉场景时，我们的视神经以每秒$10^8$比特的速度接收信息，远远超出了我们大脑所能完全处理的范围。幸运的是，我们的祖先已经从经验(也称为数据)中了解到，*并不是所有的感觉输入都是平等的*。纵观人类历史，只将注意力引导到感兴趣的一小部分信息上的能力使我们的大脑能够更灵活地分配资源来生存、成长和社交，比如检测捕食者、猎物和配偶。

## 生物学中的注意线索

为了解释我们的注意力是如何在视觉世界中展开的，一个由两个组件组成的框架应运而生，并得到了普及。这个想法可以追溯到19世纪90年代的威廉·詹姆斯，他被认为是“美国心理学之父”:cite:`James.2007`。在这个框架中，受试者使用*非意志线索*和*意志线索*选择性地引导注意力的聚光灯。

非意志性线索是基于环境中物体的显著程度和显着性的。想象一下你面前有五个物体：一份报纸、一份研究论文、一杯咖啡、一本笔记本和一本书，比如:numref:`fig_eye-coffee`。虽然所有的纸制品都是黑白印刷的，但咖啡杯是红色的。换句话说，这种咖啡在这个视觉环境中本质上是突出和显眼的，自动和不由自主地吸引了人们的注意。所以你把中心凹(黄斑中心，视觉敏锐度最高的地方)放到咖啡上，如图:numref:`fig_eye-coffee`所示。

![Using the nonvolitional cue based on saliency (red cup, non-paper), attention is involuntarily directed to the coffee.](../img/eye-coffee.svg)
:width:`400px`
:label:`fig_eye-coffee`

喝完咖啡后，你会变得咖啡因，想看书。所以你转过头，重新聚焦你的眼睛，看着这本书，就像:numref:`fig_eye-book`中描述的那样。与:numref:`fig_eye-coffee`中的情况不同，在这种依赖任务的情况下，你是在认知和意志的控制下选择书的，而在这种情况下，咖啡会让你偏向于根据显著性进行选择。使用基于变量选择标准的意志线索，这种注意形式更加深思熟虑。在受试者的自愿努力下，它也更加强大。

![Using the volitional cue (want to read a book) that is task-dependent, attention is directed to the book under volitional control.](../img/eye-book.svg)
:width:`400px`
:label:`fig_eye-book`

## 查询、键和值

受解释注意部署的非意志和意志注意线索的启发，下面我们将描述一个框架，通过合并这两个注意线索来设计注意机制。

首先，考虑更简单的情况，即只有非意志线索可用。要将选择偏向于感觉输入，我们可以简单地使用参数化的完全连接层，甚至使用非参数化的最大或平均池。

因此，将注意机制与那些完全相连的层或汇合层区分开来的是包含了意志线索。在注意机制的上下文中，我们将意志线索称为*查询*。在给定任何查询的情况下，注意机制通过*注意汇集*将选择偏向于感觉输入(例如，中间特征表示)。在注意力机制的背景下，这些感觉输入被称为“值”。更广泛地说，每个值都与一个*键*配对，它可以被认为是感官输入的非意志性提示。如:numref:`fig_qkv`中所示，我们可以设计注意力集中，以便给定的查询(意志线索)可以与按键(非意志线索)交互，从而引导对值(感官输入)的偏向选择。

![Attention mechanisms bias selection over values (sensory inputs) via attention pooling, which incorporates queries (volitional cues) and keys (nonvolitional cues).](../img/qkv.svg)
:label:`fig_qkv`

请注意，对于注意力机制的设计，有许多可供选择的方案。例如，我们可以设计可以使用强化学习方法:cite:`Mnih.Heess.Graves.ea.2014`训练的不可微注意力模型。鉴于框架在:numref:`fig_qkv`的主导地位，此框架下的模型将是我们在本章关注的中心。

## 注意力的可视化

平均合并可以视为输入的加权平均，其中权重是一致的。在实践中，注意池使用加权平均值聚合值，其中权重是在给定查询和不同键之间计算的。

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

为了可视化注意力权重，我们定义了`show_heatmaps`函数。其输入`matrices`具有形状(用于显示的行数、用于显示的列数、查询数、键数)。

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

为了进行演示，我们考虑一个简单的情况，其中只有当查询和键相同时，注意力权重才为1；否则为零。

```{.python .input}
#@tab all
attention_weights = d2l.reshape(d2l.eye(10), (1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
```

在接下来的小节中，我们将经常调用此函数来可视化注意力权重。

## 摘要

* 人类的注意力是一种有限的、有价值的、稀缺的资源。
* 受试者有选择地使用非意志性和意志性线索来引导注意力。前者基于显著性，后者依赖于任务。
* 由于包含了意志线索，注意机制不同于完全连通层或汇合层。
* 注意机制通过注意汇集使选择偏向于值(感觉输入)，它结合了询问(意志线索)和键(非意志线索)。键和值是成对的。
* 我们可以可视化查询和关键字之间的关注度权重。

## 练习

1. 在机器翻译中，当逐个令牌解码序列令牌时，什么是自愿提示？非意志的暗示和感觉输入是什么？
1. 随机生成$10 \times 10$矩阵，并使用SOFTMAX运算来确保每行都是有效的概率分布。将输出注意力权重可视化。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1596)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1592)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1710)
:end_tab:
