# 模型选择、不足拟合和过度拟合
:label:`sec_model_selection`

作为机器学习科学家，我们的目标是发现*模式*。但是，我们如何才能确定我们已经真正发现了一种“一般”模式，而不是简单地记住了我们的数据呢？例如，假设我们想要在将患者与他们的痴呆症状态联系起来的遗传标记中寻找模式，其中标签是从集合$\{\text{dementia}, \text{mild cognitive impairment}, \text{healthy}\}$中提取的。因为每个人的基因都是唯一识别他们的(忽略完全相同的兄弟姐妹)，所以有可能记住整个数据集。

我们不想让我们的模型说
*“那是鲍勃！我记得他！他有痴呆症！”*
原因很简单。当我们将来部署该模型时，我们会遇到该模型从未见过的患者。只有当我们的模型真正发现了一种“一般”模式时，我们的预测才会有用。

更正式地说，我们的目标是发现模式，这些模式捕捉到了我们训练集所来自的潜在人群中的规律性。如果我们在这一努力中取得成功，那么我们就可以成功地评估风险，即使是对我们以前从未遇到过的个人来说也是如此。这个问题-如何发现“泛化”的模式-是机器学习的基本问题。

危险在于，当我们训练模型时，我们只能访问一小部分数据样本。最大的公共图像数据集包含大约一百万张图像。更多时候，我们只能从数千或数万个数据例子中学习。在大型医院系统中，我们可能会访问数十万份医疗记录。在处理有限的样本时，我们可能会冒着这样的风险，即当我们收集更多数据时，我们可能会发现明显的关联，而这些关联最终证明是站不住脚的。

将训练数据拟合得比潜在分布更接近的现象称为“过度拟合”，用于对抗过度拟合的技术称为“正则化”。在前面的部分中，您可能已经在试验Fashion-MNIST数据集时观察到了这种效果。如果在实验期间更改了模型结构或超参数，您可能已经注意到，如果有足够的神经元、层和训练周期，即使测试数据的准确性下降，模型最终也可以在训练集上达到完美的精度。

## 训练误差和泛化误差

为了更正式地讨论这一现象，我们需要区分训练误差和泛化误差。*训练误差*是我们的模型在训练数据集上计算的误差，而*推广误差*是我们将其应用于从与原始样本相同的底层数据分布中提取的无限多个附加数据示例的情况下模型误差的预期。

从问题上讲，我们永远不能准确地计算出泛化误差。这是因为无限数据流是一个虚构的对象。在实践中，我们必须通过将我们的模型应用于一个独立的测试集来“估计”泛化误差，该测试集由从我们的训练集中保留的随机选择的数据示例组成。

下面的三个思维实验将有助于更好地说明这种情况。假设一个大学生正在努力准备期末考试。一个勤奋的学生会努力练习好，并利用往年的考试来测试自己的能力。尽管如此，在过去的考试中取得好成绩并不能保证他会在重要的时候出类拔萃。例如，学生可能试图通过死记硬背考题的答案来做准备。这需要学生记住很多东西。她甚至可以完全记住过去考试的答案。另一名学生可能会通过试图理解给出某些答案的原因来做准备。在大多数情况下，后一个学生会做得更好。

同样，考虑一个简单地使用查找表回答问题的模型。如果允许的输入集合是离散的并且相当小，那么也许在查看*许多**训练示例之后，该方法将执行得很好。尽管如此，当面对它从未见过的例子时，这个模型没有比随机猜测更好的能力了。实际上，输入空间太大了，无法记住对应于每一个可以想到的输入的答案。例如，考虑黑白$28\times28$图像。如果每个像素可以取$256$个灰度值中的一个，则有$256^{784}$个可能的图像。这意味着低分辨率的灰度缩略图大小的图像比宇宙中的原子要多得多。即使我们可以遇到这样的数据，我们也永远负担不起存储查找表的费用。

最后，考虑尝试根据一些可能可用的上下文特征对掷硬币的结果(0类：正面，1类：反面)进行分类的问题。假设硬币是公平的。无论我们想出什么算法，泛化误差始终是$\frac{1}{2}$。然而，对于大多数算法，我们应该预期我们的训练误差会相当低，这取决于抽签的运气，即使我们没有任何功能！考虑数据集{0，1，1，1，0，1}。我们的无特征算法将不得不依赖于总是预测*多数类*，从我们有限的样本来看，它似乎是*1*。在这种情况下，总是预测类1的模型将产生$\frac{1}{3}$的误差，这比我们的泛化误差要好得多。随着数据量的增加，头部比例明显偏离$\frac{1}{2}$的可能性降低，我们的训练误差将与推广误差相匹配。

### 统计学习理论

由于泛化是机器学习中的基本问题，了解到许多数学家和理论家毕生致力于开发描述这一现象的形式理论，您可能不会感到惊讶。在他们的[同名theorem](https://en.wikipedia.org/wiki/Glivenko%E2%80%93Cantelli_theorem)，]中，格里文科和坎特利导出了训练误差收敛到泛化误差的速率。在一系列开创性的论文中，[Vapnik和Chervonenkis](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_theory)将这一理论扩展到更一般的函数类。这项工作为统计学习理论奠定了基础。

在标准的监督学习设置中，我们到目前为止一直在讨论这一问题，并将在本书的大部分内容中坚持使用，我们假设训练数据和测试数据都是从*相同的*分布中“独立”提取的。这通常被称为*I.I.D.。假设*，这意味着对我们的数据进行采样的过程没有内存。换句话说，抽取的第二个示例和第三个样本并不比抽取的第二个样本和第200万个样本的相关性更强。

要成为一名优秀的机器学习科学家需要批判性的思考，而且你应该已经在这个假设中找出漏洞，找出假设失败的常见情况。如果我们根据从加州大学旧金山分校医学中心的患者收集的数据培训死亡风险预报器，并将其应用于马萨诸塞州综合医院的患者，会怎么样？这些分布完全不一样。此外，抽签可能在时间上是相关的。如果我们对Tweet的主题进行分类呢？新闻周期会在正在讨论的话题中产生时间依赖性，这违反了任何独立的假设。

有时候我们只要轻微违反身份证就可以逍遥法外。假设和我们的模型将继续运行得非常好。毕竟，几乎每个现实世界的申请都至少涉及到一些轻微的身份识别违规行为。然而，我们有许多有用的工具可用于各种应用，如人脸识别、语音识别和语言翻译。

其他违规行为肯定会带来麻烦。例如，想象一下，如果我们试图训练一个人脸识别系统，只针对大学生进行培训，然后想要将其部署为一种工具，用于监测疗养院人口中的老年病。这不太可能起到很好的作用，因为大学生看起来往往与老年人有很大的不同。

在接下来的章节中，我们将讨论因违反身份证而引起的问题。假设。目前，即使拿到身份证。假设是理所当然的，理解泛化是一个可怕的问题。此外，阐明可能解释为什么深层神经网络泛化得如此好的精确理论基础，继续困扰着学习理论中的最伟大的人。

当我们训练我们的模型时，我们试图搜索一个尽可能符合训练数据的函数。如果该函数如此灵活，以至于它可以像捕捉真实关联一样容易地捕捉到虚假模式，那么它可能执行得“太好了”，而不会产生一个对看不见的数据进行很好概括的模型。这正是我们想要避免的，或者至少是想要控制的。深度学习中的许多技术都是启发式的和旨在防止过度适应的技巧。

### 模型复杂性

当我们有简单的模型和大量的数据时，我们期望泛化误差与训练误差相似。当我们处理更复杂的模型和更少的示例时，我们预计训练误差会下降，但泛化差距会增大。模型复杂性的确切构成是一个复杂的问题。一个模型是否能很好地推广，取决于很多因素。例如，具有更多参数的模型可能被认为更复杂。其参数可以采用更大范围值的模型可能更为复杂。通常，对于神经网络，我们认为需要更多训练迭代的模型比较复杂，而需要“提前停止”(较少训练迭代)的模型就不那么复杂。

很难比较本质上不同模型类的成员之间的复杂性(例如，决策树与神经网络)。就目前而言，一条简单的经验法则相当有用：统计学家认为，能够轻松解释任意事实的模型是复杂的，而表达能力有限但仍能很好地解释数据的模型可能更接近真相。在哲学上，这与波普尔的科学理论的可证伪性标准密切相关：如果一个理论符合数据，如果有具体的测试可以用来证明它是错误的，那么它就是好的。这一点很重要，因为所有的统计估计都是
*邮寄*，
也就是说，我们在观察事实之后进行估计，因此容易受到相关谬误的影响。目前，我们将把哲学放在一边，坚持更切实的问题。

在本节中，为了给您一些直观的印象，我们将重点介绍几个倾向于影响模型类的通用性的因素：

1. 可调参数的数量。当可调参数(有时称为*自由度*)的数量很大时，模型往往更容易过度拟合。
1. 参数采用的值。当权重的取值范围较大时，模型可能更容易过度拟合。
1. 训练样例的数量。即使您的模型很简单，覆盖只包含一个或两个示例的数据集也是非常容易的。但是，用数百万个示例来过度拟合一个数据集需要一个极其灵活的模型。

## 选型

在机器学习中，我们通常在评估几个候选模型后选择最终的模型。这个过程叫做“选型”。有时，需要进行比较的模型在本质上是完全不同的(比如，决策树与线性模型)。在其他时间，我们比较已经用不同的超参数设置训练的同一类模型的成员。

例如，对于MLP，我们可能希望比较具有不同数量的隐藏层、不同数量的隐藏单元以及应用于每个隐藏层的激活函数的各种选择的模型。为了确定候选模型中的最佳模型，我们通常会使用验证数据集。

### 验证数据集

原则上，在我们选择了所有的超参数之前，我们不应该接触我们的测试集。如果我们在模型选择过程中使用测试数据，可能会有过度拟合测试数据的风险。那我们就麻烦大了。如果我们过度匹配我们的训练数据，总会有对测试数据的评估来保持我们的诚实。但是如果我们过度拟合测试数据，我们怎么知道呢？

因此，我们决不能依赖试验数据进行模型选择。然而，我们也不能仅仅依靠训练数据来选择模型，因为我们不能估计我们用来训练模型的数据的泛化误差。

在实际应用中，情况变得更加模糊。虽然理想情况下我们只会触摸测试数据一次，以评估最好的模型或将少数模型相互比较，但现实世界的测试数据很少在使用一次后被丢弃。我们很少能负担得起每一轮实验的新测试集。

解决此问题的常见做法是将我们的数据分成三种方式，除了训练和测试数据集之外，还合并一个*验证数据集*(或*验证集*)。结果是一种模糊的实践，验证和测试数据之间的边界模糊得令人担忧。除非另有明确说明，否则在这本书的实验中，我们实际上是在使用应该被正确地称为训练数据和验证数据的东西，没有真正的测试集。因此，书中每次实验报告的准确度都是真正的验证准确度，而不是真正的测试集准确度。

### $K$倍交叉验证

当训练数据稀缺时，我们甚至可能没有能力提供足够的数据来构成一个适当的验证集。这个问题的一个流行的解决方案是采用$K$倍交叉验证*。这里，原始训练数据被分成$K$个不重叠的子集。然后执行$K$次模型训练和验证，每次在$K-1$个子集上进行训练，并在不同的子集(在该轮中没有用于训练的子集)上进行验证。最后，通过对$K$个实验的结果进行平均来估计训练和验证误差。

## 不太合身还是太合身？

当我们比较训练和验证错误时，我们要注意两种常见的情况。首先，我们要注意这样的情况：我们的训练错误和验证错误都很严重，但它们之间有一点差距。如果模型不能减少训练错误，这可能意味着我们的模型过于简单(即，表达能力不足)，无法捕获我们试图建模的模式。此外，由于我们的训练和验证错误之间的“泛化差距”很小，我们有理由相信我们可以用一个更复杂的模型逃脱惩罚。这种现象被称为“不合身”。

另一方面，正如我们上面所讨论的，当我们的训练误差明显低于我们的验证误差，表明严重的“过度拟合”时，我们要注意这种情况。请注意，过度贴身并不总是一件坏事。特别是在深度学习方面，众所周知，最好的预测模型在训练数据上的表现往往比在抵抗数据上好得多。最终，我们通常更关心验证错误，而不是训练错误和验证错误之间的差距。

我们是否过度匹配可能取决于我们模型的复杂性和可用训练数据集的大小，这两个主题将在下面讨论。

### 模型复杂性

为了说明一些关于过拟合和模型复杂性的经典直觉，我们用多项式给出了一个例子。给定由单个特征$x$和对应的实值标签$y$组成的训练数据，我们试图找到次数为$d$的多项式

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

以估计标签$y$。这只是一个线性回归问题，我们的特征是$x$的幂给出的，模型的权重是$w_i$给出的，偏差是$w_0$给出的，因为所有的$x$都是$x^0 = 1$。由于这只是一个线性回归问题，我们可以使用平方误差作为我们的损失函数。

由于高次多项式的参数较多，模型函数的选择范围较广，因此高次多项式函数比低次多项式函数复杂得多。在固定训练数据集的情况下，高次多项式函数相对于低次多项式的训练误差应该始终更低(最坏情况下是相等的)。事实上，只要每个数据样本都有$x$的不同值，次数等于数据样本数量的多项式函数就可以很好地拟合训练集。在:numref:`fig_capacity_vs_error`中，我们直观地描述了多项式次数和欠拟合与过拟合之间的关系。

![Influence of model complexity on underfitting and overfitting](../img/capacity_vs_error.svg)
:label:`fig_capacity_vs_error`

### 数据集大小

另一个需要牢记的重要因素是数据集的大小。修正我们的模型，训练数据集中的样本越少，我们遇到过拟合的可能性(也就越严重)。随着训练数据量的增加，泛化误差通常会减小。此外，一般来说，更多的数据不会有什么坏处。对于固定的任务和数据分布，通常在模型复杂性和数据集大小之间存在关系。给出更多的数据，我们可能会尝试拟合一个更复杂的模型，这可能是有益的。如果没有足够的数据，简单的模型可能更难击败。对于许多任务，深度学习只有在有数千个训练示例可用时才优于线性模型。在一定程度上，深度学习目前的成功要归功于互联网公司、廉价存储、互联设备以及广泛的经济数字化带来的海量数据集。

## 多项式回归

我们现在可以通过将多项式拟合到数据来交互地探索这些概念。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import math
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import numpy as np
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np
import math
```

### 生成数据集

首先，我们需要数据。给定$x$，我们将使用以下三次多项式来生成训练和测试数据的标签：

$$y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.1^2).$$

噪声项$\epsilon$服从均值为0且标准差为0.1的正态分布。对于优化，我们通常希望避免非常大的渐变值或损失。这就是将*功能*从$x^i$重新调整为$\frac{x^i}{i！}$的原因。它允许我们避免大指数$i$的非常大的值。我们将为训练集和测试集各合成100个样本。

```{.python .input}
#@tab all
max_degree = 20  # Maximum degree of the polynomial
n_train, n_test = 100, 100  # Training and test dataset sizes
true_w = np.zeros(max_degree)  # Allocate lots of empty space
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # `gamma(n)` = (n-1)!
# Shape of `labels`: (`n_train` + `n_test`,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)
```

同样，存储在`poly_features`中的单项式由GAMMA函数重新缩放，其中$\GAMMA(N)=(n-1)！$。看一下生成的数据集中的前2个样本。从技术上讲，值1是一个特征，即与偏置相对应的恒定特征。

```{.python .input}
#@tab pytorch, tensorflow
# Convert from NumPy ndarrays to tensors
true_w, features, poly_features, labels = [d2l.tensor(x, dtype=
    d2l.float32) for x in [true_w, features, poly_features, labels]]
```

```{.python .input}
#@tab all
features[:2], poly_features[:2, :], labels[:2]
```

### 对模型进行培训和测试

让我们首先实现一个函数来评估给定数据集的损失。

```{.python .input}
#@tab mxnet, tensorflow
def evaluate_loss(net, data_iter, loss):  #@save
    """Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        l = loss(net(X), y)
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_loss(net, data_iter, loss):  #@save
    """Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        y = d2l.reshape(y, out.shape)
        l = loss(out, y)
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]
```

现在定义训练函数。

```{.python .input}
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = gluon.loss.L2Loss()
    net = nn.Sequential()
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net.add(nn.Dense(1, use_bias=False))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    test_iter = d2l.load_array((test_features, test_labels), batch_size,
                               is_train=False)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': 0.01})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data().asnumpy())
```

```{.python .input}
#@tab pytorch
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())
```

```{.python .input}
#@tab tensorflow
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = tf.losses.MeanSquaredError()
    input_shape = train_features.shape[-1]
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1, use_bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    test_iter = d2l.load_array((test_features, test_labels), batch_size,
                               is_train=False)
    trainer = tf.keras.optimizers.SGD(learning_rate=.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net.get_weights()[0].T)
```

### 三次多项式函数拟合(正态)

我们将首先使用三阶多项式函数，它与数据生成函数的阶数相同。结果表明，该模型能有效降低训练损失和测试损失。学习的模型参数也接近真值$w = [5, 1.2, -3.4, 5.6]$。

```{.python .input}
#@tab all
# Pick the first four dimensions, i.e., 1, x, x^2/2!, x^3/3! from the
# polynomial features
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])
```

### 线性函数拟合(欠拟合)

让我们再看看线性函数拟合。在经历了早期的衰落之后，进一步减少该模式的训练损失变得困难起来。在最后一个历元迭代完成后，训练损失仍然很高。当用来拟合非线性模式(如这里的三次多项式函数)时，线性模型容易拟合不足。

```{.python .input}
#@tab all
# Pick the first two dimensions, i.e., 1, x, from the polynomial features
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])
```

### 高次多项式函数拟合(过拟合)

现在，让我们尝试使用过高的多项式来训练模型。这里，没有足够的数据来了解高次系数应该具有接近于零的值。因此，我们的过于复杂的模型是如此敏感，以至于它受到训练数据中的噪声的影响。虽然训练损失可以有效地降低，但测试损失仍然很高。结果表明，复杂模型对数据的拟合效果较差。

```{.python .input}
#@tab all
# Pick all the dimensions from the polynomial features
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)
```

在接下来的章节中，我们将继续讨论过度安装的问题和处理这些问题的方法，例如体重衰减和辍学。

## 摘要

* 由于不能基于训练误差来估计泛化误差，因此简单地最小化训练误差并不一定意味着泛化误差的减小。机器学习模型需要注意防止过拟合，以使泛化误差最小化。
* 验证集可以用于模型选择，前提是不能过于随意地使用它。
* 欠拟合是指模型不能减少训练误差。当训练误差远小于验证误差时，存在过拟合。
* 我们应该选择一个适当复杂的模型，避免使用不足的训练样本。

## 练习

1. 你能准确地解决这个多项式回归问题吗？提示：使用线性代数。
1. 考虑多项式的模型选择：
    1. 绘制训练损失与模型复杂度(多项式的次数)的关系图。你观察到了什么？您需要多项式的次数才能将训练损失减少到0？
    1. 在这种情况下绘制测试损失图。
    1. 根据数据量生成相同的曲线图。
1. 如果您放弃规范化($1/i！$) of the polynomial features $x^i$？你能用其他方法解决这个问题吗？
1. 您能期望看到零泛化错误吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/234)
:end_tab:
