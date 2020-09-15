# 选型、欠拟合和过拟合
:label:`sec_model_selection`

作为机器学习科学家，我们的目标是发现*模式*。但是，我们怎样才能确定我们已经真正发现了一个*一般*模式，而不是简单地记忆我们的数据呢？例如，想象一下，我们想在基因标记中寻找将病人与痴呆状态联系起来的模式，其中的标签来自$\{\text{dementia}, \text{mild cognitive impairment}, \text{healthy}\}$。因为每个人的基因唯一地识别他们（忽略相同的兄弟姐妹），所以有可能记住整个数据集。

我们不想让我们的模型说
*“那是鲍勃！我记得他！他得了痴呆症！”*
原因很简单。当我们将来部署该模型时，我们将遇到模型从未见过的患者。我们的预测只有在我们的模型真正发现了一个*一般*模式的情况下才会有用。

为了更正式地重述一下，我们的目标是发现捕捉基础人群中规律性的模式，我们的训练集就是从中提取的。如果我们在这方面取得了成功，那么我们甚至可以成功地评估我们从未遇到过的个人的风险。这个问题——如何发现泛化的模式——是机器学习的基本问题。

危险在于，当我们训练模型时，我们只访问一小部分数据样本。最大的公共图像数据集包含大约一百万张图像。更多的时候，我们必须从成千上万的数据示例中学习。在一个大型医院系统中，我们可能会访问数十万份医疗记录。当我们使用有限的样本时，我们可能会冒着这样的风险：当我们收集更多的数据时，我们可能会发现明显的关联性，而这些关联最终却无法成立。

将我们的训练数据拟合得比我们对潜在分布拟合得更紧密的现象称为“过度拟合”，而用于防止过度拟合的技术称为“正则化”。在前面的部分中，您可能在使用时尚MNIST数据集时观察到了这种效果。如果你在实验过程中改变了模型结构或超参数，你可能已经注意到，有了足够的神经元、层和训练时间，模型最终可以在训练集上达到完美的精度，即使测试数据的精确度下降了。

## 训练误差和泛化误差

为了更正式地讨论这一现象，我们需要区分训练误差和泛化误差。*training error*是我们在训练数据集上计算的模型误差，而*generalization error*是我们对模型误差的预期，如果我们将其应用于从与原始样本相同的基础数据分布中提取的无限多个额外数据示例。

有问题的是，我们永远无法精确计算泛化误差。这是因为无限数据流是一个虚构的对象。在实践中，我们必须通过将我们的模型应用于一个独立的测试集，该测试集由从我们的训练集中截取的数据样本组成。

以下三个思考实验将有助于更好地说明这种情况。假设一个大学生正在准备期末考试。一个勤奋的学生会努力练习好，用往年的考试来检验自己的能力。尽管如此，在过去的考试中取得好成绩并不能保证他会在重要的时候取得优异成绩。例如，学生可以通过死记硬背来准备考试问题的答案。这需要学生记住很多东西。她甚至可以完美地记住过去考试的答案。另一个学生可能会试图理解给出某些答案的原因。在大多数情况下，后一个学生会做得更好。

同样，考虑一个简单地使用查找表来回答问题的模型。如果允许输入的集合是离散的并且相当小，那么在查看了*许多*训练示例之后，这种方法将表现良好。然而，当面对以前从未见过的例子时，这个模型没有比随机猜测更好的能力。实际上，输入空间太大，无法记住与每个可想象的输入相对应的答案。例如，考虑黑白$28\times28$图像。如果每个像素可以取$256$灰度值中的一个，则存在$256^{784}$个可能的图像。这意味着低分辨率灰度缩略图像的数量远远超过宇宙中原子的数量。即使我们能遇到这样的数据，我们也永远付不起存储查找表的费用。

最后，考虑一下根据一些可能可用的上下文特征对掷硬币的结果（0级：正面，1级：反面）进行分类的问题。假设硬币是公平的。不管我们提出什么算法，泛化误差总是$\frac{1}{2}$。然而，对于大多数算法，我们应该期望我们的训练误差会大大降低，这取决于抽签的运气，即使我们没有任何特征！考虑数据集{0，1，1，1，0，1}。我们的无特征算法必须依赖于总是预测*多数类*，从我们有限的样本来看，它似乎是*1*。在这种情况下，总是预测类1的模型将产生$\frac{1}{3}$的错误，这比我们的泛化错误要好得多。随着数据量的增加，头的分数显著偏离$\frac{1}{2}$的概率减小，我们的训练误差将与泛化误差相匹配。

### 统计学习理论

由于泛化是机器学习中的基本问题，您可能不会惊讶于许多数学家和理论家毕生致力于发展形式化理论来描述这种现象。在他们的[同名定理](https://en.wikipedia.org/wiki/Glivenko%E2%80%93Cantelli_定理)Glivenko和Cantelli导出了训练误差收敛到泛化误差的速率。在一系列开创性的论文中(https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_理论)把这个理论推广到更一般的函数类。这项工作奠定了统计学习理论的基础。

在标准的监督学习设置中，我们到目前为止一直在讨论，并且在本书的大部分内容中，我们假设训练数据和测试数据都是从相同的分布中独立提取的。这通常被称为“i.i.d.假设”，这意味着对数据进行采样的过程没有内存。换句话说，抽取的第二个样本和第三个样本的相关性并不比第二个样本和第两百万个样本的相关性高。

作为一个好的机器学习科学家需要批判性的思考，你应该已经在这个假设上捅了个洞，想出一些假设失败的常见案例。如果我们根据从加州大学旧金山分校医学中心收集的数据训练一个死亡率风险预测因子，并将其应用于马萨诸塞州综合医院的患者，会怎么样？这些分布完全不同。此外，绘制可能在时间上是相关的。如果我们把推特的话题分类呢？新闻周期会在所讨论的主题中产生时间依赖性，违反任何独立性假设。

有时，我们可以逃脱对i.i.d.假设的轻微违反，我们的模型将继续非常有效地工作。毕竟，几乎每一个现实世界中的应用程序都至少涉及到一些轻微的违反i.i.d.假设的情况，然而我们有许多有用的工具用于各种应用，如人脸识别、语音识别和语言翻译。

其他违规行为肯定会引起麻烦。试想一下，例如，如果我们试图训练一个面部识别系统，只对大学生进行训练，然后想把它作为一个工具来监测养老院人群中的老年病。这不太可能奏效，因为大学生看起来往往与老年人大不相同。

在接下来的章节中，我们将讨论因违反i.i.d.假设而产生的问题。目前，即使把i.i.d.假设视为理所当然，理解泛化也是一个令人生畏的问题。此外，解释精确的理论基础，也许可以解释为什么深层神经网络会像他们那样泛化，继续困扰学习理论中最伟大的头脑。

当我们训练我们的模型时，我们试图寻找一个尽可能适合训练数据的函数。如果函数非常灵活，它可以像捕捉真正的关联一样容易地捕捉到虚假的模式，那么它可能会执行得*太好*而不会生成一个能很好地推广到不可见数据的模型。这正是我们想要避免或至少控制的。深度学习中的许多技术都是旨在防止过度适应的启发式和技巧。

### 模型复杂性

当模型简单，数据丰富时，我们期望泛化误差与训练误差相似。当我们使用更复杂的模型和更少的例子时，我们期望训练误差减小，而泛化差距增大。精确地构成模型复杂性的是一个复杂的问题。一个模型是否能很好地推广有许多因素。例如，具有更多参数的模型可能会被认为更复杂。一个参数可以取更大范围值的模型可能更复杂。通常对于神经网络，我们认为一个需要更多训练迭代的模型更复杂，而一个需要提前停止（较少的训练迭代）的模型就不那么复杂。

很难比较本质上不同的模型类成员之间的复杂性（例如，决策树与神经网络）。就目前而言，一个简单的经验法则是相当有用的：一个能轻易解释任意事实的模型是统计学家认为复杂的，而表达能力有限但仍能很好地解释数据的模型可能更接近事实。在哲学上，这与波普尔关于科学理论可证伪性的标准密切相关：如果一个理论符合数据，并且有具体的测试可以用来证明它是正确的。这一点很重要，因为所有的统计估计
*事后*，
i、 我们在观察事实之后进行估计，因此容易受到相关谬论的影响。目前，我们将把哲学放在一边，坚持更具体的问题。

在本节中，为了给您一些直观的印象，我们将集中讨论一些影响模型类泛化的因素：

1. 可调参数的数目。当可调参数（有时称为“自由度”）的数量很大时，模型更容易过度拟合。
1. 参数获取的值。当权重可以取更大范围的值时，模型更容易过度拟合。
1. 训练实例的数量。即使模型很简单，也很容易过度拟合只包含一个或两个示例的数据集。但是，用数百万个例子过度拟合数据集需要一个非常灵活的模型。

## 选型

在机器学习中，我们通常在评估几个候选模型后选择最终模型。此过程称为*型号选择*。有时，需要比较的模型本质上是不同的（比如，决策树与线性模型）。其他时候，我们比较的是同一类模型的成员，这些模型已经过不同的超参数设置训练。

例如，对于mlp，我们可能希望比较具有不同数量的隐藏层、不同数量的隐藏单元以及应用于每个隐藏层的激活函数的各种选择的模型。为了确定候选模型中的最佳模型，我们通常会使用一个验证数据集。

### 验证数据集

原则上，在我们选择了所有的超参数之后，我们才应该接触我们的测试集。如果我们在模型选择过程中使用测试数据，则有可能过度拟合测试数据。那我们就麻烦大了。如果我们过度拟合我们的训练数据，总会有对测试数据的评估来保持我们的诚实。但是如果我们过度拟合测试数据，我们怎么会知道呢？

因此，我们决不能依赖试验数据来选择模型。然而，我们也不能仅仅依靠训练数据来选择模型，因为我们不能根据我们用来训练模型的数据来估计泛化误差。

在实际应用中，图像变得更加模糊。虽然理想情况下我们只接触一次测试数据，以评估最佳模型或将少量模型相互比较，但实际测试数据很少会在一次使用后被丢弃。我们很少能负担得起每一轮实验的新测试设备。

解决这个问题的常见做法是将我们的数据分成三种方式，除了训练和测试数据集外，还合并一个*验证数据集*（或*验证集*）。结果是一个模糊的实践，验证和测试数据之间的界限令人担忧地模糊不清。除非另有明确说明，在这本书中的实验中，我们实际使用的是正确的训练数据和验证数据，没有真正的测试集。因此，本书每次实验报告的准确度实际上是验证准确度，而不是真实的测试集准确度。

### $K$折叠交叉验证

当训练数据不足时，我们甚至可能无法提供足够的数据来构成一个适当的验证集。解决这个问题的一个流行的解决方案是使用$K$*折叠交叉验证*。在这里，原始训练数据被分成$K$个不重叠的子集。然后进行$K$次模型训练和验证，每次训练$K-1$个子集，并在不同的子集上验证（该轮中没有用于训练的子集）。最后，通过对$K$个实验结果的平均值来估计训练和验证误差。

## 不合身还是过度合身？

当我们比较培训和验证错误时，我们需要注意两种常见情况。首先，我们希望注意这样的情况：我们的培训错误和验证错误都是巨大的，但它们之间有一点差距。如果模型不能减少训练误差，这可能意味着我们的模型过于简单（即，表达能力不足），无法捕捉我们试图建模的模式。此外，由于我们的训练和验证误差之间的“泛化差距”很小，我们有理由相信我们可以用一个更复杂的模型逃脱惩罚。这种现象被称为“不合身”。

另一方面，正如我们上面所讨论的，当我们的训练误差明显低于我们的验证误差时，我们要注意这种情况，这表明严重的“过度拟合”。请注意，过拟合并不总是坏事。特别是在深入学习的情况下，众所周知，最好的预测模型在训练数据上的表现往往比在保留数据上表现得好得多。最终，我们通常更关心验证错误，而不是培训和验证错误之间的差距。

我们是否过拟合或欠拟合都取决于模型的复杂性和可用训练数据集的大小，这两个主题将在下面讨论。

### 模型复杂性

为了说明一些关于过拟合和模型复杂性的经典直觉，我们给出了一个使用多项式的例子。给定由单一特征$x$和相应的实值标签$x$组成的训练数据，我们试图找到$d$次多项式

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

估计标签$y$。这只是一个线性回归问题，我们的特征由$x$的幂次给出，模型的权重由$w_i$给出，而偏差是$x^0 = 1$以来的$x$。因为这是一个线性损失函数。

高阶多项式函数比低阶多项式函数更复杂，因为高阶多项式具有更多的参数，且模型函数的选择范围更广。在固定训练数据集的情况下，高阶多项式函数相对于低阶多项式的训练误差总是较低的（最坏的情况是相等的）。事实上，当每个数据示例都有一个不同的值$x$时，一个次数等于数据示例数的多项式函数可以完美地拟合训练集。在:numref:`fig_capacity_vs_error`中，我们将多项式次数与欠拟合与过拟合之间的关系可视化。

![Influence of model complexity on underfitting and overfitting](../img/capacity_vs_error.svg)
:label:`fig_capacity_vs_error`

### 数据集大小

另一个需要记住的重要因素是数据集的大小。修正我们的模型，训练数据集中的样本越少，我们就越有可能（而且更严重）遇到过度拟合。随着训练数据量的增加，泛化误差通常会减小。此外，总的来说，更多的数据不会造成伤害。对于固定的任务和数据分布，模型复杂度和数据集大小之间通常存在关系。如果有更多的数据，我们可以尝试拟合一个更复杂的模型。如果没有足够的数据，更简单的模型可能更难被击败。对于许多任务，深度学习只有在成千上万个训练实例可用时才能优于线性模型。在一定程度上，深度学习目前的成功归功于互联网公司、廉价存储、联网设备以及经济的广泛数字化所带来的海量数据集。

## 多项式回归

我们现在可以通过拟合多项式来交互式地研究这些概念。

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

首先我们需要数据。给定$x$，我们将使用以下三次多项式生成训练和测试数据上的标签：

$$y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.1^2).$$

噪声项$\epsilon$服从正态分布，平均值为0，标准偏差为0.1。对于优化，我们通常希望避免非常大的梯度值或损失。这就是为什么*features*从$x^i$重新缩放到$\frac{x^i}{i！}$. 它允许我们避免大指数$i$的很大值。我们将合成训练集和测试集各100个样本。

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

同样，存储在`poly_features`中的单项式被gamma函数重新缩放，其中$\gamma（n）=（n-1）！$. 从生成的数据集中查看前2个示例。值1在技术上是一个特征，即与偏差相对应的常量特征。

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

### 培训和测试模型

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

现在定义培训功能。

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

### 三阶多项式函数拟合（正态）

我们将首先使用三阶多项式函数，它与数据生成函数的阶数相同。结果表明，该模型能有效地减少训练损失和测试损失。学习的模型参数也接近真实值$w = [5, 1.2, -3.4, 5.6]$。

```{.python .input}
#@tab all
# Pick the first four dimensions, i.e., 1, x, x^2/2!, x^3/3! from the
# polynomial features
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])
```

### 线性函数拟合（欠拟合）

让我们再来看看线性函数拟合。在经历了早期衰退之后，要进一步降低这种模型的训练损失就变得困难了。最后一次历元迭代完成后，训练损失仍然很大。当用于拟合非线性模式（如这里的三阶多项式函数）时，线性模型容易欠拟合。

```{.python .input}
#@tab all
# Pick the first two dimensions, i.e., 1, x, from the polynomial features
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])
```

### 高阶多项式函数拟合（过拟合）

现在让我们试着用高次多项式来训练模型。这里，没有足够的数据来了解高次系数的值应该接近于零。因此，我们过于复杂的模型非常容易受到训练数据中噪声的影响。虽然可以有效地减少训练损失，但测试损失仍然很大。结果表明，复杂模型对数据拟合过度。

```{.python .input}
#@tab all
# Pick all the dimensions from the polynomial features
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)
```

在后面的章节中，我们将继续讨论过度拟合的问题和处理它们的方法，例如重量衰减和脱落。

## 摘要

* 由于不能根据训练误差来估计泛化误差，所以简单地最小化训练误差并不一定意味着泛化误差的减少。机器学习模型需要注意防止过度拟合，以尽量减少泛化误差。
* 一个验证集可以用于模型选择，前提是它的使用不是太自由。
* 欠拟合意味着模型不能减少训练误差。当训练误差远小于验证误差时，存在过拟合现象。
* 应适当选择样本不足的样本进行训练。

## 练习

1. 你能准确地解决多项式回归问题吗？提示：使用线性代数。
1. 考虑多项式的模型选择：
    1. 绘制训练损失与模型复杂度（多项式次数）的关系图。你观察到了什么？将训练损失减少到0需要多少次多项式？
    1. 在这种情况下绘制测试损失图。
    1. 生成与数据量函数相同的绘图。
1. 如果你放弃正常化（$1/i）会怎么样！$) of the polynomial features $x^i$？你能用别的方法解决这个问题吗？
1. 你能期望看到零泛化误差吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/234)
:end_tab:
