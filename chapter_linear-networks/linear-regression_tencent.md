# 线性回归
:label:`sec_linear_regression`

*回归*指的是一组建模方法
一个或多个自变量与因变量之间的关系。在自然科学和社会科学中，回归的目的往往是
*描述输入和输出之间的关系。
另一方面，机器学习最关心的是“预测”。

每当我们想要预测一个数值时，回归问题就会出现。常见的例子包括预测价格(房屋、股票等)、预测住院时间(针对住院患者)、需求预测(针对零售额)等等。并不是每个预测问题都是经典的回归问题。在接下来的部分中，我们将介绍分类问题，其中的目标是预测一组类别中的成员资格。

## 线性回归的基本要素

*线性回归*可能既是最简单的
并且在回归的标准工具中最受欢迎。追溯到19世纪的黎明，线性回归源于几个简单的假设。首先，我们假设自变量$\mathbf{x}$和因变量$y$之间的关系是线性的，即，给定观测中的一些噪声，$y$可以表示为$\mathbf{x}$中的元素的加权和。其次，我们假设任何噪声都是良好的(遵循高斯分布)。

为了激励这种方法，让我们从一个运行的示例开始。假设我们希望根据面积(平方英尺)和楼龄(年)来估计房价(以美元为单位)。要真正开发一个预测房价的模型，我们需要获得一个包含销售量的数据集，该数据集包含我们知道每套房屋的销售价格、面积和年限。在机器学习的术语中，数据集被称为*训练数据集*或*训练集*，并且每一行(这里对应于一个销售的数据)被称为*示例*(或*数据点*、*数据实例*、*样本*)。我们试图预测(价格)的东西叫做*标签*(或*目标*)。预测所依据的自变量(年龄和面积)称为*特征*(或*协变量*)。

通常，我们将使用$n$来表示数据集中的示例数量。我们将数据示例编入$i$的索引，将每个输入表示为$\mathbf{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}]^\top$，将对应的标签表示为$y^{(i)}$。

### 线性模型
:label:`subsec_linear_model`

线性假设只是说目标(价格)可以表示为特征(面积和年龄)的加权和：

$$\mathrm{price} = w_{\mathrm{area}} \cdot \mathrm{area} + w_{\mathrm{age}} \cdot \mathrm{age} + b.$$
:eqlabel:`eq_price-area`

在:eqref:`eq_price-area`中，$w_{\mathrm{area}}$和$w_{\mathrm{age}}$称为*权重*，$b$称为*偏差*(也称为*偏移*或*截取*)。权重决定了每个特征对我们预测的影响，而偏差只表示当所有特征取值0时，预测价格应该取什么值。即使我们永远看不到面积为零的房屋，或者正好是零岁的房屋，我们仍然需要偏见，否则我们将限制我们模型的表现力。严格地说，:eqref:`eq_price-area`是输入特征的“仿射变换”，其特征是通过加权和对特征进行“线性变换”，并通过增加的偏差进行*平移*。

给定一个数据集，我们的目标是选择权重$\mathbf{w}$和偏差$b$，以便平均而言，根据我们的模型做出的预测最符合在数据中观察到的真实价格。其输出预测由输入特征的仿射变换确定的模型是*线性模型*，其中仿射变换由所选择的权重和偏差指定。

在通常只关注只有几个特性的数据集的学科中，像这样显式地表达模型的长格式是很常见的。在机器学习中，我们通常处理高维数据集，因此使用线性代数表示法更方便。当我们的输入由$d$个要素组成时，我们将预测$\hat{y}$(通常“帽子”符号表示估计值)表示为

$$\hat{y} = w_1  x_1 + ... + w_d  x_d + b.$$

将所有特征收集到向量$\mathbf{x} \in \mathbb{R}^d$中，并将所有权重收集到向量$\mathbf{w} \in \mathbb{R}^d$中，我们可以使用点积来紧凑地表示我们的模型：

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b.$$
:eqlabel:`eq_linreg-y`

在:eqref:`eq_linreg-y`中，向量$\mathbf{x}$对应于单个数据示例的特征。我们经常会发现，通过*设计矩阵*$\mathbf{X} \in \mathbb{R}^{n \times d}$引用包含$n$个示例的整个数据集的功能非常方便。在这里，$\mathbf{X}$为每个示例包含一行，为每个特性包含一列。

对于特征$\mathbf{X}$的集合，预测$\hat{\mathbf{y}} \in \mathbb{R}^n$可以通过矩阵向量乘积来表示：

$${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b,$$

其中在求和期间应用广播(见:numref:`subsec_broadcasting`)。给定训练数据集$\mathbf{X}$的特征和对应的(已知)标签$\mathbf{y}$，线性回归的目标是找到权重向量$\mathbf{w}$和偏差项$b$，其给定从与$\mathbf{X}$相同的分布采样的新数据示例的特征，将(在期望中)以最低误差预测新示例的标签。

即使我们认为预测$y$(给定$\mathbf{x}$)的最佳模型是线性的，我们也不会期望找到包含$n$个示例的真实世界数据集，其中$y^{(i)}$正好等于所有$1 \leq i \leq n$个示例的$\mathbf{w}^\top \mathbf{x}^{(i)}+b$。例如，我们用来观察特征$\mathbf{X}$和标签$\mathbf{y}$的任何仪器都可能遭受少量的测量误差。因此，即使我们确信潜在的关系是线性的，我们也会加入噪声项来解释这些错误。

在我们可以着手搜索最佳*参数*(或*模型参数*)$\mathbf{w}$和$b$之前，我们还需要两件事：(I)某个给定模型的质量度量；(Ii)更新模型以改进其质量的过程。

### 损失函数

在我们开始考虑如何将数据与我们的模型“匹配”之前，我们需要确定一个“适合度”的度量。损失函数*量化目标的“真实”值和“预测”值之间的距离。损失通常是一个非负数，值越小越好，完美预测的损失为0。回归问题中最常用的损失函数是误差平方。当我们对示例$i$的预测是$\hat{y}^{(i)}$并且对应的真实标签是$y^{(i)}$时，平方误差由下式给出：

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$

常量$\frac{1}{2}$没有真正的区别，但在公证上是方便的，当我们取损失的导数时，它就会被抵消。由于训练数据集是给我们的，因此不受我们的控制，所以经验误差只是模型参数的函数。为了使事情更具体，请考虑下面的示例，在该示例中，我们为一维情况绘制一个回归问题，如:numref:`fig_fit_linreg`中所示。

![Fit data with a linear model.](../img/fit-linreg.svg)
:label:`fig_fit_linreg`

注意，由于二次相关性，估计$\hat{y}^{(i)}$和观测$y^{(i)}$之间的大差异导致对损失的更大贡献。为了在包含$n$个示例的整个数据集上衡量模型的质量，我们只需对训练集上的损失进行平均(或等效地求和)。

$$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

在训练模型时，我们希望找到使所有训练示例的总损失最小化的参数($\mathbf{w}^*, b^*$)：

$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b).$$

### 解析解

线性回归恰好是一个异常简单的最优化问题。与我们在本书中将遇到的大多数其他模型不同，线性回归可以通过应用一个简单的公式进行解析求解。首先，我们可以通过将一列附加到由所有1组成的设计矩阵来将偏置$b$包含到参数$\mathbf{w}$中。那么我们的预测问题是最小化$\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$。在损耗面上只有一个临界点，它对应于整个区域的最小损耗。取损耗相对于$\mathbf{w}$的导数，并将其设置为零，可得到解析(闭合形式)解：

$$\mathbf{w}^* = (\mathbf X^\top \mathbf X)^{-1}\mathbf X^\top \mathbf{y}.$$

虽然像线性回归这样的简单问题可能会有解析解，但你不应该习惯这么好的运气。虽然解析解可以进行很好的数学分析，但是解析解的要求是非常严格的，它会将所有的深度学习都排除在外。

### 小批量随机梯度下降

即使在我们不能解析模型的情况下，事实证明我们仍然可以在实践中有效地训练模型。此外，对于许多任务来说，那些难以优化的模型被证明要好得多，以至于弄清楚如何训练它们最终是非常值得的。

优化几乎任何深度学习模型的关键技术，我们将在整本书中介绍，包括通过在递增地降低损失函数的方向上更新参数来迭代地减少误差。这种算法被称为“梯度下降”。

梯度下降最天真的应用是取损失函数的导数，该导数是对数据集中每个示例计算的损失的平均值。实际上，这可能非常慢：在进行单个更新之前，我们必须遍历整个数据集。因此，我们经常满足于在每次需要计算更新时对随机小批量示例进行抽样，这是一种称为“小批量随机梯度下降”的变体。

在每次迭代中，我们首先随机抽样由固定数量的训练样本组成的小批次$\mathcal{B}$。然后我们计算小批量平均损失关于模型参数的导数(梯度)。最后，我们将梯度乘以预定的正值$\eta$，并从当前参数值中减去结果项。

我们可以用数学方式表示更新，如下所示($\partial$表示偏导数)：

$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).$$

总而言之，该算法的步骤如下：(I)我们初始化模型参数的值，通常是随机的；(Ii)我们迭代地从数据中随机抽样小批量，在负梯度方向上更新参数。对于二次损失和仿射变换，我们可以明确地写出如下内容：

$$\begin{aligned} \mathbf{w} &\leftarrow \mathbf{w} -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),\\ b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_b l^{(i)}(\mathbf{w}, b)  = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}$$
:eqlabel:`eq_linreg_batch_update`

请注意，$\mathbf{w}$和$\mathbf{x}$是:eqref:`eq_linreg_batch_update`中的矢量。在这里，更优雅的向量表示法使数学比用系数(比方说$w_1, w_2, \ldots, w_d$)表示的数学更具可读性。集合基数$|\mathcal{B}|$表示每个小批次中的示例数量(*批次大小*)，并且$\eta$表示*学习率*。我们强调，批次大小和学习率的值是手动预先指定的，通常不是通过模型训练学习的。这些在训练循环中可调但不更新的参数称为*超参数*。
*超参数调整*是选择超参数的过程，
并且通常需要我们基于在单独的*验证数据集*(或*验证集*)上评估的训练循环的结果来调整它们。

在训练了一些预定次数的迭代之后(或者直到满足一些其他停止标准)，我们记录估计的模型参数，表示为$\hat{\mathbf{w}}, \hat{b}$。请注意，即使我们的函数是真正的线性和无噪声的，这些参数也不会是损失的精确最小化，因为尽管算法向最小化缓慢收敛，但它不能在有限的步骤中精确地实现它。

线性回归碰巧是一个学习问题，在整个域上只有一个最小值。然而，对于更复杂的模型，如深层网络，损失曲面包含许多极小值。幸运的是，由于尚未完全了解的原因，深度学习从业者很少努力寻找将“训练集”上的损失降至最低的参数。更艰巨的任务是找到能够实现我们以前从未见过的低数据损失的参数，这是一项称为“泛化”的挑战。我们在整本书中都会回到这些主题。

### 用学习模型进行预测

给定学习的线性回归模型$\hat{\mathbf{w}}^\top \mathbf{x} + \hat{b}$，我们现在可以在给定面积$x_1$和年限$x_2$的情况下估计新房的价格(没有包含在训练数据中)。对给定特征的目标进行估计通常称为“预测”或“推断”。

我们将努力坚持“预测”，因为将这一步骤称为“推理”，尽管它已成为深度学习中的标准术语，但却有点用词不当。在统计学中，*推理*更多地表示基于数据集估计参数。当深度学习从业者与统计学家交谈时，这种术语的误用是念力的常见来源。

## 速度的矢量化

在训练我们的模型时，我们通常希望同时处理整个小批量的示例。要有效地做到这一点，我们需要将计算矢量化并利用快速线性代数库，而不是用Python编写代价高昂的for循环。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np
import time
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import numpy as np
import time
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
import numpy as np
import time
```

为了说明这一点如此重要的原因，我们可以考虑两种添加向量的方法。首先，我们实例化两个包含全部1的10000维向量。在一种方法中，我们将使用Python for循环遍历向量。在另一种方法中，我们将依赖于对`+`的单个调用。

```{.python .input}
#@tab all
n = 10000
a = d2l.ones(n)
b = d2l.ones(n)
```

由于我们将在本书中频繁地对运行时间进行基准测试，因此让我们定义一个计时器。

```{.python .input}
#@tab all
class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
```

现在，我们可以对工作负载进行基准测试。首先，我们使用for循环将它们相加，一次一个坐标。

```{.python .input}
#@tab mxnet, pytorch
c = d2l.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
f'{timer.stop():.5f} sec'
```

```{.python .input}
#@tab tensorflow
c = tf.Variable(d2l.zeros(n))
timer = Timer()
for i in range(n):
    c[i].assign(a[i] + b[i])
f'{timer.stop():.5f} sec'
```

或者，我们依靠重新加载的`+`运算符来计算元素求和。

```{.python .input}
#@tab all
timer.start()
d = a + b
f'{timer.stop():.5f} sec'
```

您可能注意到，第二种方法比第一种方法快得多。向量化代码通常会带来数量级的加速。此外，我们将更多的数学推送到库中，不需要自己编写那么多的计算，从而降低了出错的可能性。

## 正态分布与平方损失
:label:`subsec_normal_distribution_and_squared_loss`

虽然您已经可以仅使用上面的信息来亲手操作，但是在下面我们可以通过关于噪声分布的假设来更正式地激励平方损失目标。

线性回归是由高斯在1795年发明的，他也发现了正态分布(也称为*高斯*)。事实证明，正态分布和线性回归之间的联系比普通的亲子关系要深得多。为了唤起您的记忆，均值为$\mu$、方差为$\sigma^2$(标准差$\sigma$)的正态分布的概率密度如下所示

$$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).$$

下面我们定义一个Python函数来计算正态分布。

```{.python .input}
#@tab all
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
```

我们现在可以想象正态分布。

```{.python .input}
#@tab all
# Use numpy again for visualization
x = np.arange(-7, 7, 0.01)

# Mean and standard deviation pairs
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

正如我们所看到的，改变均值对应于沿$x$轴的移动，增加方差会分散分布，降低其峰值。

用均方误差损失函数(或简单的平方损失)激励线性回归的一种方法是正式假设观测值来自有噪声的观测值，其中噪声的正态分布如下：

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon \text{ where } \epsilon \sim \mathcal{N}(0, \sigma^2).$$

因此，我们现在可以写出看到给定$y$ VIA的特定$\mathbf{x}$的“可能性”

$$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right).$$

现在，根据最大似然原则，参数$\mathbf{w}$和$b$的最佳值是那些最大化整个数据集的*似然*的值：

$$P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x}^{(i)}).$$

根据最大似然原则选择的估计量称为“最大似然估计量”。虽然最大化许多指数函数的乘积可能看起来很困难，但我们可以在不改变目标的情况下，通过最大化可能性的对数来显著简化事情。由于历史原因，优化更多地表示为最小化而不是最大化。因此，在不做任何更改的情况下，我们可以将“负对数似然”最小化*$-\log P(\mathbf y \mid \mathbf X)$。算出数学公式给了我们：

$$-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2.$$

现在我们只需要再做一个假设，$\sigma$是某个固定常数。因此，我们可以忽略第一项，因为它不依赖于$\mathbf{w}$或$b$。现在，除了乘法常数$\frac{1}{\sigma^2}$之外，第二项与前面介绍的平方误差损失相同。幸运的是，解决方案并不依赖于$\sigma$。在加性高斯噪声的假设下，最小化均方误差等价于线性模型的最大似然估计。

## 从线性回归到深度网络

到目前为止，我们只讨论了线性模型。虽然神经网络涵盖了更丰富的模型家族，但我们可以通过用神经网络语言表示线性模型来开始将其视为神经网络。首先，让我们从用“层”符号重写内容开始。

### 神经网络图

深度学习实践者喜欢绘制图表来可视化他们的模型中正在发生的事情。在:numref:`fig_single_neuron`中，我们将线性回归模型描述为神经网络。请注意，这些图表突出显示了连接模式，例如每个输入如何连接到输出，而不是权重或偏差所取的值。

![Linear regression is a single-layer neural network.](../img/singleneuron.svg)
:label:`fig_single_neuron`

对于:numref:`fig_single_neuron`中所示的神经网络，输入为$x_1, \ldots, x_d$，因此输入层中的*输入数量*(或*特征维度*)为$d$。:numref:`fig_single_neuron`中网络的输出是$o_1$，因此输出层的*输出数*是1。请注意，输入值都是*给定的*，并且只有一个*计算*神经元。集中在计算发生的地方，通常我们在计算图层时不考虑输入层。也就是说，:numref:`fig_single_neuron`版的神经网络的“层数”是1。我们可以将线性回归模型看作是由单个人工神经元组成的神经网络，也可以看作是单层神经网络。

由于对于线性回归，每一个输入都连接到每一个输出(在本例中只有一个输出)，所以我们可以将这个变换(:numref:`fig_single_neuron`中的输出层)视为一个*完全连通层*或*密集层*。在下一章中，我们将更多地讨论由这些层组成的网络。

### 生物学

由于线性回归(发明于1795年)早于计算神经科学，将线性回归描述为神经网络似乎是不合时宜的。当控制学家/神经生理学家沃伦·麦卡洛克和沃尔特·皮茨开始开发人工神经元模型时，要了解为什么线性模型是一个自然的起点，请考虑:numref:`fig_Neuron`的一幅生物神经元的卡通图片，其中包括
*树枝状*(输入端子)，
*核*(CPU)、*轴突*(输出线)和*轴突终末*(输出终末)能够通过*突触*与其他神经元连接。

![The real neuron.](../img/neuron.svg)
:label:`fig_Neuron`

来自其他神经元(或诸如视网膜的环境传感器)的信息$x_i$在树突中被接收。具体地说，该信息由*突触权重*$w_i$加权，确定输入的效果(例如，通过乘积$x_i w_i$激活或抑制)。来自多个源的加权输入在核中被聚集为加权和$y = \sum_i x_i w_i + b$，然后该信息被发送用于轴突$y$中的进一步处理，通常在经过$\sigma(y)$的一些非线性处理之后。从那里，它要么到达目的地(例如肌肉)，要么通过树突被馈送到另一个神经元。

当然，许多这样的单元可以与正确的连通性和正确的学习算法拼凑在一起，产生比任何一个神经元单独表达的更有趣和更复杂的行为，这一高级想法归功于我们对真实生物神经系统的研究。

与此同时，今天大多数深度学习的研究在神经科学中几乎没有直接的启发。我们引用斯图尔特·罗素(Stuart Russell)和彼得·诺维格(Peter Norvig)在他们的经典人工智能教科书中
*人工智能: A Modern Approach* :cite:`Russell.Norvig.2016`，
他指出，尽管飞机可能受到鸟类的“启发”，但几个世纪以来，鸟类学并不是航空创新的主要驱动力。同样，如今深度学习的灵感同样或更多地来自数学、统计学和计算机科学。

## 摘要

* 机器学习模型的关键成分是训练数据、损失函数、优化算法，很明显，还有模型本身。
* 矢量化使一切变得更好(主要是数学)和更快(主要是代码)。
* 最小化目标函数和执行最大似然估计可能意味着相同的事情。
* 线性回归模型也是神经网络。

## 练习

1. 假设我们有一些数据$x_1, \ldots, x_n \in \mathbb{R}$。我们的目标是找到一个常数$b$，使$\sum_i (x_i - b)^2$最小化。
    1. 找出最佳值$b$的解析解。
    1. 这个问题及其解决方案与正态分布有何关系？
1. 求出了具有平方误差的线性回归优化问题的解析解。为了简单起见，您可以从问题中省略偏差$b$(我们可以有原则地将一列添加到由所有1组成的$\mathbf X$列中)。
    1. 用矩阵和向量表示法写出优化问题(将所有数据视为单个矩阵，将所有目标值视为单个向量)。
    1. 计算损失相对于$w$的梯度。
    1. 将梯度设为零，求解矩阵方程，求解析解。
    1. 什么时候这可能比使用随机梯度下降更好呢？这种方法什么时候会被打破呢？
1. 假设支配加性噪声$\epsilon$的噪声模型是指数分布。就是$p(\epsilon) = \frac{1}{2} \exp(-|\epsilon|)$。
    1. 写出模型$-\log P(\mathbf y \mid \mathbf X)$下的数据的负对数似然。
    1. 你能找到封闭形式的解决方案吗？
    1. 提出了一种随机梯度下降算法来解决这一问题。可能会出什么问题(提示：当我们不断更新参数时，固定点附近会发生什么情况)？你能把这个修好吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/40)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/258)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/259)
:end_tab:
