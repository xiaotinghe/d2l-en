# 线性回归
:label:`sec_linear_regression`

*回归*是指一组建模方法
一个或多个自变量与因变量之间的关系。在自然科学和社会科学中，回归的目的通常是为了
*描述*输入和输出之间的关系。
另一方面，机器学习通常与预测有关。

每当我们想预测一个数值时，就会出现回归问题。常见的例子包括预测价格（房屋、股票等）、预测住院时间（针对住院患者）、需求预测（针对零售额），等等。不是所有的预测问题都是经典的回归问题。在后面的章节中，我们将介绍分类问题，目标是预测一组类别中的成员。

## 线性回归的基本要素

*线性回归*可能是最简单的
在回归的标准工具中最流行。回溯到19世纪初，线性回归是由一些简单的假设产生的。首先，我们假设自变量$\mathbf{x}$和因变量$y$之间的关系是线性的，即$y$可以表示为$\mathbf{x}$中元素的加权和，在给定观测噪声的情况下。第二，我们假设任何噪声都表现良好（遵循高斯分布）。

为了激励这种方法，让我们从一个运行的例子开始。假设我们希望根据房屋面积（平方英尺）和年龄（年）来估计房屋的价格（美元）。要真正开发出一个预测房价的模型，我们需要获得一个由销售数据集组成的数据集，我们知道每套房子的销售价格、面积和年龄。在机器学习的术语中，数据集被称为*训练数据集*或*训练集*，每一行（这里对应于一次销售的数据）被称为*示例*（或*数据点*，*数据实例*，*样本*）。我们试图预测（价格）的东西叫做*标签*（或*目标*）。预测所依据的自变量（年龄和面积）称为*特征*（或*协变量*）。

通常，我们将使用$n$来表示数据集中的示例数。我们用$i$索引数据示例，表示每个输入为$\mathbf{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}]^\top$，相应的标签为$y^{(i)}$。

### 线性模型
:label:`subsec_linear_model`

线性假设只是说目标（价格）可以表示为特征（面积和年龄）的加权和：

$$\mathrm{price} = w_{\mathrm{area}} \cdot \mathrm{area} + w_{\mathrm{age}} \cdot \mathrm{age} + b.$$
:eqlabel:`eq_price-area`

在:eqref:`eq_price-area`中，$w_{\mathrm{area}}$和$w_{\mathrm{age}}$被称为*权重*，而$b$被称为*偏移*（也称为*偏移*或*截距*）。权重决定了每个特征对我们的预测的影响，而偏差只是说明当所有特征值取0时，预测价格应该取什么值。即使我们永远也看不到任何面积为零的房屋，或是零年的房屋，我们仍然需要偏差，否则我们将限制模型的表达能力。严格地说，:eqref:`eq_price-area`是输入特征的仿射变换，其特征是通过加权和对特征进行*线性变换*和*通过附加偏差*进行*平移*。

给定一个数据集，我们的目标是选择权重$\mathbf{w}$和偏差$b$，以便平均而言，根据我们的模型所做的预测最符合数据中观察到的真实价格。其输出预测由输入特征的仿射变换决定的模型是*线性模型*，其中仿射变换由所选权重和偏差指定。

在通常只关注一些特性的数据集的领域中，像这样显式地表达长形式的模型是很常见的。在机器学习中，我们通常使用高维数据集，因此使用线性代数表示法更方便。当我们的输入包含$d$个特征时，我们将预测$\hat{y}$（通常“帽子”符号表示估计值）表示为

$$\hat{y} = w_1  x_1 + ... + w_d  x_d + b.$$

将所有特征集合到向量$\mathbf{x} \in \mathbb{R}^d$中，将所有权重集合到向量$\mathbf{w} \in \mathbb{R}^d$中，我们可以使用点积简洁地表示我们的模型：

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b.$$
:eqlabel:`eq_linreg-y`

在:eqref:`eq_linreg-y`中，向量$\mathbf{x}$对应于单个数据示例的特征。我们通常会发现，通过*设计矩阵*$\mathbf{X} \in \mathbb{R}^{n \times d}$，可以方便地引用$n$示例数据集的特征。这里，$\mathbf{X}$为每个示例包含一行，为每个特性包含一列。

对于特征$\mathbf{X}$的集合，预测$\hat{\mathbf{y}} \in \mathbb{R}^n$可以通过矩阵向量积表示：

$${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b,$$

在求和过程中应用广播（见:numref:`subsec_broadcasting`）。给定训练数据集$\mathbf{X}$的特征和相应的（已知）标签$\mathbf{y}$，线性回归的目标是找到权重向量$\mathbf{w}$和偏倚项$b$，其中给定了从与$\mathbf{X}$相同分布中抽样的新数据示例的特征，新示例的标签将（在预期中）以最小的误差进行预测。

即使我们认为预测$y$（给定$\mathbf{x}$）的最佳模型是线性的，我们也不会期望找到一个包含$n$个示例的真实数据集，其中$y^{(i)}$完全等于732293612的$\mathbf{w}^\top \mathbf{x}^{(i)}+b$。例如，无论我们使用什么仪器来观察特性$\mathbf{X}$和标签$\mathbf{y}$，都可能受到少量测量误差的影响。因此，即使我们确信潜在的关系是线性的，我们也会加入一个噪声项来解释这种误差。

在我们开始寻找最佳的*参数*（或*模型参数*）$\mathbf{w}$和$b$之前，我们还需要两件事：（i）某个给定模型的质量度量；（ii）更新模型以提高其质量的过程。

### 损失函数

在我们开始考虑如何将数据与我们的模型相适应之前，我们需要确定一个适合度的度量。*损失函数*量化目标的*实际*和*预测*值之间的距离。损失通常是一个非负数，数值越小越好，完美的预测损失为0。回归问题中最常见的损失函数是平方误差。当我们对示例$i$的预测为$\hat{y}^{(i)}$且对应的真标签为$y^{(i)}$时，平方误差由下式给出：

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$

常数$\frac{1}{2}$没有实际的区别，但会证明非常方便，当我们取损失的导数时会抵消掉。由于训练数据集是给我们的，因此我们无法控制，经验误差只是模型参数的函数。为了使事情更具体，考虑下面的例子，我们为一维情况绘制回归问题，如:numref:`fig_fit_linreg`所示。

![Fit data with a linear model.](../img/fit-linreg.svg)
:label:`fig_fit_linreg`

请注意，估计值$\hat{y}^{(i)}$与观测值$y^{(i)}$之间的巨大差异导致损失的贡献更大，这是由于二次相关。为了在$n$个示例的整个数据集上测量模型的质量，我们只需平均（或等效地求和）训练集上的损失。

$$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

在训练模型时，我们希望找到使所有训练示例中的总损失最小化的参数（$\mathbf{w}^*, b^*$）：

$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b).$$

### 解析解

线性回归恰好是一个非常简单的优化问题。与我们将在本书中遇到的大多数其他模型不同，线性回归可以通过应用一个简单的公式进行分析求解。首先，我们可以将偏差$b$包含在参数$\mathbf{w}$中，方法是在包含所有参数的设计矩阵中添加一列。那么我们的预测问题是最小化$\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$。在损耗面上只有一个临界点，它对应于整个区域的损耗最小值。取$\mathbf{w}$的损失导数并将其设为零，得到解析（闭合形式）解：

$$\mathbf{w}^* = (\mathbf X^\top \mathbf X)^{-1}\mathbf X^\top \mathbf{y}.$$

虽然像线性回归这样的简单问题可能需要解析解，但你不应该习惯这样的好运。虽然解析解可以进行很好的数学分析，但对解析解的要求是如此的严格，以至于它会排除所有的深入学习。

### 小批量随机梯度下降

即使在我们无法解析地求解模型的情况下，我们仍然可以在实践中有效地训练模型。此外，对于许多任务来说，那些难以优化的模型结果会好得多，以至于弄清楚如何训练它们最终是值得的。

优化几乎任何深度学习模型的关键技术，我们在本书中都会用到，它包括通过在逐渐降低损失函数的方向上更新参数来迭代减少误差。这种算法叫做梯度下降法。

梯度下降最朴素的应用是获取损失函数的导数，它是在数据集中每个实例上计算的损失的平均值。实际上，这可能非常慢：在进行一次更新之前，我们必须传递整个数据集。因此，每次我们需要计算更新时，我们通常会选择随机的小批量样本，这个变量称为*小批量随机梯度下降*。

在每次迭代中，我们首先随机抽取一个由固定数量的训练示例组成的小批量$\mathcal{B}$。然后，我们计算小批量平均损失相对于模型参数的导数（梯度）。最后，我们将梯度乘以预定的正值$\eta$，并从当前参数值中减去得到的项。

我们可以用数学方法表示更新如下（$\partial$表示偏导数）：

$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).$$

综上所述，该算法的步骤如下：（i）初始化模型参数的值，通常是随机的；（ii）我们从数据中迭代地随机抽样，沿着负梯度方向更新参数。对于二次损失和仿射变换，我们可以明确地写出如下：

$$\begin{aligned} \mathbf{w} &\leftarrow \mathbf{w} -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),\\ b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_b l^{(i)}(\mathbf{w}, b)  = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}$$
:eqlabel:`eq_linreg_batch_update`

注意，$\mathbf{w}$和$\mathbf{x}$是:eqref:`eq_linreg_batch_update`中的向量。在这里，更优雅的向量表示法使数学比用系数表示更具可读性，比如$w_1, w_2, \ldots, w_d$。设置的基数$|\mathcal{B}|$表示每个小批量中的示例数（*批大小*），$\eta$表示*学习率*。我们强调批大小和学习率的值是手动预先指定的，而不是通常通过模型训练来学习的。这些可调但在训练循环中未更新的参数称为*hyperparameters*。
*超参数调整*是选择超参数的过程，
通常要求我们根据训练循环的结果调整它们，这些结果是在一个单独的*验证数据集*（或*验证集*）上评估的。

在对一些预先确定的迭代次数进行训练之后（或者直到满足其他一些停止条件），我们记录估计的模型参数，表示为$\hat{\mathbf{w}}, \hat{b}$。虽然我们的算法不能很慢地收敛到最小的参数，但是我们的算法不能很慢地收敛到最小。

线性回归恰好是一个学习问题，在整个领域中只有一个最小值。然而，对于更复杂的模型，如深网络，损失曲面包含许多极小值。幸运的是，由于一些尚未完全理解的原因，深度学习实践者很少努力寻找能够将训练集损失最小化的参数。更艰巨的任务是找到能够在我们以前从未见过的数据上实现低损失的参数，这是一个称为*泛化*的挑战。我们在整本书中都会回到这些主题。

### 用所学模型进行预测

考虑到所学的线性回归模型$\hat{\mathbf{w}}^\top \mathbf{x} + \hat{b}$，我们现在可以估计一栋新房的价格（不包含在培训数据中），因为它的面积为$x_1$，年龄为$x_2$。根据给定的特征估计目标通常被称为*预测*或*推断*。

我们将坚持“预测”，因为称这一步为“推断”，尽管在深度学习中已成为标准行话，但这有点用词不当。在统计学中，*推断*通常表示基于数据集的估计参数。当深入学习的实践者与统计学家交谈时，术语的滥用是一个常见的混淆来源。

## 速度矢量化

在训练模型时，我们通常希望同时处理整个小批量的示例。要有效地做到这一点，我们需要将计算矢量化并利用快速的线性代数库，而不是在Python中编写代价高昂的for循环。

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

为了说明这一点为什么如此重要，我们可以考虑两种添加向量的方法。首先，我们实例化两个10000维向量，其中包含所有向量。在一种方法中，我们将使用Python for循环遍历向量。在另一个方法中，我们将依赖于对`+`的单个调用。

```{.python .input}
#@tab all
n = 10000
a = d2l.ones(n)
b = d2l.ones(n)
```

由于我们将在本书中频繁地对运行时间进行基准测试，所以让我们定义一个计时器。

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

现在我们可以对工作负载进行基准测试。首先，我们使用for循环一次添加一个坐标。

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

或者，我们依赖重新加载的`+`运算符来计算元素和。

```{.python .input}
#@tab all
timer.start()
d = a + b
f'{timer.stop():.5f} sec'
```

你可能注意到第二种方法比第一种方法快得多。矢量化代码通常会产生数量级的加速。此外，我们将更多的数学推送到库中，不需要自己编写那么多的计算，从而减少了出错的可能性。

## 正态分布与平方损失
:label:`subsec_normal_distribution_and_squared_loss`

虽然仅使用上述信息，您已经可以弄脏您的手，在下面，我们可以更正式地通过假设噪声分布来激励平方损失目标。

线性回归是由高斯在1795年发明的，他也发现了正态分布（也称为高斯分布）。结果表明，正态分布和线性回归之间的联系比普通亲子关系更深。为了刷新您的记忆，正态分布的概率密度（平均值$\mu$和方差$\sigma^2$）（标准偏差$\sigma$）如下所示

$$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).$$

下面我们定义一个Python函数来计算正态分布。

```{.python .input}
#@tab all
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
```

我们现在可以看到正态分布。

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

如我们所见，改变平均值对应于沿$x$轴的移动，增加方差会分散分布，降低峰值。

用均方误差损失函数（或简单的平方损失）来激励线性回归的一种方法是正式假设观测值来自噪声观测值，其中噪声的正态分布如下：

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon \text{ where } \epsilon \sim \mathcal{N}(0, \sigma^2).$$

因此，我们现在可以写出给定$\mathbf{x}$通孔看到特定$y$的可能性

$$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right).$$

现在，根据最大似然原理，参数$\mathbf{w}$和$b$的最佳值是使整个数据集的*似然*最大的值：

$$P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x}^{(i)}).$$

根据极大似然原理选择的估计量称为*极大似然估计量*。虽然使许多指数函数的乘积最大化看起来很困难，但是我们可以通过最大化似然对数来显著简化事情，而不改变目标。由于历史原因，优化通常表示为最小化而不是最大化。所以，在不改变任何东西的情况下，我们可以最小化负对数似然。计算出数学公式可以给我们：

$$-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2.$$

现在我们只需要再假设$\sigma$是某个固定常数。因此，我们可以忽略第一项，因为它不依赖于$\mathbf{w}$或$b$。现在第二项与前面介绍的平方误差损失相同，除了乘法常数$\frac{1}{\sigma^2}$。幸运的是，解决方案并不依赖于$\sigma$。在最大似然高斯模型下，它遵循最小的平均方差的线性模型。

## 从线性回归到深层网络

到目前为止，我们只讨论线性模型。虽然神经网络涵盖了更丰富的模型家族，但我们可以通过用神经网络的语言来表达线性模型来将其视为神经网络。首先，让我们用“层”符号重写内容。

### 神经网络图

深度学习的实践者喜欢绘制图表来可视化模型中正在发生的事情。在:numref:`fig_single_neuron`中，我们将线性回归模型描述为一个神经网络。请注意，这些图突出显示了连接模式，例如每个输入如何连接到输出，但没有显示权重或偏差所取的值。

![Linear regression is a single-layer neural network.](../img/singleneuron.svg)
:label:`fig_single_neuron`

对于:numref:`fig_single_neuron`所示的神经网络，输入为$x_1, \ldots, x_d$，因此输入层中的*输入数目*（或*特征维数*）为$d$。:numref:`fig_single_neuron`中的网络输出为$o_1$，因此输出层的*输出数量*为1。注意输入值都是给定的，只有一个计算的神经元。针对计算发生的地方，我们通常在计算层时不考虑输入层。也就是说，:numref:`fig_single_neuron`中神经网络的*层数*是1。我们可以把线性回归模型看作是由单个人工神经元组成的神经网络，或者是单层神经网络。

因为对于线性回归，每个输入都与每个输出相连接（在这种情况下只有一个输出），我们可以将此转换（:numref:`fig_single_neuron`中的输出层）视为*完全连接层*或*密集层*。在下一章中，我们将更多地讨论由这些层组成的网络。

### 生物

由于线性回归（发明于1795年）早于计算神经科学，所以将线性回归描述为神经网络似乎是不合时宜的。当控制论者/神经生理学家沃伦·麦库洛奇和沃尔特·皮茨开始开发人工神经元模型时，为什么线性模型是一个自然的起点，考虑:numref:`fig_Neuron`年一个生物神经元的卡通图片，它包括
*树枝晶*（输入端子），
核*（CPU）、*轴突*（输出线）和*轴突终端*（输出终端），通过*突触*与其他神经元连接。

![The real neuron.](../img/neuron.svg)
:label:`fig_Neuron`

树突中接收来自其他神经元（或视网膜等环境传感器）的信息$x_i$。具体而言，该信息通过*突触权重*$w_i$来加权，以确定输入的影响（例如，通过产物$x_i w_i$激活或抑制）。来自多个源的加权输入在核中聚合为加权和$y = \sum_i x_i w_i + b$，然后该信息被发送到轴突$y$中进行进一步处理，通常在通过$\sigma(y)$进行一些非线性处理之后。从那里，它要么到达目的地（例如肌肉），要么通过树突进入另一个神经元。

当然，高层次的想法，许多这样的单位可以拼凑在一起与正确的连接性和正确的学习算法，产生远比任何一个神经元单独能够表达的有趣和复杂的行为，这归功于我们对真实生物神经系统的研究。

同时，目前大多数关于深度学习的研究几乎没有从神经科学中获得直接的灵感。我们引用Stuart Russell和Peter Norvig在他们经典的人工智能教科书中
*人工智能: A Modern Approach* :cite:`Russell.Norvig.2016`，
他指出，虽然飞机可能是受鸟类的启发，但鸟类学并不是几个世纪以来航空创新的主要驱动力。同样地，如今深度学习的灵感同样来自于数学、统计学和计算机科学。

## 摘要

* 机器学习模型的关键要素是训练数据、损失函数、优化算法，很明显，还有模型本身。
* 矢量化使一切变得更好（主要是数学）和更快（主要是代码）。
* 最小化目标函数和执行最大似然估计可能意味着相同的事情。
* 线性回归模型也是神经网络。

## 练习

1. 假设我们有一些数据$x_1, \ldots, x_n \in \mathbb{R}$。我们的目标是找到一个常数$b$，使$\sum_i (x_i - b)^2$最小化。
    1. 找到最佳值$b$的解析解。
    1. 这个问题及其解决方案与正态分布有什么关系？
1. 推导了具有平方误差的线性回归优化问题的解析解。为了简单起见，您可以省略问题中的偏差$b$（我们可以通过在$\mathbf X$中添加一列（包含所有列）来原则性地进行此操作）。
    1. 用矩阵和向量表示法写出优化问题（将所有数据视为单个矩阵，将所有目标值视为单个向量）。
    1. 计算关于$w$的损耗梯度。
    1. 通过设置梯度为零并求解矩阵方程，求出解析解。
    1. 什么时候这会比使用随机梯度下降更好呢？这种方法何时会失效？
1. 假设控制加性噪声的噪声模型$\epsilon$是指数分布。也就是$p(\epsilon) = \frac{1}{2} \exp(-|\epsilon|)$。
    1. 写出$-\log P(\mathbf y \mid \mathbf X)$模型下数据的负对数似然。
    1. 你能找到一个封闭形式的解决方案吗？
    1. 提出一种随机梯度下降算法来解决这个问题。可能出什么问题（提示：当我们不断更新参数时，在固定点附近会发生什么情况）？你能修好这个吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/40)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/258)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/259)
:end_tab:
