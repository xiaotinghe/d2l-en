# Softmax回归
:label:`sec_softmax`

在:numref:`sec_linear_regression`中，我们引入了线性回归，在:numref:`sec_linear_scratch`中从头开始实现，在:numref:`sec_linear_concise`中再次使用深度学习框架的高级api来完成繁重的工作。

回归是我们想要回答的锤子，多少？*或者*多少？*问题。如果你想预测一栋房子的售价，或者一支棒球队可能赢得的胜利数，或者病人出院前住院的天数，那么你可能在寻找一个回归模型。

实际上，我们更感兴趣的是分类：不是问“多少”，而是“哪一个”：

* 这封电子邮件属于垃圾邮件文件夹还是收件箱？
* 这个客户更可能*注册*还是*不注册*订阅服务？
* 这是一只驴，还是一只猫，一只猫？
* 阿斯顿接下来最有可能看哪部电影？

通俗地说，机器学习的实践者用“分类”这个词来描述两个微妙的不同问题：（i）我们只对类别（类）的示例硬赋值感兴趣；以及（ii）我们希望进行软任务的问题，即评估每个类别应用的概率。这种区别往往变得模糊，部分原因是，即使我们只关心硬任务，我们仍然使用软任务的模型。

## 分类问题
:label:`subsec_classification-problem`

为了让我们的脚湿，让我们从一个简单的图像分类问题开始。这里，每个输入由$2\times2$灰度图像组成。我们可以用一个标量来表示每个像素值，给我们四个特性$x_1, x_2, x_3, x_4$。此外，让我们假设每个图像属于“猫”、“鸡”和“狗”中的一个类别。

接下来，我们必须选择如何表示标签。我们有两个明显的选择。也许最自然的冲动是选择$y \in \{1, 2, 3\}$，其中整数分别代表73229365。这是一种在计算机上存储此类信息的好方法。如果这些类别之间有一些自然的顺序，比如说如果我们试图预测$\{\text{baby}, \text{toddler}, \text{adolescent}, \text{young adult}, \text{adult}, \text{geriatric}\}$，那么将这个问题转化为回归并保持这种格式的标签可能是有意义的。

但是，一般的分类问题并不是由类之间的自然顺序引起的。幸运的是，分类统计学家很久以前发明了一种简单的数据编码方法。一个热编码是一个向量，它包含的组件和我们拥有的类别一样多。与特定实例的类别对应的组件设置为1，所有其他组件设置为0。在我们的例子中，标签$y$是一个三维向量，其中$(1, 0, 0)$对应于“cat”，$(0, 1, 0)$对应于“chicken”，$(0, 0, 1)$对应于“dog”：

$$y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}.$$

## 网络体系结构

为了估计与所有可能类相关的条件概率，我们需要一个具有多个输出的模型，每个类一个输出。为了解决线性模型的分类问题，我们需要尽可能多的仿射函数作为输出。每个输出将对应于它自己的仿射函数。在我们的例子中，因为我们有4个特性和3个可能的输出类别，我们需要12个标量来表示权重（$w$带下标），3个标量来表示偏差（$b$带下标）。我们为每个输入计算这三个*逻辑*、$o_1, o_2$和$o_3$：

$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}
$$

我们可以用:numref:`fig_softmaxreg`所示的神经网络图来描述这种计算。与线性回归一样，softmax回归也是一个单层神经网络。由于每个输出$o_1, o_2$和$o_3$的计算依赖于所有输入$x_1$、$x_2$、$x_3$和$x_4$，所以softmax回归的输出层也可以描述为全连接层。

![Softmax regression is a single-layer neural network.](../img/softmaxreg.svg)
:label:`fig_softmaxreg`

为了更简洁地表达模型，我们可以使用线性代数表示法。在向量形式中，我们得到了$\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$，这是一种更适合数学和编写代码的形式。请注意，我们已经将所有权重集合到一个$3 \times 4$矩阵中，对于给定数据示例$\mathbf{x}$的特征，我们的输出是由我们的输入特征加上我们的偏差$\mathbf{b}$的权重的矩阵向量乘积给出的。

## Softmax操作
:label:`subsec_softmax_operation`

我们在这里要采取的主要方法是将模型的输出解释为概率。我们将优化我们的参数以产生使观测数据的可能性最大化的概率。然后，为了生成预测，我们将设置一个阈值，例如，选择具有最大预测概率的标签。

从形式上讲，我们希望任何输出$\hat{y}_j$被解释为给定项目属于类别$j$的概率。然后我们可以选择输出值最大的类作为我们的预测$\operatorname*{argmax}_j y_j$。例如，如果$\hat{y}_1$、$\hat{y}_2$和$\hat{y}_3$分别为0.1、0.8和0.1，那么我们预测类别2，它（在我们的例子中）表示“鸡”。

您可能会建议我们直接将logits $o$解释为我们感兴趣的输出。然而，有一些问题直接解释为线性输出。一方面，没有什么能限制这些数字的和为1。另一方面，根据输入，它们可以取负值。这些违反了:numref:`sec_prob`中提出的概率的基本公理

为了将我们的输出解释为概率，我们必须保证（即使是在新数据上），它们将是非负的，并且总和为1。此外，我们需要一个训练目标，鼓励模型忠实地估计概率。在分类器输出0.5的所有实例中，我们希望这些示例中有一半实际上属于预测类。这是一个名为*校准*的属性。

社会科学家R.Duncan-Luce在1959年在“选择模型”的背景下发明的*softmax函数*正是这样做的。为了将逻辑变换为非负并求和为1，同时要求模型保持可微，我们首先对每个logit求幂（确保非负性），然后除以它们的和（确保它们和为1）：

$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{where}\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}. $$
:eqlabel:`eq_softmax_y_and_o`

很容易看到$\hat{y}_1 + \hat{y}_2 + \hat{y}_3 = 1$和$j$的$\hat{y}_1 + \hat{y}_2 + \hat{y}_3 = 1$。因此，$\hat{\mathbf{y}}$是一个适当的概率分布，其元素值可以据此进行解释。注意，softmax操作不会改变logits $\mathbf{o}$之间的顺序，logits $\mathbf{o}$只是确定分配给每个类的概率的pre-softmax值。因此，在预测过程中，我们仍然可以通过

$$
\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j.
$$

尽管softmax是一个非线性函数，但softmax回归的输出仍然由输入特征的仿射变换决定，因此，softmax回归是一个线性模型。

## 小批量的矢量化
:label:`subsec_softmax_vectorization`

为了提高计算效率并利用gpu，我们通常对小批量数据进行向量计算。假设我们得到了一个小批量$\mathbf{X}$示例，其特征维数（输入数量）为$d$，批量大小为$n$。此外，假设输出中有$q$个类别。然后，小批量特征$\mathbf{X}$在$\mathbb{R}^{n \times d}$中，权重$\mathbf{W} \in \mathbb{R}^{d \times q}$，偏差满足$\mathbf{b} \in \mathbb{R}^{1\times q}$。

$$ \begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned} $$
:eqlabel:`eq_minibatch_softmax_reg`

这加速了主导运算，变成了矩阵矩阵积$\mathbf{X} \mathbf{W}$，而不是我们一次处理一个例子所要执行的矩阵向量积。因为$\mathbf{X}$中的每一行代表一个数据示例，所以softmax操作本身可以*rowwise*计算：对于$\mathbf{O}$的每一行，将所有条目求幂，然后通过求和将它们规范化。在:eqref:`eq_minibatch_softmax_reg`中的和$\mathbf{X} \mathbf{W} + \mathbf{b}$期间触发广播，小批量逻辑$\mathbf{O}$和输出概率$\hat{\mathbf{Y}}$都是$n \times q$矩阵。

## 损失函数

接下来，我们需要一个损失函数来衡量我们预测的概率的质量。我们将依赖最大似然估计，这是我们在为线性回归中的均方误差目标提供概率证明时遇到的相同概念（:numref:`subsec_normal_distribution_and_squared_loss`）。

### 对数似然

我们给出了22736的条件输入向量的估计值。假设整个数据集$\{\mathbf{X}, \mathbf{Y}\}$有$n$个示例，其中由$i$索引的示例由一个特征向量$\mathbf{x}^{(i)}$和一个热标签向量$\mathbf{y}^{(i)}$组成。我们可以根据我们的模型，通过检查实际类的可能性，将估计值与实际值进行比较，前提是：

$$
P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}).
$$

根据最大似然估计，我们最大化$P(\mathbf{Y} \mid \mathbf{X})$，相当于最小化负对数似然：

$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}),
$$

其中，对于$q$类上的任何一对标签$\mathbf{y}$和模型预测$\hat{\mathbf{y}}$，损失函数$l$为

$$ l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j. $$
:eqlabel:`eq_l_cross_entropy`

出于后面解释的原因，:eqref:`eq_l_cross_entropy`中的损失函数通常称为*交叉熵损失*。由于$\mathbf{y}$是长度为$q$的一个热向量，所以除了一个项之外，所有坐标$j$上的和都会消失。因为所有$\hat{y}_j$都是预测概率，所以它们的对数永远不会大于$0$。因此，如果我们用*确定性*正确预测实际标签，即，如果实际标签$P(\mathbf{y} \mid \mathbf{x}) = 1$的预测概率$P(\mathbf{y} \mid \mathbf{x}) = 1$，则无法进一步最小化损失函数。请注意，这通常是不可能的。例如，数据集中可能存在标签噪声（有些示例可能标签错误）。当输入特征没有足够的信息来完美地分类每一个例子时，这也可能是不可能的。

### Softmax及其衍生物
:label:`subsec_softmax_and_derivatives`

由于softmax和相应的损耗非常常见，因此有必要更好地了解它是如何计算的。将:eqref:`eq_softmax_y_and_o`插入:eqref:`eq_l_cross_entropy`中的损耗定义，并使用softmax的定义，我们得到：

$$
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j\\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}
$$

为了更好地理解发生了什么，考虑一下与任何logit $o_j$相关的导数。我们得到了

$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.
$$

换句话说，导数是由我们的模型分配的概率（用softmax操作表示）与实际发生的情况（由一个热标签向量中的元素表示）之间的差值。从这个意义上讲，它与我们在回归中看到的非常相似，其中梯度是观察值$y$和估计值$\hat{y}$之间的差值。这不是巧合。在任何指数族（见[online appendix on distributions](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/distributions.html)）模型中，对数似然的梯度都是由这个项精确给出的。这一事实使计算梯度在实践中变得容易。

### 交叉熵损失

现在考虑这样一个例子，我们观察到的不仅仅是一个结果，而是整个结果分布。我们可以使用与之前相同的表示法来标记$\mathbf{y}$。唯一的区别是，不是一个只包含二进制项的向量，比如$(0, 0, 1)$，我们现在有一个通用的概率向量，比如$(0.1, 0.2, 0.7)$。我们之前用来定义$l$在:eqref:`eq_l_cross_entropy`中的损失的数学计算仍然很好，只是解释稍微更一般。它是在标签上分布的损失的期望值。这种损失称为交叉熵损失，它是分类问题中最常用的损失之一。我们可以通过介绍信息理论的基础知识来揭开这个名字的神秘面纱。如果你想了解更多关于信息论的细节，你可以进一步参考[online appendix on information theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html)。

## 信息论基础
:label:`subsec_info_theory_basics`

*信息论研究的是编码、解码、传输，
以及以尽可能简洁的形式处理信息（也称为数据）。

### 熵

信息论的核心思想是量化数据中的信息内容。这个数量限制了我们压缩数据的能力。在信息论中，这个量被称为分布$P$的*熵*，它可以通过以下等式得到：

$$H[P] = \sum_j - P(j) \log P(j).$$
:eqlabel:`eq_softmax_reg_entropy`

我们需要对随机抽取的一个数据的基本分布进行编码。如果您想知道“nat”是什么，它相当于bit，但是当使用以$e$为基数的代码时，而不是使用以2为基数的代码时。因此，一个nat是$\frac{1}{\log(2)} \approx 1.44$位。

### 出人意料的

你可能想知道压缩和预测有什么关系。假设我们有一个要压缩的数据流。如果我们总是很容易预测下一个令牌，那么这个数据就很容易压缩了！举一个极端的例子，流中的每个令牌总是取相同的值。这是一个非常无聊的数据流！不仅无聊，而且很容易预测。因为它们总是相同的，所以我们不必传输任何信息来传递流的内容。易于预测，易于压缩。

然而，如果我们不能完美地预测每一个事件，那么我们有时可能会感到惊讶。当我们分配一个事件的概率较低时，我们的惊喜更大。克劳德·香农决定用$\log \frac{1}{P(j)} = -\log P(j)$来量化一个人在观察一个事件时的“惊喜”，$j$赋予它一个（主观）概率$P(j)$。:eqref:`eq_softmax_reg_entropy`中定义的熵就是当我们分配了正确的概率，与数据生成过程真正匹配时，预期的惊喜。

### 交叉熵再探

所以，如果熵是知道真实概率的人所经历的惊喜程度，那么你可能会想知道，什么是交叉熵？从*$P$*到*$Q$的交叉熵*表示为$H(P, Q)$，是主观概率为$Q$的观察者在看到根据概率$P$实际生成的数据时的预期惊喜。当$P=Q$达到最小交叉熵时。在这种情况下，从$P$到$Q$的交叉熵是$H(P, P)= H(P)$。

简言之，我们可以从两个方面考虑交叉熵分类目标：（i）最大化观测数据的可能性；（ii）最小化传递标签所需的意外（以及由此产生的比特数）。

## 模型预测与评价

在训练了softmax回归模型后，在给定任何实例特征的情况下，我们可以预测每个输出类的概率。通常，我们使用预测概率最高的类作为输出类。如果预测与实际类（标签）一致，则预测是正确的。在下一部分的实验中，我们将使用*精度*来评估模型的性能。这等于正确预测数与预测总数之比。

## 摘要

* softmax操作获取一个向量并将其映射为概率。
* Softmax回归适用于分类问题。它在softmax操作中使用输出类的概率分布。
* 交叉熵是两个概率分布之间差异的一个很好的度量。它测量给定模型的数据编码所需的比特数。

## 练习

1. 我们可以更深入地探讨指数族和softmax之间的联系。
    1. 计算softmax的交叉熵损失$l(\mathbf{y},\hat{\mathbf{y}})$的二阶导数。
    1. 计算$\mathrm{softmax}(\mathbf{o})$给出的分布的方差，并证明它与上面计算的二阶导数相匹配。
1. 假设我们有三类概率相等的类，即概率向量为$(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$。
    1. 如果我们试图为它设计一个二进制代码，会有什么问题？
    1. 你能设计一个更好的代码吗？提示：如果我们试图编码两个独立的观察结果，会发生什么？如果我们联合编码$n$个观测值呢？
1. 对于上面介绍的映射来说，Softmax是一个错误的名称（但是每个深入学习的人都使用它）。真正的softmax被定义为$\mathrm{RealSoftMax}(a, b) = \log (\exp(a) + \exp(b))$。
    1. 证明$\mathrm{RealSoftMax}(a, b) > \mathrm{max}(a, b)$。
    1. 证明这适用于$\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b)$，前提是$\lambda > 0$。
    1. 显示$\lambda \to \infty$有$\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b) \to \mathrm{max}(a, b)$个。
    1. 软敏长什么样？
    1. 把这个扩展到两个以上的数字。

[Discussions](https://discuss.d2l.ai/t/46)
