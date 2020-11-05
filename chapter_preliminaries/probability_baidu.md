# 概率
:label:`sec_prob`

在某种形式下，机器学习就是做出预测。考虑到患者的临床病史，我们可能需要预测他们明年心脏病发作的概率。在异常检测中，我们可能需要评估飞机喷气发动机的一组读数，如果它正常工作的话。在强化学习中，我们需要一个智能体在一个环境中智能地行动。这意味着我们需要考虑在每一个可用的行动下获得高回报的可能性。当我们建立推荐系统时，我们还需要考虑概率。例如，假设我们为一家大型在线书店工作。我们可能需要估计特定用户购买特定书籍的概率。为此，我们需要使用概率的语言。整个课程，专业，论文，职业生涯，甚至是系，都致力于概率论。所以自然，我们在这一部分的目标不是教整个学科。取而代之的是，我们希望能让你脱离现实，教给你足够的知识，让你可以开始建立你的第一个深度学习模型，并给你足够的味道，你可以开始自己探索，如果你愿意的话。

我们已经在前面的章节中引用了概率，但没有明确说明它们是什么，也没有给出具体的例子。现在让我们更认真地考虑第一个案例：根据照片区分猫和狗。这听起来可能很简单，但实际上却是一个艰巨的挑战。首先，问题的难度可能取决于图像的分辨率。

![Images of varying resolutions ($10 \times 10$, $20 \times 20$, $40 \times 40$, $80 \times 80$, and $160 \times 160$ pixels).](../img/cat-dog-pixels.png)
:width:`300px`
:label:`fig_cat_dog`

如:numref:`fig_cat_dog`所示，虽然在$160 \times 160$像素的分辨率下，人类很容易识别猫和狗，但在$40 \times 40$像素的分辨率下，它变得具有挑战性，在$10 \times 10$像素时几乎不可能。换言之，我们在很远的距离（因此分辨率较低）分辨猫和狗的能力可能接近于无知的猜测。概率给了我们一种正式的方式来推理我们的确定程度。如果我们完全确定图像描绘的是一只猫，那么我们认为对应的标签$y$是“猫”，表示为$P(y=$“猫”$)$等于$1$。如果我们没有证据表明$y =$是“猫”还是$y =$“狗”，那么我们可以说这两种可能性是相等的
*很可能*表达为$P(y=$“猫”$) = P(y=$“狗”$) = 0.5$。如果我们合理的话
自信，但不确定图像中描绘的是一只猫，我们可以指定一个概率$0.5  < P(y=$“猫”$) < 1$。

现在考虑第二种情况：根据一些天气监测数据，我们想预测明天台北下雨的可能性。如果是夏天，下雨的概率可能是0.5。

在这两种情况下，我们都有一定的价值。在这两种情况下，我们都不确定结果。但这两种情况有一个关键的区别。在第一种情况下，图像实际上不是狗就是猫，我们只是不知道是哪一种。在第二种情况下，结果实际上可能是一个随机事件，如果你相信这样的事情（大多数物理学家都相信）。因此，概率是一种灵活的语言，用于推理我们的确定程度，并且它可以有效地应用于广泛的环境中。

## 基本概率论

假设我们掷骰子，想知道看到1的几率有多大，而不是看到另一个数字。如果死亡是公平的，那么所有六个结果$\{1, \ldots, 6\}$都有同样的可能性发生，因此我们将看到每六个案例中有一个是$1$。我们正式声明$1$发生的概率为$\frac{1}{6}$。

对于一个真正的模具，我们从工厂收到，我们可能不知道这些比例，我们需要检查它是否被污染。研究模具的唯一方法是多次铸造并记录结果。对于每个铸模，我们将观察到$\{1, \ldots, 6\}$的值。考虑到这些结果，我们想调查观察每个结果的概率。

对于每个值，一个自然的方法是取每个值的单个计数，然后除以总投掷数。这给了我们一个给定事件概率的估计值。大数定律告诉我们，随着投掷次数的增加，这个估计值将越来越接近真实的潜在概率。在详细介绍这方面的内容之前，让我们先来试试。

首先，让我们导入必要的包。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch.distributions import multinomial
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
```

下一步，我们将希望能够铸造模具。在统计学中，我们把这个从概率分布中提取例子的过程称为抽样。将概率分配给多个离散选择的分布称为
*多项式分布*。我们将给出一个更正式的定义
*不过，作为一个高层次的分配，以后再想想
事件发生的概率。

要绘制单个样本，我们只需传入一个概率向量。输出是另一个长度相同的向量：它在索引$i$处的值是采样结果对应于$i$的次数。

```{.python .input}
fair_probs = [1.0 / 6] * 6
np.random.multinomial(1, fair_probs)
```

```{.python .input}
#@tab pytorch
fair_probs = torch.ones([6]) / 6
multinomial.Multinomial(1, fair_probs).sample()
```

```{.python .input}
#@tab tensorflow
fair_probs = tf.ones(6) / 6
tfp.distributions.Multinomial(1, fair_probs).sample()
```

如果你运行采样器多次，你会发现每次都会得到随机值。与估计模具的公平性一样，我们通常希望从同一分布中生成多个样本。使用python`for`循环进行此操作的速度非常慢，因此我们使用的函数支持一次绘制多个样本，返回任意形状的独立样本数组。

```{.python .input}
np.random.multinomial(10, fair_probs)
```

```{.python .input}
#@tab pytorch
multinomial.Multinomial(10, fair_probs).sample()
```

```{.python .input}
#@tab tensorflow
tfp.distributions.Multinomial(10, fair_probs).sample()
```

现在我们知道了如何对模具的辊进行取样，我们可以模拟1000个辊。然后，我们可以检查并计算每1000卷之后，每个数字被滚动了多少次。具体来说，我们计算相对频率作为真实概率的估计。

```{.python .input}
counts = np.random.multinomial(1000, fair_probs).astype(np.float32)
counts / 1000
```

```{.python .input}
#@tab pytorch
# Store the results as 32-bit floats for division
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts / 1000  # Relative frequency as the estimate
```

```{.python .input}
#@tab tensorflow
counts = tfp.distributions.Multinomial(1000, fair_probs).sample()
counts / 1000
```

因为我们从一个公平的骰子中生成了数据，我们知道每个结果的真实概率为$\frac{1}{6}$，大约为$0.167$，所以上面的输出估计值看起来不错。

我们还可以想象这些概率如何随着时间的推移收敛到真实的概率。让我们进行500组实验，每组抽取10个样本。

```{.python .input}
counts = np.random.multinomial(10, fair_probs, size=500)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].asnumpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
#@tab tensorflow
counts = tfp.distributions.Multinomial(10, fair_probs).sample(500)
cum_counts = tf.cumsum(counts, axis=0)
estimates = cum_counts / tf.reduce_sum(cum_counts, axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

每一条实体曲线对应于模具的六个值之一，并给出我们在每组实验后评估的模具出现该值的概率。虚线黑线给出了真实的潜在概率。当我们通过更多的实验得到更多的数据时，$6$条实曲线向真实概率收敛。

### 概率论公理

当处理骰子的滚动时，我们将集合$\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$称为*样本空间*或*结果空间*，其中每个元素都是*结果*。*event*是给定样本空间的一组结果。例如，“看到$5$”（$\{5\}$）和“看到奇数”（$\{1, 3, 5\}$）都是有效的滚动模具事件。注意，如果随机实验的结果是在事件$\mathcal{A}$中，那么事件$\mathcal{A}$已经发生。也就是说，如果$3$点在掷骰子后朝上，那么从$3 \in \{1, 3, 5\}$开始，我们可以说“看到奇数”事件已经发生。

形式上，*概率*可以被认为是一个函数，它将一个集合映射到一个实数。给定样本空间$\mathcal{A}$中事件$\mathcal{A}$的概率，表示为$P(\mathcal{A})$，满足以下性质：

* 对于任何事件$\mathcal{A}$，它的概率永远不是负的，即$P(\mathcal{A}) \geq 0$；
* 整个样本空间的概率为$1$，即$P(\mathcal{S}) = 1$；
* 对于任何互斥的$\mathcal{A}_1, \mathcal{A}_2, \ldots$事件序列（$i \neq j$为$i \neq j$），任何事件发生的概率等于其各自概率的总和，即$P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$。

这些也是Kolmogorov在1933年提出的概率论的公理。由于这个公理系统，我们可以避免任何关于随机性的哲学争论；相反，我们可以用数学语言进行严格的推理。例如，让事件$\mathcal{A}_1$是整个样本空间，$i > 1$是$P(\emptyset) = 0$，我们可以证明$P(\emptyset) = 0$，即不可能事件的概率是$0$。

### 随机变量

在我们铸造模具的随机实验中，我们引入了*随机变量*的概念。一个随机变量可以是几乎任何数量，而且不是确定性的。它可以从随机实验中的一组可能性中取一个值。考虑一个随机变量$X$，其值在轧制模具的样本空间$\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$中。我们可以将“看到$5$”事件表示为$\{X = 5\}$或$X = 5$，其概率为$P(\{X = 5\})$或$P(X = 5)$。通过$P(X = a)$，我们区分了随机变量$X$和$X$可以取的值（例如$a$）。然而，这样的迂腐导致了一个繁琐的符号。对于一个紧凑的符号，一方面，我们可以把$P(X)$表示为随机变量$X$的*分布：这个分布告诉我们$X$取任何值的概率。另一方面，我们可以简单地写$P(a)$来表示随机变量取值$a$的概率。由于概率论中的事件是样本空间的一组结果，我们可以指定随机变量的取值范围。例如，$P(1 \leq X \leq 3)$表示事件$\{1 \leq X \leq 3\}$的概率，即$\{X = 1, 2, \text{or}, 3\}$。等效地，$P(1 \leq X \leq 3)$表示随机变量$X$可以从$\{1, 2, 3\}$取值的概率。

注意，离散的随机变量和连续的随机变量（比如模具的侧面）和连续的随机变量（比如人的体重和身高）之间有细微的差别。问两个人的身高是否完全相同是没有意义的。如果我们进行足够精确的测量，你会发现地球上没有两个人的身高完全相同。事实上，如果我们做一个足够精确的测量，当你醒来和睡觉的时候，你的身高就不一样了。所以，我们没有必要去问一个人身高是1.80139278291028719210196740527486202米的概率。考虑到世界人口，概率几乎为0。在这种情况下，问一个人的身高是否落在给定的间隔内，比如1.79到1.81米之间，就更有意义了。在这些情况下，我们量化了将某个值视为*密度*的可能性。精确到1.80米的高度是不可能的，但是密度不是零。在任何两个不同高度之间的间隔中，我们有非零概率。在本节的其余部分中，我们将讨论离散空间中的概率。关于连续随机变量的概率，你可以参考:numref:`sec_random_variables`。

## 处理多个随机变量

通常情况下，我们会一次考虑多个随机变量。例如，我们可能需要建立疾病和症状之间的关系模型。给定一种疾病和一种症状，比如说“流感”和“咳嗽”，可能发生在病人身上，也可能不发生。虽然我们希望两者的概率接近于零，但我们可能需要估计这些概率及其相互关系，以便我们可以应用我们的推论来实现更好的医疗保健。

作为一个更复杂的例子，图像包含数百万像素，因此有数百万随机变量。在很多情况下，图像都会附带一个标签，用来识别图像中的对象。我们也可以把标签看作一个随机变量。我们甚至可以把所有元数据看作随机变量，比如位置、时间、光圈、焦距、ISO、焦距和相机类型。所有这些都是共同发生的随机变量。当我们处理多个随机变量时，会有几个有趣的量。

### 联合概率

第一个被称为联合概率$P(A = a, B=b)$。给定任何值$a$和$b$，联合概率让我们回答，$A=a$和$B=b$同时发生的概率是多少？注意，对于任何值$a$和$b$，$P(A=a, B=b) \leq P(A=a)$。必须是这样，因为$A=a$和$B=b$必须发生，$A=a$必须发生，$B=b$也必须发生（反之亦然）。因此，$A=a$和$B=b$不可能比单独的$A=a$或$B=b$大。

### 条件概率

这个比率很有趣。我们把这个比率称为条件概率，用$P(B=b \mid A=a)$表示：它是$B=b$的概率，前提是$A=a$已经发生。

### 贝叶斯定理

利用条件概率的定义，我们可以导出统计学中最有用和最著名的方程之一：*贝叶斯定理*。情况如下。通过构造，我们得到了*乘法规则，即$P(A, B) = P(B \mid A) P(A)$。根据对称性，这也适用于$P(A, B) = P(A \mid B) P(B)$。假设$P(B) > 0$。求解我们得到的一个条件变量

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}.$$

注意，这里我们使用更紧凑的符号，其中$P(A, B)$是*联合分布*而$P(A \mid B)$是*条件分布*。这种分布可以针对特定值$A = a, B=b$进行评估。

### 边缘化

如果我们想从另一件事中推断出一件事，比如因果关系，那么Bayes定理是非常有用的，但是我们只知道反方向的性质，这一节后面我们会看到。我们需要的一个重要的行动，就是边缘化。它是从$P(A, B)$确定$P(B)$的操作。我们可以看到，$B$的概率相当于说明了$A$的所有可能选择，并将所有这些选择的联合概率加在一起：

$$P(B) = \sum_{A} P(A, B),$$

也被称为*求和规则*。边缘化的概率或分布称为边际概率或边际分布。

### 独立性

另一个有用的属性是*dependency*vs.*independence*。两个随机变量$A$和$B$是独立的意味着一个$A$事件的发生并不揭示任何关于$B$事件发生的信息。本案为$P(B \mid A) = P(B)$。统计学家通常将其表述为$A \perp  B$。根据Bayes定理，它紧跟着$P(A \mid B) = P(A)$。在所有其他情况下，我们称$A$和$B$为从属关系。例如，一个模具的两个连续辊是独立的。相比之下，电灯开关的位置和房间里的亮度则不是（虽然它们不是完全确定的，因为我们可能总是灯泡坏了，停电了，或者开关坏了）。

由于$P(A \mid B) = \frac{P(A, B)}{P(B)} = P(A)$相当于$P(A, B) = P(A)P(B)$，两个随机变量是独立的当且仅当它们的联合分布是各自分布的乘积。同样地，两个随机变量$A$和$B$是*条件独立*给另一个随机变量$C$当且仅当$P(A, B \mid C) = P(A \mid C)P(B \mid C)$。这表示为$A \perp B \mid C$。

### 应用
:label:`subsec_probability_hiv_app`

让我们考验一下我们的技能。假设一个医生对病人进行HIV测试。这个测试是相当准确的，如果病人是健康的但报告他有病，它失败的概率只有1%。此外，如果病人真的感染了艾滋病病毒，它也不会漏掉。我们使用$D_1$来表示诊断（$1$如果阳性，$0$如果阴性）和$H$来表示HIV状态（$1$如果阳性，$0$如果阴性）。:numref:`conditional_prob_D1`列出了这样的条件概率。

：条件概率为$P(D_1 \mid H)$。

| Conditional probability | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_1 = 1 \mid H)$|            1 |         0.01 |
|$P(D_1 = 0 \mid H)$|            0 |         0.99 |
:label:`conditional_prob_D1`

注意，列和都是1（但是行和不是），因为条件概率需要和概率一样加到1。让我们计算出如果检测结果呈阳性，患者感染艾滋病毒的概率，即$P(H = 1 \mid D_1 = 1)$。显然，这将取决于这种疾病的普遍程度，因为它会影响假警报的数量。假设人口相当健康，例如$P(H=1) = 0.0015$。为了应用Bayes定理，我们需要应用边缘化和乘法法则来确定

$$\begin{aligned}
&P(D_1 = 1) \\
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

因此，我们得到

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1)\\ =& \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} \\ =& 0.1306 \end{aligned}.$$

换言之，尽管使用了非常精确的检测方法，但患者感染艾滋病毒的几率只有13.06%。正如我们所见，概率可能是违反直觉的。

病人在收到如此可怕的消息后该怎么办？很可能，患者会要求医生进行另一项检查以获得清晰的信息。第二个测试有不同的特性，它不如第一个测试好，如:numref:`conditional_prob_D2`所示。

：条件概率为$P(D_2 \mid H)$。

| Conditional probability | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_2 = 1 \mid H)$|            0.98 |         0.03 |
|$P(D_2 = 0 \mid H)$|            0.02 |         0.97 |
:label:`conditional_prob_D2`

不幸的是，第二次检测也呈阳性。让我们通过假设条件独立性来计算调用Bayes定理的必要概率：

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1 \mid H = 0) \\
=& P(D_1 = 1 \mid H = 0) P(D_2 = 1 \mid H = 0)  \\
=& 0.0003,
\end{aligned}
$$

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1 \mid H = 1) \\
=& P(D_1 = 1 \mid H = 1) P(D_2 = 1 \mid H = 1)  \\
=& 0.98.
\end{aligned}
$$

现在我们可以应用边缘化和乘法法则：

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1) \\
=& P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
=& P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
=& 0.00176955.
\end{aligned}
$$

最后，在两种阳性测试中，病人感染HIV的概率是

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1, D_2 = 1)\\
=& \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)} \\
=& 0.8307.
\end{aligned}
$$

也就是说，第二次测试让我们获得了更高的信心，认为并非所有的都是好的。尽管第二次测试的准确度远低于第一次测试，但它仍然大大提高了我们的估计。

## 期望与方差

为了总结概率分布的主要特征，我们需要一些度量。随机变量的平均值＊61734

$$E[X] = \sum_{x} x P(X = x).$$

当函数$f(x)$的输入是从具有不同值$f(x)$的分布$f(x)$提取的随机变量时，$f(x)$的期望值计算为

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x).$$

在许多情况下，我们想通过随机变量$X$偏离其期望值的程度来衡量。这可以用方差来量化

$$\mathrm{Var}[X] = E\left[(X - E[X])^2\right] =
E[X^2] - E[X]^2.$$

它的平方根称为*标准差*。随机变量函数的方差衡量函数偏离函数期望的程度，因为随机变量的不同值$x$从其分布中取样：

$$\mathrm{Var}[f(x)] = E\left[\left(f(x) - E[f(x)]\right)^2\right].$$

## 摘要

* 我们可以从概率分布中取样。
* 我们可以使用联合分布、条件分布、Bayes定理、边缘化和独立性假设来分析多个随机变量。
* 期望和方差为总结概率分布的关键特征提供了有用的度量方法。

## 练习

1. 我们进行了$m=500$组实验，每组抽取$n=10$个样本。更改$m$和$n$。观察并分析实验结果。
1. 给定概率为$P(\mathcal{A})$和$P(\mathcal{B})$的两个事件，计算$P(\mathcal{A} \cup \mathcal{B})$和$P(\mathcal{A} \cap \mathcal{B})$的上下界。（提示：使用[Venn Diagram](https://en.wikipedia.org/wiki/Venn_diagram)显示情况。）
1. 假设我们有一系列随机变量，比如$A$、$B$和$C$，其中$B$只依赖于$A$，$C$只依赖于$B$，你能简化联合概率$P(A, B, C)$吗？（提示：这是[Markov Chain](https://en.wikipedia.org/wiki/Markov_chain)。）
1. 在:numref:`subsec_probability_hiv_app`中，第一个测试更准确。为什么不运行第一个测试两次，而不是同时运行第一个和第二个测试？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/36)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/37)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/198)
:end_tab:
