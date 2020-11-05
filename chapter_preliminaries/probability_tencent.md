# 概率
:label:`sec_prob`

以这样或那样的形式，机器学习都是关于做出预测的。考虑到患者的临床病史，我们可能想要预测患者明年心脏病发作的“概率”。在异常检测中，我们可能想要评估一组来自飞机喷气发动机的读数的“可能性”有多大，如果它正常运行的话。在强化学习中，我们希望Agent在环境中智能地行动。这意味着我们需要考虑在每种可用行动下获得高额奖励的概率。当我们建立推荐系统时，我们也需要考虑概率。例如，假设我们为一家大型在线书商工作。我们可能想要估计特定用户购买特定图书的概率。为此，我们需要使用概率语言。整个课程，专业，论文，职业，甚至系，都致力于概率。因此，很自然，我们在这一部分的目标不是教授整个课程。相反，我们希望让您起步，教给您足够的知识，以便您可以开始构建您的第一个深度学习模型，并给您足够的兴趣，以便您可以开始自己探索这门学科(如果您愿意的话)。

我们已经在前面的章节中引用了概率，但没有详细说明它们到底是什么，也没有给出具体的例子。现在让我们更认真地考虑第一个案例：根据照片区分猫和狗。这听起来可能很简单，但实际上是一个艰巨的挑战。首先，问题的难度可能取决于图像的分辨率。

![Images of varying resolutions ($10 \times 10$, $20 \times 20$, $40 \times 40$, $80 \times 80$, and $160 \times 160$ pixels).](../img/cat-dog-pixels.png)
:width:`300px`
:label:`fig_cat_dog`

如:numref:`fig_cat_dog`所示，虽然人类在$160 \times 160$像素的分辨率下很容易识别猫和狗，但在$40 \times 40$像素的分辨率下就变得很有挑战性，而在$10 \times 10$像素的分辨率下几乎是不可能的。换句话说，我们在很远的距离上区分猫和狗的能力(因此分辨率很低)可能接近于不知情的猜测。概率为我们提供了一种关于确定性水平的正式推理方式。如果我们完全确定图像描绘了一只猫，我们说对应的标签$y$是“猫”(表示为$P(y=$“猫”$)$)的*概率*等于$1$。如果我们没有证据表明$y =$的“猫”或$y =$的“狗”，那么我们可以说这两种可能性是相等的。
*可能*表示为$P(y=$“猫”$) = P(y=$“狗”$) = 0.5$。如果我们合理地
有信心，但不确定图像描绘的是一只猫，我们可能会将概率指定为$0.5  < P(y=$“猫”$) < 1$。

现在再来考虑第二个案例，在一些天气监测数据的情况下，我们想预测一下台北明天下雨的概率。如果现在是夏季，下雨的概率可能是0.5。

在这两种情况下，我们都有一定的利益价值。在这两种情况下，我们都不确定结果。但这两个案例之间有一个关键的不同之处。在第一种情况下，图像实际上是一只狗或一只猫，我们只是不知道是哪一种。在第二种情况下，如果你相信这样的事情(大多数物理学家都相信)，结果实际上可能是一个随机事件。因此，概率是一种灵活的语言，可以用来推理我们的确定性水平，它可以有效地应用于广泛的背景中。

## 基本概率论

假设我们掷骰子，想知道看到1而不是另一个数字的机会有多大。如果骰子是公平的，所有六个结果$\{1, \ldots, 6\}$的可能性都是相等的，因此我们将看到每六个案例中就有一个是$1$。从形式上说，$1$发生的概率为$\frac{1}{6}$。

对于我们从工厂收到的真正的模具，我们可能不知道这些比例，我们需要检查它是否被污染。调查模具的唯一方法是多次铸造并记录结果。对于骰子的每一次投掷，我们将在$\{1, \ldots, 6\}$中观察到一个值。考虑到这些结果，我们想要调查观察到每个结果的可能性。

对于每个值，一种自然的方法是取该值的单个计数，然后将其除以抛出的总次数。这给了我们对给定*事件*的概率的“估计”。“大数定律”告诉我们，随着掷硬币次数的增加，这一估计将越来越接近真实的潜在概率。在详细介绍这里发生的事情之前，让我们先试一试。

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

接下来，我们希望能够掷骰子。在统计学中，我们称之为从概率分布中抽取样本的过程*抽样*。将概率分配给若干离散选择的分布称为
*多项分布*。我们将给出一个更正式的定义
*分发*稍后，但在更高的级别上，可以将其看作是对
事件的概率。

要绘制单个样本，我们只需传入一个概率向量。输出是另一个相同长度的向量：它在索引$i$处的值是采样结果对应于$i$的次数。

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

如果您多次运行采样器，您会发现每次都会得到随机值。与估计骰子的公平性一样，我们通常希望从相同的分布中生成许多样本。使用Python`for`循环执行此操作将慢得令人无法忍受，因此我们使用的函数支持一次绘制多个样本，返回任意形状的独立样本数组。

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

既然我们知道了如何对骰子的卷进行取样，我们就可以模拟1000卷了。然后，我们可以在1000卷中的每一卷之后仔细数一数，每个数字被滚动了多少次。具体地说，我们计算相对频率作为真实概率的估计。

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

因为我们是从公平的骰子中生成数据的，所以我们知道每个结果都有$\frac{1}{6}$的真实概率，大约是$0.167$，所以上面的产量估计看起来不错。

我们还可以想象这些概率是如何随时间向真实概率收敛的。让我们进行500组实验，每组抽取10个样本。

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

每条立体曲线对应于骰子的六个值中的一个，并给出了我们估计的骰子在每组实验后发现该值的概率。黑虚线给出了真实的潜在概率。随着我们通过进行更多的实验获得更多的数据，$6$条固体曲线向真实概率收敛。

### 概率论公理

在处理骰子的滚动时，我们将集合$\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$称为“样本空间”或“结果空间”，其中每个元素都是一个“结果”。事件*是给定样本空间的一组结果。例如，“看到$5$”($\{5\}$)和“看到奇数”($\{1, 3, 5\}$)都是掷骰子的有效事件。请注意，如果随机实验的结果在事件$\mathcal{A}$中，则事件$\mathcal{A}$已经发生。也就是说，如果掷骰子后$3$个点朝上，从$3 \in \{1, 3, 5\}$开始，我们可以说发生了“见奇数”事件。

形式上，*概率*可以被认为是将集合映射到真实值的函数。事件$\mathcal{A}$在给定样本空间$\mathcal{S}$中的概率(表示为$P(\mathcal{A})$)满足以下属性：

* 对于任何事件$\mathcal{A}$，其概率从不为负，即$P(\mathcal{A}) \geq 0$；
* 整个样本空间的概率为$1$，即$P(\mathcal{S}) = 1$；
* 对于*互斥*的任何可数序列事件$\mathcal{A}_1, \mathcal{A}_2, \ldots$(对于所有事件$i \neq j$都是$\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$)，任何发生的概率等于它们各自概率的总和，即$P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$。

这些也是科尔莫戈罗夫在1933年提出的概率论公理。多亏了这个公理体系，我们可以避免任何关于随机性的哲学争论；相反，我们可以用数学语言进行严格的推理。例如，通过将事件$\mathcal{A}_1$设为整个样本空间，并将所有$i > 1$设为$\mathcal{A}_i = \emptyset$，我们可以证明$P(\emptyset) = 0$，即不可能事件的概率为$0$。

### 随机变量

在我们的随机铸模实验中，我们引入了“随机变量”的概念。随机变量几乎可以是任何量，并且不是确定性的。它可以在随机实验的一组可能性中取一个值。考虑随机变量$X$，其值在滚动骰子的样本空间$\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$中。我们可以将事件“看到$5$”表示为$\{X = 5\}$或$X = 5$，其概率为$P(\{X = 5\})$或$P(X = 5)$。通过$P(X = a)$，我们在随机变量$X$和$X$可以取的值(例如，$a$)之间进行区分。然而，这种迂腐的做法导致了繁琐的记号。对于一个紧凑的符号，一方面，我们可以将$P(X)$表示为随机变量$X$上的*分布*：该分布告诉我们$X$取任意值的概率。另一方面，我们可以简单地写成$P(a)$来表示随机变量取值$a$的概率。由于概率论中的事件是样本空间的一组结果，因此我们可以指定随机变量的取值范围。例如，$P(1 \leq X \leq 3)$表示事件$\{1 \leq X \leq 3\}$的概率，这意味着$\{X = 1, 2, \text{or}, 3\}$。等效地，$P(1 \leq X \leq 3)$表示随机变量$X$可以从$\{1, 2, 3\}$取值的概率。

请注意，像骰子侧面这样的“离散”随机变量和像人的体重和身高这样的“连续”随机变量之间有微妙的区别。问两个人的身高是否完全相同没有什么意义。如果我们测量得足够精确，你会发现地球上没有两个人的身高完全相同。事实上，如果我们测量得足够精细，当你醒来和睡觉的时候，你的身高就不会一样了。因此，询问某人身高1.80139278291028719210196740527486202米的概率是没有意义的。考虑到世界人口的数量，这个概率几乎为0。在这种情况下，询问某人的身高是否落在给定的区间内更有意义，比如在1.79米到1.81米之间。在这些情况下，我们将看到一个值的可能性量化为*密度*。恰好是1.80米的高度没有概率，但密度不为零。在任意两个不同高度之间的区间内，我们有非零概率。在本节的睡觉中，我们考虑离散空间中的概率。关于连续随机变量的概率，你可以参考:numref:`sec_random_variables`。

## 多随机变量的处理

通常，我们希望一次考虑多个随机变量。例如，我们可能想要对疾病和症状之间的关系进行建模。给定一种疾病和一种症状，比如“流感”和“咳嗽”，患者可能会发生，也可能不会发生。虽然我们希望两者的概率都接近于零，但我们可能想要估计这些概率及其相互之间的关系，以便我们可以应用我们的推断来实现更好的医疗保健。

作为一个更复杂的例子，图像包含数百万个像素，因此包含数百万个随机变量。在许多情况下，图像会附带一个标签，用来标识图像中的对象。我们也可以把标签看作是一个随机变量。我们甚至可以将所有元数据视为随机变量，如位置、时间、光圈、焦距、ISO、焦距和相机类型。所有这些都是共同出现的随机变量。当我们处理多个随机变量时，有几个量是令人感兴趣的。

### 联合概率

第一个称为“联合概率*$P(A = a, B=b)$”。给定任意值$a$和$b$，联合概率让我们回答，$A=a$和$B=b$同时出现的概率是多少？请注意，对于任何值$a$和$b$,$P(A=a, B=b) \leq P(A=a)$。这是必须的，因为对于$A=a$和$B=b$,$A=a$必须发生*而*$B=b$也必须发生(反之亦然)。因此，$A=a$和$B=b$不能分别大于$A=a$或$B=b$。

### 条件概率

这给我们带来了一个有趣的比率：$0 \leq \frac{P(A=a, B=b)}{P(A=a)} \leq 1$。我们称这个比率为“条件概率”，并用$P(B=b \mid A=a)$表示：它是$B=b$的概率，假设$A=a$已经发生。

### 贝叶斯定理

利用条件概率的定义，我们可以推导出统计学中最有用和最著名的方程之一：*贝叶斯定理*。它是这样进行的。通过构造，我们得到了$P(A, B) = P(B \mid A) P(A)$的“乘法法则”。根据对称性，这也适用于$P(A, B) = P(A \mid B) P(B)$。假设是$P(B) > 0$。我们得到的一个条件变量的解

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}.$$

请注意，这里我们使用更紧凑的表示法，其中$P(A, B)$是“联合分布”，$P(A \mid B)$是“条件分布”。可以针对特定值$A = a, B=b$评估这样的分布。

### 边缘化

如果我们想从一件事推断另一件事，比如因果，那么贝叶斯定理是非常有用的，但是我们只知道相反方向的属性，就像我们将在本节后面看到的那样。要实现这一点，我们需要的一个重要行动是“边缘化”。它是从$P(B)$确定$P(A, B)$的运算。我们可以看到，$B$的概率相当于考虑了$A$的所有可能选择，并汇总了所有这些选项的联合概率：

$$P(B) = \sum_{A} P(A, B),$$

这也被称为*和规则*。边际化的概率或分布称为“边际概率”或“边际分布”。

### 独立

另一个需要检查的有用属性是“依赖”与“独立”。两个随机变量$A$和$B$是独立的意味着$A$的一个事件的发生不揭示关于$B$的事件的发生的任何信息。在这种情况下是$P(B \mid A) = P(B)$。统计学家通常将这一数字表示为$A \perp  B$。从贝叶斯定理可以直接得出，也是$P(A \mid B) = P(A)$。在所有其他情况下，我们称$A$和$B$为从属。例如，骰子的两个连续卷是独立的。相反，电灯开关的位置和房间里的亮度不是完全确定的(不过，它们不是完全确定的，因为我们可能总是会有灯泡坏了、电源故障或开关坏了)。

因为$P(A \mid B) = \frac{P(A, B)}{P(B)} = P(A)$等于$P(A, B) = P(A)P(B)$，所以两个随机变量是独立的当且仅当它们的联合分布是它们各自分布的乘积。同样，两个随机变量$A$和$B$在给定另一个随机变量$C$的情况下是“条件独立的”当且仅当$P(A, B \mid C) = P(A \mid C)P(B \mid C)$。这表示为$A \perp B \mid C$。

### 应用程序
:label:`subsec_probability_hiv_app`

让我们来考验一下我们的技能吧。假设一名医生对一名病人进行艾滋病病毒检测。这个测试相当准确，如果病人是健康的，但报告他是有病的，它失败的可能性只有1%。此外，如果患者真的感染了艾滋病病毒，它永远不会错过检测。我们使用$D_1$表示诊断($1$表示阳性，$0$表示阴性)，$H$表示艾滋病毒状态($1$表示阳性，$0$表示阴性)。:numref:`conditional_prob_D1`列出了这样的条件概率。

：条件概率为$P(D_1 \mid H)$。

| Conditional probability | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_1 = 1 \mid H)$|            1 |         0.01 |
|$P(D_1 = 0 \mid H)$|            0 |         0.99 |
:label:`conditional_prob_D1`

请注意，列和都是1(但行和不是)，因为条件概率需要与概率一样求和为1。让我们计算一下，如果检测结果呈阳性，即$P(H = 1 \mid D_1 = 1)$，病人感染艾滋病毒的可能性是多少。显然，这将取决于这种疾病的普遍程度，因为它会影响错误警报的数量。假设人口相当健康，例如，$P(H=1) = 0.0015$。要应用贝叶斯定理，我们需要应用边际化和乘法法则来确定

$$\begin{aligned}
&P(D_1 = 1) \\
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

因此，我们得到了

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1)\\ =& \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} \\ =& 0.1306 \end{aligned}.$$

换句话说，尽管使用了非常准确的检测，但患者实际上感染艾滋病毒的可能性只有13.06%。正如我们所看到的，概率可能是违反直觉的。

病人收到这样可怕的消息该怎么办呢？很可能，病人会要求医生再进行一次测试，以获得明确的结果。第二个测试有不同的特点，不如第一个测试好，如:numref:`conditional_prob_D2`所示。

：条件概率为$P(D_2 \mid H)$。

| Conditional probability | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_2 = 1 \mid H)$|            0.98 |         0.03 |
|$P(D_2 = 0 \mid H)$|            0.02 |         0.97 |
:label:`conditional_prob_D2`

不幸的是，第二次检测结果也呈阳性。让我们通过假设条件独立性来计算调用贝叶斯定理所需的概率：

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

现在我们可以应用边际化和乘法规则：

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1) \\
=& P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
=& P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
=& 0.00176955.
\end{aligned}
$$

最后，两种检测结果均呈阳性的患者感染艾滋病病毒的概率为

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1, D_2 = 1)\\
=& \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)} \\
=& 0.8307.
\end{aligned}
$$

也就是说，第二次测试让我们获得了更高的信心，即并不是所有的事情都很好。尽管第二次测试的精确度比第一次低得多，但它仍然显著改善了我们的估计。

## 期望与方差

为了总结概率分布的关键特征，我们需要一些措施。随机变量$X$的*期望*(或平均值)表示为

$$E[X] = \sum_{x} x P(X = x).$$

当函数$f(x)$的输入是从分布$P$提取的具有不同值$x$的随机变量时，$f(x)$的期望被计算为

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x).$$

在许多情况下，我们希望通过随机变量$X$偏离其预期的程度来衡量。这可以通过方差来量化

$$\mathrm{Var}[X] = E\left[(X - E[X])^2\right] =
E[X^2] - E[X]^2.$$

它的平方根称为*标准差*。当随机变量的不同值$x$从其分布中被采样时，随机变量的函数的方差通过该函数偏离该函数的期望的程度来度量：

$$\mathrm{Var}[f(x)] = E\left[\left(f(x) - E[f(x)]\right)^2\right].$$

## 摘要

* 我们可以从概率分布中抽样。
* 我们可以使用联合分布、条件分布、贝叶斯定理、边际化和独立性假设来分析多个随机变量。
* 期望和方差为总结概率分布的关键特征提供了有用的度量。

## 练习

1. 我们进行了$m=500$组实验，每组抽取$n=10$个样本。变化$m$和$n$。对实验结果进行观察和分析。
1. 给定概率为$P(\mathcal{A})$和$P(\mathcal{B})$的两个事件，计算$P(\mathcal{A} \cup \mathcal{B})$和$P(\mathcal{A} \cap \mathcal{B})$的上界和下界。(提示：使用[Venn Diagram](https://en.wikipedia.org/wiki/Venn_diagram)显示情况。)
1. 假设我们有一个随机变量序列，比如说$A$、$B$和$C$，其中$B$只依赖于$A$,$C$只依赖于$B$，您能简化联合概率$P(A, B, C)$吗？(提示：这是一辆[Markov Chain](https://en.wikipedia.org/wiki/Markov_chain)。)
1. 在:numref:`subsec_probability_hiv_app`中，第一次测试更准确。为什么不运行第一个测试两次，而不是同时运行第一个和第二个测试呢？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/36)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/37)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/198)
:end_tab:
