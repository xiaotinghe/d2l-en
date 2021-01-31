# 语言模型与数据集
:label:`sec_language_model`

在:numref:`sec_text_preprocessing`中，我们看到了如何将文本数据映射到标记中，其中这些标记可以看作是一系列离散的观察结果，例如单词或字符。假设长度为$T$的文本序列中的令牌依次为$x_1, x_2, \ldots, x_T$。然后，在文本序列中，$x_t$（$1 \leq t \leq T$）可被视为时间步骤$t$处的观察或标签。给定这样一个文本序列，*语言模型*的目标是估计序列的联合概率

$$P(x_1, x_2, \ldots, x_T).$$

语言模型非常有用。例如，一个理想的语言模型能够自己生成自然文本，只需每次绘制一个标记$x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1)$。与使用打字机的猴子完全不同，从这种模型中产生的所有文本都将作为自然语言传递，例如英语文本。此外，只要将文本限定在前面的对话片段上，就足以生成一个有意义的对话。显然，我们离设计这样一个系统还很遥远，因为它需要“理解”文本，而不仅仅是生成语法上合理的内容。

尽管如此，语言模型即使在其有限的形式下也能发挥巨大的作用。例如，短语“识别语音”和“破坏美丽的海滩”听起来非常相似。这可能会导致语音识别中的歧义，这很容易通过一种语言模型来解决，这种语言模型拒绝将第二个译文视为稀奇古怪。同样，在文档摘要算法中，“狗咬人”比“人咬狗”要频繁得多，或者“我想吃奶奶”是一个相当令人不安的语句，而“我想吃，奶奶”则要温和得多。

## 学习语言模型

显而易见的问题是，我们应该如何建模文档，甚至是一系列标记。假设我们在单词级别标记文本数据。我们可以利用:numref:`sec_sequence`中我们应用于序列模型的分析。让我们从应用基本概率规则开始：

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1}).$$

例如，包含四个单词的文本序列的概率如下所示：

$$P(\text{deep}, \text{learning}, \text{is}, \text{fun}) =  P(\text{deep}) P(\text{learning}  \mid  \text{deep}) P(\text{is}  \mid  \text{deep}, \text{learning}) P(\text{fun}  \mid  \text{deep}, \text{learning}, \text{is}).$$

为了计算语言模型，我们需要计算单词的概率和给定前几个单词的单词的条件概率。这些概率本质上是语言模型参数。

在这里，我们假设训练数据集是一个大型文本语料库，例如所有维基百科条目、[Project Gutenberg](https://en.wikipedia.org/wiki/Project_Gutenberg)以及所有发布在Web上的文本。单词的概率可以通过训练数据集中给定单词的相对词频来计算。例如，估计值$\hat{P}(\text{deep})$可以计算为以单词“deep”开头的任何句子的概率。稍微不太准确的方法是计算“deep”一词的所有出现次数，然后除以语料库中的单词总数。这很有效，特别是对于频繁使用的单词。接下来，我们可以尝试估计

$$\hat{P}(\text{learning} \mid \text{deep}) = \frac{n(\text{deep, learning})}{n(\text{deep})},$$

其中$n(x)$和$n(x, x')$分别是单例和连续字对的出现次数。不幸的是，估计一个词对的概率有点困难，因为“深度学习”的出现要少很多。特别是，对于一些不寻常的单词组合，可能很难找到足够的出现次数来获得准确的估计。对于三个词的组合和其他词，情况会变得更糟。在我们的数据集中，我们可能看不到许多似是而非的三词组合。除非我们提供一些解决方案来分配这样的单词组合非零计数，否则我们将无法在语言模型中使用它们。如果数据集很小，或者单词非常罕见，我们甚至可能找不到一个单词。

常用的策略是执行某种形式的拉普拉斯平滑。解决办法是在所有计数中加一个小常数。用$n$表示训练集中的总字数，用$m$表示唯一字数。此解决方案有助于解决单例问题，例如，通过

$$\begin{aligned}
	\hat{P}(x) & = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1}, \\
	\hat{P}(x' \mid x) & = \frac{n(x, x') + \epsilon_2 \hat{P}(x')}{n(x) + \epsilon_2}, \\
	\hat{P}(x'' \mid x,x') & = \frac{n(x, x',x'') + \epsilon_3 \hat{P}(x'')}{n(x, x') + \epsilon_3}.
\end{aligned}$$

这里$\epsilon_1,\epsilon_2$和$\epsilon_3$是超参数。以$\epsilon_1$为例：当$\epsilon_1 = 0$时，不应用平滑；当$\epsilon_1$接近正无穷大时，$\hat{P}(x)$接近均匀概率$1/m$。以上是其他技术可以实现的:cite:`Wood.Gasthaus.Archambeau.ea.2011`的一个相当原始的变体。

不幸的是，由于以下原因，这样的模型很快就会变得笨拙。首先，我们需要存储所有计数。其次，这完全忽略了词语的含义。例如，“猫”和“猫”应该出现在相关的上下文中。这是很难调整这种模式，以额外的环境，然而，深入学习为基础的语言模型是非常适合考虑到这一点。最后，长单词序列几乎肯定是新颖的，因此一个简单地计算先前看到的单词序列的频率的模型在那里的表现肯定很差。

## 马尔可夫模型与$n$克

在讨论涉及深度学习的解决方案之前，我们需要更多的术语和概念。回想一下我们在:numref:`sec_sequence`中对马尔可夫模型的讨论。让我们来应用这个语言。序列上的分布满足一阶马氏性if $P(x_{t+1} \mid x_t, \ldots, x_1) = P(x_{t+1} \mid x_t)$。高阶对应较长的依赖关系。这导致了我们可以应用于序列建模的许多近似值：

$$
\begin{aligned}
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3).
\end{aligned}
$$

涉及一个、两个和三个变量的概率公式通常分别称为“单变量模型”、“双变量模型”和“三变量模型”。在下面，我们将学习如何设计更好的模型。

## 自然语言统计

让我们看看它是如何在真实数据上工作的。我们根据:numref:`sec_text_preprocessing`中介绍的时间机器数据集构建了一个词汇表，并打印出前10个最常用的单词。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
#@tab all
tokens = d2l.tokenize(d2l.read_time_machine())
# Since each text line is not necessarily a sentence or a paragraph, we
# concatenate all text lines 
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
vocab.token_freqs[:10]
```

正如我们所看到的，最流行的词实际上是相当无聊的看。它们通常被称为“停止词”，因此被过滤掉。尽管如此，它们仍然具有意义，我们仍将使用它们。此外，很明显词频衰减得相当快。$10^{\mathrm{th}}$最常用的单词少于$1/5$最常用的单词。为了更好的理解，我们绘制了词频的图。

```{.python .input}
#@tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

我们在这里看到了一些非常基本的东西：词频以一种明确的方式迅速衰减。将前几个单词作为例外处理后，所有剩余的单词大致沿着对数图上的一条直线。这意味着单词符合Zipf定律，即$i^\mathrm{th}$最常用单词的频率$n_i$为：

$$n_i \propto \frac{1}{i^\alpha},$$
:eqlabel:`eq_zipf_law`

相当于

$$\log n_i = -\alpha \log i + c,$$

其中$\alpha$是表征分布的指数，$c$是常数。如果我们想通过计数统计和平滑对单词进行建模，这应该已经让我们暂停了。毕竟，我们会明显高估尾部的频率，也就是不常出现的词。但是其他的单词组合呢，比如bigrams，trigrams，等等？让我们看看二元图频率的行为方式是否与单元图频率相同。

```{.python .input}
#@tab all
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

有一点值得注意。在十个最常见的词对中，有九个是由两个停止词组成的，只有一个与实际的书——《时间》有关。此外，让我们看看三角形频率是否以同样的方式表现。

```{.python .input}
#@tab all
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

最后，让我们将这三个模型中的标记频率可视化：单图、双图和三元图。

```{.python .input}
#@tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

这个数字令人兴奋，原因有很多。首先，除了单字外，单词序列似乎也遵循齐普夫定律，尽管:eqref:`eq_zipf_law`中的指数$\alpha$更小，这取决于序列长度。其次，$n$克的数量并没有那么大。这给了我们希望，语言中有相当多的结构。第三，许多$n$克很少出现，这使得拉普拉斯平滑相当不适合语言建模。相反，我们将使用基于深度学习的模型。

## 读取长序列数据

由于序列数据本质上是连续的，我们需要解决处理它的问题。我们在:numref:`sec_sequence`以一种相当特别的方式这样做。当序列太长而无法一次由模型处理时，我们可能希望拆分这些序列以便读取。现在让我们来描述一下总体策略。在介绍模型之前，我们假设我们将使用神经网络来训练语言模型，其中网络一次处理一小批具有预定义长度的序列，比如$n$个时间步。现在的问题是如何随机读取小批量的特性和标签。

首先，由于文本序列可以是任意长的，例如整个*时间机器*书籍，我们可以将这样长的序列划分为具有相同时间步数的子序列。在训练我们的神经网络时，一小批这样的子序列将被输入到模型中。假设网络一次处理$n$个时间步的子序列。:numref:`fig_timemachine_5gram`展示了从原始文本序列获得子序列的所有不同方法，其中$n=5$和每个时间步的令牌对应于一个字符。注意，我们有相当大的自由度，因为我们可以选择一个任意的偏移量来指示初始位置。

![Different offsets lead to different subsequences when splitting up text.](../img/timemachine-5gram.svg)
:label:`fig_timemachine_5gram`

因此，我们应该从:numref:`fig_timemachine_5gram`中选择哪一个呢？其实，他们都一样好。然而，如果我们只选择一个偏移量，那么训练我们的网络的所有可能的子序列的覆盖范围是有限的。因此，我们可以从随机偏移量开始对序列进行分区，以获得*覆盖*和*随机性*。在下面，我们将描述如何为这两个应用程序实现这一点
*随机抽样*和*顺序分区*策略。

### 随机抽样

在随机抽样中，每个示例都是在原始长序列上任意捕获的子序列。在迭代过程中，来自两个相邻随机小批量的子序列不一定在原始序列上相邻。对于语言建模，目标是根据我们目前看到的标记预测下一个标记，因此标签是原始序列，由一个标记移位。

下面的代码每次都从数据中随机生成一个minibatch。这里，参数`batch_size`指定每个minibatch中的子序列示例数，`num_steps`是每个子序列中预定义的时间步数。

```{.python .input}
#@tab all
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using random sampling."""
    # Start with a random offset (inclusive of `num_steps - 1`) to partition a
    # sequence
    corpus = corpus[random.randint(0, num_steps - 1):]
    # Subtract 1 since we need to account for labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # The starting indices for subsequences of length `num_steps`
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # In random sampling, the subsequences from two adjacent random
    # minibatches during iteration are not necessarily adjacent on the
    # original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return a sequence of length `num_steps` starting from `pos`
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, `initial_indices` contains randomized starting indices for
        # subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield d2l.tensor(X), d2l.tensor(Y)
```

让我们手动生成一个从0到34的序列。我们假设批量大小和时间步数分别为2和5。这意味着我们可以生成$\lfloor (35 - 1) / 5 \rfloor= 6$个特征标签子序列对。小批量大小为2时，我们只能得到3个小批量。

```{.python .input}
#@tab all
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

### 顺序分区

除了对原始序列进行随机抽样外，我们还可以保证迭代过程中两个相邻小批量的子序列在原始序列上是相邻的。这种策略在对小批进行迭代时保留了拆分子序列的顺序，因此称为顺序分区。

```{.python .input}
#@tab mxnet, pytorch
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = d2l.tensor(corpus[offset: offset + num_tokens])
    Ys = d2l.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

```{.python .input}
#@tab tensorflow
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = d2l.tensor(corpus[offset: offset + num_tokens])
    Ys = d2l.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs = d2l.reshape(Xs, (batch_size, -1))
    Ys = d2l.reshape(Ys, (batch_size, -1))
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

使用相同的设置，让我们为顺序分区读取的每个小批量子序列打印特征`X`和标签`Y`。请注意，在迭代过程中，来自两个相邻小批的子序列在原始序列上确实是相邻的。

```{.python .input}
#@tab all
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

现在我们将上述两个采样函数包装到一个类中，以便以后可以将其用作数据迭代器。

```{.python .input}
#@tab all
class SeqDataLoader:  #@save
    """An iterator to load sequence data."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
```

最后，我们定义了一个函数`load_data_time_machine`，它返回数据迭代器和词汇表，因此我们可以像使用`load_data`前缀的其他函数一样使用它，比如:numref:`sec_fashion_mnist`中定义的`d2l.load_data_fashion_mnist`。

```{.python .input}
#@tab all
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """Return the iterator and the vocabulary of the time machine dataset."""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
```

## 摘要

* 语言模型是自然语言处理的关键。
* $n$克通过截断依赖性提供了一个处理长序列的方便模型。
* 长序列的问题是它们很少出现或从未出现过。
* 齐普夫定律不仅适用于单字，也适用于其他$n$字。
* 通过拉普拉斯平滑法可以有效地处理不常见的词组合，但结构复杂，频率不够。
* 读取长序列的主要选择是随机抽样和序列分割。后者可以保证迭代过程中相邻两个小批量的子序列在原序列上是相邻的。

## 练习

1. 假设训练数据集中有$100,000$个单词。四克需要存储多少字频率和多字相邻频率？
1. 你将如何模拟对话？
1. 估计单图、双图和三元图的齐普夫定律指数。
1. 你还能想到什么方法来读取长序列数据？
1. 考虑我们用来读取长序列的随机偏移量。
    1. 为什么随机偏移是个好主意？
    1. 它真的会导致文档序列上的完全均匀分布吗？
    1. 你要怎么做才能使事情变得更加统一？
1. 如果我们想让一个序列示例成为一个完整的句子，那么这在小批量采样中会引入什么样的问题呢？我们怎样才能解决这个问题？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1049)
:end_tab:
