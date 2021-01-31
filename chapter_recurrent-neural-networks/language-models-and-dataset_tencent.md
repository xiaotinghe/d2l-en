# 语言模型和数据集
:label:`sec_language_model`

在:numref:`sec_text_preprocessing`中，我们将了解如何将文本数据映射到记号中，其中这些记号可以被视为一系列离散的观察，例如单词或字符。假设长度为$T$的文本序列中的记号依次为$x_1, x_2, \ldots, x_T$。然后，在文本序列中，$x_t$($1 \leq t \leq T$)可以被认为是时间步骤$t$处的观察或标签。给定这样的文本序列，*语言模型*的目标是估计序列的联合概率

$$P(x_1, x_2, \ldots, x_T).$$

语言模型非常有用。例如，理想的语言模型将能够仅通过简单地一次绘制一个记号来仅凭其自身生成自然文本$x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1)$。与猴子使用打字机非常不同的是，从这样的模型中出现的所有文本都将作为自然语言来传递，例如英语文本。此外，只需将文本限制在前面的对话片断上，就足以生成有意义的对话框。显然，我们离设计这样的系统还很远，因为它需要“理解”文本，而不仅仅是生成语法上合理的内容。

尽管如此，语言模型即使在有限的形式下也是非常有用的。例如，短语“识别语音”和“破坏一个美丽的海滩”听起来非常相似。这可能会导致语音识别中的歧义，这很容易通过语言模型来解决，该语言模型拒绝第二个翻译，认为它很奇怪。同样，在文档摘要算法中，值得知道的是“狗咬人”比“人咬狗”频繁得多，或者“我想吃奶奶”是一个相当令人不安的语句，而“我想吃东西，奶奶”要温和得多。

## 学习语言模型

显而易见的问题是，我们应该如何建模一个文档，甚至是一系列令牌。假设我们在单词级别对文本数据进行标记化。我们可以求助于我们在:numref:`sec_sequence`中应用于序列模型的分析。让我们从应用基本概率规则开始：

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1}).$$

例如，文本序列包含四个单词的概率将被给出为：

$$P(\text{deep}, \text{learning}, \text{is}, \text{fun}) =  P(\text{deep}) P(\text{learning}  \mid  \text{deep}) P(\text{is}  \mid  \text{deep}, \text{learning}) P(\text{fun}  \mid  \text{deep}, \text{learning}, \text{is}).$$

为了计算语言模型，我们需要计算单词的概率和给定前面几个单词的单词的条件概率。这样的概率本质上是语言模型参数。

这里，我们假设训练数据集是一个大型文本语料库，比如所有维基百科条目，[Project Gutenberg](https://en.wikipedia.org/wiki/Project_Gutenberg)，以及发布在网络上的所有文本。可以根据训练数据集中给定词的相对词频来计算词的概率。例如，可以将估计值$\hat{P}(\text{deep})$计算为任何以单词“Deep”开头的句子的概率。一种稍微不太准确的方法是统计单词“Deep”的所有出现次数，然后将其除以语料库中的单词总数。这很有效，特别是对于频繁出现的单词。接下来，我们可以尝试估计

$$\hat{P}(\text{learning} \mid \text{deep}) = \frac{n(\text{deep, learning})}{n(\text{deep})},$$

其中$n(x)$和$n(x, x')$分别是单个单词和连续单词对的出现次数。不幸的是，由于“深度学习”的出现频率要低得多，所以估计词对的概率要困难得多。特别是，对于一些不寻常的单词组合，可能很难找到足够的出现次数来获得准确的估计。对于三个字的组合和以后的情况，情况变得更糟了。将会有许多我们可能在我们的数据集中看不到的看似合理的三字组合。除非我们提供一些解决方案来将这些单词组合指定为非零计数，否则我们将无法在语言模型中使用它们。如果数据集很小，或者如果单词非常罕见，我们可能甚至找不到其中的一个。

一种常见的策略是执行某种形式的*拉普拉斯平滑*。解决方案是在所有计数中添加一个小常量。用$n$表示训练集中的单词总数，用$m$表示唯一单词的数量。此解决方案有助于处理单例，例如VIA

$$\begin{aligned}
	\hat{P}(x) & = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1}, \\
	\hat{P}(x' \mid x) & = \frac{n(x, x') + \epsilon_2 \hat{P}(x')}{n(x) + \epsilon_2}, \\
	\hat{P}(x'' \mid x,x') & = \frac{n(x, x',x'') + \epsilon_3 \hat{P}(x'')}{n(x, x') + \epsilon_3}.
\end{aligned}$$

其中，$\epsilon_1,\epsilon_2$和$\epsilon_3$是超参数。以$\epsilon_1$为例：当为$\epsilon_1 = 0$时，不应用平滑；当$\epsilon_1$接近正无穷大时，$\hat{P}(x)$接近均匀概率$1/m$。以上是其他技术可以实现:cite:`Wood.Gasthaus.Archambeau.ea.2011`的一个相当原始的变体。

不幸的是，像这样的模型很快就会变得笨拙，原因如下。首先，我们需要存储所有计数。第二，这完全忽略了单词的意思。例如，“猫”和“猫”应该出现在相关的上下文中。很难将这些模型调整到额外的上下文中，而基于深度学习的语言模型很适合考虑这一点。最后，长单词序列几乎肯定是新奇的，因此简单地统计以前看到的单词序列的频率的模型在那里肯定表现不佳。

## 马尔可夫模型与$n$克

在我们讨论涉及深度学习的解决方案之前，我们需要更多的术语和概念。回想一下我们在:numref:`sec_sequence`中对马尔可夫模型的讨论。让我们将其应用于语言建模。序列上的分布满足一阶IF $P(x_{t+1} \mid x_t, \ldots, x_1) = P(x_{t+1} \mid x_t)$的马尔可夫性质。阶数越高，对应的依赖关系就越长。这导致了许多我们可以应用于对序列建模的近似：

$$
\begin{aligned}
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3).
\end{aligned}
$$

涉及一个、两个和三个变量的概率公式通常分别称为“一元图”、“二元图”和“三元图”模型。在下面，我们将学习如何设计更好的模型。

## 自然语言统计

让我们看看这是如何对真实数据起作用的。我们根据:numref:`sec_text_preprocessing`中介绍的时光机数据集构建词汇表，并打印最常用的10个单词。

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

正如我们所看到的，最流行的词实际上看起来很无聊。它们通常被称为“停用词”，因此被过滤掉了。尽管如此，它们仍然有意义，我们仍然会使用它们。此外，很明显，词频衰减得相当快。$10^{\mathrm{th}}$个最常用的单词还不到最流行单词的$1/5$。为了得到一个更好的概念，我们画出了词频的数字。

```{.python .input}
#@tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

我们在这里谈到了一件非常基本的事情：单词频率以一种明确的方式迅速衰减。在将前几个单词作为例外处理之后，其余所有单词大致沿着对数-对数曲线上的一条直线前进。这意味着单词满足*齐普夫定律*，该定律指出，$i^\mathrm{th}$个最频繁的单词的频率为$n_i$：

$$n_i \propto \frac{1}{i^\alpha},$$
:eqlabel:`eq_zipf_law`

这相当于

$$\log n_i = -\alpha \log i + c,$$

其中$\alpha$是表征分布的指数，$c$是常数。如果我们想要通过计数统计和平滑来建模单词，这应该已经让我们停下来了。毕竟，我们会大大高估尾巴的频率，也就是所谓的不常用单词。但是其他的单词组合呢，比如二元语法、三元语法等等呢？让我们看看双字频率是否与单字频率的行为方式相同。

```{.python .input}
#@tab all
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

这里有一件事值得注意。在十个最频繁的词对中，有九个是由两个停用词组成的，只有一个与实际的书相关-“时间”。此外，让我们看看三元频率是否以相同的方式运行。

```{.python .input}
#@tab all
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

最后，让我们直观地看一下这三种模型中的记号频率：单字、双字和三字。

```{.python .input}
#@tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

这个数字相当令人兴奋，原因有很多。首先，除了单字词，单词序列似乎也遵循齐夫定律，尽管根据序列长度的不同，:eqref:`eq_zipf_law`的指数较小，为$\alpha$。第二，不同的$n$克的数量不是那么多。这给了我们希望，语言中有相当多的结构。第三，很多$n$克很少出现，这使得拉普拉斯平滑非常不适合语言建模。相反，我们将使用基于深度学习的模型。

## 读取长序列数据

由于序列数据本质上是连续的，我们需要解决处理它的问题。我们在:numref:`sec_sequence`以一种相当特别的方式做到了这一点。当序列变得太长而不能被模型一次全部处理时，我们可能希望拆分这样的序列以供阅读。现在让我们描述一下总体策略。在介绍该模型之前，让我们假设我们将使用神经网络来训练语言模型，其中该网络一次处理具有预定义长度的一小批序列，例如$n$个时间步。现在的问题是如何随机读取小批量的特征和标签。

首先，由于文本序列可以是任意长的，例如整个“时光机”书，我们可以将这样长的序列划分为具有相同时间步数的子序列。当训练我们的神经网络时，这样的子序列的小批量将被输入到模型中。假设网络一次处理$n$个时间步长的子序列。:numref:`fig_timemachine_5gram`示出了从原始文本序列获得子序列的所有不同方式，其中$n=5$和每个时间步的标记对应于一个字符。请注意，我们有相当大的自由度，因为我们可以选择指示初始位置的任意偏移量。

![Different offsets lead to different subsequences when splitting up text.](../img/timemachine-5gram.svg)
:label:`fig_timemachine_5gram`

因此，我们应该从:numref:`fig_timemachine_5gram`个中选择一个呢？事实上，它们都是一样好的。然而，如果我们只选择一个偏移量，那么用于训练网络的所有可能的子序列的覆盖范围都是有限的。因此，我们可以从随机偏移量开始划分序列，以获得*覆盖*和*随机性*。在以下内容中，我们将介绍如何为这两个应用程序实现这一点
*随机抽样*和*顺序分区*策略。

### 随机抽样

在随机采样中，每个示例都是在原始长序列上任意捕获的子序列。迭代期间来自两个相邻随机小批次的子序列不一定在原始序列上相邻。对于语言建模，目标是根据我们到目前为止看到的令牌来预测下一个令牌，因此标签是原始序列，移位了一个令牌。

下面的代码每次从数据随机生成一个小批量。这里，自变量`batch_size`指定每个小批次中的子序列示例的数目，并且`num_steps`是每个子序列中的预定时间步数。

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

让我们手动生成从0到34的序列。我们假设批次大小和时间步数分别为2和5。这意味着我们可以生成$\lfloor (35 - 1) / 5 \rfloor= 6$个特征-标签子序列对。如果小批量是2，我们只有3个小批量。

```{.python .input}
#@tab all
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

### 顺序分区

除了对原始序列进行随机抽样外，我们还可以确保迭代过程中来自两个相邻小批次的子序列在原始序列上是相邻的。这种策略在小批量迭代时保持了拆分子序列的顺序，因此被称为顺序分区。

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

使用相同的设置，让我们为通过顺序分区读取的每个小批次的子序列打印特征`X`和标签`Y`。请注意，迭代期间来自两个相邻小批次的子序列实际上在原始序列上是相邻的。

```{.python .input}
#@tab all
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

现在，我们将上述两个采样函数包装到一个类中，以便稍后可以将其用作数据迭代器。

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

最后，我们定义了一个函数`load_data_time_machine`，它同时返回数据迭代器和词汇表，因此我们可以与其他带有`load_data`前缀的函数(如:numref:`sec_fashion_mnist`中定义的`d2l.load_data_fashion_mnist`)类似地使用它。

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
* $n$-GRAM通过截断相关性，为处理长序列提供了一种方便的模型。
* 长序列有一个问题，那就是它们很少出现或从不出现。
* 齐普夫定律不仅规定了单字形的单词分布，而且还规定了其他$n$克的单词分布。
* 结构很多，但频率不够高，无法通过拉普拉斯平滑有效地处理不常用的词组合。
* 读取长序列的主要选择是随机采样和顺序分区。后者可以保证迭代过程中来自两个相邻小批次的子序列在原始序列上是相邻的。

## 练习

1. 假设训练数据集中有$100,000$个单词。一个四元词需要存储多少词频和多词邻频？
1. 你将如何模拟对话？
1. 估计单星、双星和三星的齐普夫定律的指数。
1. 您还能想到哪些读取长序列数据的其他方法？
1. 考虑一下我们用于读取长序列的随机偏移量。
    1. 为什么随机偏移量是个好主意？
    1. 它真的会在文档上的序列上实现完美均匀的分布吗？
    1. 你要怎么做才能让事情变得更加统一呢？
1. 如果我们希望一个序列示例是一个完整的句子，那么这在小批量抽样中会带来什么样的问题呢？我们怎样才能解决这个问题呢？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1049)
:end_tab:
