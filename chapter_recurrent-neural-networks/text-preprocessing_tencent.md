# 文本预处理
:label:`sec_text_preprocessing`

我们回顾和评估了序列数据的统计工具和预测挑战。此类数据可以采用多种形式。具体地说，正如我们将在本书的许多章节中重点介绍的那样，文本是序列数据最流行的示例之一。例如，一篇文章可以简单地视为一个单词序列，甚至是一个字符序列。为了便于我们将来对序列数据的实验，我们将专门用这一节来解释文本的常见预处理步骤。通常，这些步骤包括：

1. 将文本作为字符串加载到内存中。
1. 将字符串拆分成标记(例如，单词和字符)。
1. 构建词汇表以将拆分的标记映射到数字索引。
1. 将文本转换为数字索引序列，以便模型可以轻松地操作它们。

```{.python .input}
import collections
from d2l import mxnet as d2l
import re
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import re
```

```{.python .input}
#@tab tensorflow
import collections
from d2l import tensorflow as d2l
import re
```

## 正在读取数据集

为了开始，我们从H.G.Wells‘[*The Time Machine*](http://www.gutenberg.org/ebooks/35)加载文本。这是一个相当小的语料库，只有30000多个单词，但对于我们想要说明的目的来说，这是很好的。更现实的文档集合包含数十亿个单词。下面的函数将数据集读取到文本行列表中，其中每行都是一个字符串。为简单起见，这里我们忽略标点符号和大写。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """Load the time machine dataset into a list of text lines."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# text lines: {len(lines)}')
print(lines[0])
print(lines[10])
```

## 标记化

以下`tokenize`函数将列表(`lines`)作为输入，其中每个列表是文本序列(例如，文本行)。每个文本序列被拆分成一个标记列表。*TOKEN*是文本的基本单位。最后，返回一个令牌列表列表，其中每个令牌都是一个字符串。

```{.python .input}
#@tab all
def tokenize(lines, token='word'):  #@save
    """Split text lines into word or character tokens."""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
```

## 词汇

令牌的字符串类型不便于接受数字输入的模型使用。现在，让我们构建一个字典，通常也称为*词汇*，将字符串标记映射到从0开始的数字索引。为此，我们首先统计训练集中所有文档中的唯一令牌，即*语料库*，然后根据每个唯一令牌的频率为每个唯一令牌分配一个数字索引。很少出现的令牌通常会被移除以降低复杂性。语料库中不存在或已删除的任何令牌都会映射到特殊的未知令牌“&lt；unk&gt；”。我们可选地添加保留标记的列表，例如用于填充的“&lt；pad&>”，表示序列开始的“&lt；bos&>”，以及表示序列结束的“&lt；Eos&>”。

```{.python .input}
#@tab all
class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = [] 
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The index for the unknown token is 0
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(tokens):  #@save
    """Count token frequencies."""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
```

我们使用时光机数据集作为语料库来构建词汇表。然后，我们打印前几个频繁使用的令牌及其索引。

```{.python .input}
#@tab all
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
```

现在，我们可以将每个文本行转换为数字索引列表。

```{.python .input}
#@tab all
for i in [0, 10]:
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])
```

## 把所有的东西放在一起

使用上述函数，我们将所有内容打包到`load_corpus_time_machine`函数中，该函数返回`corpus`(标记索引列表)和`vocab`(时间机器语料库的词汇表)。我们在这里所做的修改是：i)我们将文本标记化为字符，而不是单词，以简化后面部分中的训练；ii)`corpus`是单个列表，而不是标记列表的列表，因为时间机器数据集中的每个文本行不一定是句子或段落。

```{.python .input}
#@tab all
def load_corpus_time_machine(max_tokens=-1):  #@save
    """Return token indices and the vocabulary of the time machine dataset."""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Since each text line in the time machine dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

## 摘要

* 文本是序列数据的一种重要形式。
* 为了预处理文本，我们通常将文本拆分成记号，构建词汇表将记号字符串映射为数字索引，并将文本数据转换为记号索引以供模型操作。

## 练习

1. 标记化是关键的预处理步骤。它因语言不同而不同。尝试找到另外三种常用的文本标记化方法。
1. 在本节的实验中，将文本标记化为单词，并改变`Vocab`实例的`Vocab`个参数。这对词汇量有什么影响？

[Discussions](https://discuss.d2l.ai/t/115)
