# 机器翻译与数据集
:label:`sec_machine_translation`

我们使用RNN设计语言模型，这是自然语言处理的关键。另一个旗舰基准是机器翻译，它是将输入序列转换为输出序列的序列转换模型的核心问题域。序列转导模型在各种现代人工智能应用中扮演着重要的角色，将成为本章剩余部分和:numref:`chap_attention`的重点。为此，本节将介绍机器翻译问题及其稍后将使用的数据集。

*机器翻译是指
从一种语言到另一种语言的序列的自动翻译。事实上，这个领域可以追溯到20世纪40年代数字计算机发明后不久，特别是考虑到在第二次世界大战中使用计算机破解语言代码。几十年来，在使用神经网络的端到端学习兴起之前，统计方法在这一领域占主导地位。后者常被称为
*神经机器翻译*
使自己与
*统计机器翻译*
这涉及翻译模型和语言模型等组成部分的统计分析。

强调端到端的学习，这本书将集中在神经机器翻译方法。与:numref:`sec_language_model`语言模型问题不同的是，机器翻译数据集是由源语言和目标语言的文本序列对组成的。因此，我们需要一种不同的方法来预处理机器翻译数据集，而不是重用语言建模的预处理例程。下面，我们将演示如何将预处理的数据加载到小批量中进行训练。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import os
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import os
```

## 数据集的下载和预处理

首先，我们下载一个由[bilingual sentence pairs from the Tatoeba Project](http://www.manythings.org/anki/)组成的英法数据集。数据集中的每一行都是一对以制表符分隔的英文文本序列和翻译的法文文本序列。请注意，每个文本序列只能是一个句子或一段多个句子。在这个将英语翻译成法语的机器翻译问题中，英语是源语言，法语是目标语言。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():
    """Load the English-French dataset."""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])
```

在下载数据集之后，我们继续对原始文本数据进行几个预处理步骤。例如，我们将不间断空格替换为空格，将大写字母转换为小写字母，并在单词和标点符号之间插入空格。

```{.python .input}
#@tab all
#@save
def preprocess_nmt(text):
    """Preprocess the English-French dataset."""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])
```

## 符号化

与:numref:`sec_language_model`中的字符级标记化不同，对于机器翻译，我们更喜欢这里的单词级标记化（最先进的模型可能使用更先进的标记化技术）。下面的`tokenize_nmt`函数对第一个`num_examples`文本序列对进行标记，其中每个标记是单词或标点符号。此函数返回两个令牌列表：`source`和`target`。具体而言，`source[i]`是源语言（此处为英语）中$i^\mathrm{th}$文本序列中的令牌列表，`target[i]`是目标语言（此处为法语）中的令牌列表。

```{.python .input}
#@tab all
#@save
def tokenize_nmt(text, num_examples=None):
    """Tokenize the English-French dataset."""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
source[:6], target[:6]
```

让我们绘制每个文本序列的标记数的直方图。在这个简单的英法数据集中，大多数文本序列的标记数少于20个。

```{.python .input}
#@tab all
d2l.set_figsize()
_, _, patches = d2l.plt.hist(
    [[len(l) for l in source], [len(l) for l in target]],
    label=['source', 'target'])
for patch in patches[1].patches:
    patch.set_hatch('/')
d2l.plt.legend(loc='upper right');
```

## 词汇

由于机器翻译数据集由成对的语言组成，我们可以分别为源语言和目标语言建立两个词汇表。单词级标记化的词汇量将显著大于字符级标记化的词汇量。为了缓解这种情况，这里我们将出现次数少于2次的不常见标记视为相同的未知（“&lt；unk&gt；”）标记。除此之外，我们还指定了其他特殊标记，例如用于在小批量中填充相同长度的（“&lt；pad&gt；”）序列，以及用于标记序列的开始（“&lt；bos&gt；”）或结束（“&lt；eos&gt；”）。这种特殊标记通常用于自然语言处理任务。

```{.python .input}
#@tab all
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)
```

## 加载数据集
:label:`subsec_mt_data_loading`

回想一下，在语言建模中，每个序列示例，一个句子的一段或多个句子的跨度，都有一个固定的长度。这是由:numref:`sec_language_model`中的`num_steps`（时间步数或令牌数）参数指定的。在机器翻译中，每个例子都是一对源文本序列和目标文本序列，其中每个文本序列可能有不同的长度。

为了提高计算效率，我们仍然可以通过*截断*和*填充*一次处理一小批文本序列。假设相同minibatch中的每个序列都应该具有相同的长度`num_steps`。如果一个文本序列的标记数少于`num_steps`，我们将继续在其末尾附加特殊的“&lt；pad&gt；”标记，直到其长度达到`num_steps`。否则，我们将通过仅获取其前`num_steps`个标记并丢弃其余标记来截断文本序列。这样，每个文本序列将具有相同的长度，以便以相同形状的小批量加载。

下面的`truncate_pad`函数如前所述截断或填充文本序列。

```{.python .input}
#@tab all
#@save
def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
```

现在我们定义一个函数来将文本序列转换成小批量进行训练。我们将特殊的“&lt；eos&gt；”标记附加到每个序列的末尾，以指示序列的结尾。当模型通过一个接一个地生成序列令牌进行预测时，“&lt；eos&gt；”令牌的生成可以表明输出序列是完整的。此外，我们还记录了每个文本序列的长度，不包括填充标记。我们稍后将介绍的一些模型需要这些信息。

```{.python .input}
#@tab all
#@save
def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches."""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = d2l.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = d2l.reduce_sum(
        d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
    return array, valid_len
```

## 把所有的东西放在一起

最后，我们定义`load_data_nmt`函数来返回数据迭代器，以及源语言和目标语言的词汇表。

```{.python .input}
#@tab all
#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset."""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab
```

让我们读一下英法数据集中的第一个小批量。

```{.python .input}
#@tab all
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', d2l.astype(X, d2l.int32))
    print('valid lengths for X:', X_valid_len)
    print('Y:', d2l.astype(Y, d2l.int32))
    print('valid lengths for Y:', Y_valid_len)
    break
```

## 摘要

* 机器翻译是指将一种语言的序列自动翻译成另一种语言。
* 使用单词级标记化，词汇量将显著大于使用字符级标记化。为了缓解这种情况，我们可以将不经常出现的标记视为相同的未知标记。
* 我们可以截断和填充文本序列，这样所有的文本序列都将具有相同的长度，以便在小批量中加载。

## 练习

1. 在`load_data_nmt`函数中尝试`num_examples`参数的不同值。这对源语言和目标语言的词汇量有何影响？
1. 某些语言（如中文和日语）中的文本没有单词边界指示符（例如空格）。对于这种情况，单词级标记化仍然是一个好主意吗？为什么不呢？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1060)
:end_tab:
