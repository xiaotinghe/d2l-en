# 巴赫达瑙注意
:label:`sec_seq2seq_attention`

我们在:numref:`sec_seq2seq`研究了机器翻译问题，设计了一种基于两个RNN的用于序列到序列学习的编解码器结构。具体地，RNN编码器将可变长度的序列转换为固定形状的上下文变量，然后RNN解码器基于所生成的令牌和上下文变量逐个令牌地生成输出(目标)序列。然而，即使不是所有的输入(源)令牌都对解码某一令牌有用，但是在每个解码步骤中仍然使用对整个输入序列进行编码的*相同*上下文变量。

在针对给定文本序列的手写生成的单独但相关的挑战中，格雷夫斯设计了一个可区分的注意力模型来将文本字符与更长的笔迹对齐，其中对齐仅在一个方向上移动:cite:`Graves.2013`。在学习对齐的想法的启发下，巴达努等人。提出了一种无严重单向排列限制的可微分注意模型:cite:`Bahdanau.Cho.Bengio.2014`。当预测令牌时，如果并非所有输入令牌都相关，则模型仅对齐(或参与)与当前预测相关的输入序列的部分。这是通过将上下文变量视为注意力集中的输出来实现的。

## 型号

当描述下面的rnn编解码器的巴达瑙注意事项时，我们将遵循:numref:`sec_seq2seq`中的相同符号。新的基于注意力的模型与:numref:`sec_seq2seq`中的相同，除了:eqref:`eq_seq2seq_s_t`中的上下文变量$\mathbf{c}$在任何解码时间步骤$t'$被$\mathbf{c}_{t'}$替换。假设输入序列中有$T$个令牌，则解码时间步骤$t'$处的上下文变量是注意力集中的输出：

$$\mathbf{c}_{t'} = \sum_{t=1}^T \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_t) \mathbf{h}_t,$$

其中解码器隐藏状态$\mathbf{s}_{t' - 1}$在时间步骤$t' - 1$是查询，而编码器隐藏状态$\mathbf{h}_t$既是关键字又是值，并且关注度权重$\alpha$如在:eqref:`eq_attn-scoring-alpha`中那样使用由:eqref:`eq_additive-attn`定义的附加注意力得分函数来计算。

与:numref:`fig_seq2seq_details`中的Vanilla RNN编解码器体系结构略有不同，同样的体系结构在:numref:`fig_s2s_attention_details`中得到了巴赫达诺的关注。

![Layers in an RNN encoder-decoder model with Bahdanau attention.](../img/seq2seq-attention-details.svg)
:label:`fig_s2s_attention_details`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn, nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## 用注意力定义解码器

要实现具有Bahdanau注意力的RNN编解码器，我们只需要重新定义解码器。为了更方便地可视化学习到的注意力权重，下面的`AttentionDecoder`个类定义了具有注意力机制的解码器的基本接口。

```{.python .input}
#@tab all
#@save
class AttentionDecoder(d2l.Decoder):
    """The base attention-based decoder interface."""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError
```

现在，让我们在接下来的`Seq2SeqAttentionDecoder`级中实现具有巴达瑙注意力的rnn解码器。解码器的状态用i)在所有时间步长处的编码器最后层隐藏状态(作为注意力的关键字和值)来初始化；ii)在最后时间步长处的编码器全层隐藏状态(以初始化解码器的隐藏状态)；以及iii)编码器有效长度(以在注意力集中中排除填充标记)。在每个解码时间步，使用前一个时间步的解码器最后一层隐藏状态作为关注的查询。结果，注意输出和输入嵌入都被级联为RNN解码器的输入。

```{.python .input}
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # Shape of `outputs`: (`num_steps`, `batch_size`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        outputs, hidden_state = enc_outputs
        return (outputs.swapaxes(0, 1), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output `X`: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).swapaxes(0, 1)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of `query`: (`batch_size`, 1, `num_hiddens`)
            query = np.expand_dims(hidden_state[0][-1], axis=1)
            # Shape of `context`: (`batch_size`, 1, `num_hiddens`)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = np.concatenate((context, np.expand_dims(x, axis=1)), axis=-1)
            # Reshape `x` as (1, `batch_size`, `embed_size` + `num_hiddens`)
            out, hidden_state = self.rnn(x.swapaxes(0, 1), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully-connected layer transformation, shape of `outputs`:
        # (`num_steps`, `batch_size`, `vocab_size`)
        outputs = self.dense(np.concatenate(outputs, axis=0))
        return outputs.swapaxes(0, 1), [enc_outputs, hidden_state,
                                        enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
#@tab pytorch
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # Shape of `outputs`: (`num_steps`, `batch_size`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output `X`: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of `query`: (`batch_size`, 1, `num_hiddens`)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # Shape of `context`: (`batch_size`, 1, `num_hiddens`)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # Reshape `x` as (1, `batch_size`, `embed_size` + `num_hiddens`)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully-connected layer transformation, shape of `outputs`:
        # (`num_steps`, `batch_size`, `vocab_size`)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]
    
    @property
    def attention_weights(self):
        return self._attention_weights
```

在接下来的测试中，我们使用7个时间步长的4个序列输入的小批量测试了所实现的具有巴达诺注意力的解码器。

```{.python .input}
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.initialize()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
decoder.initialize()
X = d2l.zeros((4, 7))  # (`batch_size`, `num_steps`)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

```{.python .input}
#@tab pytorch
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.eval()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
decoder.eval()
X = d2l.zeros((4, 7), dtype=torch.long)  # (`batch_size`, `num_steps`)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

## 培训

类似于:numref:`sec_seq2seq_training`，我们在这里指定超参数，实例化一个编码器和一个解码器，并训练这个模型用于机器翻译。由于新增加的注意机制，这次训练比没有注意机制的:numref:`sec_seq2seq_training`训练要慢得多。

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = d2l.Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

在对模型进行训练后，我们使用该模型将几个英语句子翻译成法语，并计算出它们的BLEU分数。

```{.python .input}
#@tab all
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
```

```{.python .input}
#@tab all
attention_weights = d2l.reshape(
    d2l.concat([step[0][0][0] for step in dec_attention_weight_seq], 0),
    (1, 1, -1, num_steps))
```

通过在翻译最后一个英语句子时可视化注意力权重，我们可以看到每个查询在键值对上分配的权重是不一致的。结果表明，在每个解码步骤中，输入序列的不同部分在注意池中被选择性地聚集。

```{.python .input}
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1],
    xlabel='Key posistions', ylabel='Query posistions')
```

```{.python .input}
#@tab pytorch
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Key posistions', ylabel='Query posistions')
```

## 摘要

* 当预测一个令牌时，如果不是所有的输入令牌都是相关的，则具有巴达诺注意力的RNN编解码器选择性地聚合输入序列的不同部分。这是通过将上下文变量视为附加注意力集中的输出来实现的。
* 在RNN编解码器中，Babdanau注意将解码器在前一个时间步长的隐藏状态作为查询，将编码器在所有时间步长的隐藏状态作为关键字和值。

## 练习

1. 在实验中，将GRU替换为LSTM。
1. 修改实验，将加法注意力评分函数替换为缩放后的点积。它对培训效率有何影响？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/347)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1065)
:end_tab:
