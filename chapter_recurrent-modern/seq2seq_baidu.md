#  序列间学习
:label:`sec_seq2seq`

正如我们在:numref:`sec_machine_translation`中看到的，在机器翻译中，输入和输出都是可变长度的序列。为了解决这类问题，我们在:numref:`sec_encoder-decoder`中设计了一个通用的编解码器结构。在本节中，我们将使用两个RNN来设计此架构的编码器和解码器，并将其应用于机器翻译:cite:`Sutskever.Vinyals.Le.2014,Cho.Van-Merrienboer.Gulcehre.ea.2014`的*序列对序列*学习。

RNN编码器遵循编解码结构的设计原则，以可变长度序列作为输入，将其转换为固定形状的隐藏状态。换言之，输入（源）序列的信息在RNN编码器的隐藏状态下被*编码*。为了逐个令牌地生成输出序列令牌，单独的RNN解码器可以基于已经看到的令牌（例如在语言建模中）或生成的令牌以及输入序列的编码信息来预测下一令牌。:numref:`fig_seq2seq`演示了如何在机器翻译中使用两个RNN进行序列到序列的学习。

![Sequence to sequence learning with an RNN encoder and an RNN decoder.](../img/seq2seq.svg)
:label:`fig_seq2seq`

在:numref:`fig_seq2seq`中，特殊的“&lt；eos&gt；”标记表示序列的结束。一旦生成此令牌，模型就可以停止进行预测。在RNN解码器的初始时间步，有两个特殊的设计决策。首先，序列“&lt；bos&gt；”标记的特殊开头是一个输入。其次，使用RNN编码器的最终隐藏状态来启动解码器的隐藏状态。在诸如:cite:`Sutskever.Vinyals.Le.2014`的设计中，这正是编码的输入序列信息被馈送到解码器以生成输出（目标）序列的方式。如编码器73615的某些这样的输入在每个时刻也被隐藏在解码器73734的某些这样的部分中。“我们可以按一个原始语言”Ils“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”SLS“，”。

下面，我们将对:numref:`fig_seq2seq`的设计进行更详细的说明。我们将在:numref:`sec_machine_translation`中介绍的英法数据集上训练这个机器翻译模型。

```{.python .input}
import collections
from d2l import mxnet as d2l
import math
from mxnet import np, npx, init, gluon, autograd
from mxnet.gluon import nn, rnn
npx.set_np()
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
from torch import nn
```

## 编码器

从技术上讲，编码器将可变长度的输入序列转换成固定形状*上下文变量*$\mathbf{c}$，并且在该上下文变量中编码输入序列信息。如:numref:`fig_seq2seq`所示，我们可以使用RNN来设计编码器。

让我们考虑一个序列示例（批大小：1）。假设输入序列是$x_1, \ldots, x_T$，使得$x_t$是输入文本序列中的$t^{\mathrm{th}}$令牌。在时间步骤$t$，RNN将用于$x_t$的输入特征向量$\mathbf{x}_t$和来自上一时间步骤的隐藏状态$\mathbf{h} _{t-1}$转换为当前隐藏状态$\mathbf{h}_t$。我们可以用一个函数$f$来表示RNN的递归层的变换：

$$\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1}). $$

通常，编码器通过定制函数$q$将所有时间步的隐藏状态转换为上下文变量：

$$\mathbf{c} =  q(\mathbf{h}_1, \ldots, \mathbf{h}_T).$$

例如，当选择$q(\mathbf{h}_1, \ldots, \mathbf{h}_T) = \mathbf{h}_T$（例如在:numref:`fig_seq2seq`中）时，上下文变量只是在最后时间步的输入序列的隐藏状态$\mathbf{h}_T$。

到目前为止，我们已经使用了一个单向RNN来设计编码器，其中隐藏状态只依赖于隐藏状态的时间步长处和之前的输入子序列。我们也可以使用双向RNN构造编码器。在这种情况下，隐藏状态取决于时间步长前后的子序列（包括当前时间步长处的输入），该子序列对整个序列的信息进行编码。

现在让我们实现RNN编码器。注意，我们使用*嵌入层*来获得输入序列中每个令牌的特征向量。嵌入层的权重是一个矩阵，其行数等于输入词汇表的大小（`vocab_size`），列数等于特征向量的维数（`embed_size`）。对于任何输入令牌索引$i$，嵌入层获取权重矩阵的$i^{\mathrm{th}}$行（从0开始）以返回其特征向量。另外，本文选择了一个多层GRU来实现编码器。

```{.python .input}
#@save
class Seq2SeqEncoder(d2l.Encoder):
    """The RNN encoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X = self.embedding(X)
        # In RNN models, the first axis corresponds to time steps
        X = X.swapaxes(0, 1)
        state = self.rnn.begin_state(batch_size=X.shape[1], ctx=X.ctx)
        output, state = self.rnn(X, state)
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state[0]` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
```

```{.python .input}
#@tab pytorch
#@save
class Seq2SeqEncoder(d2l.Encoder):
    """The RNN encoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X = self.embedding(X)
        # In RNN models, the first axis corresponds to time steps
        X = X.permute(1, 0, 2)
        # When state is not mentioned, it defaults to zeros
        output, state = self.rnn(X)
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
```

回归层的返回变量已在:numref:`sec_rnn-concise`中解释。让我们仍然使用一个具体的例子来说明上述编码器实现。下面我们将实例化一个隐藏单元数为16的两层GRU编码器。给定一小批序列输入`X`（批大小：4，时间步数：7），所有时间步最后一层的隐藏状态（`output`由编码器的循环层返回）是形状张量（时间步数，批大小，隐藏单元数）。

```{.python .input}
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.initialize()
X = d2l.zeros((4, 7))
output, state = encoder(X)
output.shape
```

```{.python .input}
#@tab pytorch
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.eval()
X = d2l.zeros((4, 7), dtype=torch.long)
output, state = encoder(X)
output.shape
```

由于这里使用GRU，所以在最后一个时间步的多层隐藏状态的形状是（隐藏层的数量、批量大小、隐藏单元的数量）。如果使用LSTM，`state`中还将包含存储单元信息。

```{.python .input}
len(state), state[0].shape
```

```{.python .input}
#@tab pytorch
state.shape
```

## 解码器
:label:`sec_seq2seq_decoder`

正如我们刚才提到的，编码器输出的上下文变量$\mathbf{c}$对整个输入序列$x_1, \ldots, x_T$进行编码。给定来自训练数据集的输出序列$y_1, y_2, \ldots, y_{T'}$，对于每个时间步$t'$（符号不同于输入序列或编码器的时间步$t$），解码器输出$y_{t'}$的概率取决于先前的输出子序列$y_1, \ldots, y_{t'-1}$和上下文变量$\mathbf{c}$，即$P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$。

为了在序列上模拟这种条件概率，我们可以使用另一个RNN作为解码器。在输出序列上的任何时间步骤$t^\prime$，RNN将来自上一时间步骤的输出$y_{t^\prime-1}$和上下文变量$\mathbf{c}$作为其输入，然后在当前时间步骤将它们和上一隐藏状态$\mathbf{s}_{t^\prime-1}$转换为隐藏状态$\mathbf{s}_{t^\prime}$。因此，我们可以使用函数$g$来表示解码器的隐藏层的变换：

$$\mathbf{s}_{t^\prime} = g(y_{t^\prime-1}, \mathbf{c}, \mathbf{s}_{t^\prime-1}).$$
:eqlabel:`eq_seq2seq_s_t`

在获得解码器的隐藏状态之后，我们可以使用输出层和softmax操作来计算时间步$t^\prime$处的输出的条件概率分布$P(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \mathbf{c})$。

在:numref:`fig_seq2seq`之后，当实现如下解码器时，我们直接使用编码器最后一个时间步的隐藏状态来初始化解码器的隐藏状态。这要求RNN编码器和RNN解码器具有相同数量的层和隐藏单元。为了进一步合并经编码的输入序列信息，上下文变量在所有时间步处与解码器输入串联。为了预测输出令牌的概率分布，在RNN解码器的最后一层使用完全连接层来变换隐藏状态。

```{.python .input}
class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # The output `X` shape: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).swapaxes(0, 1)
        # `context` shape: (`batch_size`, `num_hiddens`)
        context = state[0][-1]
        # Broadcast `context` so it has the same `num_steps` as `X`
        context = np.broadcast_to(context, (
            X.shape[0], context.shape[0], context.shape[1]))
        X_and_context = d2l.concat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).swapaxes(0, 1)
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state[0]` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
```

```{.python .input}
#@tab pytorch
class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # The output `X` shape: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).permute(1, 0, 2)
        # Broadcast `context` so it has the same `num_steps` as `X`
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = d2l.concat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
```

为了说明实现的解码器，下面我们用前面提到的编码器中相同的超参数来实例化它。如我们所见，解码器的输出形状变为（批量大小、时间步数、词汇表大小），其中张量的最后一个维度存储预测的令牌分布。

```{.python .input}
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.initialize()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
output.shape, len(state), state[0].shape
```

```{.python .input}
#@tab pytorch
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
output.shape, state.shape
```

总之，上述RNN编解码器模型中的各层如:numref:`fig_seq2seq_details`所示。

![Layers in an RNN encoder-decoder model.](../img/seq2seq-details.svg)
:label:`fig_seq2seq_details`

## 损失函数

在每个时间步，解码器预测输出令牌的概率分布。类似于语言建模，我们可以使用softmax来获得分布，并计算交叉熵损失进行优化。回想一下:numref:`sec_machine_translation`，特殊填充标记被附加到序列的末尾，因此不同长度的序列可以有效地以相同形状的小批量加载。但是，应该将填充令牌的预测排除在损失计算之外。

为此，我们可以使用下面的`sequence_mask`函数用零值屏蔽不相关的条目，以便以后任何不相关的预测与零的乘积等于零。例如，如果两个序列（不包括填充标记）的有效长度分别为1和2，则第一个和前两个条目之后的剩余条目将被清除为零。

```{.python .input}
X = np.array([[1, 2, 3], [4, 5, 6]])
npx.sequence_mask(X, np.array([1, 2]), True, axis=1)
```

```{.python .input}
#@tab pytorch
#@save
def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
sequence_mask(X, torch.tensor([1, 2]))
```

我们还可以屏蔽最后几个轴上的所有条目。如果愿意，甚至可以指定用非零值替换这些条目。

```{.python .input}
X = d2l.ones((2, 3, 4))
npx.sequence_mask(X, np.array([1, 2]), True, value=-1, axis=1)
```

```{.python .input}
#@tab pytorch
X = d2l.ones(2, 3, 4)
sequence_mask(X, torch.tensor([1, 2]), value=-1)
```

现在我们可以扩展softmax交叉熵损失来掩盖不相关的预测。最初，所有预测令牌的掩码都设置为1。一旦给定了有效长度，与任何填充令牌对应的掩码将被清除为零。最后，将所有令牌的丢失乘以掩码，以过滤掉丢失中填充令牌的不相关预测。

```{.python .input}
#@save
class MaskedSoftmaxCELoss(gluon.loss.SoftmaxCELoss):
    """The softmax cross-entropy loss with masks."""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        # `weights` shape: (`batch_size`, `num_steps`, 1)
        weights = np.expand_dims(np.ones_like(label), axis=-1)
        weights = npx.sequence_mask(weights, valid_len, True, axis=1)
        return super(MaskedSoftmaxCELoss, self).forward(pred, label, weights)
```

```{.python .input}
#@tab pytorch
#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks."""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
```

对于健全性检查，我们可以创建三个相同的序列。然后我们可以指定这些序列的有效长度分别为4、2和0。因此，第一个序列的损耗应为第二个序列的两倍，而第三个序列的损耗应为零。

```{.python .input}
loss = MaskedSoftmaxCELoss()
loss(d2l.ones((3, 4, 10)), d2l.ones((3, 4)), np.array([4, 2, 0]))
```

```{.python .input}
#@tab pytorch
loss = MaskedSoftmaxCELoss()
loss(d2l.ones(3, 4, 10), d2l.ones((3, 4), dtype=torch.long),
     torch.tensor([4, 2, 0]))
```

## 培训
:label:`sec_seq2seq_training`

在下面的训练循环中，我们连接特殊的序列起始令牌和原始输出序列（不包括最终令牌），作为解码器的输入，如:numref:`fig_seq2seq`所示。这被称为“教师强制”，因为原始输出序列（令牌标签）被送入解码器。或者，我们也可以将来自上一时间步的*预测*令牌作为当前输入馈送到解码器。

```{.python .input}
#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence."""
    net.initialize(init.Xavier(), force_reinit=True, ctx=device)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    loss = MaskedSoftmaxCELoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [
                x.as_in_ctx(device) for x in batch]
            bos = np.array(
                [tgt_vocab['<bos>']] * Y.shape[0], ctx=device).reshape(-1, 1)
            dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            with autograd.record():
                Y_hat, _ = net(X, dec_input, X_valid_len)
                l = loss(Y_hat, Y, Y_valid_len)
            l.backward()
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            trainer.step(num_tokens)
            metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence."""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
```

现在我们可以在机器翻译数据集上创建和训练一个RNN编解码器模型，用于序列到序列的学习。

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

## 预测

为了逐个令牌预测输出序列令牌，在每个解码器时间步处，将来自前一时间步的预测令牌作为输入馈入解码器。与训练类似，在初始时间步，序列的开始（“&lt；bos&gt；”）令牌被馈送到解码器。该预测过程如:numref:`fig_seq2seq_predict`所示。当序列结束（“&lt；eos&gt；”）标记被预测时，输出序列的预测就完成了。

![Predicting the output sequence token by token using an RNN encoder-decoder.](../img/seq2seq-predict.svg)
:label:`fig_seq2seq_predict`

我们将在:numref:`sec_beam-search`中介绍不同的序列生成策略。

```{.python .input}
#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Predict for sequence to sequence."""
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = np.array([len(src_tokens)], ctx=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = np.expand_dims(np.array(src_tokens, ctx=device), axis=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = np.expand_dims(np.array([tgt_vocab['<bos>']], ctx=device), axis=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(axis=2)
        pred = dec_X.squeeze(axis=0).astype('int32').item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
```

```{.python .input}
#@tab pytorch
#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Predict for sequence to sequence."""
    # Set `net` to eval mode for inference
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
```

## 预测序列的评估

我们可以通过与标签序列（基本真值）进行比较来评估预测序列。BLEU（双语评估替补）虽然最初被提出用于评估机器翻译结果:cite:`Papineni.Roukos.Ward.ea.2002`，但已被广泛用于测量不同应用的输出序列的质量。原则上，对于预测序列中的任何$n$克，BLEU评估该$n$克是否出现在标签序列中。

用$p_n$表示$n$克的精度，它是预测序列和标签序列中匹配的$n$克的数量与预测序列中$n$克的数量的比率。为了解释，给定标签序列$A$、$B$、$C$、$D$、$E$、$F$和预测序列$A$、$B$、$B$、$C$、$D$，我们有$p_1 = 4/5$、$p_2 = 3/4$、$p_3 = 1/3$和$p_4 = 0$。另外，让$\mathrm{len}_{\text{label}}$和$\mathrm{len}_{\text{pred}}$分别是标签序列和预测序列中的令牌数。那么，BLEU的定义是

$$ \exp\left(\min\left(0, 1 - \frac{\mathrm{len}_{\text{label}}}{\mathrm{len}_{\text{pred}}}\right)\right) \prod_{n=1}^k p_n^{1/2^n},$$
:eqlabel:`eq_bleu`

其中$k$是最长的$n$克进行匹配。

根据:eqref:`eq_bleu`中BLEU的定义，当预测序列与标签序列相同时，BLEU为1。此外，由于匹配更长的$n$克更加困难，BLEU为更长的$n$克精度分配了更大的权重。具体来说，当$p_n$固定时，$p_n^{1/2^n}$会随着$n$的增长而增加（原始纸张使用$p_n^{1/n}$）。此外，由于预测较短的序列倾向于获得较高的$p_n$值，因此:eqref:`eq_bleu`中乘法项之前的系数惩罚较短的预测序列。例如，当$k=2$给定标签序列$A$、$B$、$C$、$D$、$E$、$F$和预测序列$A$、$B$时，尽管$p_1 = p_2 = 1$，惩罚因子$\exp(1-6/2) \approx 0.14$降低BLEU。

我们实施BLEU措施如下。

```{.python .input}
#@tab all
def bleu(pred_seq, label_seq, k):  #@save
    """Compute the BLEU."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```

最后，利用训练好的RNN编解码器将几个英语句子翻译成法语，并计算结果的BLEU。

```{.python .input}
#@tab all
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
```

## 摘要

* 在编解码结构的设计之后，我们可以使用两个RNN来设计一个序列到序列学习的模型。
* 在实现编码器和解码器时，我们可以使用多层RNN。
* 我们可以使用遮罩来过滤不相关的计算，例如在计算损失时。
* 在编解码器训练中，教师强制方法将原始输出序列（与预测相反）输入解码器。
* BLEU是一种常用的评估输出序列的方法，它通过在预测序列和标签序列之间匹配$n$克来实现。

## 练习

1. 你能调整超参数来提高翻译效果吗？
1. 在损失计算中不使用遮罩重新运行实验。你观察到什么结果？为什么？
1. 如果编码器和解码器的层数或隐藏单元数不同，如何初始化解码器的隐藏状态？
1. 在训练中，用将前一时间步的预测输入解码器来代替教师强制。这对性能有何影响？
1. 用LSTM替换GRU重新运行实验。
1. 有没有其他方法来设计解码器的输出层？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/345)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1062)
:end_tab:
