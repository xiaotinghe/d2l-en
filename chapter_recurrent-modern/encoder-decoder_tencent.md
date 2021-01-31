# 编解码器体系结构
:label:`sec_encoder-decoder`

正如我们在:numref:`sec_machine_translation`中讨论的那样，机器翻译是序列转导模型的一个主要问题领域，其输入和输出都是可变长度的序列。要处理这种类型的输入和输出，我们可以设计一个包含两个主要组件的体系结构。第一个组件是*编码器*：它接受可变长度的序列作为输入，并将其转换为具有固定形状的状态。第二个组件是*解码器*：它将固定形状的编码状态映射到可变长度序列。这称为*编码器-解码器*架构，如:numref:`fig_encoder_decoder`中所述。

![The encoder-decoder architecture.](../img/encoder-decoder.svg)
:label:`fig_encoder_decoder`

让我们以从英语到法语的机器翻译为例。给定英文输入序列：“They”、“are”、“Watch”、“.”，该编码器-解码器体系结构首先将可变长度的输入编码成状态，然后对该状态进行解码以逐个令牌生成翻译后的序列令牌作为输出：“ils”、“regerdent”、“.”。由于编码器-解码器体系结构构成了后续部分中不同序列转导模型的基础，因此本节将把该体系结构转换为稍后将实现的接口。

## 编码器

在编码器接口中，我们只指定编码器接受可变长度序列作为输入`X`。实现将由继承此基础`Encoder`类的任何模型提供。

```{.python .input}
from mxnet.gluon import nn

#@save
class Encoder(nn.Block):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
#@tab pytorch
from torch import nn

#@save
class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

## 解码器

在下面的解码器接口中，我们添加了额外的`init_state`函数，将编码器输出(`enc_outputs`)转换为编码状态。请注意，此步骤可能需要额外的输入，如:numref:`subsec_mt_data_loading`中解释的输入的有效长度。为了逐个令牌地生成可变长度序列令牌，每次解码器可以将输入(例如，在前一时间步长生成的令牌)和编码状态映射到当前时间步长的输出令牌中。

```{.python .input}
#@save
class Decoder(nn.Block):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
#@tab pytorch
#@save
class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

## 将编码器和解码器放在一起

最后，编码器-解码器体系结构包含编码器和解码器，可选地带有额外的参数。在前向传播中，编码器的输出被用来产生编码状态，该状态将被解码器进一步用作其输入之一。

```{.python .input}
#@save
class EncoderDecoder(nn.Block):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

```{.python .input}
#@tab pytorch
#@save
class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

编码器-解码器体系结构中的术语“状态”可能启发您使用带状态的神经网络来实现此体系结构。在下一节中，我们将看到如何应用RNNs来设计基于此编码器-解码器体系结构的序列转导模型。

## 摘要

* 该编解码器结构可以处理可变长度序列的输入和输出，因此适合于机器翻译等序列转换问题。
* 编码器将可变长度的序列作为输入，并将其转换为具有固定形状的状态。
* 解码器将固定形状的编码状态映射到可变长度序列。

## 练习

1. 假设我们使用神经网络来实现编解码器体系结构。编码器和解码器必须是同一类型的神经网络吗？
1. 除了机器翻译，您还能想到可以应用编解码器架构的其他应用吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/341)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1061)
:end_tab:
