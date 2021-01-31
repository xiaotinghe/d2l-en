# 时间反向传播
:label:`sec_bptt`

到目前为止，我们已经多次提到像这样的事情
*爆炸梯度*，
*消失的渐变*，
以及需要
*分离RNN的渐变*。
例如，在:numref:`sec_rnn_scratch`中，我们对序列调用了`detach`函数。为了能够快速构建模型并了解它是如何工作的，所有这些都没有真正得到充分的解释。在本节中，我们将更深入地研究序列模型的反向传播的细节，以及数学为什么(以及如何工作)。

当我们第一次实现RNN(:numref:`sec_rnn_scratch`)时，我们遇到了梯度爆炸的一些影响。特别是，如果您解决了这些练习，您将会看到渐变裁剪对于确保适当的收敛至关重要。为了更好地理解这个问题，本节将回顾如何为序列模型计算梯度。请注意，它的工作方式在概念上没有什么新意。毕竟，我们仍然只是应用链式规则来计算梯度。尽管如此，在再次回顾反向传播(:numref:`sec_backprop`)时还是值得的。

我们已经在:numref:`sec_backprop`的MLP中描述了正向和反向传播以及计算图。RNN中的前向传播相对简单。
*穿越时间的反向传播*实际上是一种特定的
反向传播在rnns :cite:`Werbos.1990`中的应用。这就要求我们一次扩展一个RNN的计算图，得到模型变量和参数之间的依赖关系。然后，在链式规则的基础上，应用反向传播算法计算和存储梯度。由于序列可能相当长，因此依赖关系可能会相当长。例如，对于1000个字符的序列，第一个令牌可能对最终位置处的令牌具有显著影响。这在计算上是不可行的(它需要太长的时间和太多的内存)，它需要超过1000个矩阵乘积才能达到非常难以捉摸的梯度。这是一个充满计算和统计不确定性的过程。在下面，我们将阐明发生了什么以及如何在实践中解决这个问题。

## 随机神经网络中的梯度分析
:label:`subsec_bptt_analysis`

我们从RNN如何工作的简化模型开始。此模型忽略有关隐藏状态的细节及其更新方式的详细信息。这里的数学表示法不像以前那样显式区分标量、向量和矩阵。这些细节对分析来说无关紧要，只会使本小节中的符号变得杂乱无章。

在该简化模型中，我们将$h_t$表示为隐藏状态，将$x_t$表示为输入，并且将$o_t$表示为时间步骤$t$处的输出。回想我们在:numref:`subsec_rnn_w_hidden_states`中的讨论，可以将输入和隐藏状态连接起来，以乘以隐藏层中的一个权重变量。因此，我们使用$w_h$和$w_o$分别表示隐藏层和输出层的权重。结果，可以将每个时间步长的隐藏状态和输出解释为

$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_h),\\o_t &= g(h_t, w_o),\end{aligned}$$
:eqlabel:`eq_bptt_ht_ot`

其中$f$和$g$分别是隐藏层和输出层的变换。因此，我们具有通过递归计算彼此依赖的值链$\{\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots\}$。前向传播相当简单。我们所需要的就是一次一个时间步长遍历$(x_t, h_t, o_t)$个三元组。然后，输出$o_t$和期望标签$y_t$之间的差异由目标函数在所有$T$个时间步长上评估为

$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_h, w_o) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$

对于反向传播，情况有点棘手，特别是当我们计算关于目标函数$w_h$的参数$L$的梯度时。具体地说，根据连锁规则，

$$\begin{aligned}\frac{\partial L}{\partial w_h}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_h}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_h)}{\partial h_t}  \frac{\partial h_t}{\partial w_h}.\end{aligned}$$
:eqlabel:`eq_bptt_partial_L_wh`

:eqref:`eq_bptt_partial_L_wh`年度产品的第一因子和第二因子很容易计算。第三个因素$\partial h_t/\partial w_h$是事情变得棘手的地方，因为我们需要递归地计算参数$w_h$对$h_t$的影响。根据:eqref:`eq_bptt_ht_ot`的递归计算，$h_t$既依赖于$h_{t-1}$又依赖于$w_h$，其中$h_{t-1}$的计算也依赖于$w_h$。因此，使用链规则会产生

$$\frac{\partial h_t}{\partial w_h}= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_recur`

为了导出上述梯度，假设我们有三个序列$\{a_{t}\},\{b_{t}\},\{c_{t}\}$满足$a_{0}=0$,$a_{t}=b_{t}+c_{t}a_{t-1}$满足$t=1, 2,\ldots$。那么对于$t\geq 1$的人来说，很容易就能表现出来

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$
:eqlabel:`eq_bptt_at`

通过将$a_t$、$b_t$和$c_t$替换为

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_h},\\
b_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}, \\
c_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}},\end{aligned}$$

:eqref:`eq_bptt_partial_ht_wh_recur`中的梯度计算满足$a_{t}=b_{t}+c_{t}a_{t-1}$。因此，对于每个:eqref:`eq_bptt_at`，我们可以使用以下命令删除:eqref:`eq_bptt_partial_ht_wh_recur`中的递归计算

$$\frac{\partial h_t}{\partial w_h}=\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_h)}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_h)}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_gen`

虽然我们可以使用链规则递归地计算$\partial h_t/\partial w_h$，但是每当$t$很大时，这个链就会变得非常长。让我们讨论一下处理这个问题的一些策略。

### 完全计算#

显然，我们只需计算:eqref:`eq_bptt_partial_ht_wh_gen`的全部和即可。然而，这是非常缓慢的，梯度可能会爆炸，因为初始条件中的细微变化可能会对结果产生很大影响。也就是说，我们可以看到类似于蝴蝶效应的情况，在这种情况下，初始条件的微小变化会导致结果的不成比例的变化。就我们想要估计的模型而言，这实际上是相当不可取的。毕竟，我们正在寻找泛化良好的稳健估计器。因此，这一策略在实践中几乎从未使用过。

### 截断时间步长#

或者，我们可以在:eqref:`eq_bptt_partial_ht_wh_gen`步之后将总和截断为$\tau$。这就是我们到目前为止一直在讨论的问题，比如我们在:numref:`sec_rnn_scratch`中分离了渐变。这将导致真正梯度的*近似*，只需将总和终止于$\partial h_{t-\tau}/\partial w_h$即可。在实践中，这是非常有效的。这就是通常所说的通过时间:cite:`Jaeger.2002`的截断反向推进。这样做的后果之一是，该模型主要关注短期影响，而不是长期后果。这实际上是“可取的”，因为它使估计偏向于更简单、更稳定的模型。

### 随机截断#

最后，我们可以用一个随机变量替换$\partial h_t/\partial w_h$，该随机变量在预期中是正确的，但是会截断序列。这是通过使用$\xi_t$和预定义的$0 \leq \pi_t \leq 1$的序列来实现的，其中$P(\xi_t = 0) = 1-\pi_t$和$P(\xi_t = \pi_t^{-1}) = \pi_t$，即$E[\xi_t] = 1$。我们使用它将:eqref:`eq_bptt_partial_ht_wh_recur`中的渐变$\partial h_t/\partial w_h$替换为

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$

它是从$\xi_t$的定义推导出来的，那就是$E[z_t] = \partial h_t/\partial w_h$。每当$\xi_t = 0$递归计算在该时间终止时，步骤$t$。这导致不同长度的序列的加权和，其中长序列很少但适当地超重。这个想法是由Tallec和Olivier :cite:`Tallec.Ollivier.2017`提出的。

### 比较策略

![Comparing strategies for computing gradients in RNNs. From top to bottom: randomized truncation, regular truncation, and full computation.](../img/truncated-bptt.svg)
:label:`fig_truncated_bptt`

:numref:`fig_truncated_bptt`说明了使用RNN的时间反向传播分析“时间机器”一书的前几个人物时的三种策略：

* 第一行是随机截断，它将文本划分为不同长度的片段。
* 第二行是规则截断，它将文本分成相同长度的子序列。这就是我们在RNN实验中一直在做的事情。
* 第三行是通过时间的完全反向传播，它导致计算上不可行的表达式。

不幸的是，虽然随机截断在理论上很有吸引力，但它的效果并不比常规截断好很多，这很可能是由于许多因素造成的。首先，在许多反向传播步骤返回到过去之后，观察的效果足以捕获实践中的依赖关系。其次，增加的方差抵消了步长越多梯度越精确的事实。第三，我们实际上“想要”只有短范围交互的模型。因此，随时间规则截断的反向传播具有可能需要的轻微规则化效果。

## 详细的时间反向传播

在讨论了一般原理之后，让我们详细讨论一下时间反向传播。与:numref:`subsec_bptt_analysis`中的分析不同，下面我们将说明如何计算目标函数关于所有分解的模型参数的梯度。为简单起见，我们考虑无偏差参数的随机神经网络，其隐含层中的激活函数使用身份映射($\phi(x)=x$)。对于时间步长$t$，设单个示例输入和标签分别为$\mathbf{x}_t \in \mathbb{R}^d$和$y_t$。隐藏状态$\mathbf{h}_t \in \mathbb{R}^h$和输出$\mathbf{o}_t \in \mathbb{R}^q$被计算为

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_{hx} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_{qh} \mathbf{h}_{t},\end{aligned}$$

其中$\mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$、$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$和$\mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$是权重参数。用$l(\mathbf{o}_t, y_t)$表示时间步骤$t$处的损失。我们的目标函数，从序列开始起的超过$T$个时间步长的损失是这样的

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$

为了在RNN的计算过程中可视化模型变量和参数之间的依赖关系，我们可以为模型画一个计算图，如:numref:`fig_rnn_bptt`所示。例如，时间步长3、$\mathbf{h}_3$的隐藏状态的计算取决于模型参数$\mathbf{W}_{hx}$和$\mathbf{W}_{hh}$、最后时间步长$\mathbf{h}_2$的隐藏状态以及当前时间步长$\mathbf{x}_3$的输入。

![Computational graph showing dependencies for an RNN model with three time steps. Boxes represent variables (not shaded) or parameters (shaded) and circles represent operators.](../img/rnn-bptt.svg)
:label:`fig_rnn_bptt`

正如刚才提到的，:numref:`fig_rnn_bptt`中的模型参数是$\mathbf{W}_{hx}$、$\mathbf{W}_{hh}$和$\mathbf{W}_{qh}$。通常，训练该模型需要关于这些参数$\partial L/\partial \mathbf{W}_{hx}$、$\partial L/\partial \mathbf{W}_{hh}$和$\partial L/\partial \mathbf{W}_{qh}$的梯度计算。根据:numref:`fig_rnn_bptt`中的依赖关系，我们可以沿箭头的相反方向遍历，依次计算和存储梯度。为了灵活地表示链规则中不同形状的矩阵、向量和标量的乘法，我们继续使用$\text{prod}$运算符，如:numref:`sec_backprop`中所述。

首先，相对于在任何时间步骤$t$的模型输出来区分目标函数是相当简单的：

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$
:eqlabel:`eq_bptt_partial_L_ot`

现在，我们可以计算目标函数相对于输出层中的参数$\mathbf{W}_{qh}$的梯度：$\partial L/\partial \mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$。基于:numref:`fig_rnn_bptt`，目标函数$L$依赖于$\mathbf{W}_{qh}$至$\mathbf{o}_1, \ldots, \mathbf{o}_T$。使用链规则会产生

$$
\frac{\partial L}{\partial \mathbf{W}_{qh}}
= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$

其中$\partial L/\partial \mathbf{o}_t$是由:eqref:`eq_bptt_partial_L_ot`给出的。

接下来，如:numref:`fig_rnn_bptt`所示，在最后的时间步骤$T$，目标函数$L$仅通过$\mathbf{o}_T$依赖于隐藏状态$\mathbf{h}_T$。因此，我们可以使用链规则容易地找到梯度$\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$：

$$\frac{\partial L}{\partial \mathbf{h}_T} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$
:eqlabel:`eq_bptt_partial_L_hT_final_step`

对于任何时间步骤$t < T$来说都变得更加棘手，其中目标函数$L$依赖于$\mathbf{h}_t$通过$\mathbf{h}_{t+1}$和$\mathbf{o}_t$。根据链规则，隐藏状态$\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$在任何时间步骤$t < T$的梯度可以递归地计算为：

$$\frac{\partial L}{\partial \mathbf{h}_t} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$
:eqlabel:`eq_bptt_partial_L_ht_recur`

对于分析，扩展任何时间步骤$1 \leq t \leq T$的递归计算给出

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_{hh}^\top\right)}^{T-i} \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$
:eqlabel:`eq_bptt_partial_L_ht`

我们可以从:eqref:`eq_bptt_partial_L_ht`中看到，这个简单的线性示例已经展示了长序列模型的一些关键问题：它涉及到潜在的非常大的$\mathbf{W}_{hh}^\top$次方。其中，小于1的特征值消失，大于1的特征值发散。这在数值上是不稳定的，它以消失和爆炸梯度的形式表现出来。解决此问题的一种方法是按照计算方便的大小截断时间步长，如:numref:`subsec_bptt_analysis`中所述。实际上，这种截断是通过在给定数量的时间步长之后分离渐变来实现的。稍后，我们将看到更复杂的序列模型(如长期短期记忆)如何进一步缓解这一问题。

最后，:numref:`fig_rnn_bptt`示出目标函数$L$通过隐藏状态$\mathbf{W}_{hx}$和$\mathbf{W}_{hh}$依赖于隐藏层中的模型参数$\mathbf{h}_1, \ldots, \mathbf{h}_T$。为了计算关于这样的参数$\partial L / \partial \mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$和$\partial L / \partial \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$的梯度，我们应用链式规则，该规则给出

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_{hx}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_{hh}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top,
\end{aligned}
$$

其中$\partial L/\partial \mathbf{h}_t$是由:eqref:`eq_bptt_partial_L_hT_final_step`和:eqref:`eq_bptt_partial_L_ht_recur`递归计算的，是影响数值稳定性的关键量。

由于通过时间的反向传播是反向传播在RNN中的应用，如我们在:numref:`sec_backprop`中所解释的，训练RNN交替使用随时间的前向传播和反向传播。此外，通过时间的反向传播依次计算并存储上述梯度。具体地说，重复使用存储的中间值以避免重复计算，例如存储$\partial L/\partial \mathbf{h}_t$以用于$\partial L / \partial \mathbf{W}_{hx}$和$\partial L / \partial \mathbf{W}_{hh}$的计算两者。

## 摘要

* 通过时间的反向传播仅仅是反向传播对具有隐藏状态的序列模型的应用。
* 为了计算方便和数值稳定，需要截断，如规则截断和随机截断。
* 矩阵的高次方可能导致特征值发散或消失。这以爆炸或消失梯度的形式表现出来。
* 为了高效计算，在时间的反向传播期间缓存中间值。

## 练习

1. 假设我们具有对称矩阵$\mathbf{M} \in \mathbb{R}^{n \times n}$，该对称矩阵具有特征值$\lambda_i$，其对应的特征向量是$\mathbf{v}_i$($i = 1, \ldots, n$)。在不丧失一般性的情况下，假设它们是按顺序$|\lambda_i| \geq |\lambda_{i+1}|$排序的。
   1. 表明$\mathbf{M}^k$具有特征值$\lambda_i^k$。
   1. 证明对于随机向量$\mathbf{x} \in \mathbb{R}^n$，高概率$\mathbf{M}^k \mathbf{x}$将与特征向量$\mathbf{v}_1$非常对准
$\mathbf{M}$的人。把这句话正式化。
   1. 上述结果对于RNN中的梯度意味着什么？
1. 除了梯度剪裁，你还能想到其他方法来应对递归神经网络中的梯度爆炸吗？

[Discussions](https://discuss.d2l.ai/t/334)
