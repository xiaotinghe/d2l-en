# 时间反向传播
:label:`sec_bptt`

到目前为止，我们已经多次提到
*爆炸梯度*，
*消失梯度*，
而且需要
*分离RNN的梯度*。
例如，在:numref:`sec_rnn_scratch`中，我们调用了序列上的`detach`函数。为了能够快速构建模型并了解其工作原理，这些都没有得到充分的解释。在本节中，我们将更深入地研究序列模型的反向传播细节，以及为什么（和如何）数学工作。

我们在第一次实现RNNs（:numref:`sec_rnn_scratch`）时遇到了梯度爆炸的一些影响。特别是，如果你解决了练习，你会看到梯度剪辑是至关重要的，以确保适当的收敛。为了更好地理解这个问题，本节将回顾如何计算序列模型的梯度。注意，它的工作原理在概念上没有什么新意。毕竟，我们仍然只是应用链式法则来计算梯度。尽管如此，值得再次回顾反向传播（:numref:`sec_backprop`）。

我们在:numref:`sec_backprop`的MLPs中描述了前向和后向传播以及计算图。RNN中的前向传播相对简单。
*通过时间*的反向传播实际上是一个特定的
反向传播在rnns:cite:`Werbos.1990`中的应用。它要求我们将RNN的计算图一次展开一个时间步长，以获得模型变量和参数之间的依赖关系。然后，基于链规则，利用反向传播算法计算和存储梯度。因为序列可能相当长，所以依赖关系可能相当长。例如，对于1000个字符的序列，第一个标记可能对最后位置的标记有重大影响。这在计算上是不可行的（它需要太长的时间和太多的内存），它需要超过1000个矩阵积，然后我们才能得到非常难以捉摸的梯度。这是一个充满计算和统计不确定性的过程。下面我们将阐明发生了什么以及如何在实践中解决这个问题。

## RNNs中的梯度分析
:label:`subsec_bptt_analysis`

我们从RNN工作原理的简化模型开始。这个模型忽略了隐藏状态的细节以及它是如何更新的。这里的数学符号并不像以前那样明确区分标量、向量和矩阵。这些细节对分析无关紧要，只会使本小节中的符号变得混乱。

在这个简化模型中，我们将$h_t$表示为隐藏状态，$x_t$表示为输入，$o_t$表示为时间步$t$处的输出。回想一下我们在:numref:`subsec_rnn_w_hidden_states`中的讨论，输入和隐藏状态可以串联起来，在隐藏层中乘以一个权重变量。因此，我们使用$w_h$和$w_o$分别表示隐藏层和输出层的权重。结果，每个时间步的隐藏状态和输出可以解释为

$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_h),\\o_t &= g(h_t, w_o),\end{aligned}$$
:eqlabel:`eq_bptt_ht_ot`

其中$f$和$g$分别是隐藏层和输出层的变换。因此，我们有一个通过循环计算相互依赖的值链$\{\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots\}$。前向传播相当简单。我们只需要在$(x_t, h_t, o_t)$中循环一次。输出$o_t$和所需标签$y_t$之间的差异然后通过跨越所有$T$时间步的目标函数进行评估，如下所示：

$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_h, w_o) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$

对于反向传播，问题有点棘手，特别是当我们计算关于目标函数$L$的参数$w_h$的梯度时。具体来说，根据链式法则，

$$\begin{aligned}\frac{\partial L}{\partial w_h}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_h}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_h)}{\partial h_t}  \frac{\partial h_t}{\partial w_h}.\end{aligned}$$
:eqlabel:`eq_bptt_partial_L_wh`

:eqref:`eq_bptt_partial_L_wh`中产品的第一和第二因子易于计算。第三个因素$\partial h_t/\partial w_h$是事情变得棘手的地方，因为我们需要反复计算参数$w_h$对$h_t$的影响。根据:eqref:`eq_bptt_ht_ot`中的递归计算，$h_t$依赖于$h_{t-1}$和$w_h$，其中$h_{t-1}$的计算也依赖于$w_h$。因此，使用链式法则

$$\frac{\partial h_t}{\partial w_h}= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_recur`

为了推导上述梯度，假设我们有三个序列$\{a_{t}\},\{b_{t}\},\{c_{t}\}$满足$a_{0}=0$和$t=1, 2,\ldots$的$a_{t}=b_{t}+c_{t}a_{t-1}$。那么对于$t\geq 1$来说，很容易表现出来

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$
:eqlabel:`eq_bptt_at`

通过根据替换$a_t$、$b_t$和$c_t$

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_h},\\
b_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}, \\
c_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}},\end{aligned}$$

:eqref:`eq_bptt_partial_ht_wh_recur`中的梯度计算满足$a_{t}=b_{t}+c_{t}a_{t-1}$。因此，根据:eqref:`eq_bptt_at`，我们可以使用

$$\frac{\partial h_t}{\partial w_h}=\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_h)}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_h)}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_gen`

虽然我们可以使用链式规则递归计算$\partial h_t/\partial w_h$，但只要$t$很大，这个链式就可能非常长。让我们讨论一些处理这个问题的策略。

### 完全计算###

很明显，我们可以计算:eqref:`eq_bptt_partial_ht_wh_gen`的全和。然而，这是非常缓慢和梯度可以爆炸，因为在初始条件的微妙变化可能会影响结果很多。也就是说，我们可以看到类似于蝴蝶效应的情况，在这种情况下，初始条件的微小变化会导致结果的不成比例的变化。就我们要估计的模型而言，这实际上是非常不可取的。毕竟，我们要寻找的是能很好地推广的稳健估计。因此，这种策略几乎从未在实践中使用过。

### 截断时间步长###

或者，我们可以在$\tau$步之后截断:eqref:`eq_bptt_partial_ht_wh_gen`中的和。这是我们到目前为止一直在讨论的，比如当我们在:numref:`sec_rnn_scratch`中分离梯度时。这导致了真正梯度的*近似值*，只需将和终止于$\partial h_{t-\tau}/\partial w_h$即可。实际上，这很有效。这就是通常所说的通过时间:cite:`Jaeger.2002`截短的反向推进。其后果之一是，该模型主要关注短期影响，而不是长期影响。这实际上是“可取的”，因为它使估计偏向于更简单和更稳定的模型。

### 随机截断###

最后，我们可以用一个随机变量来代替$\partial h_t/\partial w_h$，这个随机变量在预期中是正确的，但会截断序列。这是通过使用$\xi_t$序列和预定义的$0 \leq \pi_t \leq 1$来实现的，其中$P(\xi_t = 0) = 1-\pi_t$和$P(\xi_t = \pi_t^{-1}) = \pi_t$，因此$E[\xi_t] = 1$。我们用它来替换:eqref:`eq_bptt_partial_ht_wh_recur`中的渐变$\partial h_t/\partial w_h$

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$

从$\xi_t$的定义可以看出$E[z_t] = \partial h_t/\partial w_h$。每当$\xi_t = 0$时，循环计算在该时间终止于步骤$t$。这导致了长度不等的序列的加权和，其中长序列很少，但适当地过重。这个想法是由塔勒克和奥利维尔:cite:`Tallec.Ollivier.2017`提出的。

### 比较策略

![Comparing strategies for computing gradients in RNNs. From top to bottom: randomized truncation, regular truncation, and full computation.](../img/truncated-bptt.svg)
:label:`fig_truncated_bptt`

:numref:`fig_truncated_bptt`举例说明了使用RNN时间反向传播分析*时间机器*书籍的前几个字符时的三种策略：

* 第一行是随机截断，它将文本划分为不同长度的段。
* 第二行是规则的截断，它将文本分成相同长度的子序列。这就是我们在RNN实验中所做的。
* 第三行是通过时间的完全反向传播，这导致计算上不可行的表达式。

不幸的是，虽然在理论上很有吸引力，但随机截断并不比常规截断有效得多，很可能是由于许多因素。首先，在对过去进行多次反向传播之后，观察的效果足以在实践中捕获依赖关系。第二，增加的方差抵消了一个事实，即梯度更准确的步骤。第三，我们实际上想要的是交互范围很短的模型。因此，通过时间的规则截断的反向传播具有轻微的可期望的正则化效果。

## 随时间的反向传播

在讨论了一般原理之后，让我们详细讨论通过时间的反向传播。与:numref:`subsec_bptt_analysis`中的分析不同，下面我们将展示如何计算目标函数相对于所有分解模型参数的梯度。为了简单起见，我们考虑了一个无偏差参数的RNN，它在隐层的激活函数使用身份映射（$\phi(x)=x$）。对于时间步$t$，让单个示例输入和标签分别为$\mathbf{x}_t \in \mathbb{R}^d$和$y_t$。隐藏状态$\mathbf{h}_t \in \mathbb{R}^h$和输出$\mathbf{o}_t \in \mathbb{R}^q$被计算为

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_{hx} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_{qh} \mathbf{h}_{t},\end{aligned}$$

其中$\mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$、$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$和$\mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$是权重参数。用$l(\mathbf{o}_t, y_t)$表示时间步$t$处的损失。我们的目标函数，从序列开始超过$T$个时间步的损失是这样的

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$

为了可视化RNN计算过程中模型变量和参数之间的依赖关系，我们可以绘制模型的计算图，如:numref:`fig_rnn_bptt`所示。例如，时间步骤3 $\mathbf{h}_3$的隐藏状态的计算取决于模型参数$\mathbf{W}_{hx}$和$\mathbf{W}_{hh}$、上一时间步骤$\mathbf{h}_2$的隐藏状态以及当前时间步骤$\mathbf{x}_3$的输入。

![Computational graph showing dependencies for an RNN model with three time steps. Boxes represent variables (not shaded) or parameters (shaded) and circles represent operators.](../img/rnn-bptt.svg)
:label:`fig_rnn_bptt`

如前所述，:numref:`fig_rnn_bptt`中的模型参数为$\mathbf{W}_{hx}$、$\mathbf{W}_{hh}$和$\mathbf{W}_{qh}$。通常，训练此模型需要对这些参数$\partial L/\partial \mathbf{W}_{hx}$、$\partial L/\partial \mathbf{W}_{hh}$和$\partial L/\partial \mathbf{W}_{qh}$进行梯度计算。根据:numref:`fig_rnn_bptt`中的依赖关系，我们可以按箭头的相反方向遍历，依次计算和存储梯度。为了灵活地表示链式规则中不同形状的矩阵、向量和标量的乘法，我们继续使用$\text{prod}$运算符，如:numref:`sec_backprop`所述。

首先，在任何时候根据模型输出区分目标函数步骤$t$相当简单：

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$
:eqlabel:`eq_bptt_partial_L_ot`

现在，我们可以计算目标函数相对于输出层$\mathbf{W}_{qh}$参数的梯度：$\partial L/\partial \mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$。基于:numref:`fig_rnn_bptt`，目标函数$L$通过$\mathbf{o}_1, \ldots, \mathbf{o}_T$依赖于$\mathbf{W}_{qh}$。使用链式法则产生

$$
\frac{\partial L}{\partial \mathbf{W}_{qh}}
= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$

其中$\partial L/\partial \mathbf{o}_t$由:eqref:`eq_bptt_partial_L_ot`给出。

接下来，如:numref:`fig_rnn_bptt`所示，在最后的时间步骤$T$，目标函数$L$仅经由$\mathbf{o}_T$依赖于隐藏状态$\mathbf{h}_T$。因此，我们可以使用链式法则很容易地找到梯度$\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$：

$$\frac{\partial L}{\partial \mathbf{h}_T} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$
:eqlabel:`eq_bptt_partial_L_hT_final_step`

对于任何时间步$t < T$，它都变得更加棘手，其中目标函数$L$通过$\mathbf{h}_{t+1}$和$\mathbf{o}_t$依赖于$\mathbf{h}_t$。根据链式规则，在任何时间步骤$t < T$处的隐藏状态$\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$的梯度可以循环地计算为：

$$\frac{\partial L}{\partial \mathbf{h}_t} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$
:eqlabel:`eq_bptt_partial_L_ht_recur`

为便于分析，将任意时间步长$1 \leq t \leq T$的递推计算展开，得到

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_{hh}^\top\right)}^{T-i} \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$
:eqlabel:`eq_bptt_partial_L_ht`

从:eqref:`eq_bptt_partial_L_ht`中我们可以看出，这个简单的线性例子已经展示了长序列模型的一些关键问题：它涉及$\mathbf{W}_{hh}^\top$的可能非常大的幂次。其中，小于1的特征值消失，大于1的特征值发散。这在数值上是不稳定的，表现为消失和爆炸梯度的形式。解决这一问题的一种方法是以计算方便的大小截断时间步长，如:numref:`subsec_bptt_analysis`所述。实际上，这种截断是通过在给定的时间步数之后分离梯度来实现的。稍后我们将看到更复杂的序列模型，如长-短期记忆，如何进一步缓解这种情况。

最后，:numref:`fig_rnn_bptt`显示目标函数$L$通过隐藏状态$\mathbf{h}_1, \ldots, \mathbf{h}_T$依赖于隐藏层中的模型参数$\mathbf{W}_{hx}$和$\mathbf{W}_{hh}$。为了计算关于这些参数$\partial L / \partial \mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$和$\partial L / \partial \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$的梯度，我们应用链式规则

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

其中，由:eqref:`eq_bptt_partial_L_hT_final_step`和:eqref:`eq_bptt_partial_L_ht_recur`循环计算的$\partial L/\partial \mathbf{h}_t$是影响数值稳定性的关键量。

由于时间反向传播是反向传播在RNN中的应用，正如我们在:numref:`sec_backprop`中所解释的，训练RNN在时间上交替进行正向传播和反向传播。此外，通过时间反向传播依次计算和存储上述梯度。具体地，存储的中间值被重用以避免重复计算，例如存储$\partial L/\partial \mathbf{h}_t$以用于$\partial L / \partial \mathbf{W}_{hx}$和$\partial L / \partial \mathbf{W}_{hh}$的计算。

## 摘要

* 通过时间的反向传播仅仅是反向传播在具有隐藏状态的序列模型中的应用。
* 为了计算方便和数值稳定性，需要进行截断，如规则截断和随机截断。
* 矩阵的高次幂会导致特征值发散或消失。这表现为爆发或消失梯度的形式。
* 为了提高计算效率，在时间反向传播期间缓存中间值。

## 练习

1. 假设我们有一个特征值为$\lambda_i$的对称矩阵$\mathbf{M} \in \mathbb{R}^{n \times n}$，其对应的特征向量为$\mathbf{v}_i$（$i = 1, \ldots, n$）。在不丧失一般性的前提下，假设它们是按照$|\lambda_i| \geq |\lambda_{i+1}|$号订单订购的。
   1. 证明了$\mathbf{M}^k$的特征值为$\lambda_i^k$。
   1. 证明了对于随机向量$\mathbf{x} \in \mathbb{R}^n$，$\mathbf{M}^k \mathbf{x}$很有可能与特征向量$\mathbf{v}_1$对齐
$\mathbf{M}$号。把这句话正式化。
   1. 对于RNN中的梯度，上述结果意味着什么？
1. 除了梯度削波，你能想出其他方法来处理递归神经网络中的梯度爆炸吗？

[Discussions](https://discuss.d2l.ai/t/334)
