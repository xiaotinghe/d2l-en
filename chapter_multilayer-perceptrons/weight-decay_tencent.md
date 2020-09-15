# 重量衰减
:label:`sec_weight_decay`

既然我们已经描述了过拟合问题的特征，我们可以介绍一些用于正规化模型的标准技术。回想一下，我们总是可以通过走出去收集更多的培训数据来缓解过度适应。这可能是昂贵的、耗时的，或者完全不受我们的控制，在短期内是不可能的。目前，我们可以假设在资源允许的情况下，我们已经拥有了尽可能多的高质量数据，并将重点放在正规化技术上。

回想一下，在我们的多项式回归示例(:numref:`sec_model_selection`)中，我们可以简单地通过调整拟合多项式的次数来限制模型的容量。实际上，限制特征的数量是缓解过度拟合的一种流行技术。然而，简单地把功能放在一边对于这项工作来说可能太迟钝了。继续使用多项式回归示例，考虑高维输入可能发生的情况。多项式对多变量数据的自然扩展称为*单项式*，它简单地说是变量幂的乘积。单项式的次数是幂的和。例如，$x_1^2 x_2$和$x_3 x_5^2$都是3次单项式。

请注意，具有$d$度的术语的数量随着$d$的增长而迅速增加。在给定$k$个变量的情况下，$d$次单项式的数量(即，$k$个多项式$d$)是${k - 1 + d} \choose {k - 1}$个。即使是阶数上的微小变化，比如从$2$到$3$，也会极大地增加我们模型的复杂性。因此，我们经常需要更细粒度的工具来调整功能复杂性。

## 范数和权重衰减

我们已经描述了$L_2$规范和$L_1$规范，它们是:numref:`subsec_lin-algebra-norms`中更一般的$L_p$规范的特例。
*权重衰减*(通常称为$L_2$正则化)，
可能是规则化参数机器学习模型的最广泛使用的技术。该技术是由这样的基本直觉驱动的，即在所有函数$f$中，函数$f = 0$(将值$0$赋给所有输入)在某种意义上是*最简单的*，并且我们可以通过函数离零的距离来测量函数的复杂性。但是，我们应该如何精确地测量函数和零之间的距离呢？没有唯一正确的答案。事实上，整个数学分支，包括部分泛函分析和Banach空间理论，都致力于回答这个问题。

一种简单的解释可以是通过线性函数$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$的权重向量的某个范数(例如，$\| \mathbf{w} \|^2$)来测量其复杂性。保证一个小的权向量的最常见的方法是将它的范数作为惩罚项添加到最小化损失的问题中。这样我们就取代了原来的目标，
*最小化训练标签上的预测损失*，
有了新的目标，
*最小化预测损失和惩罚项之和*。
现在，如果我们的权重向量变得太大，我们的学习算法可能会专注于最小化权重范数$\| \mathbf{w} \|^2$而不是最小化训练误差。这正是我们想要的。为了用代码来说明问题，让我们从:numref:`sec_linear_regression`开始复习前面的线性回归示例。在那里，我们的损失是由

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

回想一下，$\mathbf{x}^{(i)}$是特征，$y^{(i)}$是所有数据示例$i$的标签，$(\mathbf{w}, b)$分别是权重和偏差参数。为了惩罚权重向量的大小，我们必须以某种方式将损失函数增加$\| \mathbf{w} \|^2$，但是模型应该如何权衡这个新的附加惩罚的标准损失呢？在实践中，我们通过*正则化常数*$\lambda$(我们使用验证数据拟合的非负超参数)来表征这种权衡：

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2,$$

对于$\lambda = 0$，我们恢复了原来的损失函数。对于$\lambda > 0$，我们将大小限制为$\| \mathbf{w} \|$。我们按惯例除以$2$：当我们取二次函数的导数时，$2$和$1/2$被抵消，从而确保更新的表达式看起来又好又简单。精明的读者可能想知道为什么我们使用平方范数而不是标准范数(即欧几里德距离)。我们这样做是为了便于计算。通过对$L_2$范数平方，我们去掉了平方根，留下了权重向量每个分量的平方和。这使得罚金的导数很容易计算：导数的和等于和的导数。

此外，您可能会问，为什么我们一开始就使用$L_2$规范，而不是，比方说，使用$L_1$规范。事实上，在整个统计过程中，其他选择都是有效和受欢迎的。$L_2$正则线性模型构成经典的“岭回归”算法，而$L_1$正则线性回归是统计学中类似的基本模型，俗称“套索回归”。

使用$L_2$规范的一个原因是它对权重向量的大分量施加了过大的惩罚。这使我们的学习算法偏向在大量特征上均匀分配权重的模型。在实践中，这可能会使它们对单个变量中的测量误差更具鲁棒性。相比之下，$L_1$的惩罚导致通过将其他权重清除为零来将权重集中在一小部分特征上的模型。这就是所谓的“特征选择”，这可能是出于其他原因而需要的。

使用:eqref:`eq_linreg_batch_update`中的相同符号，$L_2$正则化回归的小批量随机梯度下降更新如下：

$$
\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
$$

和以前一样，我们根据我们的估计与观测结果不同的金额更新了$\mathbf{w}$。然而，我们也将$\mathbf{w}$的规模缩小到接近零。这就是为什么这种方法有时被称为“权重衰减”：仅在给定惩罚项的情况下，我们的优化算法在训练的每一步都会“衰减”权重。与特征选择不同，权重衰减为我们提供了一种调整函数复杂性的连续机制。较小的值$\lambda$对应较少的约束$\mathbf{w}$，而较大的值$\lambda$则更明显地约束$\mathbf{w}$。

我们是否包括相应的偏差惩罚$b^2$可以在不同的实现中变化，并且可以在神经网络的各个层上变化。通常，我们不将网络输出层的偏置项正则化。

## 高维线性回归

我们可以通过一个简单的合成例子来说明重量衰减的好处。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torch.nn as nn
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

首先，我们像以前一样生成一些数据

$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$

我们选择我们的标签是我们输入的线性函数，被具有零均值和标准差0.01的高斯噪声破坏。为了使过度拟合的效果明显，我们可以将问题的维度增加到$d = 200$，并使用仅包含20个示例的小训练集。

```{.python .input}
#@tab all
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
```

## 从头开始实施

在下面，我们将从头开始实现权重衰减，只需将$L_2$的平方惩罚添加到原始目标函数中即可。

### 正在初始化模型参数

首先，我们将定义一个函数来随机初始化模型参数。

```{.python .input}
def init_params():
    w = np.random.normal(scale=1, size=(num_inputs, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    return [w, b]
```

```{.python .input}
#@tab pytorch
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]
```

```{.python .input}
#@tab tensorflow
def init_params():
    w = tf.Variable(tf.random.normal(mean=1, shape=(num_inputs, 1)))
    b = tf.Variable(tf.zeros(shape=(1, )))
    return [w, b]
```

### 定义$L_2$定额罚款

也许实施这一处罚最方便的方法是将所有条款放在适当的位置，并将其汇总。

```{.python .input}
def l2_penalty(w):
    return (w**2).sum() / 2
```

```{.python .input}
#@tab pytorch
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
```

```{.python .input}
#@tab tensorflow
def l2_penalty(w):
    return tf.reduce_sum(tf.pow(w, 2)) / 2
```

### 定义培训循环

下面的代码适合训练集上的一个模型，并在测试集中评估它。线性网络和平方损耗自:numref:`chap_linear`以来没有变化，因此我们将仅通过`d2l.linreg`和`d2l.squared_loss`导入它们。这里唯一的变化是我们的损失现在包括了罚金期限。

```{.python .input}
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                # The L2 norm penalty term has been added, and broadcasting
                # makes `l2_penalty(w)` a vector whose length is `batch_size`
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', np.linalg.norm(w))
```

```{.python .input}
#@tab pytorch
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                # The L2 norm penalty term has been added, and broadcasting
                # makes `l2_penalty(w)` a vector whose length is `batch_size`
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', torch.norm(w).item())
```

```{.python .input}
#@tab tensorflow
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # The L2 norm penalty term has been added, and broadcasting
                # makes `l2_penalty(w)` a vector whose length is `batch_size`
                l = loss(net(X), y) + lambd * l2_penalty(w)
            grads = tape.gradient(l, [w, b])
            d2l.sgd([w, b], grads, lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', tf.norm(w).numpy())
```

### 不正规化训练

我们现在使用`lambd = 0`运行此代码，禁用重量衰减。请注意，我们的过度拟合非常严重，减少了训练误差，但没有减少测试误差-这是一个过度拟合的文本化情况。

```{.python .input}
#@tab all
train(lambd=0)
```

### 使用权重衰减

下面，我们运行时会有相当大的重量衰减。注意，训练误差增加，但测试误差减小。这正是我们期待的正规化效果。

```{.python .input}
#@tab all
train(lambd=3)
```

## 简明实施

由于权重衰减在神经网络优化中普遍存在，深度学习框架使其变得特别方便，它将权重衰减集成到优化算法本身中，以便与任何损失函数结合使用。此外，这种集成具有计算优势，允许实现技巧在不增加任何计算开销的情况下增加算法的权重衰减。由于更新的权重衰减部分仅取决于每个参数的当前值，因此优化器无论如何都必须触及每个参数一次。

:begin_tab:`mxnet`
在下面的代码中，我们在实例化`wd`时直接通过`Trainer`指定权重衰减超参数。默认情况下，胶子会同时衰退重量和偏移。注意，当更新模型参数时，超参数`wd`将乘以`wd_mult`。因此，如果我们将`wd_mult`设置为零，则偏置参数$b$将不会衰减。
:end_tab:

:begin_tab:`pytorch`
在下面的代码中，我们在实例化优化器时直接通过`weight_decay`指定权重衰减超参数。默认情况下，PyTorch会同时衰退权重和偏移。在这里，我们仅将权重设置为`weight_decay`，因此偏移参数$b$不会衰减。
:end_tab:

:begin_tab:`tensorflow`
在下面的代码中，我们使用权重衰减超参数`wd`创建了一个$L_2$的正则化函数，并通过`kernel_regularizer`参数将其应用于层。
:end_tab:

```{.python .input}
def train_concise(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    loss = gluon.loss.L2Loss()
    num_epochs, lr = 100, 0.003
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': wd})
    # The bias parameter has not decayed. Bias names generally end with "bias"
    net.collect_params('.*bias').setattr('wd_mult', 0)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', np.linalg.norm(net[0].weight.data()))
```

```{.python .input}
#@tab pytorch
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    # The bias parameter has not decayed
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                trainer.zero_grad()
                l = loss(net(X), y)
            l.backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', net[0].weight.norm().item())
```

```{.python .input}
#@tab tensorflow
def train_concise(wd):
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(wd)))
    net.build(input_shape=(1, num_inputs))
    w, b = net.trainable_variables
    loss = tf.keras.losses.MeanSquaredError()
    num_epochs, lr = 100, 0.003
    trainer = tf.keras.optimizers.SGD(learning_rate=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # `tf.keras` requires retrieving and adding the losses from
                # layers manually for custom training loop.
                l = loss(net(X), y) + net.losses
            grads = tape.gradient(l, net.trainable_variables)
            trainer.apply_gradients(zip(grads, net.trainable_variables))
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', tf.norm(net.get_weights()[0]).numpy())
```

这些情节看起来与我们从头开始实施重量衰减时的情况完全相同。但是，它们的运行速度要快得多，而且更容易实现，对于较大的问题，这一优势将变得更加明显。

```{.python .input}
#@tab all
train_concise(0)
```

```{.python .input}
#@tab all
train_concise(3)
```

到目前为止，我们只触及了构成简单线性函数的一个概念。此外，什么构成一个简单的非线性函数可能是一个更复杂的问题。例如，[再生核希尔伯特空间(RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space)]允许人们在非线性环境中应用为线性函数引入的工具。不幸的是，基于RKHS的算法往往不能很好地扩展到大型、高维数据。在本书中，我们将默认使用简单的启发式方法，即在深层网络的所有层上应用权重衰减。

## 摘要

* 正则化是处理过拟合的常用方法。该算法在训练集的损失函数中加入惩罚项，降低了学习模型的复杂度。
* 保持模型简单的一个特殊选择是重量衰减，使用$L_2$的惩罚。这导致学习算法的更新步骤中的权重衰减。
* 深度学习框架的优化器中提供了权重衰减功能。
* 在同一训练循环内，不同的参数集可以具有不同的更新行为。

## 练习

1. 在本节的估计问题中使用值$\lambda$进行试验。绘制训练和测试精度与$\lambda$的函数关系图。你观察到了什么？
1. 使用验证集查找最佳值$\lambda$。这真的是最优值吗？这有关系吗？
1. 如果我们使用$\|\mathbf{w}\|^2$而不是$\sum_i |w_i|$作为我们选择的惩罚($L_1$正则化)，更新方程式会是什么样子？
1. 我们知道那$\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$。你能为矩阵找到一个类似的方程式(参见:numref:`subsec_lin-algebra-norms`中的弗罗贝尼乌斯范数)吗？
1. 回顾训练误差和泛化误差之间的关系。除了体重下降，增加训练，使用适当复杂的模型外，你还能想到其他什么方法来处理过度适应的问题吗？
1. 在贝叶斯统计中，我们使用先验和可能性的乘积来得出后验通孔$P(w \mid x) \propto P(x \mid w) P(w)$。你怎么能把$P(w)$等同于正规化呢？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/236)
:end_tab:
