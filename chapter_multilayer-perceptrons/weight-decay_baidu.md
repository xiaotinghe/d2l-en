# 重量衰减
:label:`sec_weight_decay`

既然我们已经描述了过度拟合的问题，我们可以介绍一些标准的正则化模型的技术。回想一下，我们总是可以通过外出收集更多的培训数据来缓解过度适应。这可能是昂贵的，耗时的，或者完全失去我们的控制，这在短期内是不可能的。现在，我们可以假设我们已经拥有尽可能多的高质量数据，并将重点放在正则化技术上。

回想一下，在我们的多项式回归示例（:numref:`sec_model_selection`）中，我们可以通过调整拟合多项式的次数来限制模型的容量。实际上，限制特性的数量是一种流行的技术，可以减轻过度拟合。然而，简单地放弃特性对于这项工作来说可能过于生硬。坚持多项式回归的例子，考虑高维输入可能会发生什么。多项式对多元数据的自然扩展称为*单项式*，它只是变量幂的乘积。单项式的度是幂和。例如，$x_1^2 x_2$和$x_3 x_5^2$都是3阶的单项式。

请注意，随着$d$的增长，$d$度的项数迅速增加。给定$k$个变量，阶数$d$（即$k$多选$d$）的个数为${k - 1 + d} \choose {k - 1}$。即使是程度上的微小变化，比如从$2$到$3$，也会显著增加我们模型的复杂性。因此，我们经常需要一个更细粒度的工具来调整函数的复杂性。

## 规范与权重衰减

我们已经描述了$L_2$规范和$L_1$规范，它们是$L_p$规范在$L_p$中的特殊情况。
*权重衰减*（通常称为$L_2$正则化），
可能是正则化参数化机器学习模型最广泛使用的技术。这项技术是基于一个基本直觉，即在所有函数$f$中，函数$f = 0$（为所有输入赋值$0$）在某种意义上是最简单的，我们可以通过函数与零的距离来衡量函数的复杂度。但是我们应该如何精确地测量一个函数和零之间的距离呢？没有一个正确的答案。事实上，整个数学分支，包括函数分析和Banach空间理论，都致力于回答这个问题。

一种简单的解释可以是通过线性函数$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$的权向量的某个范数来度量其复杂性，例如$\| \mathbf{w} \|^2$。保证小权向量最常用的方法是将其范数作为惩罚项加到损失最小化的问题中。这样我们就取代了原来的目标，
*最小化训练标签上的预测损失*，
有了新的目标，
*最小化预测损失和惩罚项*之和。
现在，如果我们的权重向量增长太大，我们的学习算法可能会集中在最小化权重范数$\| \mathbf{w} \|^2$与最小化训练误差。这正是我们想要的。为了在代码中说明一些事情，让我们重温一下:numref:`sec_linear_regression`中的线性回归示例。我们的损失是由

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

回想一下$\mathbf{x}^{(i)}$是特征，$y^{(i)}$是所有数据的标签示例$i$和$(\mathbf{w}, b)$分别是权重和偏差参数。为了惩罚权重向量的大小，我们必须在损失函数中添加$\| \mathbf{w} \|^2$，但是模型应该如何权衡标准损失来获得新的附加惩罚呢？实际上，我们通过*正则化常数*$\lambda$来描述这种权衡，这是一个非负超参数，我们使用验证数据拟合：

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2,$$

对于$\lambda = 0$，我们恢复了原来的损失函数。对于$\lambda > 0$，我们限制$\| \mathbf{w} \|$的大小。我们按照惯例除以$2$：当我们取一个二次函数的导数时，$2$和$1/2$会相消，以确保更新表达式看起来既漂亮又简单。精明的读者可能会想知道为什么我们使用平方范数而不是标准范数（即欧几里得距离）。我们这样做是为了便于计算。通过平方$L_2$范数，我们去掉平方根，留下权重向量每个分量的平方和。这使得惩罚的导数很容易计算：导数的和等于和的导数。

此外，您可能会问为什么我们首先使用$L_2$规范，而不是$L_1$规范。事实上，其他选择在整个统计数据中都是有效的和受欢迎的。当$L_2$正则化线性模型构成经典的*岭回归*算法时，$L_1$正则化线性回归是统计学中类似的基本模型，通常被称为*套索回归*。

使用$L_2$范数的一个原因是它对权重向量的大分量施加了巨大的惩罚。这使得我们的学习算法偏向于在大量特征上均匀分布权重的模型。在实践中，这可能使它们对单个变量中的测量误差更为稳健。相比之下，$L_1$惩罚会导致模型通过将其他权重清除为零而将权重集中在一小部分特性上。这称为*特征选择*，这可能是由于其他原因而需要的。

使用:eqref:`eq_linreg_batch_update`中的相同符号，$L_2$正则化回归的小批量随机梯度下降更新如下：

$$
\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
$$

如前所述，我们根据我们的估计值与观察值之间的差异来更新$\mathbf{w}$。然而，我们也将$\mathbf{w}$的大小缩小到零。这就是为什么这种方法有时被称为“权重衰减”：仅考虑惩罚项，我们的优化算法*在训练的每一步*衰减*权重。与特征选择相比，权重衰减为我们提供了一种连续的机制来调整函数的复杂度。较小的$\lambda$值对应较少约束的$\mathbf{w}$，而较大的$\lambda$值对$\mathbf{w}$的约束更大。

我们是否包括相应的偏差惩罚$b^2$可以在不同的实现中变化，并且可以在神经网络的各个层之间变化。通常，我们不正则化网络输出层的偏差项。

## 高维线性回归

我们可以通过一个简单的例子来说明综合权重的衰减。

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

我们选择我们的标签是我们输入的线性函数，被高斯噪声破坏，平均值为零，标准偏差为0.01。为了使过度拟合的效果更加明显，我们可以将问题的维数增加到$d = 200$，并使用一个只包含20个示例的小训练集。

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

在下面，我们将从头开始实现权重衰减，只需将$L_2$的平方惩罚添加到原始目标函数中。

### 初始化模型参数

首先，我们将定义一个函数来随机初始化我们的模型参数。

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

### 定义$L_2$标准罚款

也许实施这一处罚最方便的方法是把所有条款都放在适当的地方，并加以总结。

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

### 定义训练循环

下面的代码适合于训练集上的模型，并在测试集中对其进行求值。自:numref:`chap_linear`以来，线性网络和平方损耗没有改变，所以我们将通过`d2l.linreg`和`d2l.squared_loss`导入它们。唯一的变化是我们的损失现在包括了惩罚条款。

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

### 无正规化培训

我们现在用`lambd = 0`运行这个代码，禁用重量衰减。注意，我们过度拟合，减少了训练误差，但没有减少测试误差——这是textook过度拟合的一个例子。

```{.python .input}
#@tab all
train(lambd=0)
```

### 使用权重衰减

下面，我们的体重大幅下降。注意，训练误差增大，但测试误差减小。这正是我们期望从正规化中得到的效果。

```{.python .input}
#@tab all
train(lambd=3)
```

## 简明实施

由于权值衰减在神经网络优化中无处不在，深度学习框架使其特别方便，将权值衰减集成到优化算法中，以便与任何损失函数结合使用。此外，这种集成还有计算上的好处，允许实现技巧在不增加任何额外的计算开销的情况下向算法中添加权重衰减。由于更新的权重衰减部分仅依赖于每个参数的当前值，因此优化器必须至少接触每个参数一次。

:begin_tab:`mxnet`
在下面的代码中，我们在实例化`Trainer`时直接通过`wd`指定weight decay超参数。默认情况下，胶子同时衰减权重和偏移。注意，更新模型参数时，超参数`wd`将乘以`wd_mult`。因此，如果我们将`wd_mult`设置为零，则偏移参数$b$将不会衰减。
:end_tab:

:begin_tab:`pytorch`
在下面的代码中，我们在实例化优化器时直接通过`weight_decay`指定weight decay超参数。默认情况下，Pythorch同时衰减权重和偏移。这里我们只为权重设置了`weight_decay`，所以bias参数$b$不会衰减。
:end_tab:

:begin_tab:`tensorflow`
在下面的代码中，我们使用权重衰减超参数$L_2$创建一个$L_2$正则化器，并通过`kernel_regularizer`参数将其应用于层。
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

这些图看起来和我们从零开始实现权重衰减时的图相同。然而，它们运行得更快，更容易实现，对于更大的问题，这一好处将变得更加明显。

```{.python .input}
#@tab all
train_concise(0)
```

```{.python .input}
#@tab all
train_concise(3)
```

到目前为止，我们只讨论了一个简单线性函数的概念。此外，什么构成一个简单的非线性函数可能是一个更复杂的问题。例如，[再生核希尔伯特空间（RKHS）](https://en.wikipedia.org/wiki/reproducting_kernel_Hilbert_空间)允许在非线性上下文中应用为线性函数引入的工具。不幸的是，基于RKHS的算法往往难以扩展到大的、高维的数据。在这本书中，我们将默认使用简单的启发式方法，即在深层网络的所有层上应用权重衰减。

## 摘要

* 正则化是处理过拟合的常用方法。在训练集的损失函数中加入惩罚项，以降低学习模型的复杂度。
* 保持模型简单的一个特别的选择是使用$L_2$惩罚的权重衰减。这会导致学习算法更新步骤中的权重衰减。
* 权重衰减功能在深度学习框架的优化器中提供。
* 在同一训练循环中，不同的参数集可以有不同的更新行为。

## 练习

1. 在本节的估计问题中，使用$\lambda$的值进行实验。绘制训练和测试精度作为$\lambda$的函数。你观察到了什么？
1. 使用验证集来找到最佳值$\lambda$。它真的是最优值吗？这有关系吗？
1. 如果我们使用$\sum_i |w_i|$作为我们选择的惩罚（$L_1$正则化），那么更新方程会是什么样子？
1. 我们知道$\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$。你能找到类似的矩阵方程吗（见:numref:`subsec_lin-algebra-norms`中的Frobenius范数）？
1. 回顾了训练误差和泛化误差之间的关系。除了体重下降、增加训练、使用适当复杂度的模型之外，你还能想出其他什么方法来处理过度拟合？
1. 在贝叶斯统计中，我们使用先验和似然的乘积通过$P(w \mid x) \propto P(x \mid w) P(w)$到达后验点。如何识别$P(w)$与正规化？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/236)
:end_tab:
