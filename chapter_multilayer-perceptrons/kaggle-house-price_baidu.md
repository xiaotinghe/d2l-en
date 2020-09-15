# 用Kaggle预测房价
:label:`sec_kaggle_house`

现在我们已经介绍了一些构建和训练深层网络的基本工具，并使用包括权重衰减和退出的技术对其进行规范化，我们已经准备好通过参加Kaggle竞赛将所有这些知识付诸实践。房价预测大赛是一个很好的起点。这些数据是相当通用的，没有显示出可能需要特殊模型（如音频或视频）的奇异结构。该数据集由Bart de Cock于2011年收集:cite:`De-Cock.2011`，涵盖了2006年至2010年期间IA艾姆斯的房价。它比著名的哈里森和鲁宾菲尔德（1978）的[Boston housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names)要大得多，它拥有更多的例子和更多的特性。

在本节中，我们将引导您详细了解数据预处理、模型设计和超参数选择。我们希望通过亲身实践的方法，你将获得一些直觉，这些直觉将指导你作为数据科学家的职业生涯。

## 下载和缓存数据集

在本书中，我们将在各种下载的数据集上训练和测试模型。在这里，我们实现了几个实用函数，以方便数据下载。首先，我们维护一个字典`DATA_HUB`，它将一个字符串（数据集的*名称*映射到一个元组，该元组包含定位数据集的URL和验证文件完整性的SHA-1键。所有这些数据集都托管在地址为`DATA_URL`的站点上。

```{.python .input}
#@tab all
import os
import requests
import zipfile
import tarfile
import hashlib

#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
```

下面的`download`函数下载数据集，将其缓存在本地目录中（默认为`../data`），并返回下载文件的名称。如果缓存目录中已经存在与此数据集对应的文件，并且其SHA-1与存储在`DATA_HUB`中的文件相匹配，则我们的代码将使用缓存文件，以避免冗余下载阻塞您的互联网。

```{.python .input}
#@tab all
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    d2l.mkdir_if_not_exist(cache_dir)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
```

我们还实现了两个附加的实用程序函数：一个是下载并提取一个zip或tar文件，另一个是将本书中使用的所有数据集从`DATA_HUB`下载到缓存目录中。

```{.python .input}
#@tab all
def download_extract(name, folder=None):  #@save
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """Download all files in the DATA_HUB."""
    for name in DATA_HUB:
        download(name)
```

## 卡格尔

[Kaggle](https://www.kaggle.com)是一个流行的平台，主办机器学习比赛。每场比赛都以一个数据集为中心，许多比赛都是由利益相关者赞助的，他们为获胜的解决方案提供奖品。该平台帮助用户通过论坛和共享代码进行互动，促进合作和竞争。虽然排行榜的追逐往往失控，研究人员目光短浅地专注于预处理步骤，而不是问一些基本问题，但客观的平台也有巨大的价值，这个平台有助于在竞争方法之间进行直接定量比较，以及代码共享，以便每个人都能学习什么起作用，什么不起作用。如果你想参加Kaggle竞赛，你首先需要注册一个帐户（见:numref:`fig_kaggle`）。

![The Kaggle website.](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

在房价预测竞争页面，如:numref:`fig_house_pricing`所示，您可以找到数据集（在“数据”选项卡下），提交预测，并查看您的排名，网址如下：

> https://www.kaggle.com/c/house-prices-advanced-regulation-technologies

![The house price prediction competition page.](../img/house_pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## 访问和读取数据集

请注意，竞赛数据分为训练集和测试集。每个记录包括房屋的属性值和属性，如街道类型、建筑年份、屋顶类型、地下室状况等。特征由各种数据类型组成。例如，建筑年份由整数表示，屋顶类型由离散类别指定表示，其他要素由浮点数表示。这就是现实使事情复杂化的地方：在一些例子中，一些数据完全丢失，缺失的值被简单地标记为“na”。每套房子的价格只包含在训练设备中（毕竟这是一场比赛）。我们需要划分训练集来创建一个验证集，但是我们只有在将预测上传到Kaggle之后才能在官方测试集中评估我们的模型。:numref:`fig_house_pricing`中竞争标签上的“数据”标签有下载数据的链接。

首先，我们将使用:numref:`sec_pandas`中介绍的`pandas`读入并处理数据。因此，在继续下一步之前，您需要确保已经安装了`pandas`。幸运的是，如果你在Jupyter读书，我们甚至不用离开笔记本就可以安装熊猫。

```{.python .input}
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd
npx.set_np()
```

```{.python .input}
#@tab pytorch
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
from d2l import torch as d2l
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
```

```{.python .input}
#@tab tensorflow
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import pandas as pd
import numpy as np
```

为了方便起见，我们可以使用上面定义的脚本下载并缓存Kaggle housing数据集。

```{.python .input}
#@tab all
DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
```

我们使用`pandas`分别加载包含训练和测试数据的两个csv文件。

```{.python .input}
#@tab all
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
```

训练数据集包括1460个示例、80个特征和1个标签，而测试数据包含1459个示例和80个特征。

```{.python .input}
#@tab all
print(train_data.shape)
print(test_data.shape)
```

让我们看看前四个和最后两个特性，以及前四个例子中的标签（SalePrice）。

```{.python .input}
#@tab all
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
```

我们可以看到，在每个示例中，第一个特性是ID，这有助于模型识别每个训练示例。虽然这很方便，但它不携带任何用于预测的信息。因此，在将数据输入模型之前，我们将其从数据集中删除。

```{.python .input}
#@tab all
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

## 数据预处理

如上所述，我们有各种各样的数据类型。在开始建模之前，我们需要对数据进行预处理。让我们从数字特征开始。首先，我们使用启发式方法，用相应特征的平均值替换所有缺失值。然后，为了将所有特征放在一个共同的尺度上，我们通过将特征重新调整为零均值和单位方差来*标准化*数据：

$$x \leftarrow \frac{x - \mu}{\sigma}.$$

为了验证这确实转换了我们的特征（变量），使其具有零均值和单位方差，请注意$E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$和$E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$。直观地说，我们标准化数据有两个原因。首先，它证明了优化的方便性。第二，因为我们不知道哪些特征是相关的，所以我们不想对分配给一个特征的系数进行惩罚。

```{.python .input}
#@tab all
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# After standardizing the data all means vanish, hence we can set missing
# values to 0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

接下来我们处理离散值。其中包括“MSZoning”等功能。我们用一个热编码来代替它们，就像我们以前把多类标签转换成向量一样（见:numref:`subsec_classification-problem`）。例如，“MSZoning”假设值为“RL”和“RM”。除去“MSZoning”功能，将创建两个新的指示符功能“MSZoning-RL”和“MSZoning-u-RM”，其值为0或1。根据一种热编码，如果“MSZoning”的原始值为“RL”，那么“MSZoning\u RL”为1，“MSZoning\u RM”为0。`pandas`包自动为我们做这件事。

```{.python .input}
#@tab all
# `Dummy_na=True` considers "na" (missing value) as a valid feature value, and
# creates an indicator feature for it
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
```

您可以看到，这种转换将特性的数量从79个增加到331个。最后，通过`values`属性，我们可以从`pandas`格式中提取NumPy格式并将其转换为张量表示进行训练。

```{.python .input}
#@tab all
n_train = train_data.shape[0]
train_features = d2l.tensor(all_features[:n_train].values, dtype=d2l.float32)
test_features = d2l.tensor(all_features[n_train:].values, dtype=d2l.float32)
train_labels = d2l.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=d2l.float32)
```

## 培训

首先，我们训练一个损失平方的线性模型。毫不奇怪，我们的线性模型不会导致竞争获胜，但它提供了一个健全的检查，看看数据中是否存在有意义的信息。如果我们不能比随机猜测做得更好，那么很有可能我们有一个数据处理错误。如果一切顺利的话，线性模型将作为一个基线，让我们对简单模型与最佳报告模型的接近程度有一些直觉，让我们知道我们应该从更漂亮的模型中获得多少收益。

```{.python .input}
loss = gluon.loss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net
```

```{.python .input}
#@tab pytorch
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.MeanSquaredError()

def get_net():
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    return net
```

对于房价，就像股票价格一样，我们关心的是相对数量，而不是绝对数量。因此，我们更关心相对误差$\frac{y - \hat{y}}{y}$，而不是绝对误差$y - \hat{y}$。例如，如果我们在俄亥俄州农村地区估计一栋房子的价格时，我们的预测偏差了10万美元，在那里，一栋典型的房子的价值是12.5万美元，那么我们可能做得很糟糕。另一方面，如果我们在加利福尼亚州洛斯阿尔托斯山的这一数字出现错误，这可能是一个惊人的准确预测（在那里，房价中位数超过400万美元）。

解决这个问题的一种方法是测量价格估计值对数的差异。事实上，这也是比赛用来评价提交质量的官方误差指标。毕竟，$|\log y - \log \hat{y}| \leq \delta$的小值$\delta$转换为$e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$。这将导致预测价格的对数与标签价格的对数之间出现以下均方根误差：

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

```{.python .input}
def log_rmse(net, features, labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = np.clip(net(features), 1, float('inf'))
    return np.sqrt(2 * loss(np.log(clipped_preds), np.log(labels)).mean())
```

```{.python .input}
#@tab pytorch
def log_rmse(net, features, labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(torch.mean(loss(torch.log(clipped_preds),
                                       torch.log(labels))))
    return rmse.item()
```

```{.python .input}
#@tab tensorflow
def log_rmse(y_true, y_pred):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = tf.clip_by_value(y_pred, 1, float('inf'))
    return tf.sqrt(tf.reduce_mean(loss(
        tf.math.log(y_true), tf.math.log(clipped_preds))))
```

与前面的部分不同，我们的培训功能将依赖于Adam优化器（我们将在后面更详细地描述它）。这种优化器的主要吸引力在于，尽管在无限制的超参数优化资源下没有做得更好（有时甚至更差），但人们倾向于发现它对初始学习速率的敏感性明显降低。

```{.python .input}
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

```{.python .input}
#@tab pytorch
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

```{.python .input}
#@tab tensorflow
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    net.compile(loss=loss, optimizer=optimizer)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(X)
                l = loss(y, y_hat)
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
        train_ls.append(log_rmse(train_labels, net(train_features)))
        if test_labels is not None:
            test_ls.append(log_rmse(test_labels, net(test_features)))
    return train_ls, test_ls
```

## $K$折叠交叉验证

您可能还记得，我们在讨论如何处理模型选择的部分中介绍了$K$折叠交叉验证（:numref:`sec_model_selection`）。这将有助于模型设计的选择和超参数的调整。我们首先需要一个函数，在$K$折叠交叉验证过程中返回$i^\mathrm{th}$倍的数据。它通过切片$i^\mathrm{th}$段作为验证数据，其余部分作为训练数据返回。请注意，这并不是处理数据的最有效方法，如果我们的数据集大得多，我们肯定会做一些更聪明的事情。但是这种增加的复杂性可能会不必要地混淆我们的代码，因此我们可以安全地忽略它，因为我们的问题很简单。

```{.python .input}
#@tab all
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = d2l.concat([X_train, X_part], 0)
            y_train = d2l.concat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
```

当我们在$K$次交叉验证中训练$K$次时，返回训练和验证误差的平均值。

```{.python .input}
#@tab all
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs+1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse',
                     legend=['train', 'valid'], yscale='log')
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k
```

## 选型

在本例中，我们选取了一组未经调整的超参数，并将其留给读者来改进模型。找到一个好的选择需要时间，这取决于优化了多少变量。在足够大的数据集和正常类型的超参数的情况下，$K$倍交叉验证对于多重测试具有相当的弹性。然而，如果我们尝试了不合理的大量选项，我们可能会幸运地发现，我们的验证性能不再代表真正的错误。

```{.python .input}
#@tab all
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-fold validation: avg train log rmse: {float(train_l):f}, '
      f'avg valid log rmse: {float(valid_l):f}')
```

请注意，有时一组超参数的训练错误数可能非常低，即使$K$倍交叉验证的错误数要高得多。这表明我们过度适应了。在整个培训过程中，您需要监控这两个数字。较少的过度拟合可能表明我们的数据可以支持一个更强大的模型。大规模的过度拟合可能意味着我们可以通过合并正则化技术来获得收益。

##  提交Kaggle的预测

既然我们知道了超参数的最佳选择是什么，我们还可以使用所有的数据对其进行训练（而不仅仅是交叉验证切片中使用的$1-1/K$个数据）。我们以这种方式获得的模型可以应用于测试集。将预测保存在csv文件中可以简化将结果上载到Kaggle的过程。

```{.python .input}
#@tab all
def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', yscale='log')
    print(f'train log rmse {float(train_ls[-1]):f}')
    # Apply the network to the test set
    preds = d2l.numpy(net(test_features))
    # Reformat it to export to Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

一个很好的健全性检查是看看测试集上的预测是否类似于$K$倍交叉验证过程的预测。如果有，是时候把它们上传到Kaggle了。下面的代码将生成一个名为`submission.csv`的文件。

```{.python .input}
#@tab all
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
```

接下来，如:numref:`fig_kaggle_submit2`所示，我们可以提交我们对Kaggle的预测，看看它们与测试集上实际房价（标签）的比较情况。步骤非常简单：

* 登录Kaggle网站，访问房价预测竞赛页面。
* 点击“提交预测”或“延迟提交”按钮（在本文撰写之际，按钮位于右侧）。
* 点击页面底部虚线框中的“上传提交文件”按钮，选择要上传的预测文件。
* 点击页面底部的“提交”按钮查看结果。

![Submitting data to Kaggle](../img/kaggle_submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## 摘要

* 实际数据通常包含不同数据类型的混合，需要进行预处理。
* 将实值数据重新调整为零均值和单位方差是一个很好的默认设置。用平均值替换缺失值也是如此。
* 将范畴特征转化为指标特征，可以使我们把它们当作一个热点向量来对待。
* 我们可以使用$K$折叠交叉验证来选择模型和调整超参数。
* 对数对于相对误差很有用。

## 练习

1. 把你对这一部分的预测提交给卡格尔。你的预测有多好？
1. 你能通过直接最小化价格对数来改进你的模型吗？如果你试图预测价格的对数而不是价格，会发生什么？
1. 用它们的平均值替换缺失值总是一个好主意吗？提示：你能构造一个不随机丢失值的情况吗？
1. 通过$K$倍交叉验证调整超参数，提高Kaggle的得分。
1. 通过改进模型来提高分数（例如，层数、重量衰减和脱落）。
1. 如果我们没有像本节所做的那样标准化连续的数值特征，会发生什么？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/106)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/107)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/237)
:end_tab:
