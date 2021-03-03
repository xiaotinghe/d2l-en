# 前言

就在几年前，还没有大公司和初创公司开发智能产品和服务的深度学习科学家军团。当我们(作者)中最年轻的人进入这个领域时，机器学习并没有占据每日报纸的头条。我们的父母根本不知道什么是机器学习，更不用说为什么我们可能更喜欢机器学习而不是从事医学或法律职业。机器学习是一门具有前瞻性的学科，在现实世界中的应用范围很窄。而那些应用，例如语音识别和计算机视觉，需要如此多的领域知识，以至于它们通常被认为是完全独立的领域，而机器学习对于这些领域来说只是一个小组件。因此，神经网络，我们在本书中关注的深度学习模型的前身，被认为是过时的工具。

就在过去的五年里，深度学习给世界带来了惊喜，推动了计算机视觉、自然语言处理、自动语音识别、强化学习和统计建模等领域的快速发展。有了这些进步，我们现在可以制造比以往任何时候都更自主的汽车(而自主程度比一些公司可能让你相信的要低)，可以自动起草最平凡的电子邮件的智能回复系统，帮助人们从巨大得令人窒息的收件箱中解脱出来的挖洞，以及在围棋等棋类游戏中主宰世界上最优秀的人类的软件代理，这一壮举曾经被认为是几十年后的事。这些工具已经对工业和社会产生了越来越广泛的影响，改变了电影制作的方式，疾病的诊断，并在基础科学中发挥着越来越大的作用-从天体物理学到生物学。

## 关于本书

这本书代表了我们让深度学习变得平易近人的尝试，教你*概念*、*上下文*和*代码*。

### 一种结合了代码、数学和HTML的媒介

要使任何计算技术发挥其全部作用，它必须得到充分的理解、良好的文档记录以及成熟的、维护良好的工具的支持。关键思想应该被清楚地提炼出来，最大限度地减少将新从业者带到最新所需的入职时间。成熟的库应该自动化常见的任务，示例代码应该使从业人员可以轻松地修改、应用和扩展常见的应用程序，以满足他们的需求。以动态Web应用为例。尽管像亚马逊这样的大量公司在20世纪90年代开发了成功的数据库驱动的Web应用程序，但在过去的10年里，这种技术帮助创意企业家的潜力已经在更大程度上得到了实现，这在一定程度上要归功于强大的、文档齐全的框架的开发。

测试深度学习的潜力带来了独特的挑战，因为任何单一的应用程序都会将不同的学科结合在一起。应用深度学习需要同时了解(I)以特定方式提出问题的动机；(Ii)给定建模方法的数学；(Iii)将模型拟合到数据的优化算法；以及(Iv)有效训练模型所需的工程，克服数值计算的缺陷，最大限度地利用现有硬件。同时教授表述问题所需的批判性思维技能，解决问题的数学知识，以及实施这些解决方案的软件工具，都是一项艰巨的挑战。我们在这本书中的目标是提供一个统一的资源，让潜在的实践者了解最新情况。

在我们开始这本书项目的时候，没有同时(I)是最新的；(Ii)涵盖了具有相当技术深度的现代机器学习的全部广度；以及(Iii)交错地阐述了人们从一本引人入胜的教科书中期望的质量，以及人们期望在实践教程中找到的干净的可运行代码。我们发现了大量关于如何使用给定的深度学习框架(例如，如何对TensorFlow中的矩阵进行基本的数值计算)或实现特定技术(例如，LeNet、AlexNet、ResNet等的代码片段)的代码示例，分散在各种博客帖子和GitHub库中。但是，这些示例通常集中在
*如何*实现给定的方法，
但忽略了“为什么”做出某些算法决策的讨论。虽然一些交互式资源已经零星地弹出以解决特定主题，例如，在网站[Distill](http://distill.pub)上发布的引人入胜的博客帖子或个人博客，但它们仅覆盖深度学习中的选定主题，并且通常缺乏关联的代码。另一方面，虽然已经出现了几本教科书，其中最著名的是:cite:`Goodfellow.Bengio.Courville.2016`本，它对深度学习背后的概念进行了全面的调查，但这些资源并没有将这些概念的描述与代码中这些概念的实现结合起来，有时会让读者对如何实现它们一无所知。此外，太多的资源隐藏在商业课程提供商的付费墙后面。

我们着手创建的资源可以：(I)每个人都可以免费获得；(Ii)提供足够的技术深度，为真正成为一名应用机器学习科学家提供起点；(Iii)包括可运行的代码，向读者展示*如何*解决实践中的问题；(Iv)允许快速更新，包括我们和整个社区；以及(V)以[forum](http://discuss.d2l.ai)作为补充，用于技术细节的互动讨论和回答问题。

这些目标经常是相互冲突的。方程式、定理和引文最好用LaTeX来管理和布局。代码最好用Python描述。并且网页在HTML和JavaScript中是原生的。此外，我们希望内容既可以作为可执行代码访问，也可以作为纸质书访问，作为可下载的PDF访问，也可以作为网站在Internet上访问。目前还没有完全适合这些需求的工具和工作流程，所以我们不得不自行组装。我们在:numref:`sec_how_to_contribute`中详细描述了我们的方法。我们选择GitHub来共享源代码并允许编辑，选择Jupyter笔记本来混合代码、方程式和文本，选择Sphinx作为渲染引擎来生成多个输出，并为论坛提供讨论。虽然我们的制度尚不完善，但这些选择在相互冲突的问题之间提供了一个很好的妥协。我们相信，这可能是第一本使用这种集成工作流程出版的书。

### 边做边学

许多教科书教授一系列的主题，每一个都非常详细。例如，克里斯·毕晓普(Chris Bishop)优秀的教科书:cite:`Bishop.2006`，对每个主题都教得如此透彻，以至于要读到关于线性回归的那一章，需要做大量的工作。虽然专家们非常喜欢这本书的彻底性，但对于初学者来说，这一特性限制了它作为介绍性文本的实用性。

在这本书中，我们将适时教授大部分概念。换句话说，您将在实现某些实际目的所需的非常时刻学习概念。虽然我们在开始时花了一些时间来教授基础基础知识，如线性代数和概率，但我们希望您在担心更深奥的概率分布之前，先品尝一下培训第一个模型的满足感。

除了提供基本数学背景速成课程的几个初步笔记本外，后续的每一章都介绍了合理数量的新概念，并提供了单个自包含的工作示例-使用真实数据集。这带来了组织上的挑战。某些型号可能在逻辑上组合在单个笔记本中。而一些想法可能最好是通过连续执行几个模型来传授。另一方面，坚持“一个工作示例，一个笔记本”的策略有一个很大的好处：这使您可以通过利用我们的代码尽可能轻松地启动您自己的研究项目。只需复制笔记本并开始修改即可。

我们将根据需要将可运行代码与背景材料交错。通常，在完整解释工具之前，我们经常会犯提供工具的错误(我们将在稍后解释背景)。例如，在充分解释它为什么有用或为什么有效之前，我们可能会使用“随机梯度下降”。这有助于给从业者提供必要的弹药来快速解决问题，但代价是要求读者信任我们，做出一些策展决定。

这本书将从头开始教授深度学习的概念。有时，我们想深入研究模型的详细信息，这些模型通常会被深度学习框架的高级抽象隐藏起来。这一点在基础教程中尤其突出，我们希望您了解在给定层或优化器中发生的一切。在这些情况下，我们通常会提供两个版本的示例：一个是我们从头开始实现一切，仅依赖于NumPy接口和自动区分；另一个是更实际的示例，其中我们使用深度学习框架的高级API编写简洁的代码。一旦我们教会了一些组件的工作原理，我们就可以在后续教程中使用高级API了。

### 内容和结构

全书大致可分为三个部分，:numref:`fig_book_org`用不同的颜色呈现：

![Book structure](../img/book-org.svg)
:label:`fig_book_org`

* 第一部分包括基础知识和预备知识。
:numref:`chap_introduction`提供深度学习的入门课程。然后，在:numref:`chap_preliminaries`中，我们将快速向您介绍实践深度学习所需的前提条件，例如如何存储和处理数据，以及如何应用基于线性代数、微积分和概率的基本概念的各种数值运算。:numref:`chap_linear`和:numref:`chap_perceptrons`涵盖了深度学习的最基本概念和技术，例如线性回归、多层感知器和正则化。

* 接下来的五章集中讨论现代深度学习技术。
:numref:`chap_computation`描述了深度学习计算的各种关键组件，并为我们随后实现更复杂的模型奠定了基础。接下来，在:numref:`chap_cnn`和:numref:`chap_modern_cnn`中，我们介绍了卷积神经网络(CNN)，这是构成大多数现代计算机视觉系统的骨干的强大工具。随后，在:numref:`chap_rnn`和:numref:`chap_modern_rnn`中，我们引入了递归神经网络(RNNs)，这是一种利用数据中的时间或序列结构的模型，通常用于自然语言处理和时间序列预测。在:numref:`chap_attention`中，我们引入了一类新的模型，它采用了一种称为注意机制的技术，最近它们已经开始在自然语言处理中取代RNN。这些部分将帮助您快速了解大多数现代深度学习应用程序背后的基本工具。

* 第三部分讨论可伸缩性、效率和应用程序。
首先，在:numref:`chap_optimization`中，我们讨论了几种用于训练深度学习模型的常用优化算法。下一章，:numref:`chap_performance`将研究影响深度学习代码计算性能的几个关键因素。在:numref:`chap_cv`中，我们说明了深度学习在计算机视觉中的主要应用。在:numref:`chap_nlp_pretrain`和:numref:`chap_nlp_app`中，我们介绍了如何对语言表示模型进行预训练，并将其应用于自然语言处理任务。

### 代码
:label:`sec_code`

本书的大部分章节都以可执行代码为特色，因为我们相信交互式学习体验在深度学习中的重要性。目前，某些直觉只能通过试错、小幅调整代码并观察结果来开发。理想情况下，一个优雅的数学理论可能会精确地告诉我们如何调整代码以达到期望的结果。不幸的是，目前，这些优雅的理论让我们望而却步。尽管我们尽了最大努力，但仍然缺乏对各种技术的正式解释，这既是因为描述这些模型的数学方法可能非常困难，也因为对这些主题的严肃调查最近才刚刚开始。我们希望，随着深度学习理论的发展，这本书的未来版本将能够在当前版本无法提供的地方提供见解。

有时，为了避免不必要的重复，我们将本书中经常导入和引用的函数、类等封装在`d2l`包中。对于要保存到包中的任何挡路，比如一个函数、一个类或者多个导入，我们都会标记为`#@save`。我们在:numref:`sec_d2l`中提供了这些函数和类的详细概述。`d2l`软件包是轻量级的，仅需要以下软件包和模块作为依赖项：

```{.python .input}
#@tab all
#@save
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import hashlib
d2l = sys.modules[__name__]
```

:begin_tab:`mxnet`
本书中的大部分代码都是基于Apache MXNet的。MXNet是深度学习的开源框架，是AWS(亚马逊网络服务)以及许多大学和公司的首选。本书中的所有代码都通过了最新MXNet版本的测试。但是，由于深度学习的快速发展，一些代码
*在印刷版*中，MXNet的未来版本可能无法正常工作。
但是，我们计划使在线版本保持最新。如果您遇到任何此类问题，请咨询:ref:`chap_installation`以更新您的代码和运行时环境。

下面是我们如何从MXNet导入模块。
:end_tab:

:begin_tab:`pytorch`
本书中的大部分代码都是基于PyTorch的。PyTorch是一个开源的深度学习框架，在研究界非常受欢迎。本书中的所有代码都在最新的PyTorch下通过了测试。但是，由于深度学习的快速发展，一些代码
*在印刷版中*在未来版本的PyTorch中可能无法正常工作。
但是，我们计划使在线版本保持最新。如果您遇到任何此类问题，请咨询:ref:`chap_installation`以更新您的代码和运行时环境。

下面是我们如何从PyTorch导入模块。
:end_tab:

:begin_tab:`tensorflow`
本书中的大部分代码都是基于TensorFlow的。TensorFlow是一个开源的深度学习框架，在研究界和产业界都非常受欢迎。本书中的所有代码都通过了最新TensorFlow的测试。但是，由于深度学习的快速发展，一些代码
*在印刷版*中，TensorFlow的未来版本可能无法正常工作。
但是，我们计划使在线版本保持最新。如果您遇到任何此类问题，请咨询:ref:`chap_installation`以更新您的代码和运行时环境。

下面是我们如何从TensorFlow导入模块。
:end_tab:

```{.python .input}
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from PIL import Image
```

```{.python .input}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```

### 目标受众

本书面向学生(本科生或研究生)、工程师和研究人员，他们希望扎实掌握深度学习的实用技术。因为我们从头开始解释每个概念，所以不需要以前的深度学习或机器学习背景。全面解释深度学习的方法需要一些数学和编程，但我们只假设您了解一些基础知识，包括线性代数、微积分、概率和Python编程(非常基础)。此外，在附录中，我们提供了本书所涵盖的大多数数学知识的复习。大多数时候，我们会优先考虑直觉和想法，而不是数学的严谨性。有许多很棒的书可以引导感兴趣的读者走得更远。例如，Bela Bollobas :cite:`Bollobas.1999`的“线性分析”非常深入地涵盖了线性代数和泛函分析。“统计:cite:`Wasserman.2013`全集”是一本很棒的统计指南。如果您以前没有使用过Python语言，那么您可能想要仔细阅读这个[Python tutorial](http://learnpython.org/)。

### 论坛

与本书相关的是，我们已经启动了一个论坛，位于[discuss.d2l.ai](https://discuss.d2l.ai/)。当您对本书的任何一节有疑问时，您可以在每一章的末尾找到相关的讨论页链接。

## 确认

我们感谢数以百计的英文和中文草稿的撰稿人。他们帮助改进了内容，并提供了宝贵的反馈。具体地说，我们感谢这份英文草稿的每一位贡献者让它对每个人都变得更好。他们的GitHub ID或名称是(没有特定顺序)：alxnorden、avinashingit、Bowen0701、brettkoonce、chaitanya Prakash Bapat、cryptonaut、Davide Fiocco、edgarroman、gkutiel、John Mitro、梁普、Rahul Agarwal、Mohamed Ali牙买加oui、Michael(Stu)Stewart、Mike Müller、NRauschmayr、Prakhar Srikhar李博文，Aarush Ahuja，Prasanth Budareddygari，brianhendee，mani2106，mtn，lkevinzc，曹吉林，lakshya，Fiete吕尔，Surbhi Vijayvargeeya，Muhyun Kim，dennismalmgren，adursun，Anirudh Dagar，liqingnz，Pedro Larroy，lgov，ati-Ozgur，J.codypenta，joseppinilla，ahmaurya，karolszk，heytitle，Peter Goetz，rigtorp，Tiep Vu，sfilip，mlxd，kale-ab Tessera，Sanjar Adilov，MatteoFerrara，hkinto，Katarzyna Biesialska，Gregory Bruss，Duy-Thanh Doan，Paulaurel，GreyTowne，Duc Pham，sl74.Tomer kaftan，liweiwp，netyster，ypandya，nishantthharani，heiligerl，sportsthu，hoa nguyen，Manuel-arno-Korfmann-webentwicklong，aterzis-Personal，nxby，晓婷He，Josiah Yoder，mathResearch，mzz2017，jroberayalas，iluu，ghejc，bhari，vkramdev，simonon

我们感谢Amazon Web Services，特别是Swami Sivasubramanian、Raju Gulabani、Charlie Bell和Andrew Jassy对撰写本书的慷慨支持。如果没有可用的时间、资源、与同事的讨论和不断的鼓励，这本书就不会出版。

## 摘要

* 深度学习使模式识别发生了革命性的变化，引入了现在支持广泛技术的技术，包括计算机视觉、自然语言处理、自动语音识别。
* 要成功地应用深度学习，您必须了解如何处理问题、建模的数学、将模型与数据拟合的算法，以及实现所有这些的工程技术。
* 这本书提供了一个全面的资源，包括散文，数字，数学和代码，都集中在一个地方。
* 要回答与本书相关的问题，请访问我们的论坛https://discuss.d2l.ai/.
* 所有笔记本电脑都可以在GitHub上下载。

## 练习

1. 在本书[discuss.d2l.ai](https://discuss.d2l.ai/)的论坛上注册帐户。
1. 在您的计算机上安装Python。
1. 沿着本节底部的链接进入论坛，在那里您可以寻求帮助，讨论这本书，并通过与作者和更广泛的社区接触来找到问题的答案。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/18)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/20)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/186)
:end_tab:
