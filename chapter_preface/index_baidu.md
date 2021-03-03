# 前言

就在几年前，还没有大批深入学习的科学家在大公司和初创公司开发智能产品和服务。当我们中最年轻的一位（作者）进入这个领域时，机器学习并没有成为日报的头条新闻。我们的父母不知道什么是机器学习，更不用说为什么我们更喜欢机器学习而不是从事医学或法律职业。机器学习是一门具有前瞻性的学科，在现实世界中的应用范围很窄。而那些应用，例如语音识别和计算机视觉，需要如此多的领域知识，以至于它们常常被视为完全独立的领域，机器学习只是其中的一个小组成部分。神经网络，我们在这本书中关注的深度学习模型的前身，被认为是过时的工具。

在过去的五年里，深度学习让世界大吃一惊，推动了计算机视觉、自然语言处理、自动语音识别、强化学习和统计建模等领域的快速发展。有了这些进步，我们现在可以制造出比以往任何时候都更能自主驾驶的汽车（而且比一些公司可能让你相信的自主性要小）、自动起草最平凡电子邮件的智能回复系统、帮助人们从令人压抑的大收件箱中挖掘出来，以及主宰世界上最好的软件代理人类玩围棋这样的棋类游戏，这一壮举曾经被认为是几十年后的事。这些工具已经对工业和社会产生了越来越广泛的影响，改变了电影的制作方式，疾病的诊断方式，并在基础科学中扮演着越来越重要的角色——从天体物理学到生物学。

## 关于这本书

这本书代表了我们试图让深度学习变得平易近人，教你概念、上下文和代码。

### 一种结合了代码、数学和HTML的媒体

任何一种计算技术要想发挥其全部影响力，都必须得到充分的理解、充分的文档记录，并得到成熟的、维护良好的工具的支持。关键的想法应该被清晰地提炼出来，尽可能减少需要让新的从业者跟上时代的入职时间。成熟的库应该自动化常见的任务，范例代码应该使实践者能够方便地修改、应用和扩展常见的应用程序以满足他们的需求。以动态web应用程序为例。尽管许多公司，如亚马逊，在20世纪90年代开发了成功的数据库驱动的web应用程序，但在过去十年中，这项技术在帮助创造性企业家方面的潜力已经得到了更大程度的发挥，部分原因是开发了功能强大、文档完整的框架。

测试深度学习的潜力带来了独特的挑战，因为任何单一的应用程序都会将不同的学科结合在一起。应用深度学习需要同时理解（i）以特定方式投射问题的动机；（ii）给定建模方法的数学；（iii）将模型与数据拟合的优化算法；以及（iv）有效训练模型所需的工程，克服数值计算的缺陷，最大限度地利用现有硬件。在一个地方教授解决问题所需的批判性思维技能、解决问题所需的数学以及实现这些解决方案所需的软件工具都是一项艰巨的挑战。我们在这本书中的目标是提出一个统一的资源，使未来的从业者的速度。

在我们开始这本书的项目时，没有资源同时（i）是最新的；（ii）涵盖了现代机器学习的全部广度和实质性的技术深度；和（iii）穿插的质量说明，人们期望从一本引人入胜的教科书与干净的可运行的代码，人们期望在实践中找到教程。我们发现了大量关于如何使用给定的深度学习框架（例如，如何使用TensorFlow中的矩阵进行基本的数值计算）或实现特定技术（例如，LeNet、AlexNet、ResNets等的代码片段）的代码示例，这些示例散布在各种博客文章和GitHub存储库中。然而，这些例子通常集中在
*如何实现给定的方法，
但是没有讨论为什么会做出某些算法决定。虽然一些互动资源偶尔出现，以解决一个特定的主题，例如，在[Distill](http://distill.pub)网站上发布的引人入胜的博客文章，或个人博客，但它们只涵盖了深度学习中选定的主题，往往缺乏相关的代码。另一方面，虽然出现了一些教科书，最著名的是:cite:`Goodfellow.Bengio.Courville.2016`，它提供了对深度学习背后概念的全面调查，但这些资源并没有将描述与代码中概念的实现结合起来，有时会让读者对如何实现它们一无所知。此外，商业课程提供商的付费墙背后隐藏着太多的资源。

我们着手创建一个资源，它可以（i）免费提供给每个人；（ii）提供足够的技术深度，为实际成为应用机器学习科学家提供一个起点；（iii）包括可运行代码，向读者展示*如何*解决实践中的问题；（iv）允许快速更新，由我们和整个社区提供；以及（v）由[forum](http://discuss.d2l.ai)进行补充，用于交互式讨论技术细节和回答问题。

这些目标经常发生冲突。方程式、定理和引文最好用乳胶进行管理和布局。代码最好用Python来描述。网页是以HTML和JavaScript为本机语言的。此外，我们希望内容可以作为可执行代码、物理书籍、可下载的PDF以及作为网站在Internet上访问。目前还没有完全适合这些需求的工具和工作流程，所以我们只能自己组装。我们在:numref:`sec_how_to_contribute`中详细描述了我们的方法。我们决定在GitHub上共享源代码并允许编辑，Jupyter笔记本用于混合代码、公式和文本，Sphinx作为生成多个输出的渲染引擎，以及论坛的讨论。虽然我们的体系还不完善，但这些选择在相互竞争的问题之间提供了一个很好的折衷方案。我们相信，这可能是第一本使用这种集成工作流出版的书。

### 在实践中学习

许多教科书教授一系列的主题，每一个都详尽无遗。例如，克里斯·毕晓普（Chris Bishop）的优秀教科书:cite:`Bishop.2006`，把每一个主题都教得非常透彻，以至于要读到线性回归这一章需要大量的工作。虽然专家们喜欢这本书正是因为它的彻底性，但对于初学者来说，这一特性限制了它作为介绍性文本的实用性。

在这本书中，我们将教授大多数概念*及时*。换言之，您将在实现某些实际目的所需的时刻学习概念。虽然我们在一开始要花一些时间来教授基本的预备知识，比如线性代数和概率，但我们希望你在担心更深奥的概率分布之前，先体验一下训练第一个模型的满足感。

除了一些提供基础数学背景速成课程的初步笔记外，接下来的每一章都会介绍一些合理数量的新概念，并提供一个独立的工作示例——使用真实的数据集。这是一个组织上的挑战。在逻辑上，某些模型可能被组合在一个笔记本中。一些想法最好是通过连续执行几个模型来传授。另一方面，坚持“一个工作示例，一个笔记本”的政策有一个很大的优势：这使得您可以尽可能容易地利用我们的代码来启动自己的研究项目。复制一个笔记本，然后开始修改它。

我们将根据需要将可运行代码与背景材料交错。一般来说，在充分解释工具之前，我们常常会在提供工具这一方面犯错误（稍后我们将通过解释背景来跟进）。例如，在充分解释它为什么有用或为什么有效之前，我们可以使用*随机梯度下降*。这有助于给从业者提供快速解决问题所需的弹药，同时要求读者相信我们的一些策展决定。

这本书将从零开始教授深度学习的概念。有时，我们想深入研究模型的细节，这些细节通常是通过深入学习框架的高级抽象对用户隐藏的。特别是在基础教程中，我们希望您了解在给定层或优化器中发生的一切。在这些情况下，我们通常会给出两个版本的示例：一个是我们从头开始实现一切，只依赖于NumPy接口和自动区分；另一个是更实际的示例，我们使用深度学习框架的高级api编写简洁的代码。一旦我们教了您一些组件是如何工作的，我们就可以在随后的教程中使用高级api了。

### 内容和结构

本书大致可分为三个部分，分别以:numref:`fig_book_org`的不同颜色呈现：

![Book structure](../img/book-org.svg)
:label:`fig_book_org`

* 第一部分包括基础知识和预备知识。
:numref:`chap_introduction`提供了深入学习的介绍。然后，在:numref:`chap_preliminaries`中，我们将快速向您介绍动手深度学习所需的先决条件，例如如何存储和操作数据，以及如何基于线性代数、微积分和概率的基本概念应用各种数值运算。:numref:`chap_linear`和:numref:`chap_perceptrons`涵盖了深度学习的最基本概念和技术，如线性回归、多层感知器和正则化。

* 接下来的五章将重点介绍现代深度学习技术。
:numref:`chap_computation`描述了深度学习计算的各种关键组件，并为我们随后实现更复杂的模型奠定了基础。接下来，在:numref:`chap_cnn`和:numref:`chap_modern_cnn`中，我们将介绍卷积神经网络（CNNs），这是构成大多数现代计算机视觉系统主干的强大工具。随后，在:numref:`chap_rnn`和:numref:`chap_modern_rnn`中，我们介绍了递归神经网络（RNN），这种模型利用数据的时间或序列结构，通常用于自然语言处理和时间序列预测。在:numref:`chap_attention`中，我们引入了一类新的模型，它采用了一种称为注意机制的技术，最近它们已经开始取代自然语言处理中的RNN。这些部分将使您了解最新的深度学习应用程序背后的基本工具。

* 第三部分讨论可伸缩性、效率和应用程序。
首先，在:numref:`chap_optimization`中，我们讨论了用于训练深度学习模型的几种常用优化算法。下一章，:numref:`chap_performance`将探讨影响深度学习代码计算性能的几个关键因素。在:numref:`chap_cv`中，我们展示了深度学习在计算机视觉中的主要应用。在:numref:`chap_nlp_pretrain`和:numref:`chap_nlp_app`中，我们展示了如何预训练语言表示模型并将其应用于自然语言处理任务。

### 代码
:label:`sec_code`

本书的大部分章节都以可执行代码为特色，因为我们相信在深度学习中交互式学习体验的重要性。目前，某些直觉只能通过反复试验、小范围调整代码和观察结果来发展。理想情况下，一个优雅的数学理论可能会精确地告诉我们如何调整代码以获得期望的结果。不幸的是，目前，这种优雅的理论还没有出现。尽管我们尽了最大的努力，但仍然缺乏对各种技术的正式解释，这既是因为描述这些模型的数学非常困难，也是因为对这些主题的认真研究最近才进入高潮。我们希望，随着深度学习理论的发展，这本书的未来版本将能够在当前版本无法提供的地方提供见解。

有时，为了避免不必要的重复，我们将本书中经常导入和引用的函数、类等封装在`d2l`包中。对于要保存在包中的任何块（如函数、类或多个导入），我们将用“#@save”标记它。我们在:numref:`sec_d2l`中提供了这些函数和类的详细概述。`d2l`软件包重量轻，仅需要以下软件包和模块作为依赖项：

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
本书中的大部分代码都基于apachemxnet。MXNet是一个用于深度学习的开源框架，是AWS（amazonwebservices）以及许多学院和公司的首选。本书中的所有代码都通过了最新MXNet版本下的测试。但是，由于深度学习的快速发展，一些代码
*在打印版中，*可能无法在将来的MXNet版本中正常工作。
但是，我们计划保持在线版本最新。如果您遇到任何此类问题，请咨询:ref:`chap_installation`以更新您的代码和运行时环境。

下面是我们如何从MXNet导入模块。
:end_tab:

:begin_tab:`pytorch`
本书中的大部分代码都是基于PyTorch的。PyTorch是一个开源的深度学习框架，在研究界非常流行。本书中的所有代码都通过了最新PyTorch下的测试。但是，由于深度学习的快速发展，一些代码
*在打印版中，*可能无法在PyTorch的未来版本中正常工作。
但是，我们计划保持在线版本最新。如果您遇到任何此类问题，请咨询:ref:`chap_installation`以更新您的代码和运行时环境。

下面是我们如何从PyTorch导入模块。
:end_tab:

:begin_tab:`tensorflow`
本书中的大部分代码都是基于TensorFlow的。TensorFlow是一个开源的深度学习框架，在研究界和业界都非常流行。本书中的所有代码都通过了最新TensorFlow下的测试。但是，由于深度学习的快速发展，一些代码
*在打印版中，*可能无法在TensorFlow的未来版本中正常工作。
但是，我们计划保持在线版本最新。如果您遇到任何此类问题，请咨询:ref:`chap_installation`以更新您的代码和运行时环境。

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

这本书是为学生（本科或研究生），工程师和研究人员，谁寻求一个深入学习的实用技术扎实掌握。因为我们从零开始解释每一个概念，所以不需要有深度学习或机器学习的背景。充分解释深度学习的方法需要一些数学和编程知识，但是我们只假设你有一些基础知识，包括线性代数、微积分、概率论和Python编程。此外，在附录中，我们对本书中涉及的大部分数学进行了复习。大多数时候，我们会优先考虑直觉和想法，而不是数学上的严谨。有许多很棒的书可以使感兴趣的读者读得更远。例如，Bela Bollobas :cite:`Bollobas.1999`的《线性分析》对线性代数和函数分析进行了深入的研究。所有的统计数据:cite:`Wasserman.2013`是一个极好的统计指南。如果您以前没有使用过Python，您可能需要仔细阅读这个[Python tutorial](http://learnpython.org/)。

### 论坛

与本书相关，我们在[discuss.d2l.ai](https://discuss.d2l.ai/)开设了一个讨论论坛。当您对本书的任何部分有疑问时，可以在每一章的末尾找到相关的讨论页链接。

## 致谢

我们对数百名投稿人的中英文草稿表示感谢。他们帮助改进了内容并提供了有价值的反馈。特别地，我们要感谢每一位英文稿的撰稿人，感谢他们为大家做得更好。他们的GitHub ID或名称（无特定顺序）：alxnorden、avinashingit、bowen0701、brettkoonce、Chaitanya Prakash Bapat、cryptonaut、Davide Fiocco、edgarroman、gkutiel、John Mitro、Liang Pu、Rahul Agarwal、Mohamed Ali Jamaoui、Michael（Stu）Stewart、Mike Müller、NRauschmayr、Prakar Srivastav、sad、Sfermiger、Sheng Zha、sundeepteki，topecongiro、tpdi、粉丝、Vishaal Kapoor、Vishwesh Ravi Shrimali、YaYaB、Yuhong Chen、Evgeniy Smirnov、lgov、Simon Corston Oliver、Igor Dzreyev、Ha Nguyen、Pmunes、Andrei Lukovenko、senorcinco、vfdev-5、dsweet、Mohammad Mahdi Rahimi、Abhishek Gupta、uwsd、DomKM、Lisa Oakley、Bowen Li、Aursh Ahuja、Prasanth Buddygari、brianhendee，mani2106、mtn、lkevinzc、Caojin、Lakshya、Fite Lüer、Surbhi Vijayvargeya、Muhyun Kim、dennismalmgren、adursun、Anirudh Dagar、liqingnz、Pedro Larroy、lgov、ati ozgur、Jun Wu、Matthias Blume、Lin Yuan、geogunow、Josh Gardner、Maximilian Böther、Rakib Islam、Leonard Lausen、Abhinav Upadhyay、rongruosong、Steve Sedlmeyer、Ruslan Baratov、RafaelSchlater、liusy182、Giannis Pappas、ati ozgur、qbaza、dchoi77、Adam Gerson、Phuc Le、Mark Atwood、christabella、vn09、Haibin Lin、jjangga0214、RichyChen、noelo、hansent、Giel Dops、dvincent1337、WhiteD3vil、Peter Kulits、codypenta、joseppinilla、ahmaurya、karolszk、Heytile、Peter Goetz、rigtorp、Tiep Vu、sfilip、mlxd、Kale ab Tessera、，Sanjar Adilov、MatteoFerrara、hsneto、Katarzyna Biesialska、Gregory Bruss、Duy–Thanh Doan、paulaurel、graytowne、Duc Pham、sl7423、Jaedong Hwang、Yida Wang、cys4、clhm、Jean Kaddour、austinmw、trebeljahr、tbaums、Cuong V.Nguyen、pavelkomarov、vzlamal、NotatherSystem、J-Arun-Mani、jancio、eldarkurtic、the great shazbot、doctorcolossus、，gducharme、cclauss、Daniel Mietchen、hoonose、biagiom、abhinavsp0730、Jonathanrandall、ysraell、Nodar Okroshiashvili、UgurKap、Jiyang Kang、StevenJokes、Tomer Kaftan、liweiwp、netyster、ypandya、NishantTharani、heiligerl、SportsTHU、Hoa Nguyen、manuel arno korfmann webentwicklung、aterzis personal、nxby、Xiaoting He、Josiah Yoder、mathresearch、，mzz2017，JROBERAYALS，iluu，ghejc，BSharmi，vkramdev，SIMONFORGJONES，LakshKD，TalNeoran，djliden，Nikhil95，Oren Barkan，GUOWIS，haozhu233，pratikhack，315930399，tayfununal，steinsag，charleybeller，Andrew Lumsdaine，JIEQUII Zhang，Deepak Pathak，Florian Donhauser，Tim Gates，ADRIAN Tijsseling，Ron Medina，Gaurav Saha，Murat Semerci，[Lei Mao](https://github.com/leimao)。

我们感谢亚马逊的网络服务，特别是斯瓦米Sivasubramanian，拉朱古拉巴尼，查理贝尔，和安德鲁贾西慷慨支持写这本书。如果没有可用的时间、资源、与同事的讨论和持续的鼓励，这本书就不会发生。

## 摘要

* 深度学习已经彻底改变了模式识别，引入了一系列技术，包括计算机视觉、自然语言处理、自动语音识别。
* 要成功地应用深度学习，您必须了解如何投射问题、建模的数学、将模型与数据拟合的算法以及实现这一切的工程技术。
* 这本书提供了一个全面的资源，包括散文，数字，数学和代码，所有在一个地方。
* 要回答与本书相关的问题，请访问我们的论坛https://discuse.d2l.ai/。
* 所有笔记本都可以在GitHub上下载。

## 练习

1. 在本书的论坛上注册一个帐号[discuss.d2l.ai](https://discuss.d2l.ai/)。
1. 在计算机上安装Python。
1. 按照本节底部的链接到论坛，在那里你将能够寻求帮助和讨论这本书，并通过与作者和更广泛的社区接触找到问题的答案。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/18)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/20)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/186)
:end_tab:
