# 基于pysyft的联邦学习

[项目地址](https://github.com/zyplanet/FederatedLearningUsingPysyft)

## 前言

### 联邦学习

联邦学习可以视为一种加密的分布式学习技术，它的核心是分布式学习算法与同态加密。分布式学习算法使得它能够从多方数据源获得模型的更新，同态加密则保证了中间过程的完全隐私性。

这里有一些便于理解的教程：

* [Federated Learning: Collaborative Machine Learning without Centralized Training Data](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
* [Building better products with on-device data and privacy by default]([https://federated.withgoogle.com](https://federated.withgoogle.com/))

### pysyft

pysyft是为安全、隐私深度学习而编写的python库，通过对pytorch深度学习框架增加新的特性来进行，它支持联邦学习、差分隐私以及多方计算。该项目由OpenMined负责，DropoutLabs、UDACITY等组织也参与其中的建设，github地址：[pysft](https://github.com/OpenMined/PySyft)

## 关于

本项目由浙江大学VAG小组的刘同学编写，基于pysyft实现了联邦学习框架下的对MNIST数据集分类任务，主要目的是供小组成员学习参考，代码编写较为规范，拥有良好的可扩展性。

### 项目组织与说明

```
|-- checkpoints/
|-- data/
|	|-- __init__.py
|	|__ dataset.py
|-- datasets/
|-- logs/
|	|__logging.txt
|-- models/
|	|-- __init__.py
|	|-- BasicModule.py
|	|__ models.py
|__ utils/
|	|-- __init__.py
|	|__ tools.py
|-- config.py
|-- main.py
|-- requirements.txt
|__ README.md
```

* checkpoints/: 存储模型参数
* data/: 数据集定义、预处理
* datasets/: 原始数据集
* logs/: 存储训练可视化产生的数据
* models/: 定义模型
* utils/: 放置需要的自定义工具函数
* config.py: 参数配置文件
* requirements.txt: 本项目的依赖库

### 使用

安装依赖库

```
pip install -r requirements.txt	#在终端中使用以通过pip与项目文件requirements.txt安装所需依赖
```

本项目使用Facebook提供的科学可视化工具[visdom](https://github.com/facebookresearch/visdom)进行训练/测试可视化

```
visdom	#在终端中使用以打开visdom，按照提示从浏览器进入其界面
```

`main.py`内置`train(), val(), help()`函数

```
python main.py help		#在终端中使用帮助函数，以查看使用方法
python main.py train	#在终端中使用训练函数，以默认参数配置开始联邦学习下的训练过程
```

