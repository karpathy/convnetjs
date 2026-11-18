# ConvNetJS

[English](Readme.md) | [简体中文](Readme.zh.md)

ConvNetJS 是一个纯 JavaScript 实现的深度学习与神经网络库，配备了直观生动的浏览器端交互式可视化演示。目前支持的功能包括：

- 通用**神经网络模块**（全连接层、非线性激活函数等）
- 分类任务（SVM、Softmax）与回归任务（L2 损失）的**损失函数**
- 支持构建与训练处理图像数据的**卷积神经网络（CNN）**
- 基于深度 Q 学习（Deep Q-Learning）的实验性**强化学习（RL）**模块

更多详细信息与完整文档，请访问官方项目主页：[convnetjs.com](http://convnetjs.com)。

> **注意**：原作者目前因精力有限不再积极维护 ConvNetJS，npm 上的历史包版本可能已年久未更新，建议直接基于源码或打包脚本构建使用。

## 在线演示 (Online Demos)

- [基于 MNIST 手写数字数据集的卷积神经网络](http://cs.stanford.edu/~karpathy/convnetjs/demo/mnist.html)
- [基于 CIFAR-10 数据集的卷积神经网络图像分类](http://cs.stanford.edu/~karpathy/convnetjs/demo/cifar10.html)
- [二维玩具数据（Toy 2D Data）决策边界分类可视化](http://cs.stanford.edu/~karpathy/convnetjs/demo/classify2d.html)
- [一维玩具数据（Toy 1D Regression）拟合回归演示](http://cs.stanford.edu/~karpathy/convnetjs/demo/regression.html)
- [基于 MNIST 手写数字数据集训练自编码器（Autoencoder）](http://cs.stanford.edu/~karpathy/convnetjs/demo/autoencoder.html)
- [基于深度 Q 学习（DQN）的 2D 智能体强化学习演示](http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html)
- [图像像素坐标回归（“AI 绘画”涂色过程）](http://cs.stanford.edu/~karpathy/convnetjs/demo/image_regression.html)
- [在 MNIST 上对比 SGD、Adagrad 与 Adadelta 优化器收敛效果](http://cs.stanford.edu/people/karpathy/convnetjs/demo/trainers.html)

## 代码范例 (Example Code)

以下是定义一个包含 20 个隐藏神经元的 **2 层神经网络**并在单个数据点上进行训练的极简范例：

```javascript
// 定义一个包含一个拥有 20 个神经元隐藏层的 2 层神经网络
var layer_defs = [];
// 输入层声明输入张量维度。此处为二维数据：
// ConvNetJS 底层统一基于三维张量 Vol(sx, sy, depth) 运算；
// 当处理非图像数据时，前两个空间维度 (sx, sy) 始终固定为 1
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
// 声明包含 20 个神经元的隐藏层，并接入 ReLU 非线性激活函数
layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
// 在隐藏层之上构建输出类别为 10 的 Softmax 线性分类器
layer_defs.push({type:'softmax', num_classes:10});

var net = new convnetjs.Net();
net.makeLayers(layer_defs);

// 将一个随机数据点输入网络进行前向传播
var x = new convnetjs.Vol([0.3, -0.5]);
var prob = net.forward(x);

// prob 输出为一个 Vol 对象。Vol 对象包含存储权重数据的 .w 字段和存储梯度的 .dw 字段
console.log('probability that x is class 0: ' + prob.w[0]); // 打印约 0.50101

var trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, l2_decay:0.001});
trainer.train(x, 0); // 训练网络，指定该样本的目标分类标签为 0

var prob2 = net.forward(x);
console.log('probability that x is class 0: ' + prob2.w[0]);
// 输出 0.50374，略高于训练前的 0.50101：
// Trainer 优化器已根据反向传播梯度完成对网络权重的更新调整，
// 提升了目标训练类别（类别 0）的预测概率
```

以下是用于图像预测的小型**卷积神经网络（CNN）**架构定义示例：

```javascript
var layer_defs = [];
// 声明输入图像尺寸为 32x32，通道数为 3（RGB）
layer_defs.push({type:'input', out_sx:32, out_sy:32, out_depth:3});
// 此时输出张量尺寸为 32x32x3

// 卷积层：包含 16 个 5x5 大小的卷积核，步长 1，全边填充 2，接入 ReLU 激活
// 四周各填充 2 像素以保证输出特征图尺寸保持一致
layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
// 此时输出特征图尺寸为 32x32x16

// 池化层：2x2 窗口，步长 2 进行下采样
layer_defs.push({type:'pool', sx:2, stride:2});
// 此时输出张量尺寸降至 16x16x16

layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
// 此时输出张量尺寸为 16x16x20

layer_defs.push({type:'pool', sx:2, stride:2});
// 此时输出张量尺寸降至 8x8x20

layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
// 此时输出张量尺寸为 8x8x20

layer_defs.push({type:'pool', sx:2, stride:2});
// 此时输出张量尺寸降至 4x4x20

layer_defs.push({type:'softmax', num_classes:10});
// 最终输出张量尺寸为 1x1x10

net = new convnetjs.Net();
net.makeLayers(layer_defs);

// 库中内置了将 DOM 图像元素直接转换为 Vol 张量的便捷辅助函数
var x = convnetjs.img_to_vol(document.getElementById('some_image'));
var output_probabilities_vol = net.forward(x);
```

## 快速入门与文档 (Getting Started)

- 项目主页提供了完整的[新手快速入门教程](http://cs.stanford.edu/people/karpathy/convnetjs/started.html)；
- 详尽的 API 接口规范可参阅[完整技术文档](http://cs.stanford.edu/people/karpathy/convnetjs/docs.html)；
- 预构建与压缩版库文件可在本项目的 **Releases** 页面下载获取，亦可直接访问以下链接获取预编译单文件：
  - [convnet.js](http://cs.stanford.edu/people/karpathy/convnetjs/build/convnet.js)（完整源码构建版本）
  - [convnet-min.js](http://cs.stanford.edu/people/karpathy/convnetjs/build/convnet-min.js)（生产混淆压缩版本）

## 源码编译：从 src/ 构建至 build/

如需对代码库进行修改或新增自定义算子特性，可在修改 `src/` 源码后将其重新编译打包至 `build/` 目录中。编译构建流水线会将 `src/` 中的源文件合并拼接并进行自动化压缩。

项目构建基于 Apache Ant 任务：通过合并 `src/` 下各源码文件生成 `build/convnet.js`，随后通过 YUI Compressor 混淆压缩生成 `build/convnet-min.js`。请确保系统已安装 **ant**（在 Ubuntu 环境下可通过 `sudo apt-get install ant` 极速安装），随后进入 `compile/` 目录并执行：

```bash
$ ant -lib yuicompressor-2.4.8.jar -f build.xml
```

编译生成的目标打包文件将自动输出至 `build/` 目录中。

## 在 Node.js 环境中使用

本项目亦可在 *Node.js* 服务端与命令行环境中运行：

1. 安装依赖包：`$ npm install convnetjs`
2. 在代码中引入：`var convnetjs = require("convnetjs");`

## 许可证 (License)

本项目采用 [MIT 许可证](LICENSE) 开源。
