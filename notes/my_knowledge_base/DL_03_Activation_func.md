# 激活函数 Activation Function

## 1. Why we need activation functions?

> reference: [Zhihu: 神经网络激励函数的作用是什么？有没有形象的解释？](神经网络激励函数的作用是什么？有没有形象的解释？)

- 没有activation function就不能非线性分类
  - 如果没有激励函数，那么神经网络的权重、偏置全是线性的**仿射变换（affine transformation）**
- 用非线性函数来进行分类
  - 激活函数变换后，数据在新空间下线性可分



## 2. Attributes

- **可微性**：计算梯度时必须要有此性质。
- **非线性**：保证数据非线性可分。
- **单调性**：保证凸函数。
- **输出值与输入值相差不会很大**：保证神经网络训练和调参高效