# Network Representation Learning: A Survey 个人总结

## 1. 概述
按照why、what、how的方式介绍NRL，以如下：
|1|unsupervised / semi-supervised |
|---|---|
|2| structure preserving / content augmented |
|3| network properties focus|
三层的方式对对NRL方法进行划分介绍。
并介绍了NRL的应用、相关数据集、未来方向等。

## 2. 贡献
提供了新的分类方法并对NRL近年的发展进行了回顾。

## 3. 内容选录
### 3.1 Introduction
现实中大量数据呈现图的结构，但缺乏有效的研究手段。NRL通常希望将图结构数据**映射**为**低维vector**，从而可作为特征(所谓*embedding*)供downstream task使用。  
然而面临如下困难：
- structure-preserving
- content-preserving
- data sparsity
- scalability

概括来说，即
1. embedding 时如何减少图的信息损失
2. 算法如何适应现实中**大量且稀疏**的数据

### 3.2 Notations and Definitions
补充了由local至global的network proximity定义，为后继分类标准提供说明。

### 3.3 Categorization
按照上述分类方式逐个介绍NRL模型。

- unsupervised
主流路线有2：
1. 基于矩阵分解，但scalability堪忧；
2. 使用DeepWalk类技巧，用random walk抽样后套入skip-gram模型学习

- semi-supervised
在unsupervised模型基础上，
1. 加入classifier（在Loss中引入label的判决）
2. modeling vertex relation
3. joint vertex label embedding

(2、3的方法不太理解)

### 4. application等
与GNN相近。（毕竟是先祖hh）