# 论文分享《Overlapping Community Detection with Graph Neural Networks》

> 论文链接：https://arxiv.org/pdf/1909.12201.pdf
>
> 论文会议：**DLG 2019**

## 1. Abstract

### 1.1 What do they do
面向**重叠社区**问题，结合 GNN 和 Bernoulli-Poisson 概率模型进行建模。

### 1.2 What's amazing points

### 1.3 Learning model
- graph：attributed graph
- supervised learning
- task：overlapped community detection
- learning model: GNN based

## 2. Motivation
当前缺少进行**重叠社区发现**的工作。

## 3. Model

### 3.1 前导简介
1. Affiliation Matrix
	$F \in \mathbb{R}_{\geq 0}^{N \times C}$, 每列表示对应节点属于该社区的强度。用以表示节点属于多社区的结果。
	
2. Bernoulli-Poisson

   伯努利-泊松模型是一个重叠社团的图生成模型。对于给定的隶属关系矩阵 $F$，可以得到邻接矩阵 Auv 采样为：

   

   $$A_{u v} \sim Bernoulli \left(1-\exp \left(-F_{u} F_{v}^{T}\right)\right)$$

   

   其中，行向量 Fu 表示节点 u 的社团隶属关系。从上式可以看出节点 u 和节点 v 之间属于相同社团的数量越多说明两节点之间存在边的概率越高。

### 3.2 模型架构

两层GCN：

$$F:=\operatorname{GCN}_{\theta}(A, X)=\operatorname{ReLU}\left(\hat{A} \operatorname{ReLU}\left(\hat{A} X W^{(1)}\right) W^{(2)}\right)$$



对于伯努利-泊松模型的负对数似然函数为：



$-\log p(A \mid F)=-\sum_{(u, v) \in E} \log \left(1-\exp \left(-F_{u} F_{v}^{T}\right)\right)+\sum_{(u, v) \notin E} F_{u} F_{v}^{T}$



由于现实中的图是非常稀疏的，因此上式中的第二项对于损失函数的贡献较大。可以通过不平衡分类的方法来消除这两项的不平衡：

$\mathcal{L}(F)=-\mathbb{E}_{(u, v) \sim P_{E}}\left[\log \left(1-\exp \left(-F_{u} F_{v}^{T}\right)\right)\right]+\mathbb{E}_{(u, v) \sim P_{N}}\left[F_{u} F_{v}^{T}\right]$

其中 $P_E$和 $P_N$ 分别表示边存在或者不存在的均匀分布。



其它论文中的方法通常是直接优化隶属矩阵 $F$ , 本文中直接通过最小化对数似然函数的值来调整神经网络的参数：

$\boldsymbol{\theta}^{\star}=\underset{\boldsymbol{\theta}}{\arg \min } \mathcal{L}\left(\operatorname{GNN}_{\boldsymbol{\theta}}(A, X)\right)$



## 4. 实验

1. 数据集

   - 标准的 Facebook 数据集
   - 作者自己提供的四个数据集：Chemistry、Computer Science、Engineering 和 Medicine。

2. 相关参数

   对于生成的隶属矩阵，作者采用隶属阈值 $ρ=0.5$ 来判定是否属于某社团。对于评价指标，文中仅使用了重叠的 NMI 值来对比不同模型的结果。NOCD-G 表示以邻接矩阵 $A$作为节点属性输入，NOCD-X 表示以原节点属性值作为输入。

3. 实验结果

   NMI 比较

   ![image-20201121084217604](.\assets\NOCD_01.png)

   GNN重要性比较

   ![](.\assets\NOCD_02.png)

## 5. 后续
经更多了解，本工作中讨论的矩阵$F$，实际是源自[BIGCLAM](https://www.researchgate.net/publication/262272761_Overlapping_community_detection_at_scale_A_nonnegative_matrix_factorization_approach) 的设计。