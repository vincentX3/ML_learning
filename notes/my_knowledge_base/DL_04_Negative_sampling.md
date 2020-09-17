## 负采样 Negative Sampling

> Overview:
>
> **Skip-gram**模型在面对**大量**语料（高位词向量矩阵），为了更有效地**更新负样本权重**，选择每次**采样**少量负样本进行更新。



## 1.  Background- Skip-gram model

- Word2Vec

  Word2Vec是从大量文本语料中以**无监督**的方式学习语义知识的一种模型，通过学习文本来用**词向量**的方式表征词的语义信息。其本质是通过神经网络学习一个**映射**，将单词从原先所属的空间（常为one-hot 编码）映射到新的低维空间中，即**embedding**。

- Skip-Gram

  **Word2Vec**模型中，主要有**Skip-Gram**和**CBOW**两种模型，Skip-Gram是用中心词来预测上下文。

  <img src="https://paperswithcode.com/media/methods/Screen_Shot_2020-05-26_at_2.04.55_PM.png" alt="skipgram" style="zoom:50%;" />

  为了理解Negative Sampling，我们需要理解Skip-gram的输入，输出及正负样本。

  不妨假设语料库中有句子:

  > Kobe is the greatest player of all time.

  我们前面提到，模型是**用中心词来预测上下文**。

  不妨设中心词为**player**，我们引入参数`skip_window=2`，用其控制上下文区间。对于中心词**player**，其上下文为`['greatest', 'player'], ['player', 'of']`。

  **Input**：

  即中心词**player**，一般用*one-hot*编码的向量表示。

  **Output**：

  预测上下文词的概率表示。若词表长度为`n`，则输出`n`维向量，每维表示对应的词的概率。

  **正样本**：

  对于**player**，其正样本即为上下文包括的词，有`['greatest', 'of']`

  **负样本**：

  除了正样本外的词皆为负样本，如`['Kobe', 'all', 'time']`

  

## 2. Negative Sampling

通过上述对**负样本**的定义，我们不难想象，当语料库**巨大**时（如百万词），每轮训练对全部的负样本的权重矩阵进行更新是不现实的。

很自然的，我们想到通过采样的方式，选取一小部分负样本，更新其对应权重矩阵，较为合理。

接下来的问题数，负样本如何选取。

Word2Vec中，作者根据unigram distribution来进行，也就是根据单词的概率，出现概率高的单词容易被选为负样本，提出公式如下：

$P\left(w_{i}\right)=\frac{f\left(w_{i}\right)}{\sum_{j=0}^{n} f\left(w_{j}\right)}$

上面的公式表示，$f()$为统计单词出现的次数。用单词$i$出现次数，除以所有单词出现的次数之和。



**One more thing**

Q：为什么我们要选出现概率高的单词作为负样本呢？

A：如果词频高的词经常出现在正样本，那么模型就会很容易记住这些正样本，那么就会偷懒/收敛成 每次都判断这些高频热点词为正，所以这时候，负采样高频词就是在适当的时候告诉模型“不是每一次这些词都是正例哦”。



## Reference

1. https://medium.com/@makcedward/how-negative-sampling-work-on-word2vec-7bf8d545b116
2. https://zhuanlan.zhihu.com/p/39684349
3. https://zhuanlan.zhihu.com/p/56106590

