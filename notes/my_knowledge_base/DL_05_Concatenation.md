# Concatenation 连接

> Overview: Concatenation操作保留feature vectors**全部信息**，尽量少引入先验知识，但其同时增加了后续的存储、计算开销。



## What is it?

Concatenation，如其名，连接，常用于向量间的操作，数学符号记为$||$。

在*pytorch*中有：

Concatenates the given sequence of `seq` tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.

Example:

```python
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 0)
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 1)
tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
         -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
         -0.5790,  0.1497]])
```

## Why it Works? When to Use?

以`concate`,`sum`为例做对比，考虑以下场景：

你和朋友在一起，每个人都有一些钱。`concate`意味着如实写出每个人拥有多少的钱，`sum`则经过统计，给出钱的总数。

如上例子，说明`concate`

- Strength
  - **Keeping every information where they are**，它提供了原始数据的全部信息，没有变换数据；
  - 其他`sum`,`mean`等aggregator意味着损失信息，在不同的任务场景中选择合适的aggregator未必容易。
- Weakness
  - **Memory Cost**，保留全部信息意味着空间开销。

## Reference

1. https://www.quora.com/What-is-the-theory-behind-the-concatenation-vs-summation-of-2-tensors-in-deep-learning-How-does-this-empirically-relate-to-information-passed
2. https://pytorch.org/docs/stable/generated/torch.cat.html