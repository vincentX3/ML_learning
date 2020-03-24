# 机器学习面试简答题小结

> 个人搜集，结合网上资料作答，仅供分享、参考。

## 目录

### 1. SVM与LR的区别

- 模型本质：
假设样本集线性可分，对于**linear SVM**与**LR**，其本质在于对**hyperplane**的约束不同。**SVM**要求取得**max margin**。

- 敏感程度：
LR考虑全部样本，SVM只有**support vector**决定。

- 拟合能力：
LR容易欠拟合，准确度低；
SVM不太容易过拟合：松弛因子+损失函数形式。

- 应用场景：
LR模型基于统计易解释，计算复杂度低；
SVM适合高维稀疏，少样本。

- 处理分类问题能力不同：
LR可以直接进行多类别分类。
SVM只能处理二类分类问题，如果要处理多类别分类，需要进行 one VS one 或one VS all建模。

### 2. XGBOOST 与 GDBT 对比

> 关于**XGBOOST**，官网的[tutorial](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)阐述的简明易懂，推荐阅读。

1. GBDT是机器学习算法，XGBoost是该算法的工程实现。
2. 在使用CART作为基分类器时，XGBoost显式地加入了正则项来控制模
型的复杂度，有利于防止过拟合，从而提高模型的泛化能力。
3. GBDT在模型训练时只使用了代价函数的一阶导数信息，XGBoost对代
价函数进行二阶泰勒展开，可以同时使用一阶和二阶导数。
4. 传统的GBDT采用CART作为基分类器，XGBoost支持多种类型的基分类
器，比如线性分类器。
5. 传统的GBDT在每轮迭代时使用全部的数据，XGBoost则采用了与随机
森林相似的策略，支持对数据进行采样。
6. 传统的GBDT没有设计对缺失值进行处理，XGBoost能够自动学习出缺
失值的处理策略。


### References
感谢各位作者、分享者：
- 《百面机器学习》
- https://www.jianshu.com/p/ace5051d0023