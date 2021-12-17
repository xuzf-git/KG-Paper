# Collective Annotation of Wikipedia Entities in Web Text

这篇论文是实体消歧领域中全局消歧方法的开山之作。

## Abstract

作者提出了一个通用的集体消歧的方法。基于“文档中的提及应该对应于知识库中相关的topic或domain的实体”的假设，作者给出了局部消歧特征的 “mention-entity compatibility“ 和全局消歧特征的 “global coher- ence between entities” 的权衡公式。

优化全局的实体分配是 NP-hard 问题。我们研究了基于局部爬山、舍入整数线性规划和预聚类实体以及簇内局部优化的实际解决方案。

## 1 Introduction

## 相关工作

**M&W**：提出一种全局消歧的有限形式。提出两个实体之间的相关性分数，以及 “上下文提及” 的概念。在对一个 mention 进行消歧时，引入 “上下文提及” 一致性分数作为给候选实体打分的一个特征。因此在一定程度上考虑了实体间的一致性问题。

***Cucerzan’s algorithm***：意识到文本中提及的实体标签存在普遍的依赖关系的第一人。使用高维向量表示实体，计算整个文档所有mention的候选实体集合的平均向量，引入候选实体与平均向量的点积特征，作为候选实体的全局一致性分数。

一致性 example：

> Our guiding premise is that documents largely refer to topically coherent entities, and this “coherence prior” can be exploited for disambiguation. While Michael Jordan and Stuart Russell can refer to seven (basketball player, foot- baller, actor, machine learning researcher, etc.) and three (politician, AI researcher, DJ) persons respectively in Wiki- pedia (as of early 2009), a page where both Michael Jordan and Stuart Russell are mentioned is almost certainly about computer science, disambiguating them completely.

作者提出了一个全局优化方法，组合了来自所选提及和知识库实体的兼容性（局部特征）和全局页面级主题连贯性（全局特征），以消除所有点的歧义 。作者将优化问题等价为 “在概率图形模型中搜索最大概率标注配置”。推理是 NP-hard 的。作者提出了基于局部爬山和线性规划松弛的启发式算法。