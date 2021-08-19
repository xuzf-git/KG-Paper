# Combining Word and Entity Embeddings for Entity Linking

论文地址：[https://perso.limsi.fr/bg/fichiers/2017/combining-word-entity-eswc2017.pdf](https://perso.limsi.fr/bg/fichiers/2017/combining-word-entity-eswc2017.pdf) (ESWC 2017)

## 1 Abstract

论文针对 EL 的第二个阶段 candidate selection 提出了新的方法——在同一空间中，联合学习文本中的 word 和知识库 entity 的嵌入表示。 *candiadate selection 是从候选实体中选出最优实体*

作者提出该方法的优势有以下三点：

1. word embedding 更靠谱，因为在训练时它的上下文可能会包含语义向量，而非以往浅层的词语排列
2. entity embedding 是在大规模文本语料库上学习的，这比只在知识库上学习 embedding 在训练时，能有更高的出现频率。
3. 由于是在同一个空间学习的 embedding，所以可以使用简单的相似度来计算 mention context 和 entity entry 之间的距离

论文的贡献主要有以下几点

* EAT 模型联合学习单词和实体嵌入；
* 一个整合 EAT 模型的全局实体链接 pipeline；
* 使用 TAC 2015 “Entity Discovery and Linking” 对模型进行评估。结果 $P(all) =0.742$ 优于 non-EAT baseline。

## 2 Combining Word and Entity Embedding

### 2.1 Extended Anchor Text

为了将 word embedding 和 entity embedding 包含在同一个空间中，作者引入了扩展锚文本 EAT 的概念。EAT 就是将锚文本 $a_i = (w_i, e_j)$ 替换成 $a_{i}^{'} = (w_{i}^{'}, e_j)$ 当 $w_i$ 不为空时，$w_{i}^{'} = w_i$ 否则，$w_{i}^{'}$ 就是 $e_j$ 中词语构成的集合。句子中的 EAT 示例如下图所示：

![image-20210819130029068](https://i.loli.net/2021/08/19/29ghYW1vTLcOIGF.png)

### 2.2 The EAT Model

引入 EAT 模型的目标是：**在同一个向量空间中表示 word 和 entity 的嵌入表示**， 类似于 word2vec 中的设置，作者定义了语料库中两个元素之间的条件概率，如下式：
$$
p(c_o | c_i) = \sum_{f_o \in c_o} \sum_{f_i \in c_i} \frac{exp(f_o^T f_i)}{\sum_{j=1}^{|\mathbb F|}exp(f_j^Tf_i)}
$$
其中，f 表示 word 或 entity 的向量表示； c 表示语料库中的 word 序列或 EAT；

当 $c$ 是一个 word 时，上面的式子就与 word2vec 完全相同了，因此通过 EAT 模型进行 word 和 entity 的表示学习只需要调整语料库即可，以下图中的例子说明：

![image-20210819131651596](https://i.loli.net/2021/08/19/dDvVGMyQiwzhreb.png) 

这样在扩展的语料库上，直接用 skip-gram 或 CBOW 就可以进行 word 和 entity 的表示学习了。

## 3 Entity Linking System

### 3.1 Generation of Candidate Entities

作者使用了启发式方法 mention 进行扩展，在文档中找到与 mention 相近的变体形式，规则如下：

1. 如果 mention 是首字母缩写，那么查找文档中其他的 mention，要求实体类型相同，且首字母与其匹配
2. 在文档中查找当前 mention 作为其子串的其他 mention

然后作者使用以下策略生成候选实体：（以下 mention 表示本来的 mention 和上一步得到的变体形式）

1. mention 与 entity 形式相同
2. mention 与 entity 的变体（翻译、别名）相同
3. mention 是 entity name 的子串
4. 根据 Levenshtein 距离计算相似度，小于 k (取为 3) 的 entity；为了提高效率，作者使用了 BK-tree
5. 使用信息检索模型：tf-idf 特征，Lucene 作为搜索引擎

### 3.2 Selection of the Best Candidate Entity

该步骤是从 candidate 集合中选出最优的 entity 作为最终的链接目标，作者设计了四种特征，作为candidate 和 mention 的相似度度量，如下所示：

1. candidate generation 阶段的特征集合，如简单匹配策略的二元特征、信息检索模型的相似度等
2. mention context 和 entity context 之间的两个文本相似度特征（Textual similarity scores）
3. Wikipedia 中 mention 和 entity 之间的链接计数（使用 log 归一化）
4. EAT embedding 的四个相似度分数（Similarity scores based on EAT embeddings）

**Textual similarity scores**

mention 记为 $q$ ; entity candidate 记为 $e$ ；定义三种文本如下： $q$ 出现的文档记为  $d(q)$，与 $e$ 对应的 Wikipedia 页面为 $w(e)$ ，知识库中与  $e$ 有关系的实体集合  $r(e)$ ；使用 tf-idf 计算出这三种文本的向量表示，例如：$d(q) = (d1, . . . , dn)$ 其中 $di =tf(ti,d)×idf(ti)$ ；文本相似度得分如下：
$$
sim_d(q,e) = cos(d(q), w(e))\\
sim_r(q,e) = cos(d(q), r(e))
$$
**Similarity scores based on EAT embeddings**

$$
EAT_{1}(e, p(q))=\frac{\sum_{w_{i} \in p(q)} \cos \left(\boldsymbol{e}, \boldsymbol{w}_{i}\right)}{\|p(q)\|} \quad EAT_{2}(e, p(q))=\cos \left(\boldsymbol{e}, \frac{\sum_{w_{i} \in p(q)} \boldsymbol{w}_{i}}{\|p(q)\|}\right)
$$

$$
EAT_{3}(e, p(q))=\frac{\sum \operatorname{argmax}_{w_{i} \in p(q); i=1 \ldots k} \cos \left(\boldsymbol{e}, \boldsymbol{w}_{i}\right)}{k} \quad E A T_{4}\left(e, w_{m}\right)=\cos \left(\boldsymbol{e}, \boldsymbol{w}_{m}\right)
$$

其中 $p(q)$ 指  $q$  的上句、当前句、下句；粗体表示 embedding；$EAT_3$ 表示最相似的前 k 个词语的相似度均值。作者将 k 取为 3。

**Classifier trained for candidate selection**

训练一个二分类器判断是否为正确的 entity；训练的正样本为 mention-correct entity；训练的负样本为错误 candidate，但是由于正负样本分布很不平衡，所以用下采样的方法将负样本限制为正样本的十倍。

作者将*没有 candidate generation*和*所有 candidate 都被分类器拒绝*的mention视为NIL。

作者尝试了几种二分类器，最终选择 Adaboost 结合决策树的模型。

