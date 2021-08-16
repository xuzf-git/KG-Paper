# Neural Entity Linking: A Survey of Models Based on Deep Learning

* 论文地址：[https://arxiv.org/abs/2006.00575](https://arxiv.org/abs/2006.00575)

## Abstract

1. 论文总结了神经实体链接通用架构的各个阶段（candidate generation, entity ranking）的典型方法。

2. 论文对各种各样的EL变形架构进行了分类，大致分成以下几个主题：
   * 联合实体识别和实体链接
   * 全局链接模型
   * 领域无关技术：零样本学习、远程监督方法
   * 跨语言方法
3. 论文简单讨论了神经实体链接的应用

## 1 Introduction

知识库中包含大量的各类实体的丰富信息，实体链接将 mention context 和 entity 进行匹配，从而实现各种语义应用。实体链接的任务是识别出文本中的 entity mention 并建立起到知识库条目的链接，将非结构化数据连接到结构化数据。

实体链接是很多信息抽取（IR）和自然语言理解（NLU）pipeline 中的重要组件，因为它消除了文本中 mention 的歧义，并确定了正确含义。

## 2 Task Description

实体链接分成“实体识别”、“实体消歧”两个阶段。实体消歧时，由于知识库很大，因此训练数据分布将极其不平衡，很多实体都没有在训练集中出现过；尽管知识库很大，但是也有一些没有合适实体对应的 mention （unlabeled mention）。EL 任务根据 mention context 的范围，分为局部 EL 和全局 EL，局部 EL 选择 mention 附近的上下文作为消歧的依据；全局 EL 以整个 document 为上下文。

实体链接在一些文献中，也被称为维基化（Wikification）实体消歧（entity disambiguation）；目前，只有少数研究提出了联合执行 ER 和 ED 的模型，大多数的论文只关注了 ED。

## 3 Neural Entity Linking

### 3.1 General Architecture

一些基于神经网络的 EL 模型将 EL 视为了多分类任务（每个实体对应一个类别）巨大的类别数目使得模型在不进行任务共享的时候，性能表现不够理想；该问题的简化方案是将 EL 问题视为排序问题，pipeline 如下所示：

![image-20210811171045058](https://i.loli.net/2021/08/16/rAPhgZfpNSDWFXv.png)

具体流程如下：

1. 实体识别：从文本中检测出 mention 的边界
2. 实体消歧：
   1. 候选实体生成：为 mention 生成可能的实体；
   2. 实体排序：例如，将每个 Candidate 编码成向量，与 mention context 进行匹配打分；
   3. NIL [optional]：确定不可链接的实体的引用；

#### 3.1.1 Candidate Generation

候选实体生成就是为 mention 提供一些可能的实体，EL 与 WSD（词义消歧）的区别在于 WSD 中每个词语都有一个 WordNet，但是 EL 中没有这样的 mention - KBs's entity 的映射，所以 EL 的搜索空间巨大，为了缩小搜索空间，需要使用 Candidate Generation 对实体集进行初步筛选。

有三类典型的算法：surface form matching、dictionary lookup、prior probability computation

1. surface matching：一些启发式算法来描述 mention 的形式或 matching 的标准，如 Levenshtein 距离、n-gram和归一化；这种方法对于 “Big Blue - IBM”这样的例子就不 work，因为 entity name 不包含 mention 的字符串。
2. dictionary lookup：通过知识库元数据（如消歧/重定向网页链接）构造的别名字典能有效提高候选实体的召回率。例如 YAGO 数据集，通过 Wikipedia 和 WordNet 自动构建，在众多关系中，YAGO 提供了mention 到 entity 的 “means” （“同义”）关系数据集，可以用于 candidate generation
3. prior probability computation：预先计算给定 mention 选择某个 entity 的概率 $p(e|m)$​​​，大部分研究都是基于 Wikipedia 的锚链接计算的先验概率；另一个广泛使用的数据集是 Cross-Wiki，这是一个通过计算网络爬虫数据中的 mention-entity 链接频率得到的大规模数据集。

以上三种 candidate generation 方法的效果如下表所示：

![image-20210811205805882](https://i.loli.net/2021/08/16/3kYCOJPRdXbst6z.png)

zero-shot 模型在不引入外部先验知识的情况下，实现 candidate generation。

#### 3.1.2 Context-mention Encoding

 使用 encoder 网络将 mention 映射到一个低维稠密的上下文向量空间中。早期的 EL 模型中，使用卷积或者 candidate context embedding 和 mention context embedding 之间的 attention 进行编码；近期的模型中有两种方法占了上风：recurrent network、self-attention

1. recurrent network：LSTM、BiLSTM、BiLSTM+pool、GRU+attention、BiLSTM+sum、BiLSTM+Position embedding、ELMo
2. self-attention：依靠 BERT 编码 mention context，然后根据 BERT layers 输出建模 mention 表示，例如：1）对 mention 中每个 token 的向量表示进行 polling 来获得 mention 的表示；2）对句子中的几个 mentions 进行 self-attention，使这几个实体之间交互；3）在 mention 的边界插入特殊标识......

#### 3.1.3 Entity Encoding

早期的 EL 模型中使用 one-hot、word2vec 等方法编码 entity，后来人们使用知识库中的实体相关性生成低维稠密的实体向量表示，一些研究将实体相关性进一步扩展，使用 Wikipedia 中类似 entity-word 共现计数等特征，在一统一的向量空间中进行 entity 和 word 的对齐。

一些研究将锚链接替换成实体描述，然后使用类似 word2vec 的方法训练实体表示。

一些研究不使用带有实体标注的问题构建实体编码，而是用远程监督的方法扩展 word2vec 模型，在 Wikipedia 的 title 和重定向的术语上进行训练。

一些研究根据 entity-entity 之间的超链接构建出一个图，然后使用 DeepWalk 的图嵌入算法。

一些模型提出了 mention encoding、entity encoding、entity ranking 联合训练的方法。

* 使用通过描述页 context、surface form 词语、entity type 等信息构建的 word2vec、BERT encoder 初始化 entity embedding，然后再 entity ranking 阶段微调参数。
* E-ELMo 方法扩展了 ELMo 模型，以一种多任务的方式进行学习，包括 1）像标准双向语言模型那样预测上一个/下一个词语；2）遇到 mention 时预测其目标 entity；最终既获得了实体表示又获得了mention encoding。

论文中给出了不同模型使用不同特征的表格（Table 2），以及是否在同一个语义空间中表示词语和实体。

#### 3.1.4 Entity Ranking

Entity Ranking：对每个 mention 的候选实体进行打分排序，通用的架构如下图：

![image-20210812162908177](https://i.loli.net/2021/08/16/51fBS8nrys9iovP.png)

mention vector 和 entity vector 的相似度一般采用点积、余弦相似度计算，然后再用 softmax 计算概率 $P(e_i|m)$​，这个相似度（或者概率）可以和 candidate generation 阶段的先验概率或者是其他特征（$f(e_i,m)$ 各种相似度、字符串匹配、实体类型）结合，最常见的一种方法是再附加上一两层的前馈神经网络 $\phi(\cdot)$​ 
$$
P(e_i|m) = softmax(s(m, e_i))\\
\Phi(e_i, m) = \phi(P(e_i | m), f(e_i,m))
$$
文献中有一些构建训练目标的方法：

1. 标准的负对数似然目标：$L(m) = −s(m,e_∗)+log \sum_{i=1}^{k} exp(s(m,e_i))$ 其中 $e_*$​ 是正确的实体
2. ranking loss 变体：在正candidate 和负 candidate 之间强制实现大于 0 的分类边界：

![image-20210812165448670](https://i.loli.net/2021/08/16/A41nTXO3jxtqKRV.png)

#### 3.1.5 Unlinkable Mention Prediction

有的 mention 在 KBs 中没有对应的实体条目，因此不能被链接到正确实体。因此 EL 系统应该能预测缺失的引用—— NIL 预测任务，一般有以下四种处理方法：

1. 候选实体生层为空
2. 设置一个阈值，如果最优 candidate 低于此值，则认为该 mention 不可预测
3. ranking 阶段引入 “NIL” 候选项
4. 训练一个额外的二分类器，输入 Ranking 阶段最佳的 mention-entity 对，以及一些其他的特征，如 linking score 等，最终决定是否为 NIL

###  Modifications of the General Architecture

![image-20210812171448865](https://i.loli.net/2021/08/16/lp6MPUWo1YBjh2F.png)

#### 3.2.1 Joint Entity Recognition and Disambiguation

1. candidate based：生成 mention 的 candidate，舍弃不正确的 mention；
2. muti-task learning：Stack-BiLSTM（NER）--- hidden state---> linker net + NIL pre net （联合训练）
3. sequence labeling：不做候选实体生成，将 EL 任务是为序列标注，每个 token 分配一个实体链接或NIL 标记。

#### 3.2.2 Global Context Architectures

实体消歧阶段有两种上下文信息：局部上下文、全局上下文。

局部上下文：每个 mention 根据其周围的词语独立地进行消歧

全局上下文：考虑上下文中多个实体的语义一致性，每个实体的消歧依赖于其他实体的决策，此时上下文指的是周围大量的词语或整个文档，示例如下图：

![image-20210816095050954](https://i.loli.net/2021/08/16/q3DM26QXNnEYi78.png)

这样的方法有两个问题：1. 候选实体组合，增大计算的时间复杂度；2. 对实体之间一致性的分数计算不能在预测前计算。

全局消歧的方法有以下几种：

1. 典型方法是构造包含上下文 mention 的候选实体的 graph，并在其上执行随机游走算法，如 PageRank。还有一些工作引入 Attention 机制只考虑目标 mention 的子图，而非文档中全部的实体。

2. 一些工作通过最大化 CRF 的势来进行全局的实体消歧，如下式，第一项是局部 entity - mention 得分，第二项是实体之间的一致性得分。

$$
g(e,m,c)= \sum_{i=1}^{n}\Phi(e_i,m_i,c_i) + \sum_{i<j}\Psi(e_i,e_j)
$$

> 上面式子的优化是一个 NP-Hard 的问题，有些工作提出了一下解决方法：
>
> * pairwise entity score 的迭代传播
> * 通过隐变量对实体之间的一致关系进行建模
> * 贪婪波束搜索最优解

3. 上述模型都存在一个问题：一个 mention 链接到错误实体，错误会影响全局的消歧，为了解决这个问题，有研究将问题定义为一个“序列决策问题”，新实体的消歧基于已经消歧的实体。实现方法有：使用 强化学习训练一个决策网络，mention 根据局部得分进行排序，具有高置信度的 mention 先进行消歧。决策网络使用 LSTM 全局编码器；可以使用attention模型利用已链接实体的知识
4. 将实体一致性组件附加到局部上下文模型中
5. 使用更大规模的上下文来捕获一致性

#### 3.2.3 Domain-Independent Architectures

Domain Independent 是 EL 系统最需要的特性，因为标注数据只存在于少数领域，在新的领域中获取标注数据需要大量的工作。最近的研究提出了远程监督学习和零样本学习的解决方案。

**远程学习**：仅使用为标注数据，使用来自于启发式表征匹配的弱监督，并将 EL 任务定义为二元多实例分类，算法学习区分正实体集合和随机负实体集合。正集合通过检索与*单词重叠度高*且*与句中其他mention的candidate有知识库关联*的实体，这需要一个描述实体关系的知识库或基于 Wikipedia 中超链接统计信息计算出的先验 entity-mention 计算。

**零样本学习**：零样本学习中唯一可以利用的实体信息是实体描述，零样本学习的关键思想是在一个有丰富的标注数据的域上训练模型，再将模型应用于有较少标注数据的域上。现有的零样本方法不需要像表面形式特征、先验 entity-mention 概率、知识库实体关系、实体类型等信息。

> 零样本方法没有预先构造的字典来生成候选实体，因此零样本的方法都使用以下策略：
>
> 1. 预先计算实体描述的表示，将其缓存
> 2. 计算 mention 的表示
> 3. 计算 mention 表示与 entity 表示的相似度，使用如 BM25、点积、余弦相似度等
>
> 零样本方法也进行 entity ranking 的步骤，使用基于 BERT 的 cross-encoder 的方法进行 entity ranking 取得了较好的表现，均衡高计算复杂度的 cross-encoder 和 快速 bi-encoder 在 entity ranking 的表现是个悬而未决的问题。

#### 3.2.4. Cross-lingual Architectures

存在不同语言之间的标注数据资源不平衡的问题，如英语的标记数据丰富，其他语言资源匮乏。因此跨语言 EL 方法被研究，Wikipedia 中跨语言链接是经常被使用的数据。

跨语言 EL 开始于候选实体生成和实体识别，因为在低资源语言中，缺少 mention 字符串到 entity 的映射关系。对于实体识别因此提出了挖掘翻译词典、训练翻译/对齐模型、字符级字符串匹配神经网络等方法。

对于 candidate ranking，有的工作训练将实体映射到英文空间的模型；有的工作提出从高资源语言上训练零样本模型，迁移到低资源语言；有的研究对上述研究进行分析得出 mention-entity 的先验概率对于跨语言 EL 影响很大。

现有的跨语言EL技术很大程度上依赖于预训练的多语言嵌入来进行实体排名。虽然在有先验概率的情况下是有效的，但在现实的零样本场景中的表现却很差。随着最近大规模预训练语言模型在零样本跨语言领域迁移的成功，跨语言自监督模型会有研究前景。

## 4 Evaluation

最广泛使用的用于评估 EL 系统的数据集是：AIDA、TAC KBP 2010、MSNBC 、AQUAINT、ACE2004 、CWEB、WW。 其中，CWEB 和 WW 是自动标注的大型数据集，而 AIDA 是手动标注的大型数据集。TAC KBP 2015 西班牙语（es）和中文（zh）数据集用于评估跨语言 EL。

下表介绍了数据集的统计信息。

![image-20210816170806753](https://i.loli.net/2021/08/16/HyREY9ZcPDrv5lo.png)

一些模型使用 GERBIL 进行评估，GERBIL 是一个用于实体识别、消歧和类型预测的 benchmark platform，具有不同类型的测试实验，例如实体消歧（D2KB），实体识别和消歧的结合（A2KB）。

## 5 Applications of Entity Linking

语义解析、问答系统、信息抽取、生物文本处理、电子健康记录挖掘、文献搜索引擎、用于信息抽取系统、知识库人口、事实验证、机器阅读。

用于训练自然语言模型：将实体链接系统集成到更大的神经网络中，如 BERT，进行联合训练。将 EL 模型整合到迁移学习的深度神经网络中进行表示学习，使得词语的表示能从知识库信息中获益。

KnowBERT 在BERT架构的顶层之间插入几个实体链接器，在以下多个任务上优化整个网络：原始 BERT 模型中的 masked 语言模型（MLM）任务和、下一个句预测（NSP）、EL。

ERNIE 用一个知识编码器（K-Encoder）扩展了 BERT 的架构，该编码器将从 self-attention network 获得的上下文词表示与来自预训练的 TransE 模型的实体表征相融合。对于模型的预训练，除了MLM任务外，作者还引入了新的预任务：保持序列中其余实体和标记，恢复给定序列中随机屏蔽的实体。这个过程被称为去噪实体自动编码器（dEA）。

## 6 Conclusion

在这篇综述中，作者分析了最近提出的神经 EL 模型，总结了一个通用的神经 EL 架构，它适用于大多数的神经 EL 系统，包括诸如 candidate generation、entity ranking、mention 和 entity coding 等组件。一般架构的各种修改被归纳为四个方向。

1. 联合实体识别和链接模型
2. 全局实体链接模型
3. 领域无关的方法，包括零样本和远距离监督方法
4. 跨语言的技术

> 大多数的研究仍然依赖于外部的知识，用于 candidate generation 步骤。编码器已经从 CNN 和 RNN 模型转变为 self-attention 的架构，并开始使用预训练语言模型，如BERT。

> 零样本：出现了一些模型，它们解决了以零样本方式将一个领域训练的模型适应于另一个领域的问题。这些方法不需要目标领域的任何标注数据，只需要这个领域的实体描述就可以实现这种迁移。

> entity-mention encoder：有几项工作表明，与独立的 mention 和 entity encoder 的模型相比，cross-encoder 的结构更为优越。许多方法依赖于预先训练好的实体表示，只有少数方法利用了 EL 模型中可训练的 entity encoder 的优势。

> global context 被广泛使用，但最近有一些研究只关注 local EL。global 模型的表现优于 local 模型

## 7 Future Directions

1. 具有 candidate generation 步骤的 End-to-end models
2. 零样本方法进一步发展来应对新型的实体：结合 NIL 预测、全局实体一致性、联合ER的EL系统
3. 使用 EL 增强的语言模型
