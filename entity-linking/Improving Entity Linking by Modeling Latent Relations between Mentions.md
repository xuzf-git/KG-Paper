# Improving Entity Linking by Modeling Latent Relations between Mentions

论文：[https://aclanthology.org/P18-1148/](https://aclanthology.org/P18-1148/)

代码：[https://github.com/lephong/mulrel-nel](https://github.com/lephong/mulrel-nel)

## Abstract

实体链接（EL）系统经常利用文档中提及之间的关系（如共指关系）来决定链接决策是否兼容。与以往依靠监督系统或启发式预测这些关系的方法不同，作者在神经 EL 模型中将关系视为隐变量。在端到端地训练EL 模型时，作者在没有任何监督信息的情况下引入提及之间的关系。作者提出的 multi-relational 模型的训练也收敛得更快，这表明引入的结构偏差有助于解释训练数据中的规律性。

## 1 Introduce

![image-20211209102412616](https://s2.loli.net/2021/12/09/QuVkRi9o8BA7HO2.png)

给文档中所有的提及指派相应的实体，受到文档语义的影响，例如上面的例子中，如果 **World Cup** 

被认为是 *FIFA_World_Cup*，那么第二个 **England** 相比于篮球队，更应该选择指派为足球队。

在以往的研究中，实体链接的全局一致性的基本假设一般定义为：<u>“来自同一个领域的实体的兼容性更好”</u>。这在经验上取得了一定的成功，但是在这样的假设下，上面的文档中出现的所有 **England** 都会倾向于映射到同一个实体。

针对上面的问题，作者提出了新的假设：<u>“提及之间的关系可以在没有（或很少）领域知识的情况下被归纳出来”</u> 。所以作者将提及之间的关系编码为隐变量，并以此提升 EL 模型表现。与其他基于表示学习的研究相同（如 [Ganea and Hofmann, 2017](https://aclanthology.org/D17-1276/)），模型也学习 mention、context、relation 的嵌入表示。

作者提出的 multi-relational 模型，相比于“关系不可知”的模型取得了很大的进步，同时模型的训练收敛时间相比于认为“关系不可知”的模型短了10倍。这也许说明引入的结构偏差有助于解释训练数据中的规律，使优化任务更容易。

> 作者将编码成隐变量的关系称为 “诱导关系” ，作者发现一部分 “诱导关系” 与共指关系密切相关，另一些则编码提及之间的语义相关性。

## 2 Background and Related work

一般的实体链接方法分为两类：局部模型&全局模型

**局部模型**：
$$
e_i^* = \arg\max_{e_i \in C_i}\Psi(e_i, c_i)
$$
链接决策只依赖于提及的上下文，不考虑其他链接。

**全局模型**：
$$
E^* = \arg\max_{E \in C_1 \times ... \times C_n}\sum_{i=1}^{n}\Psi(e_i, c_i)+\Phi(E,D)
$$
全局模型考虑链接实体之间的一致性。其中（2）式的第二项表示链接决策的一致性，当它选择最简单的形式时，式子变成：
$$
E^{*}=\underset{E \in C_{1} \times \ldots \times C_{n}}{\arg \max } \sum_{i=1}^{n} \Psi\left(e_{i}, c_{i}\right)+\sum_{i \neq j}\Phi(e_i, e_j, D)
$$
（3）式的求解是NP-hard的问题，可以使用循环置信传播（LBP）等方法进行近似求解

如何定义局部得分函数 $\Psi$ 和成对得分函数 $\Phi$ ，作者参考了 2017 年 Ganea 和 Hofmann 的基于表示学习的方法，如下所示：
$$
\Psi(e_i, c_i) = e_i^T \bold B f(c_i)\\
\Phi(e_i, e_j, D) = \frac{1}{n-1}e_i^T\bold Re_j
$$
其中 $e_i, e_j$ 都是实体嵌入，函数 $f(c_i)$ 将注意机制应用于 $c_i$ 中的上下文词，以获得上下文的特征表示，$\bold B, \bold R$ 都是对角矩阵。目前成对一致性的全局项是不能体现实体之间的关系和顺序的，作者认为成对一致性应该考虑到以关系嵌入作为表示形式的提及之间的关系信息。

## 3 Multi-relational models

作者一共提出了三种形式分别为：general、rel-norm、ment-norm，下面分别对这三种模型进行说明。

### 3.1 General form

作者假设存在 $K$ 个隐关系，每个关系 $k$ 被赋予给每个提及对 $(m_i, m_j)$, 还给出相应的非负数的置信度 $\alpha_{ijk}$ ，则成对得分表示为特定关系得分的加权求和：
$$
\Phi(e_i, e_j, D) = \sum_{k=1}^K\alpha_{ijk}\Phi_k(e_i, e_j, D)
$$
其中 $\Phi_k(e_i, e_j, D)$ 作者表示如下，其中 $\bold R_k$ 是一个对角矩阵，表示关系 k
$$
\Phi_k(e_i, e_j, D) = e_i^T \bold R_k e_j
$$
权重 $\alpha_{ijk}$ 表示为归一化分数，其中 $\bold D_k$ 是一个对角矩阵，$Z_{ijk}$ 是归一化因子，$f()$函数将 $(m_i, c_i)$ 映射为向量
$$
\alpha_{ijk} = \frac{1}{Z_{ijk}} \exp \{\frac{f^T(m_i, c_i)\bold D_k f^T(m_j, c_j)}{\sqrt{d}}\}
$$
**Note**：

> 1. 作者选择单层神经网络作为 $f$ 函数(LSTM 出现严重过拟合，效果较差)。
> 2. 因为 $\alpha_{ijk}$ 有索引 $j$ 和 $k$ 因此可以选择，按照关系$(k)$进行归一化，或者按照提及$(j)$进行归一化，归一化因子不同是这三种形式的主要区别。

![image-20211216230754372](../../../../../Pictures/typora-imgs/image-20211216230754372.png)

### 3.2 Rel-norm: Relation-wise normalization

对于每个提及对，按照一定的概率 $\alpha_{ijk}$ 从关系池中选出相应的关系，并依赖关系嵌入计算相似性得分。从这个理解角度上来说，选择关系的概率应该具有归一性，即 $\alpha_{ijk}$ 在关系 k 上应该是归一化的：
$$
Z_{ijk} = \sum_{k^\prime = 1}^{K} \exp \{\frac{f^T(m_i, c_i)\bold D_{k^\prime} f^T(m_j, c_j)}{\sqrt{d}}\}
\\
\sum_{k=1}^{K} \alpha_{ijk} = 1
\\
\Phi(e_i, e_j, D) = e_i^T (\sum_{k=1}^{K}\alpha_{ijk}\bold R_k) e_j
$$
实际上，可以不依赖关系嵌入矩阵 $\bold R_k$ 的线性组合，而是直接预测特定于上下文的关系嵌入$R_{ij}=diag \{g(m_i，c_i，m_j，c_j)\}$，其中 $g$ 是一个神经网络。然而在初步实验中，作者发现这会导致过拟合结果变差。因此，作者选择了使用固定数量的关系作为约束模型和改进泛化的方法。

### 3.3 Ment-norm: Mention-wise normalization

ment-norm 中的 $\alpha_{ijk}$ 可以理解为：对于某种关系 $k$，提及 $m_i$ 寻找与其满足该关系的 提及，其中 $m_j$ 和 $m_i$ 在关系 $k$ 上的匹配程度即为 $\alpha_{ijk}$ 。因此，$\alpha_{ijk}$ 需要在除 $m_i$ 外的所有提及上归一化，即在 $j$ 上进行归一化:
$$
Z_{ijk} = \sum_{j^\prime = 1,j^\prime\neq i}^{n} \exp \{\frac{f^T(m_i, c_i)\bold D_k f^T(m_j, c_j)}{\sqrt{d}}\}
\\
\sum_{j = 1,j \neq i}^{n} \alpha_{ijk} = 1
\\
\Phi(e_i, e_j, D) = \sum_{k=1}^K\alpha_{ijk}e_i^T\bold R_k e_j
$$
可以发现，当 $\alpha_{ijk}$ 为均匀分布，即 $\alpha_{ijk} = \frac{1}{n-1}$ 时，如果 $K=1$ ，Ment-norm 的多关系模型就退化成了 2017年 Ganea 和 Hofmann 的模型。

分析当采取 ment-norm 的设置时，对于一对提及$(m_i, m_j)$可能会存在以下两种与 rel-norm 不同的情况：

1. $\alpha_{ijk}$ 对于所有的 $k$ 都比较小，这表示 $m_i$和 $m_j$ 之间不存在任何关系
2. $\alpha_{ijk}$ 对于一个或多个 $k$ 都比较大，这表示 $m_i$ 和 $m_j$ 之间预测为存在一个或多个关系

ment-norm 符合注意力机制的特点，对于每个提及 $m_i$ 和每个 $k$，可以将 $\alpha_{ijk}$  解释为在文档中的提及集合中选择一个提及 $m_j$ 的概率。因为有 $K$ 个关系，所有每个提及 最多有 $K$ 个提及要关注，对应于多头注意力中的每个头。

**Mention padding**

ment-norm 存在一个问题，无论这 $K$ 种关系是否都存在，都要找出对应的提及，这是因为归一化条件 $\sum_{j = 1,j \neq i}^{n} \alpha_{ijk} = 1$ , 为了解决这个问题，作者提出在每个文章中添加一个链接到 padding 实体 $e_{pad}$ 的 padding 提及 $m_{pad}$，通过这种方式，模型可以通过使用 $m_{pad}$ 来吸收概率，从而降低跟其他提及的无关关系的概率值。

### 3.4 Implementation

作者定义了条件随机场 CRF 如下：
$$
q(E|D) \propto \exp \{\sum_{i=1}^{n} \Psi(e_i, c_i) + \sum_{i \neq j} \Phi(e_i, e_j, D)\}
\\
\hat q_i(e_i|D)\approx \max_{e_1, ..., e_{i-1}, e_{i+1},...,e_n}q(E|D)
$$
对于每个提及 $m_i$，它的最终得分通过下式给出：
$$
\rho_i(e)=g(\hat q_i(e|D), \hat p(e|m_i))
$$
其中，$\hat p(e|m_i)$ 表示为先验概率通过统计计数得到，$g(*)$ 是一个两层的神经网络。

最小化下面的 ranking loss
$$
L(\theta) = \sum_{D \in \mathcal{D}}\sum_{m_i \in D}\sum_{e\in C_i} h(m_i, e)
\\
h(m_i, e) = \max (0, \gamma-\rho_i(e_i^*) + \rho_i(e))
$$
其中 $\theta$ 是模型参数，$\mathcal{D}$  是训练集，$e_i^*$ 是 ground-truth，使用 Adam 作为优化器。

为了鼓励模型探索更多不同的关系，作者在上面的损失中加入以下正则项：
$$
\lambda_1\sum_{i,j}dist(\bold R_i, \bold R_j) + \lambda_2\sum_{i,j}dist(\bold D_i, \bold D_j)
$$
 在实验中，作者将 $\lambda_1, \lambda_2$ 都取为 $-10^{-7}$；$dist$ 如下：
$$
dist(x, y) = \Arrowvert \frac{x}{\| x \|_2}- \frac{y}{\| y \|_2}\Arrowvert_2
$$
这两个正则项使得最终的关系嵌入不会全都很像，保证了关系的多样性。



## 4 Experiments

候选实体生成：现根据先验概率选择了 30 个候选实体，保留先验最高的四个候选，再从剩下的里面选出三个 $\bold e^T(\sum_{w \in d_i}\bold w)$ 得分最高的候选，其中 $d_i$ 选提及附近的 50 个词，求他们的嵌入表示的和，在点积求相似度。

![image-20211217160959214](../../../../../Pictures/typora-imgs/image-20211217160959214.png)

![image-20211217161026518](../../../../../Pictures/typora-imgs/image-20211217161026518.png)

## 5 Conclusion and Future work

作者展示了在实体链接中使用关系的好处。作者提出的模型认为关系是潜在可变的，因此不需要任何额外的监督。表示学习用于学习关系嵌入，避免了特征工程的需要。

在未来工作中，作者希望使用句法和话语结构（例如，提及之间的句法依赖路径）来鼓励模型发现更丰富的关系集合等。
