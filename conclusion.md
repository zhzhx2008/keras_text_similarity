# keras_text_similarity

Reference：
- https://www.jiqizhixin.com/articles/2019-10-18-14?utm_source=tuicool&utm_medium=referral

Task：
- 相似度计算&复述识别（textual similarity&paraphrase identification）
- 问答匹配（answer selection）
- 对话匹配（response selection）
- 自然语言推理/文本蕴含识别（Natural Language Inference/Textual Entailment）
- 信息检索中的匹配
- 机器阅读理解

tips:
- 两个句子合并成一个句子，然后看成是文本分类的问题，参考bert
- 两个句子分来处理，然后再合并做分类（一般情况下比合并处理效果好）

model architecture
- 基于表示
    - encoder
        - InferSent
        - SSE
    - 相似度计算函数
        - SiamCNN
        - SiamLSTM
        - Multi-view
- 基于交互
    - MatchCNN
    - DecAtt
    - ESIM
    - DAM
    - HCAN


preprocess：
- 参考https://github.com/zhzhx2008/keras_text_classification/blob/master/text_classification_conclusion.md
- 数据集扩充：q1=q2, q2=q3 -->  q1=q3， q1=q2, q2 != q3 --> q1 != q3(不一定)


feature extraction（参考kaggle：quora question pairs）：
'''
特征1：两个问题共同词的权重之和/两个问题所有词权重之和
特征2：两个问题共同词的个数之和/（两个问题所有词之和 - 共同词个数之和）
特征3：两个问题共同词的个数之和
特征4：问题1停用词个数/问题1所有词个数
特征5：问题2停用词个数/问题2所有词个数
特征6：两个问题共同bi-gram词个数之和/两个问题所有bi-gram个数之和
特征7：cosin距离（分子是numpy.dot(shared_weights, shared_weights)）
特征8：两个问题的汉明距离

特征9：问题1停用词频次 - 问题2停用词频次
特征10：问题1句子的长度
特征11：问题2句子的长度
特征12：问题1句子的长度 - 问题2句子的长度

特征13：问题1大写词的个数
特征14：问题2大写词的个数
特征15：问题1大写词的个数 - 问题2大写词的个数

特征16：问题1字的个数
特征17：问题2字的个数
特征18：问题1字的个数 - 问题2字的个数

特征19：问题1词的个数
特征20：问题2词的个数
特征21：问题1词的个数 - 问题2词的个数

特征22：问题1字符的长度/问题1词的长度
特征23：问题2字符的长度/问题2词的长度
特征24：上述两个的比值

特征25：两句话是不是完全一样

特征26-47
特征: how,what,why,when,where,which,who是不是存在问题1中
特征: how,what,why,when,where,which,who是不是存在问题2中
特征: 上述两者之积

特征：两个问题共同的uni-gram长度
特征：两个问题共同的uni-gram长度 / 两个问题uni-gram之和
特征：两个问题共同的bi-gram长度
特征：两个问题共同的bi-gram长度 / 两个问题bi-gram之和
特征：两个问题共同的tri-gram长度
特征：两个问题共同的tri-gram长度 / 两个问题tri-gram之和

1.Text Mining Feature，比如句子长度；两个句子的文本相似度，如N-gram的编辑距离，Jaccard距离等；两个句子共同的名词，动词，疑问词等。
2.Embedding Feature，预训练好的词向量相加求出句子向量，然后求两个句子向量的距离，比如余弦相似度、欧式距离等等。
3.Vector Space Feature，用TF-IDF矩阵来表示句子，求相似度。
4.Magic Feature，是Forum上一些选手通过思考数据集构造过程而发现的Feature，这种Feature往往与Label有强相关性，可以大大提高预测效果。

data leakly：出现次数越多的问题，越有可能是重复的问题。
'''