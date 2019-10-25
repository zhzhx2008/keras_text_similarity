# coding=utf-8

# @Author  : zhzhx2008
# @Time    : 18-10-8


import warnings

import jieba
import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

seed = 2019
np.random.seed(seed)


def get_datas(input_file):
    datas = []
    with open(input_file, 'r') as fin:
        for line in fin:
            line = line.strip()
            if '' == line:
                continue
            split = line.split('\t')
            if 4 != len(split):
                continue
            q1 = split[1].strip()
            q1_word = ' '.join(jieba.cut(q1))
            q1_char = ' '.join(list(q1))
            q2 = split[2].strip()
            q2_word = ' '.join(jieba.cut(q2))
            q2_char = ' '.join(list(q2))
            label = int(split[3].strip())
            datas.append((q1_word, q2_word, q1_char, q2_char, label))
    return datas


input_file = './data/atec_nlp_sim_train.csv'
input_file_add = './data/atec_nlp_sim_train_add.csv'
datas = get_datas(input_file)
datas.extend(get_datas(input_file_add))

np.random.shuffle(datas)
datas = datas[:1000]

datas, datas_test = train_test_split(datas, test_size=0.3, shuffle=True)
datas_train, datas_dev = train_test_split(datas, test_size=0.3, shuffle=True)

q1_word_train = [x[0] for x in datas_train]
q1_word_dev = [x[0] for x in datas_dev]
q1_word_test = [x[0] for x in datas_test]
q2_word_train = [x[1] for x in datas_train]
q2_word_dev = [x[1] for x in datas_dev]
q2_word_test = [x[1] for x in datas_test]
q1_char_train = [x[2] for x in datas_train]
q1_char_dev = [x[2] for x in datas_dev]
q1_char_test = [x[2] for x in datas_test]
q2_char_train = [x[3] for x in datas_train]
q2_char_dev = [x[3] for x in datas_dev]
q2_char_test = [x[3] for x in datas_test]
label_train = [x[4] for x in datas_train]
label_dev = [x[4] for x in datas_dev]
label_test = [x[4] for x in datas_test]

# sklearn extract feature
# feature1: count(csr_matrix)
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b",
                             ngram_range=(1, 1))  # token_pattern must remove \w, or single char not counted
vectorizer.fit(q1_word_train + q2_word_train)
q1_train_count = vectorizer.transform(q1_word_train)
q1_dev_count = vectorizer.transform(q1_word_dev)
q1_test_count = vectorizer.transform(q1_word_test)
q2_train_count = vectorizer.transform(q2_word_train)
q2_dev_count = vectorizer.transform(q2_word_dev)
q2_test_count = vectorizer.transform(q2_word_test)

# feature2: binary(csr_matrix)
q1_train_binary = q1_train_count.copy()
q1_dev_binary = q1_dev_count.copy()
q1_test_binary = q1_test_count.copy()
q1_train_binary[q1_train_binary > 0] = 1.0
q1_dev_binary[q1_dev_binary > 0] = 1.0
q1_test_binary[q1_test_binary > 0] = 1.0
q2_train_binary = q2_train_count.copy()
q2_dev_binary = q2_dev_count.copy()
q2_test_binary = q2_test_count.copy()
q2_train_binary[q2_train_binary > 0] = 1.0
q2_dev_binary[q2_dev_binary > 0] = 1.0
q2_test_binary[q2_test_binary > 0] = 1.0

# feature3: tf-idf(csr_matrix)
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 1))
vectorizer.fit(q1_word_train + q2_word_train)
q1_train_tfidf = vectorizer.transform(q1_word_train)
q1_dev_tfidf = vectorizer.transform(q1_word_dev)
q1_test_tfidf = vectorizer.transform(q1_word_test)
q2_train_tfidf = vectorizer.transform(q2_word_train)
q2_dev_tfidf = vectorizer.transform(q2_word_dev)
q2_test_tfidf = vectorizer.transform(q2_word_test)

# keras extract feature
tokenizer = Tokenizer()
tokenizer.fit_on_texts(q1_word_train + q2_word_train)
# feature1: count
q1_train_count = tokenizer.texts_to_matrix(q1_word_train, mode='count')
q1_dev_count = tokenizer.texts_to_matrix(q1_word_dev, mode='count')
q1_test_count = tokenizer.texts_to_matrix(q1_word_test, mode='count')
q2_train_count = tokenizer.texts_to_matrix(q2_word_train, mode='count')
q2_dev_count = tokenizer.texts_to_matrix(q2_word_dev, mode='count')
q2_test_count = tokenizer.texts_to_matrix(q2_word_test, mode='count')

# feature2: binary
q1_train_binary = tokenizer.texts_to_matrix(q1_word_train, mode='binary')
q1_dev_binary = tokenizer.texts_to_matrix(q1_word_dev, mode='binary')
q1_test_binary = tokenizer.texts_to_matrix(q1_word_test, mode='binary')
q2_train_binary = tokenizer.texts_to_matrix(q2_word_train, mode='binary')
q2_dev_binary = tokenizer.texts_to_matrix(q2_word_dev, mode='binary')
q2_test_binary = tokenizer.texts_to_matrix(q2_word_test, mode='binary')

# feature3: tf-idf
q1_train_tfidf = tokenizer.texts_to_matrix(q1_word_train, mode='tfidf')
q1_dev_tfidf = tokenizer.texts_to_matrix(q1_word_dev, mode='tfidf')
q1_test_tfidf = tokenizer.texts_to_matrix(q1_word_test, mode='tfidf')
q2_train_tfidf = tokenizer.texts_to_matrix(q2_word_train, mode='tfidf')
q2_dev_tfidf = tokenizer.texts_to_matrix(q2_word_dev, mode='tfidf')
q2_test_tfidf = tokenizer.texts_to_matrix(q2_word_test, mode='tfidf')

# feature4: freq
q1_train_freq = tokenizer.texts_to_matrix(q1_word_train, mode='freq')
q1_dev_freq = tokenizer.texts_to_matrix(q1_word_dev, mode='freq')
q1_test_freq = tokenizer.texts_to_matrix(q1_word_test, mode='freq')
q2_train_freq = tokenizer.texts_to_matrix(q2_word_train, mode='freq')
q2_dev_freq = tokenizer.texts_to_matrix(q2_word_dev, mode='freq')
q2_test_freq = tokenizer.texts_to_matrix(q2_word_test, mode='freq')

# feature5: word index for deep learning
q1_train_word_index = tokenizer.texts_to_sequences(q1_word_train)
q1_dev_word_index = tokenizer.texts_to_sequences(q1_word_dev)
q1_test_word_index = tokenizer.texts_to_sequences(q1_word_test)
q2_train_word_index = tokenizer.texts_to_sequences(q2_word_train)
q2_dev_word_index = tokenizer.texts_to_sequences(q2_word_dev)
q2_test_word_index = tokenizer.texts_to_sequences(q2_word_test)
max_word_length = max(
    [max(len(q1_idx), len(q2_idx)) for q1_idx in q1_train_word_index for q2_idx in q2_train_word_index])
q1_train_word_index = sequence.pad_sequences(q1_train_word_index, maxlen=max_word_length)
q1_dev_word_index = sequence.pad_sequences(q1_dev_word_index, maxlen=max_word_length)
q1_test_word_index = sequence.pad_sequences(q1_test_word_index, maxlen=max_word_length)
q2_train_word_index = sequence.pad_sequences(q2_train_word_index, maxlen=max_word_length)
q2_dev_word_index = sequence.pad_sequences(q2_dev_word_index, maxlen=max_word_length)
q2_test_word_index = sequence.pad_sequences(q2_test_word_index, maxlen=max_word_length)
