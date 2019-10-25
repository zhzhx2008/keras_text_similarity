# coding=utf-8

# @Author  : zhzhx2008
# @Date    : 2019/10/22

# from:
# https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf
#
# reference：
# https://github.com/airalcorn2/Deep-Semantic-Similarity-Model/blob/master/deep_semantic_similarity_keras.py


from keras import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *





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



voc_char_size = q1_train_binary.shape[1]

q1_input = Input(name='q1', shape=(voc_char_size,))
q2_input = Input(name='q2', shape=(voc_char_size,))

dense1 = Dense(300, activation='tanh')
dense2 = Dense(300, activation='tanh')
dense3 = Dense(128, activation='tanh')

q1 = dense1(q1_input)
q1 = dense2(q1)
q1 = dense3(q1)

q2 = dense1(q2_input)
q2 = dense2(q2)
q2 = dense3(q2)

molecular = Lambda(lambda x: K.abs(K.sum(x[0] * x[1], axis=-1, keepdims=True)))([q1, q2])
denominator = Lambda(lambda x: K.sqrt(K.sum(K.square(x[0]), axis=-1, keepdims=True)) * K.sqrt(K.sum(K.square(x[1]), axis=-1, keepdims=True)))(
    [q1, q2])
out = Lambda(lambda x: x[0] / x[1])([molecular, denominator])

model = Model(inputs=[q1_input, q2_input], outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())


model_weight_file = './model_dssm.h5'
model_file = './model_dssm.model'
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(model_weight_file, save_best_only=True, save_weights_only=True)
model.fit([q1_train_binary, q2_train_binary],
          label_train,
          batch_size=32,
          epochs=1000,
          verbose=2,
          callbacks=[early_stopping, model_checkpoint],
          validation_data=([q1_dev_binary, q2_dev_binary], label_dev),
          shuffle=True)

model.load_weights(model_weight_file)
model.save(model_file)
evaluate = model.evaluate([q1_test_binary, q2_test_binary], label_test, batch_size=32, verbose=2)
print('loss value=' + str(evaluate[0]))
print('metrics value=' + str(evaluate[1]))
