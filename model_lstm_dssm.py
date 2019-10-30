# coding=utf-8

# @Author  : zhzhx2008
# @Date    : 2019/10/22

# from:
# https://arxiv.org/abs/1412.6629
#


# from keras import Model
# from keras import backend as K
# from keras.layers import *
# from keras.optimizers import Adam
#
# maxlen = 128
# voc_char_size = 6000
#
# # tri-gram
# q1_input = Input(name='q1', shape=(maxlen-2, voc_char_size * 3, ))
# q2_input = Input(name='q2', shape=(maxlen-2, voc_char_size * 3, ))
#
# lstm = LSTM(300, activation='tanh')
# dense = Dense(128, activation='tanh')
#
# q1 = lstm(q1_input)
# q1 = dense(q1)
#
# q2 = lstm(q2_input)
# q2 = dense(q2)
#
# molecular = Lambda(lambda x: K.abs(K.sum(x[0] * x[1], axis=-1)))([q1, q2])
# denominator = Lambda(lambda x: K.sqrt(K.sum(K.square(x[0]), axis=-1)) * K.sqrt(K.sum(K.square(x[1]), axis=-1)))(
#     [q1, q2])
# out = Lambda(lambda x: x[0] / x[1])([molecular, denominator])
#
# model = Model(inputs=[q1_input, q2_input], outputs=out)
# model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy',
#               metrics=['categorical_crossentropy', 'accuracy'])
# print(model.summary())


# coding=utf-8

# @Author  : zhzhx2008
# @Date    : 2019/10/22

# from:
# http://www.iro.umontreal.ca/~lisa/pointeurs/ir0895-he-2.pdf
#
# reference：
# https://www.cnblogs.com/guoyaohua/p/9229190.html
#


from keras import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *
from keras.optimizers import Adam

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


def idx_processed(ary, voc_size, ngram):
    ary3 = np.eye(voc_size + 1)[ary]
    nums = ary3.shape[0]
    maxlen = ary3.shape[1]
    dim = ary3.shape[2]
    res = np.zeros((nums, maxlen + 1 - ngram, dim * ngram))
    for i in range(nums):
        for j in range(0, maxlen - ngram + 1):
            t = []
            for k in range(0, ngram):
                t.extend(list(ary3[i, j + k, :]))
            res[i, j, :] = t
    return res


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

# keras extract feature
tokenizer = Tokenizer()
tokenizer.fit_on_texts(q1_word_train + q2_word_train)

# feature2: binary
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

voc_char_size = len(tokenizer.word_index)
ngram = 3

q1_train_word_matrix = idx_processed(q1_train_word_index, voc_char_size, ngram)
q1_dev_word_matrix = idx_processed(q1_dev_word_index, voc_char_size, ngram)
q1_test_word_matrix = idx_processed(q1_test_word_index, voc_char_size, ngram)
q2_train_word_matrix = idx_processed(q2_train_word_index, voc_char_size, ngram)
q2_dev_word_matrix = idx_processed(q2_dev_word_index, voc_char_size, ngram)
q2_test_word_matrix = idx_processed(q2_test_word_index, voc_char_size, ngram)

# tri-gram
q1_input = Input(name='q1', shape=(max_word_length - ngram + 1, (voc_char_size + 1) * ngram,))
q2_input = Input(name='q2', shape=(max_word_length - ngram + 1, (voc_char_size + 1) * ngram,))

lstm = LSTM(300, activation='tanh')
dense = Dense(128, activation='tanh')

q1 = lstm(q1_input)
q1 = dense(q1)

q2 = lstm(q2_input)
q2 = dense(q2)

molecular = Lambda(lambda x: K.abs(K.sum(x[0] * x[1], axis=-1, keepdims=True)))([q1, q2])
denominator = Lambda(lambda x: K.sqrt(K.sum(K.square(x[0]), axis=-1, keepdims=True)) * K.sqrt(
    K.sum(K.square(x[1]), axis=-1, keepdims=True)))(
    [q1, q2])
out = Lambda(lambda x: x[0] / x[1])([molecular, denominator])

model = Model(inputs=[q1_input, q2_input], outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

model_weight_file = './model_lstm_dssm.h5'
model_file = './model_lstm_dssm.model'
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(model_weight_file, save_best_only=True, save_weights_only=True)
model.fit([q1_train_word_matrix, q2_train_word_matrix],
          label_train,
          batch_size=32,
          epochs=1000,
          verbose=2,
          callbacks=[early_stopping, model_checkpoint],
          validation_data=([q1_dev_word_matrix, q2_dev_word_matrix], label_dev),
          shuffle=True)

model.load_weights(model_weight_file)
model.save(model_file)
evaluate = model.evaluate([q1_test_word_matrix, q2_test_word_matrix], label_test, batch_size=32, verbose=2)
print('loss value=' + str(evaluate[0]))
print('metrics value=' + str(evaluate[1]))
