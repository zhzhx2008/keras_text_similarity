# coding=utf-8

# @Author  : zhzhx2008
# @Date    : 2019/10/12

# from:
# https://arxiv.org/abs/1708.02312，《Shortcut-Stacked Sentence Encoders for Multi-Domain Inference》


import warnings

import jieba
import numpy as np
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras import backend as K

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

# keras extract feature
tokenizer = Tokenizer()
tokenizer.fit_on_texts(q1_word_train + q2_word_train)

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

maxlen = max_word_length
voc_size = len(tokenizer.word_index)
embedding_dim = 300
drop_out = 0.5
lstm_dim = 600
dense_dim = 800

q1_input = Input(name='q1', shape=(maxlen,))
q2_input = Input(name='q2', shape=(maxlen,))

embed = Embedding(voc_size + 1, embedding_dim)
lstm = Bidirectional(LSTM(lstm_dim, return_sequences=True))
lstm_1 = Bidirectional(LSTM(lstm_dim, return_sequences=True))
lstm_2 = Bidirectional(LSTM(lstm_dim, return_sequences=True))
global_max_pooling = Lambda(lambda x: K.max(x, axis=1))

p_s1 = embed(q1_input)
p_s2 = embed(q2_input)

s1_layer1_out = lstm(p_s1)
s2_layer1_out = lstm(p_s2)
s1_layer2_in = Concatenate()([p_s1, s1_layer1_out])
s2_layer2_in = Concatenate()([p_s2, s2_layer1_out])
s1_layer2_out = lstm_1(s1_layer2_in)
s2_layer2_out = lstm_1(s2_layer2_in)
s1_layer3_in = Concatenate()([p_s1, Lambda(lambda x: x[0] + x[1])([s1_layer1_out, s1_layer2_out])])
s2_layer3_in = Concatenate()([p_s2, Lambda(lambda x: x[0] + x[1])([s2_layer1_out, s2_layer2_out])])
s1_layer3_out = lstm_2(s1_layer3_in)
s2_layer3_out = lstm_2(s2_layer3_in)
s1_layer3_maxout = global_max_pooling(s1_layer3_out)
s2_layer3_maxout = global_max_pooling(s2_layer3_out)
s_layer3_maxout_abs = Lambda(lambda x: K.abs(x[0] - x[1]))([s1_layer3_maxout, s2_layer3_maxout])
features = Concatenate()([s1_layer3_maxout, s2_layer3_maxout,
             s_layer3_maxout_abs,
             multiply([s1_layer3_maxout, s2_layer3_maxout])])
dense = Dense(dense_dim, activation='relu')(GlobalMaxPool1D()(s1_layer3_in))
dense = Dropout(0.1)(dense)
dense = Dense(dense_dim, activation='relu')(dense)
dense = Dropout(0.1)(dense)
out = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[q1_input, q2_input], outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

model_weight_file = './model_sse.h5'
model_file = './model_sse.model'
early_stopping = EarlyStopping(monitor='val_loss', patience=1)
model_checkpoint = ModelCheckpoint(model_weight_file, save_best_only=True, save_weights_only=True)
model.fit([q1_train_word_index, q2_train_word_index],
          label_train,
          batch_size=32,
          epochs=1000,
          verbose=2,
          callbacks=[early_stopping, model_checkpoint],
          validation_data=([q1_dev_word_index, q2_dev_word_index], label_dev),
          shuffle=True)

model.load_weights(model_weight_file)
model.save(model_file)
evaluate = model.evaluate([q1_test_word_index, q2_test_word_index], label_test, batch_size=32, verbose=2)
print('loss value=' + str(evaluate[0]))
print('metrics value=' + str(evaluate[1]))
