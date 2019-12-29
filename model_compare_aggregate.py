# coding=utf-8

# @Author  : zhzhx2008
# @Date    : 2019/12/29

# from:
# https://arxiv.org/abs/1611.01747，《A COMPARE-AGGREGATE MODEL FOR MATCHING TEXT SEQUENCES》


import warnings

import jieba
import numpy as np
from keras import Model, regularizers, constraints, initializers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras import backend as K

warnings.filterwarnings("ignore")

seed = 2019
np.random.seed(seed)


class Comapre_Aggregate(Layer):
    def __init__(self, hidden_size, comparison='nn', bias = True,
                 initial = 'glorot_uniform',
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 **kwargs):
        self.hidden_size = hidden_size

        self.comparision = comparison

        self.init = initializers.get(initial)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias

        super(Comapre_Aggregate, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Wi = self.add_weight(name='Wi',
                                  shape=(input_shape[0][-1], self.hidden_size),
                                  initializer=self.init,
                                  regularizer = self.W_regularizer,
                                  trainable=True,
                                  constraint=self.W_constraint)
        self.Wu = self.add_weight(name='Wu',
                                  shape=(input_shape[0][-1], self.hidden_size),
                                  initializer=self.init,
                                  regularizer = self.W_regularizer,
                                  trainable=True,
                                  constraint=self.W_constraint)
        self.Wg = self.add_weight(name='Wg',
                                  shape=(self.hidden_size, self.hidden_size),
                                  initializer=self.init,
                                  regularizer = self.W_regularizer,
                                  trainable=True,
                                  constraint=self.W_constraint)
        if self.comparision == 'nn':
            self.Wt = self.add_weight(name='Wt',
                                      shape=(self.hidden_size * 2, self.hidden_size),
                                      initializer=self.init,
                                      regularizer=self.W_regularizer,
                                      trainable=True,
                                      constraint=self.W_constraint)
        elif self.comparision == 'ntn':
            self.WT = self.add_weight(name='WT',
                                      shape=(self.hidden_size, self.hidden_size, self.hidden_size),
                                      initializer=self.init,
                                      regularizer=self.W_regularizer,
                                      trainable=True,
                                      constraint=self.W_constraint)
        if self.bias:
            self.bi = self.add_weight(name='bi',
                                      shape=(self.hidden_size,),
                                      initializer='zero',
                                      regularizer=self.b_regularizer,
                                      trainable=True,
                                      constraint=self.b_constraint)
            self.bu = self.add_weight(name='bu',
                                      shape=(self.hidden_size,),
                                      initializer='zero',
                                      regularizer=self.b_regularizer,
                                      trainable=True,
                                      constraint=self.b_constraint)
            self.bg = self.add_weight(name='bg',
                                      shape=(self.hidden_size,),
                                      initializer='zero',
                                      regularizer=self.b_regularizer,
                                      trainable=True,
                                      constraint=self.b_constraint)
            if self.comparision == 'nn':
                self.bt = self.add_weight(name='bt',
                                          shape=(self.hidden_size,),
                                          initializer='zero',
                                          regularizer=self.b_regularizer,
                                          trainable=True,
                                          constraint=self.b_constraint)
            elif self.comparision == 'ntn':
                self.bT = self.add_weight(name='bT',
                                          shape=(self.hidden_size,),
                                          initializer='zero',
                                          regularizer=self.b_regularizer,
                                          trainable=True,
                                          constraint=self.b_constraint)
        super(Comapre_Aggregate, self).build(input_shape)


    def call(self, x, **kwargs):
        assert len(x) == 2
        Q, A = x

        Q = K.sigmoid(K.bias_add(K.dot(Q, self.Wi), self.bi)) * K.tanh(K.bias_add(K.dot(Q, self.Wu), self.bu))
        A = K.sigmoid(K.bias_add(K.dot(A, self.Wi), self.bi)) * K.tanh(K.bias_add(K.dot(A, self.Wu), self.bu))

        G = K.batch_dot(K.bias_add(K.dot(Q, self.Wg), self.bg), A, axes=[-1, -1])
        G = K.softmax(G, axis=1)
        H = K.batch_dot(G, Q, axes=[1, 1])

        T = None
        if self.comparision == 'nn':
            T = concatenate([A, H])
            T = K.dot(T, self.Wt)
            T = K.bias_add(T, self.bt)
            T = K.relu(T)
        elif self.comparision == 'ntn':
            T = K.dot(A, self.WT)
            T = K.batch_dot(T, H, axes=[-1, -1])
            T = K.bias_add(T, self.bT)
            T = K.relu(T)
        elif self.comparision == 'eu_cos':
            T1 = K.sum(K.square(K.abs(A - H)))
            T2 = K.sum(A, H) / K.sqrt(K.square(A)) * K.sqrt(K.square(H))
            T = K.concatenate([T1, T2])
        elif self.comparision == 'sub':
            T = (A - H) * (A - H)
        elif self.comparision == 'mult':
            T = A * H
        elif self.comparision == 'sub_mult_nn':
            T = K.relu(K.bias_add(K.dot(K.concatenate([(A - H) * (A - H), A * H]), self.Wt), self.bt))
        else:
            pass

        return T


    def compute_output_shape(self, input_shape):
        return (input_shape[1][0], input_shape[1][1], self.hidden_size)



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
drop_out = 0.2

hidden_size = 150


q_input = Input(name='q', shape=(maxlen,))
a_input = Input(name='a', shape=(maxlen,))

# 1. Input Encoding
embedding = Embedding(voc_size + 1, embedding_dim)
spatialdropout = SpatialDropout1D(drop_out)
q_embed = embedding(q_input)
a_embed = embedding(a_input)
q_embed = spatialdropout(q_embed)
a_embed = spatialdropout(a_embed)
com_agg = Comapre_Aggregate(hidden_size)([q_embed, a_embed])

cnn1 = Conv1D(hidden_size, 3, padding='same', strides=1, activation='relu')(com_agg)
cnn1 = GlobalMaxPool1D()(cnn1)
cnn2 = Conv1D(hidden_size, 4, padding='same', strides=1, activation='relu')(com_agg)
cnn2 = GlobalMaxPool1D()(cnn2)
cnn3 = Conv1D(hidden_size, 5, padding='same', strides=1, activation='relu')(com_agg)
cnn3 = GlobalMaxPool1D()(cnn3)
cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)

out = Dense(1, activation='sigmoid')(cnn)

model = Model(inputs=[q_input, a_input], outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

model_weight_file = './model_compare_aggregate.h5'
model_file = './model_compare_aggregate.model'
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
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
