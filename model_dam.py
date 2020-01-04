# coding=utf-8

# @Author  : zhzhx2008
# @Date    : 2020/01/03

# from:
# https://www.aclweb.org/anthology/P18-1103.pdf，《Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network》


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


class Attention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)


    def call(self, x, **kwargs):

        Query, Key, Value = x

        Query = K.dot(Query, self.WQ)
        Key = K.dot(Key, self.WK)
        Value = K.dot(Value, self.WV)

        QK = K.batch_dot(Query, Key, axes=[-1, -1]) / self.output_dim ** 0.5
        QK = K.softmax(QK, axis=-1)
        R = K.batch_dot(QK, Value, axes=(-1, 1))
        return R

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


voc_size = 6000
embedding_dim = 300
max_turn_num = 3  # u1, u2, u3
max_turn_len = 72
stack_num = 2
in_channels = 0
out_channels_0, out_channels_1 = 32, 16

embedding = Embedding(voc_size + 1, embedding_dim)
us = []
for i in range(max_turn_num):
    us.append(Input(shape=(max_turn_len,), name='u' + str(i)))
r = Input(shape=(max_turn_len,), name='r')
us.append(r)
Hr = embedding(r)
Hr_stack = [Hr]
for index in range(stack_num):
    Hr_stack.append(Attention(embedding_dim)([Hr, Hr, Hr]))
sim_turns = []
for i in range(max_turn_num):
    Hu = us[i]
    Hu = embedding(Hu)
    Hu_stack = [Hu]
    for _ in range(stack_num):
        Hu_stack.append(Attention(embedding_dim)([Hu, Hu, Hu]))
    t_a_r_stack = []
    r_a_t_stack = []
    for index in range(stack_num + 1):
        t_a_r = Attention(embedding_dim)([Hu_stack[index], Hr_stack[index], Hr_stack[index]])
        r_a_t = Attention(embedding_dim)([Hr_stack[index], Hu_stack[index], Hu_stack[index]])
        t_a_r_stack.append(t_a_r)
        r_a_t_stack.append(r_a_t)
    t_a_r_stack.extend(Hu_stack)
    r_a_t_stack.extend(Hr_stack)
    # t_a_r_stack_expand = []
    # r_a_t_stack_expand = []
    pq_stack = []
    for p, q in zip(t_a_r_stack, r_a_t_stack):
        pq = Lambda(lambda x : K.batch_dot(x[0], x[1], axes=(-1, -1)))([p, q])
        pq = Lambda(lambda x : K.expand_dims(x, axis=-1))(pq)
        pq_stack.append(pq)
    sim = concatenate(pq_stack, axis=-1)
    sim = Lambda(lambda x: K.expand_dims(x, axis=1))(sim)
    in_channels = sim.shape[-1]
    sim_turns.append(sim)
sim = Lambda(lambda x: concatenate(x, axis=1))(sim_turns)
sim = Conv3D(out_channels_0, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='elu')(sim)
sim = MaxPooling3D(pool_size=(3, 3, 3), strides=(3, 3, 3), padding='same')(sim)
sim = Conv3D(out_channels_1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='elu')(sim)
sim = MaxPooling3D(pool_size=(3, 3, 3), strides=(3, 3, 3), padding='same')(sim)
sim = Flatten()(sim)
out = Dense(1, activation='sigmoid')(sim)

model = Model(inputs=us, outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
