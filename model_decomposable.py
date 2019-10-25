# coding=utf-8

# @Author  : zhzhx2008
# @Date    : 2019/10/12

# from:
# https://arxiv.org/abs/1606.01933
#
# reference:
# https://github.com/explosion/spaCy/blob/master/examples/keras_parikh_entailment/keras_decomposable_attention.py


import warnings

import jieba
import numpy as np
from keras import Input, Model, Sequential
from keras import backend as K
from keras.activations import softmax
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Dot, Lambda, Permute, Dense, Dropout, dot, concatenate, TimeDistributed
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
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
dense_dim = 200

q1 = Input(name='q1', shape=(maxlen,))
q2 = Input(name='q2', shape=(maxlen,))

# Embedding
embedding = Embedding(voc_size + 1, embedding_dim)
q1_embed = embedding(q1)
q2_embed = embedding(q2)

# Attend
F = Sequential(
    [
        Dense(dense_dim, activation="relu"),
        Dropout(drop_out),
        Dense(dense_dim, activation="relu"),
        Dropout(drop_out),
    ]
)
e = dot([F(q1_embed), F(q2_embed)], axes=-1)
e_1 = Lambda(lambda x: softmax(x, axis=1))(e)
e_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(e))
q2_aligned = Dot(axes=1)([e_1, q1_embed])
q1_aligned = Dot(axes=1)([e_2, q2_embed])
G = Sequential(
    [
        Dense(dense_dim, activation="relu"),
        Dropout(drop_out),
        Dense(dense_dim, activation="relu"),
        Dropout(drop_out),
    ]
)
comp1 = concatenate([q1_embed, q1_aligned])
comp2 = concatenate([q2_embed, q2_aligned])
v1 = TimeDistributed(G)(comp1)
v2 = TimeDistributed(G)(comp2)
# step 3: aggregate
v1_sum = Lambda(lambda x: K.sum(x, axis=1))(v1)
v2_sum = Lambda(lambda x: K.sum(x, axis=1))(v2)
v_sum = concatenate([v1_sum, v2_sum])
H = Sequential(
    [
        Dense(dense_dim, activation="relu"),
        Dropout(drop_out),
        Dense(dense_dim, activation="relu"),
        Dropout(drop_out),
    ]
)
out = H(v_sum)
out = Dense(1, activation="sigmoid")(out)

model = Model(inputs=[q1, q2], outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

model_weight_file = './model_decomposable.h5'
model_file = './model_decomposable.model'
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
