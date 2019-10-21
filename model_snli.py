# coding=utf-8

# @Author  : zhzhx2008
# @Date    : 2019/10/21

# from:
# http://nlp.stanford.edu/pubs/snli_paper.pdf


from keras import Input, Model
from keras.activations import softmax
from keras.layers import *
from keras.optimizers import Adam
from keras import backend as K

maxlen = 128
voc_size = 6000
embedding_dim = 300
drop_out = 0.5
dense_dim = 100

q1_input = Input(name='q1', shape=(maxlen,))
q2_input = Input(name='q2', shape=(maxlen,))

embedding = Embedding(voc_size, embedding_dim)
spatialdropout = SpatialDropout1D(drop_out)
dense = Dense(dense_dim, activation='tanh')

# sum or rnn or lstm
# encode = Lambda(lambda x: K.sum(x, axis=1, keepdims=False))
# encode = SimpleRNN(dense_dim)
encode = LSTM(dense_dim)

q1_embed = dense(spatialdropout(embedding(q1_input)))
q2_embed = dense(spatialdropout(embedding(q2_input)))
q1_embed = encode(q1_embed)
q2_embed = encode(q2_embed)
con = concatenate([q1_embed, q2_embed])
con = Dense(200, activation='tanh')(con)
con = Dense(200, activation='tanh')(con)
con = Dense(200, activation='tanh')(con)
con = Dropout(drop_out)(con)
out = Dense(3, activation='softmax')(con)

model = Model(inputs=[q1_input, q2_input], outputs=out)
model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['categorical_crossentropy', 'accuracy'])
print(model.summary())
