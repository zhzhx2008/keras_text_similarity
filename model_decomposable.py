# coding=utf-8

# @Author  : zhaoxi9
# @Date    : 2019/10/12


from keras import Input, Model, Sequential
from keras import backend as K
from keras.activations import softmax
from keras.layers import Embedding, Dot, Lambda, Permute, Dense, Dropout, dot, concatenate

maxlen = 30
voc_size = 6000
embedding_dim = 300
drop_out = 0.2
lstm_dim = 128
dense_dim = 100

# from: https://arxiv.org/abs/1606.01933
# reference: https://github.com/explosion/spaCy/blob/master/examples/keras_parikh_entailment/keras_decomposable_attention.py
q1 = Input(name='q1', shape=(maxlen,))
q2 = Input(name='q2', shape=(maxlen,))

# Embedding
embedding = Embedding(voc_size, embedding_dim)
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
v1 = G(comp1)
v2 = G(comp2)
# step 3: aggregate
v1_sum = Lambda(lambda x : K.sum(x, axis=1))(v1)
v2_sum = Lambda(lambda x : K.sum(x, axis=1))(v2)
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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_crossentropy', 'accuracy'])
print(model.summary())
