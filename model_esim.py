# coding=utf-8

# @Author  : zhzhx2008
# @Date    : 2019/10/12


from keras import Input, Model
from keras.activations import softmax
from keras.layers import Embedding, Bidirectional, LSTM, Dot, Lambda, Permute, Concatenate, Multiply, GlobalAvgPool1D, GlobalMaxPool1D, Dense, Subtract
from keras.optimizers import Adam

maxlen = 30
voc_size = 6000
embedding_dim = 300
drop_out = 0.2
lstm_dim = 128
dense_dim = 100

# from: https://arxiv.org/abs/1609.06038
# reference: https://www.kaggle.com/lamdang/dl-models
q1_input = Input(name='q1', shape=(maxlen,))
q2_input = Input(name='q2', shape=(maxlen,))

# 1. Input Encoding
embedding = Embedding(voc_size, embedding_dim)
encode = Bidirectional(LSTM(lstm_dim, return_sequences=True))
q1_embed = embedding(q1_input)
q2_embed = embedding(q2_input)
q1_encoded = encode(q1_embed)
q2_encoded = encode(q2_embed)

# 2. Local Inference Modeling
e = Dot(axes=-1)([q1_encoded, q2_encoded])
e_1 = Lambda(lambda x: softmax(x, axis=1))(e)
e_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(e))
q2_aligned = Dot(axes=1)([e_1, q1_encoded])
q1_aligned = Dot(axes=1)([e_2, q2_encoded])
q1_combined = Concatenate()([
    q1_encoded,
    q1_aligned,
    Subtract()([q1_encoded, q1_aligned]),
    Multiply()([q1_encoded, q1_aligned])
])
q2_combined = Concatenate()([
    q2_encoded,
    q2_aligned,
    Subtract()([q2_encoded, q2_aligned]),
    Multiply()([q2_encoded, q2_aligned])
])

# 3. Inference Composition
compose = Bidirectional(LSTM(lstm_dim, return_sequences=True))
q1_compare = compose(q1_combined)
q2_compare = compose(q2_combined)

# 4. Prediction
merged = Concatenate()([
    GlobalAvgPool1D()(q1_compare),
    GlobalMaxPool1D()(q1_compare),
    GlobalAvgPool1D()(q2_compare),
    GlobalMaxPool1D()(q2_compare)
])
dense = Dense(dense_dim, activation='tanh')(merged)
out = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[q1_input, q2_input], outputs=out)
model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['binary_crossentropy', 'accuracy'])
print(model.summary())
