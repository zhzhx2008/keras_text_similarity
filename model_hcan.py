# coding=utf-8

# @Author  : zhzhx2008
# @Date    : 2020/01/07

# from:
# https://jinfengr.github.io/publications/Rao_etal_EMNLP2019.pdf
# https://github.com/jinfengr/hcan

from keras import backend as K, initializers, regularizers
from keras.engine import Layer
from keras.initializers import RandomUniform
from keras.layers import Input, Convolution1D, Lambda, Reshape, Bidirectional, LSTM
from keras.layers.advanced_activations import Softmax
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.merge import Dot, Concatenate
from keras.models import Model


class BiAttentionLayer(Layer):
    def __init__(self, input_dim, max_sent1, max_sent2,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None, **kwargs):
        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(BiAttentionLayer, self).__init__(**kwargs)
        self.max_sent1 = max_sent1
        self.max_sent2 = max_sent2
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        # self.bias_regularizer = regularizers.get(bias_regularizer)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.W1 = self.add_weight(shape=(self.input_dim, 1),
                                  initializer=self.kernel_initializer,
                                  name='W1',
                                  regularizer=self.kernel_regularizer)
        self.W2 = self.add_weight(shape=(self.input_dim, 1),
                                  initializer=self.kernel_initializer,
                                  name='W2',
                                  regularizer=self.kernel_regularizer)
        self.bilinear_weights = self.add_weight(shape=(self.input_dim, self.input_dim),
                                                initializer=self.kernel_initializer,
                                                name='bilinear_weights',
                                                regularizer=self.kernel_regularizer)
        self.trainable_weights = [self.W1, self.W2, self.bilinear_weights]

    def call(self, inputs):
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('BiAttentionLayer must be called on a list of tensors '
                            '(at least 2). Got: ' + str(inputs))
        if K.ndim(inputs[0]) < 3 or K.ndim(inputs[1]) < 3:
            raise Exception('input tensors with insufficent dimensions:'
                            + str(K.shape(inputs[0])) + str(K.shape(inputs[1])))

        # s1, s2: batch_size * time_steps * input_dim
        s1, s2 = inputs[0], inputs[1]
        batch_size = K.shape(s1)[0]
        # print(K.shape(s1))
        attention1 = K.dot(s1, self.W1)
        attention2 = K.dot(s2, self.W2)
        # print(attention1, attention2)
        bilinear_attention = K.batch_dot(s1, K.dot(s2, self.bilinear_weights), axes=2)
        # print(bilinear_attention)
        rep_attention1 = K.repeat_elements(attention1, self.max_sent2, -1)
        reshape_attention2 = K.reshape(attention2, (batch_size, 1, self.max_sent2))
        rep_attention2 = K.repeat_elements(reshape_attention2, self.max_sent1, -2)
        return rep_attention1 + rep_attention2 + bilinear_attention

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return (batch_size, self.max_sent1, self.max_sent2)

    def test(self):
        from keras.layers import Input
        from keras.models import Model
        import numpy as np
        input_a = Input(shape=(3, 4,))
        input_b = Input(shape=(5, 4,))
        bi_attention = BiAttentionLayer(4, 3, 5)([input_a, input_b])
        m = Model([input_a, input_b], bi_attention)
        print(m.summary())
        tensor_a = np.ones((2, 3, 4))
        tensor_b = np.ones((2, 5, 4))
        m.predict([tensor_a, tensor_b])


max_query_len = 72
max_doc_len = 60
vocab_size = 6000
nb_filters = 256
nb_layers = 5
kernel_size = 2
embed_size = 300
dropout_rate = 0.1
num_classes = 2
encoder_option = "deepconv"
join = 'hcan'
att = 'BiDAF'

query_word_input = Input(shape=(max_query_len,), name="query_word_input")
doc_word_input = Input(shape=(max_doc_len,), name="doc_word_input")
embed = Embedding(input_dim=vocab_size, output_dim=embed_size, trainable=True, mask_zero=False,
                  embeddings_initializer=RandomUniform(-0.05, 0.05))
query_embedding = embed(query_word_input)
doc_embedding = embed(doc_word_input)
output_list = [[query_embedding, doc_embedding]]
if encoder_option == "deepconv":
    for i in range(nb_layers):
        conv_layer = Convolution1D(filters=nb_filters, kernel_size=kernel_size, padding='same', activation='relu',
                                   strides=1)
        dropout_layer = Dropout(dropout_rate)
        query_conv_tensor, doc_conv_tensor = conv_layer(output_list[i][0]), conv_layer(output_list[i][1])
        query_dropout_tensor = dropout_layer(query_conv_tensor)
        doc_dropout_tensor = dropout_layer(doc_conv_tensor)
        output_list.append([query_dropout_tensor, doc_dropout_tensor])
elif encoder_option == "wideconv":
    for i in range(nb_layers):
        conv_layer = Convolution1D(filters=nb_filters, kernel_size=(kernel_size - 1) * i + kernel_size, padding='same',
                                   activation='relu', strides=1)
        dropout_layer = Dropout(dropout_rate)
        query_conv_tensor, doc_conv_tensor = conv_layer(output_list[i][0]), conv_layer(output_list[i][1])
        query_dropout_tensor = dropout_layer(query_conv_tensor)
        doc_dropout_tensor = dropout_layer(doc_conv_tensor)
        output_list.append([query_dropout_tensor, doc_dropout_tensor])
elif encoder_option == "bilstm":
    for i in range(nb_layers):
        bilstm_layer = Bidirectional(
            LSTM(int(nb_filters / 2), recurrent_dropout=dropout_rate, dropout=dropout_rate, return_sequences=True))
        query_lstm_tensor, doc_lstm_tensor = bilstm_layer(output_list[i][0]), bilstm_layer(output_list[i][1])
        output_list.append([query_lstm_tensor, doc_lstm_tensor])
else:
    print('invalid encoder choice!')
    exit(0)
norm_sim_list, max_sim_list, mean_sim_list = [], [], []
attention_emb_list = []
for i in range(nb_layers):
    query_embedding, doc_embedding = output_list[i][0], output_list[i][1]
    if i > 0:
        if join == 'biattention' or join == 'hcan':
            biattention_matrix = BiAttentionLayer(nb_filters, max_query_len, max_doc_len)(
                [query_embedding, doc_embedding])
            if att == 'BiDAF':

                norm_biattention = Softmax(axis=-2)(biattention_matrix)
                # Activation('softmax', axis=-2)(biattention_matrix)
                reshape_norm_biatt = Reshape((max_doc_len, max_query_len,))(norm_biattention)
                attentive_doc_embedding = Dot(axes=[-1, -2])([reshape_norm_biatt, query_embedding])

                max_biattention = Lambda(lambda x: K.max(x, axis=1),
                                         output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2],))(biattention_matrix)
                norm_biatt = Activation('softmax')(max_biattention)
                reshape_doc_embedding = Reshape((nb_filters, max_doc_len,))(doc_embedding)
                context_emb = Dot(axes=-1)([reshape_doc_embedding, norm_biatt])

                reshape_context_emb = Reshape((1, nb_filters))(context_emb)
                rep_context_emb = Lambda(lambda x: K.repeat_elements(x, max_doc_len, -2),
                                         output_shape=lambda inp_shp: (inp_shp[0], max_doc_len, nb_filters,))(
                    reshape_context_emb)
                attentive_doc_emb_prod = Lambda(lambda x: x[0] * x[1])([doc_embedding, attentive_doc_embedding])
                context_doc_emb_prod = Lambda(lambda x: x[0] * x[1])([rep_context_emb, attentive_doc_embedding])
                concat_doc_emb = Concatenate(axis=-1)([doc_embedding, attentive_doc_embedding,
                                                       attentive_doc_emb_prod, context_doc_emb_prod])
                all_doc_emb = Activation('relu')(concat_doc_emb)
                attention_emb = Bidirectional(
                    LSTM(int(nb_filters / 2), recurrent_dropout=dropout_rate, dropout=dropout_rate,
                         return_sequences=True, name="biattention-lstm%d" % i))(all_doc_emb)
                attention_sim1 = Lambda(lambda x: K.max(x, axis=1),
                                        output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2],))(
                    attention_emb)
                attention_sim2 = Lambda(lambda x: K.mean(x, axis=1),
                                        output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2],))(
                    attention_emb)
                attention_emb_list.extend([attention_sim1, attention_sim2])
            elif att == 'ESIM':

                norm_biattention = Softmax(axis=-2)(biattention_matrix)
                # Activation('softmax', axis=-2)(biattention_matrix)
                reshape_norm_biatt = Reshape((max_doc_len, max_query_len,))(norm_biattention)
                attentive_doc_embedding = Dot(axes=[-1, -2])([reshape_norm_biatt, query_embedding])

                subtract_doc_emb = Lambda(lambda x: x[0] - x[1])([attentive_doc_embedding, doc_embedding])
                prod_doc_emb = Lambda(lambda x: x[0] * x[1])([attentive_doc_embedding, doc_embedding])
                merge_doc_emb = Concatenate(axis=-1)([attentive_doc_embedding, doc_embedding,
                                                      subtract_doc_emb, prod_doc_emb])
                seq_doc_emb = Bidirectional(LSTM(nb_filters, recurrent_dropout=dropout_rate, dropout=dropout_rate,
                                                 return_sequences=True, name="biattention-lstm%d" % (2 * i - 1)))(
                    merge_doc_emb)
                max_doc_emb = Lambda(lambda x: K.max(x, axis=1),
                                     output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2],))(seq_doc_emb)

                norm_biattention = Softmax(axis=-2)(biattention_matrix)
                reshape_norm_biatt = Reshape((max_doc_len, max_query_len,))(norm_biattention)
                attentive_query_embedding = Dot(axes=[-1, -2])([reshape_norm_biatt, doc_embedding])

                subtract_query_emb = Lambda(lambda x: x[0] - x[1])([attentive_query_embedding, query_embedding])
                prod_query_emb = Lambda(lambda x: x[0] * x[1])([attentive_query_embedding, query_embedding])
                merge_query_emb = Concatenate(axis=-1)([attentive_query_embedding, query_embedding,
                                                        subtract_query_emb, prod_query_emb])
                seq_query_emb = Bidirectional(LSTM(nb_filters, recurrent_dropout=dropout_rate, dropout=dropout_rate,
                                                   return_sequences=True, name="biattention-lstm%d" % (2 * i)))(
                    merge_query_emb)
                max_query_emb = Lambda(lambda x: K.max(x, axis=1),
                                       output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2],))(
                    seq_query_emb)
                attention_emb_list.extend([max_query_emb, max_doc_emb])
            elif att == 'DecompAtt':
                pass
            else:
                print('invalid attention choice!')
                exit(0)

    dot_prod = Dot(axes=-1)([doc_embedding, query_embedding])
    norm_sim = Activation('softmax')(dot_prod)
    max_sim = Lambda(lambda x: K.max(x, axis=1), output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2],))(norm_sim)
    mean_sim = Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2],))(norm_sim)

    norm_sim_list.append(norm_sim)
    max_sim_list.append(max_sim)
    mean_sim_list.append(mean_sim)

if join == 'matching':
    if len(max_sim_list) == 1:
        feature_vector = max_sim_list[0]
    else:
        feature_vector = Concatenate(axis=-1, name="feature_vector")(max_sim_list)
elif join == 'biattention':
    if len(attention_emb_list) == 1:
        feature_vector = attention_emb_list[0]
    else:
        feature_vector = Concatenate(axis=-1, name="feature_vector")(attention_emb_list)
elif join == 'hcan':
    max_sim_list.extend(attention_emb_list)
    feature_vector = Concatenate(axis=-1, name="feature_vector")(max_sim_list)
else:
    raise Exception('invalid join method!')

feature_vector1 = Dense(150, activation='relu', name="feature_vector1")(feature_vector)
feature_vector2 = Dense(50, activation='relu', name="feature_vector2")(feature_vector1)
prediction = Dense(num_classes, activation='softmax', name="prediction")(feature_vector2)
model = Model([query_word_input, doc_word_input], [prediction])
print(model.summary())
