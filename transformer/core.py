import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.engine import Layer
from keras.initializers import Ones, Zeros
from keras.layers import Dropout, Lambda, Softmax, Dense, Add, Embedding, Conv1D


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x, **kwargs):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention:

    def __init__(self, attention_dropout=0.0):
        self.attention_dropout = attention_dropout

    def __call__(self, q, k, v, attn_mask=None, scale=1.0):
        """

        :param q: Queries 张量，形状为[N, T_q, D_q]
        :param k: Keys 张量，形状为[N, T_k, D_k]
        :param v: Values 张量，形状为[N, T_v, D_v]
        :param attn_mask: 注意力掩码，形状为[N, T_q, T_k]
        :param scale: 缩放因子，浮点标量
        :return: 上下文张量和注意力张量
        """

        attention = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(2, 2)) * scale)([q, k])  # [N, T_q, T_k]

        if attn_mask is not None:
            # 为需要掩码的地方设置一个负无穷，softmax之后就会趋近于0
            attention = Lambda(lambda x: (-1e+10) * (1 - x[0]) + x[1])([attn_mask, attention])
        attention = Softmax(axis=-1)(attention)
        attention = Dropout(self.attention_dropout)(attention)  # [N, T_q, T_k]
        context = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(2, 1)))([attention, v])  # [N, T_q, D_q]
        return context, attention


def _split(x, dim_per_head, num_heads):
    shape = K.shape(x)  # [N, T_q, dim_per_head * num_heads]
    x = K.reshape(x, (shape[0], shape[1], num_heads, dim_per_head))
    x = K.permute_dimensions(x, pattern=(0, 2, 1, 3))  # [N, num_heads, T_q, dim_per_head]
    x = K.reshape(x, (-1, shape[1], dim_per_head))  # [N * num_heads, T_q, dim_per_head]
    return x


def _concat(x, dim_per_head, num_heads):
    shape = K.shape(x)  # [N * num_heads, T_q, dim_per_head]
    x = K.reshape(x, (-1, num_heads, shape[1], dim_per_head))  # [N, num_heads, T_q, dim_per_head]
    x = K.permute_dimensions(x, [0, 2, 1, 3])  # [N, T_q, num_heads, dim_per_head]
    x = K.reshape(x, (-1, shape[1], num_heads * dim_per_head))  # [N, T_q, num_heads * dim_per_head]
    return x


class MultiHeadAttention:

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = Dense(self.dim_per_head * num_heads, use_bias=False)
        self.linear_v = Dense(self.dim_per_head * num_heads, use_bias=False)
        self.linear_q = Dense(self.dim_per_head * num_heads, use_bias=False)
        self.linear_final = Dense(model_dim, use_bias=False)
        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)
        self.split = Lambda(lambda x: _split(x, self.dim_per_head, self.num_heads),
                            output_shape=(None, self.dim_per_head))
        self.concat = Lambda(lambda x: _concat(x, self.dim_per_head, self.num_heads),
                             output_shape=(None, self.num_heads * self.dim_per_head))

    def __call__(self, query, key, value, attn_mask=None):
        """

        :param query: shape of [N, T_q, D_q]
        :param key: shape of [N, T_k, D_k]
        :param value: shape of [N, T_v, D_v]
        :param attn_mask: shape of [N, T_q, T_k]
        :return:
        """

        residual = query

        # In order to reduce information loss, linear projection is used
        query = self.linear_q(query)  # [N, T_q, dim_per_head * num_heads]
        key = self.linear_k(key)
        value = self.linear_v(value)

        # It is divided into num_heads parts
        query = self.split(query)  # [N * num_heads, T_q, dim_per_head]
        key = self.split(key)
        value = self.split(value)

        if attn_mask is not None:
            attn_mask = Lambda(lambda x: K.repeat_elements(x, self.num_heads, axis=0))(attn_mask)
        scale = self.dim_per_head ** -0.5

        context, attention = self.dot_product_attention(query, key, value, attn_mask, scale)
        context = self.concat(context)
        context = self.linear_final(context)

        # dropout
        output = self.dropout(context)

        # add residual
        output = Add()([residual, output])

        # apply layer normalize
        output = self.layer_norm(output)

        return output, attention


def padding_mask(seq_q, seq_k):
    """
    A sentence is filled with 0, which is not what we need to pay attention to
    :param seq_k: shape of [N, T_k], T_k is length of sequence
    :param seq_q: shape of [N, T_q]
    :return: a tensor with shape of [N, T_q, T_k]
    """

    q = K.expand_dims(K.ones_like(seq_q, dtype="float32"), axis=-1)  # [N, T_q, 1]
    k = K.cast(K.expand_dims(K.not_equal(seq_k, 0), axis=1), dtype='float32')  # [N, 1, T_k]
    return K.batch_dot(q, k, axes=[2, 1])


def sequence_mask(seq):
    """

    :param seq: shape of [N, T_q]
    :return:
    """
    seq_len = K.shape(seq)[1]
    batch_size = K.shape(seq)[:1]
    return K.cast(K.cumsum(tf.eye(seq_len, batch_shape=batch_size), axis=1), dtype='float32')


class PositionalEncoding:

    def __init__(self, max_seq_len, d_model=512):
        """

        :param d_model:
        :param max_seq_len:
        """

        position_encoding = np.array([
            [pos / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_seq_len + 1)
        ])  # [max_seq_len + 1, d_model]

        position_encoding[1:, 0::2] = np.sin(position_encoding[1:, 0::2])
        position_encoding[1:, 1::2] = np.cos(position_encoding[1:, 1::2])

        self.position_encoding = Embedding(max_seq_len + 1, d_model,
                                           weights=[position_encoding], trainable=False)

    def __call__(self, x):
        """

        :param x: a tensor with shape of [N, max_seq_len]
        :return: position encoding
        """
        pos_seq = Lambda(self.get_pos_seq)(x)
        return self.position_encoding(pos_seq)

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), dtype="int32")
        pos = K.cumsum(K.ones_like(x, dtype='int32'), axis=1)
        return mask * pos


class PositionalWiseFeedForward:

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        self.w1 = Conv1D(ffn_dim, kernel_size=1, activation="relu")
        self.w2 = Conv1D(model_dim, kernel_size=1)
        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNormalization()

    def __call__(self, x):
        """

        :param x: a tensor with shape of [N, T_q, D_q]
        :return:
        """

        output = self.w2(self.w1(x))
        output = self.dropout(output)
        output = self.layer_norm(Add()([output, x]))
        return output


class EncoderLayer:

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def __call__(self, inputs, attn_mask=None):
        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder:

    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        self.encoder_layers = [EncoderLayer(model_dim, num_heads, ffn_dim, dropout)
                               for _ in range(num_layers)]
        self.seq_embedding = Embedding(vocab_size + 1, model_dim)
        self.pos_embedding = PositionalEncoding(max_seq_len, model_dim)

    def __call__(self, inputs, mask=None):

        rank = len(K.int_shape(inputs))

        if rank == 2:
            seq_emb = self.seq_embedding(inputs)
            pos_emb = self.pos_embedding(inputs)

            output = Add()([seq_emb, pos_emb])

            self_attention_mask = Lambda(lambda x: padding_mask(x, x))(inputs)

        elif rank == 3:
            output = inputs
            self_attention_mask = mask
        else:
            raise ValueError("Rank must be 2 or 3!")

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        return output, attentions


class DecoderLayer:

    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def __call__(self, dec_inputs, enc_outputs, self_attn_mask=None, context_attn_mask=None):
        """

        :param dec_inputs: [N, T_t, dim_model]
        :param enc_outputs: [N, T_s, dim_model]
        :param self_attn_mask: [N, T_t, T_t]
        :param context_attn_mask: [N, T_t, T_s]
        :return:
        """

        # self attention, all inputs are decoder inputs , [N, T_t, dim_model]
        dec_output, self_attention = self.attention(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)

        # context attention
        # query is decoder's inputs, key and value are encoder's inputs
        dec_output, context_attention = self.attention(dec_output, enc_outputs, enc_outputs, context_attn_mask)

        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention


class Decoder:

    def __init__(self, vocab_size,
                 max_seq_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        self.num_layers = num_layers
        self.decoder_layers = [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        self.seq_embedding = Embedding(vocab_size + 1, model_dim)
        self.pos_embedding = PositionalEncoding(max_seq_len, model_dim)

    def __call__(self, inputs, enc_output, context_attn_mask=None):
        """

        :param inputs: [N, T_t]
        :param enc_output: [N, T_s, dim_model]
        :param context_attn_mask: [N, T_t, T_s]
        :return:
        """

        seq_emb = self.seq_embedding(inputs)  # [N, T_t, dim_model]
        pos_emb = self.pos_embedding(inputs)

        output = Add()([seq_emb, pos_emb])  # [N, T_t, dim_model]

        self_attention_padding_mask = Lambda(lambda x: padding_mask(x, x),
                                             name="self_attention_padding_mask")(inputs)  # [N, T_t, T_t]
        seq_mask = Lambda(lambda x: sequence_mask(x),
                          name="sequence_mask")(inputs)
        # self_attn_mask = Add(name="self_attn_mask")([self_attention_padding_mask, seq_mask])  # [N, T_t, T_t]
        self_attn_mask = Lambda(lambda x: K.minimum(x[0], x[1]))(
            [self_attention_padding_mask, seq_mask])  # [N, T_t, T_t]

        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(output, enc_output, self_attn_mask, context_attn_mask)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)

        return output, self_attentions, context_attentions


def _get_loss(args):
    y_pred, y_true = args
    y_true = tf.cast(y_true, 'int32')
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
    loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
    loss = K.mean(loss)
    return loss


def _get_accuracy(args):
    y_pred, y_true = args
    mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
    corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
    corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
    return K.mean(corr)
