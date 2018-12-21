import json
import re
from concurrent.futures import ThreadPoolExecutor

import keras.backend as K
import numpy as np
from keras import Input, Model
from keras.layers import Dense, Embedding
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras_contrib.layers import CRF
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras_transformer.position import TransformerCoordinateEmbedding
from keras_transformer.transformer import TransformerBlock, TransformerACT

from segmenter.utils import load_dictionary


def label_smoothing_loss(y_true, y_pred):
    shape = K.int_shape(y_pred)
    n_class = shape[2]
    eps = 0.1
    y_true = y_true * (1 - eps) + eps / n_class
    return categorical_crossentropy(y_true, y_pred)


class TFSegmenter:

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 max_seq_len,
                 model_dim=256,
                 max_depth=8,
                 num_heads=8,
                 residual_dropout=0.0,
                 attention_dropout=0.0,
                 compression_window_size: int = None,
                 use_masking: bool = True,
                 use_crf=True,
                 label_smooth=False,
                 optimizer=Adam(),
                 src_tokenizer: Tokenizer = None,
                 tgt_tokenizer: Tokenizer = None,
                 weights_path=None,
                 num_gpu=1):

        """

        :param src_vocab_size: 源词汇量大小
        :param tgt_vocab_size: 标签词汇量大小
        :param max_seq_len: 最大句子长度
        :param model_dim:   输入大小
        :param num_heads:   多头注意力头数
        :param use_crf:     是否使用随机向量场层作为最后的输出
        :param optimizer:   优化函数
        :param src_tokenizer:   源字典
        :param tgt_tokenizer:   目标字典
        :param weights_path:    权重载入路径
        :param num_gpu:         gpu数量
        """

        self.optimizer = optimizer
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.max_depth = max_depth
        self.label_smooth = label_smooth
        self.num_gpu = num_gpu
        self.model_dim = model_dim
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        self.use_masking = use_masking
        self.compression_window_size = compression_window_size
        self.use_crf = use_crf
        self.linear = Dense(tgt_vocab_size + 1, use_bias=False, activation="softmax")
        self.crf = CRF(tgt_vocab_size + 1, sparse_target=False)

        self.model, self.parallel_model = self.__build_model()

        if weights_path is not None:
            try:
                self.model.load_weights(weights_path)
            except Exception as e:
                print(e)
                print("Fail to load weights, create a new model!")

    def __build_model(self):
        assert self.max_depth >= 1, "The parameter max_depth is at least 1"

        src_seq_input = Input(shape=(self.max_seq_len,), dtype="int32", name="src_seq_input")

        src_seq_embed = Embedding(self.src_vocab_size + 1, self.model_dim)(src_seq_input)

        transformer_block = TransformerBlock(
            name='transformer',
            num_heads=self.num_heads,
            residual_dropout=self.residual_dropout,
            attention_dropout=self.attention_dropout,
            use_masking=self.use_masking)

        add_coordinate_embedding = TransformerCoordinateEmbedding(
            self.max_depth,
            name='coordinate_embedding')

        act_layer = TransformerACT()

        next_input = src_seq_embed
        for step in range(self.max_depth):
            next_input = add_coordinate_embedding(next_input, step=step)
            next_input = transformer_block(next_input)
            next_input, act_weighted_output = act_layer(next_input)
        act_layer.finalize()

        if self.use_crf:
            y_pred = self.crf(act_weighted_output)
        else:
            y_pred = self.linear(act_weighted_output)

        model = Model(src_seq_input, y_pred)
        parallel_model = model
        if self.num_gpu > 1:
            parallel_model = multi_gpu_model(model, gpus=self.num_gpu)

        if self.use_crf:
            parallel_model.compile(self.optimizer, loss=self.crf.loss_function, metrics=[self.crf.accuracy])
        else:
            if self.label_smooth:
                parallel_model.compile(optimizer=self.optimizer, loss=label_smoothing_loss, metrics=['accuracy'])
            else:
                parallel_model.compile(optimizer=self.optimizer, loss=categorical_crossentropy, metrics=['accuracy'])

        return model, parallel_model

    def decode_sequences(self, sequences):
        sequences = self._seq_to_matrix(sequences)
        output = self.model.predict_on_batch(sequences)  # [N, -1, chunk_size + 1]
        output = np.argmax(output, axis=2)
        return self.tgt_tokenizer.sequences_to_texts(output)

    def _single_decode(self, args, noun_conjoin=True):
        sent, tag = args
        cur_sent, cur_tag = [], []
        tag = tag.split(' ')
        t1, pre_pos = [], None
        for i in range(len(sent)):
            tokens = tag[i].split('-')
            if len(tokens) == 2:
                c, pos = tokens
            else:
                c = 'i'
                pos = "<UNK>"

            word = sent[i]
            if c == 's':
                if len(t1) != 0:
                    cur_sent.append(''.join(t1))
                    cur_tag.append(pre_pos)
                t1 = [word]
                pre_pos = pos
            elif c == 'i':
                t1.append(word)
                pre_pos = pos
            elif c == 'b':
                if len(t1) != 0:
                    cur_sent.append(''.join(t1))
                    cur_tag.append(pre_pos)
                t1 = [word]
                pre_pos = pos

        if len(t1) != 0:
            cur_sent.append(''.join(t1))
            cur_tag.append(pre_pos)

        if noun_conjoin:
            return self.__noun_conjoin(cur_sent, cur_tag)
        return cur_sent, cur_tag

    def __noun_conjoin(self, sent, tags):
        ret_sent = []
        ret_tags = []
        pre_word = None
        pre_tag = None
        for word, tag in zip(sent, tags):
            if tag == 'vn':
                if pre_word is not None:
                    ret_sent.append(pre_word)
                    ret_tags.append(pre_tag)
                pre_word = word
                pre_tag = tag
                continue
            elif tag == 'n':
                if pre_word is not None:
                    pre_word += word
                    pre_tag = tag
                else:
                    ret_sent.append(word)
                    ret_tags.append(tag)
            else:
                if pre_word is not None:
                    ret_sent.append(pre_word)
                    ret_tags.append(pre_tag)
                    pre_word = None
                    pre_tag = None
                ret_sent.append(word)
                ret_tags.append(tag)

        return ret_sent, ret_tags

    def decode_texts(self, texts, noun_conjoin=True):
        sents = []
        with ThreadPoolExecutor() as executor:
            for text in executor.map(lambda x: list(re.subn("\s+", "", x)[0]), texts):
                sents.append(text)
        sequences = self.src_tokenizer.texts_to_sequences(sents)
        tags = self.decode_sequences(sequences)

        ret = []
        with ThreadPoolExecutor() as executor:
            for cur_sent, cur_tag in executor.map(lambda x: self._single_decode(x, noun_conjoin),
                                                  zip(sents, tags)):
                ret.append((cur_sent, cur_tag))

        return ret

    def _seq_to_matrix(self, sequences):
        # max_len = len(max(sequences, key=len))
        return pad_sequences(sequences, maxlen=self.max_seq_len, padding="post")

    def get_config(self):
        return {
            'src_vocab_size': self.src_vocab_size,
            'tgt_vocab_size': self.tgt_vocab_size,
            'max_seq_len': self.max_seq_len,
            'max_depth': self.max_depth,
            'model_dim': self.model_dim,
            'residual_dropout': self.residual_dropout,
            'attention_dropout': self.attention_dropout,
            'compression_window_size': self.compression_window_size,
            'use_masking': self.use_masking,
            'num_heads': self.num_heads,
            'use_crf': self.use_crf,
            'label_smooth': self.label_smooth
        }

    __singleton = None

    @staticmethod
    def get_or_create(config, src_dict_path=None,
                      tgt_dict_path=None,
                      weights_path=None,
                      num_gpu=1,
                      optimizer=Adam(),
                      encoding="utf-8"):
        if TFSegmenter.__singleton is None:
            if type(config) == str:
                with open(config, encoding=encoding) as file:
                    config = dict(json.load(file))
            elif type(config) == dict:
                config = config
            else:
                raise ValueError("Unexpect config type!")

            if src_dict_path is not None:
                src_tokenizer = load_dictionary(src_dict_path, encoding)
                config['src_tokenizer'] = src_tokenizer
            if tgt_dict_path is not None:
                config['tgt_tokenizer'] = load_dictionary(tgt_dict_path, encoding)

            config["num_gpu"] = num_gpu
            config['weights_path'] = weights_path
            config['optimizer'] = optimizer
            TFSegmenter.__singleton = TFSegmenter(**config)
        return TFSegmenter.__singleton


get_or_create = TFSegmenter.get_or_create


def save_config(obj, config_path, encoding="utf-8"):
    with open(config_path, mode="w+", encoding=encoding) as file:
        json.dump(obj.get_config(), file)
