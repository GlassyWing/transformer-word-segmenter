import json
import re
from concurrent.futures import ThreadPoolExecutor

import keras.backend as K
import numpy as np
from keras import Input, Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras_contrib.layers import CRF
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy

from segmenter.tools import load_dictionary
from transformer.core import Encoder


def _get_loss(args, crf_loss):
    y_pred, y_true = args
    loss = crf_loss(y_true, y_pred)
    loss = K.mean(loss)
    return loss


def _get_accuracy(args, crf_corr):
    y_pred, y_true = args
    corr = crf_corr(y_true, y_pred)
    return K.mean(corr)


class TFSegmenter:

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 max_seq_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0,
                 optimizer=Adam(),
                 src_tokenizer: Tokenizer = None,
                 tgt_tokenizer: Tokenizer = None,
                 weights_path=None,
                 num_gpu=1
                 ):
        self.optimizer = optimizer
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.num_gpu = num_gpu
        self.encoder = Encoder(src_vocab_size, max_seq_len, num_layers, model_dim, num_heads, ffn_dim, dropout)
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
        src_seq_input = Input(shape=(None,), dtype="int32", name="src_seq_input")

        enc_output, _ = self.encoder(src_seq_input)
        y_pred = self.linear(enc_output)
        # y_pred = self.crf(y_pred)

        model = Model(src_seq_input, y_pred)
        parallel_model = model
        if self.num_gpu > 1:
            parallel_model = multi_gpu_model(model, gpus=self.num_gpu)

        parallel_model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        # parallel_model.compile(self.optimizer, loss=self.crf.loss_function, metrics=[self.crf.accuracy])

        return model, parallel_model

    def decode_sequences(self, sequences):
        sequences = self._seq_to_matrix(sequences)
        output = self.model.predict_on_batch(sequences)  # [N, -1, chunk_size + 1]
        output = np.argmax(output, axis=2)
        return self.tgt_tokenizer.sequences_to_texts(output)

    def _single_decode(self, args):
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

        return cur_sent, cur_tag

    def decode_texts(self, texts):
        sents = []
        with ThreadPoolExecutor() as executor:
            for text in executor.map(lambda x: list(re.subn("\s+", "-", x)[0]), texts):
                sents.append(text)
        sequences = self.src_tokenizer.texts_to_sequences(sents)
        tags = self.decode_sequences(sequences)

        ret = []
        with ThreadPoolExecutor() as executor:
            for cur_sent, cur_tag in executor.map(self._single_decode, zip(sents, tags)):
                ret.append((cur_sent, cur_tag))

        return ret

    def _seq_to_matrix(self, sequences):
        max_len = len(max(sequences, key=len))
        return pad_sequences(sequences, maxlen=max_len, padding="post")

    def get_config(self):
        return {
            'src_vocab_size': self.src_vocab_size,
            'tgt_vocab_size': self.tgt_vocab_size,
            'max_seq_len': self.max_seq_len,
            'num_layers': self.num_layers,
            'model_dim': self.model_dim,
            'num_heads': self.num_heads,
            'ffn_dim': self.ffn_dim,
            'dropout': self.dropout
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


if __name__ == '__main__':
    TFSegmenter(10, 10, 100, num_layers=2, num_heads=1).model.summary(line_length=120)
