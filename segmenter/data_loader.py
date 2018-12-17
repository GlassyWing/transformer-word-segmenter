import os

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from segmenter.tools import load_dictionary


class DataLoader:

    def __init__(self,
                 src_dict_path,
                 tgt_dict_path,
                 batch_size=64,
                 max_len=999,
                 word_delimiter=' ',
                 sent_delimiter='\t',
                 encoding="utf-8",
                 sparse_target=False):
        self.src_tokenizer = load_dictionary(src_dict_path, encoding)
        self.tgt_tokenizer = load_dictionary(tgt_dict_path, encoding)
        self.batch_size = batch_size
        self.max_len = max_len
        self.word_delimiter = word_delimiter
        self.sent_delimiter = sent_delimiter
        self.src_vocab_size = self.src_tokenizer.num_words
        self.tgt_vocab_size = self.tgt_tokenizer.num_words
        self.sparse_target = sparse_target

    def generator(self, file_path, encoding="utf-8"):
        if os.path.isdir(file_path):
            while True:
                for sent, chunk in self.load_sents_from_dir(file_path):
                    yield sent, chunk
        while True:
            for sent, chunk in self.load_sents_from_file(file_path, encoding):
                yield sent, chunk

    def load_sents_from_dir(self, source_dir, encoding="utf-8"):
        for root, dirs, files in os.walk(source_dir):
            for name in files:
                file = os.path.join(root, name)
                for sent, chunk in self.load_sents_from_file(file, encoding=encoding):
                    yield sent, chunk

    def load_sents_from_file(self, file_path, encoding):
        with open(file_path, encoding=encoding) as f:
            sent, chunk = [], []
            for line in f:
                line = line[:-1]
                chars, tags = line.split(self.sent_delimiter)
                sent.append(chars.split(self.word_delimiter))
                chunk.append(tags.split(self.word_delimiter))
                if len(sent) >= self.batch_size:
                    sent = self.src_tokenizer.texts_to_sequences(sent)
                    chunk = self.tgt_tokenizer.texts_to_sequences(chunk)
                    sent, chunk = self._pad_seq(sent, chunk)
                    if not self.sparse_target:
                        chunk = to_categorical(chunk, num_classes=self.tgt_vocab_size + 1)
                    yield sent, chunk
                    sent, chunk = [], []

    def _pad_seq(self, sent, chunk):
        len_sent = min(len(max(sent, key=len)), self.max_len)
        len_chunk = min(len(max(chunk, key=len)), self.max_len)
        sent = pad_sequences(sent, maxlen=len_sent, padding='post')
        chunk = pad_sequences(chunk, maxlen=len_chunk, padding='post')
        return sent, chunk
