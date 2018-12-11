from bilm.data_process.vocabulary import Vocabulary
import numpy as np
from bilm.utils.recursive_cut import recursive_cut


class UnicodeCharsVocabulary(Vocabulary):
    def __init__(self, filename, max_word_length, **kwargs):
        super(UnicodeCharsVocabulary, self).__init__(filename, **kwargs)
        self._max_word_length = max_word_length

        self.bos_char = 256
        self.eos_char = 257
        self.bow_char = 258
        self.eow_char = 259
        self.pad_char = 260
        self.count2 = 0
        num_words = len(self._id_to_word)

        self._word_char_ids = np.zeros([num_words, max_word_length],
                                       dtype=np.int32)

        def _make_bos_eos(c):
            r = np.zeros([self._max_word_length], dtype=np.int32)
            r[:] = self.pad_char
            r[0] = self.bow_char
            r[1] = c
            r[2] = self.eow_char
            return r

        self.bos_chars = _make_bos_eos(self.bos_char)
        self.eos_chars = _make_bos_eos(self.eos_char)

        for i, word in enumerate(self._id_to_word):
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)

        self._word_char_ids[self.bos] = self.bos_chars
        self._word_char_ids[self.eos] = self.eos_chars

    @property
    def word_char_ids(self):
        return self._word_char_ids

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = self.pad_char

        word_encoded = word.encode('utf-8', 'ignore')[:(self.max_word_length-2)]
        code[0] = self.bow_char
        for k, chr_id in enumerate(word_encoded, start=1):
            code[k] = chr_id
        code[len(word_encoded) + 1] = self.eow_char
        return code

    def word_to_char_ids(self, word):
        if word in self._word_to_id:
            return self._word_char_ids[self._word_to_id[word]]
        else:
            return self._convert_word_to_char_ids(word)

    def encode_chars(self, sentence, reverse=False, split=True):
        if split:
            chars_ids = [self.word_to_char_ids(cur_word)
                         for cur_word in sentence]
        else:
            chars_ids = [self.word_to_char_ids(cur_word)
                         for cur_word in sentence]
        self.count2 += 1
        if reverse:
            return np.vstack([self.eos_chars] + chars_ids + [self.bos_chars])
        else:
            return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])
