from bilm.data_process.unicodecharsvocabulary import UnicodeCharsVocabulary
from bilm.data_process.vocabulary import Vocabulary
from typing import List
import numpy as np


class Batcher(object):
    def __init__(self, lm_vocab_file: str, max_token_length: int):
        self._lm_vocab = UnicodeCharsVocabulary(lm_vocab_file, max_token_length)
        self._max_token_length = max_token_length

    def batch_sentences(self, sentences):
        n_sentences = len(sentences)
        max_length = max(len(sentence) for sentence in sentences) + 2

        X_char_ids = np.zeros((n_sentences, max_length, self._max_token_length),
                              dtype=np.int64)

        for k, sent in enumerate(sentences):
            length = len(sent) + 2
            char_ids_without_mask = self._lm_vocab.encode_chars(sent, split=False)
            X_char_ids[k, :length, :] = char_ids_without_mask + 1

        return X_char_ids

