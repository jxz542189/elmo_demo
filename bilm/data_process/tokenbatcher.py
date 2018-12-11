from bilm.data_process.vocabulary import Vocabulary
from typing import List
import numpy as np


class TokenBatcher(object):
    def __init__(self, lm_vocab_file:str):
        self._lm_vocab = Vocabulary(lm_vocab_file)

    def batch_sentences(self, sentences: List[List[str]]):
        n_sentences = len(sentences)
        max_length = max(len(sentence) for sentence in sentences) + 2

        X_ids = np.zeros((n_sentences, max_length), dtype=np.int64)

        for k, sent in enumerate(sentences):
            length = len(sent) + 2
            ids_without_mask = self._lm_vocab.encode(sent, split=False)
            X_ids[k, :length] = ids_without_mask + 1
        return X_ids

