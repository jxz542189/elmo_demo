from bilm.data_process.lmdataset import LMDataset, _get_batch


class BidirectionalLMDataset(object):
    def __init__(self, filepattern, vocab, test=False, shuffle_on_load=False):
        self.data_forward = LMDataset(filepattern, vocab, reverse=False, test=test,
                                       shuffle_on_load=shuffle_on_load)
        self.data_reverse = LMDataset(filepattern, vocab, reverse=True, test=test,
                                       shuffle_on_load=shuffle_on_load)

    def iter_batches(self, batch_size, num_steps):
        max_word_length = self.data_forward.max_word_length

        for X, Xr in zip(_get_batch(self.data_forward.get_sentence(), batch_size,
                                    num_steps, max_word_length),
                         _get_batch(self.data_reverse.get_sentence(), batch_size,
                                    num_steps, max_word_length)):
            for k, v in Xr.items():
                X[k + '_reverse'] = v
            yield X


class InvalidNumberOfCharacters(Exception):
    pass