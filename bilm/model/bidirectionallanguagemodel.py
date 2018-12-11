import numpy as np
import tensorflow as tf
import h5py
from bilm.model.bidirectionallanguagemodelgraph import BidirectionalLanguageModelGraph
import json
from bilm.data_process.batcher import Batcher
from bilm.data_process.bidirectionallmdataset import InvalidNumberOfCharacters
from bilm.data_process.unicodecharsvocabulary import UnicodeCharsVocabulary


DTYPE = 'float32'
DTYPE_INT = 'int64'


class BidirectionalLanguageModel(object):
    def __init__(self, options_file:str,
                 weight_file:str,
                 use_character_inputs=True,
                 embedding_weight_file=None,
                 max_batch_size=128):
        with open(options_file, 'r') as fin:
            options = json.load(fin)
        options = tf.contrib.training.HParams(**options)
        if not use_character_inputs:
            if embedding_weight_file is None:
                raise ValueError(
                    "embedding_weight_file is required input with "
                    "not use_character_inputs"
                )
        self._options = options
        self._weight_file = weight_file
        self._embedding_weight_file = embedding_weight_file
        self._use_character_inputs = use_character_inputs
        self._max_batch_size = max_batch_size

        self._ops = {}
        self._graphs = {}

    def __call__(self, ids_placeholder):
        if ids_placeholder in self._ops:
            ret = self._ops[ids_placeholder]
        else:
            if len(self._ops) == 0:
                lm_graph = BidirectionalLanguageModelGraph(
                    self._options,
                    self._weight_file,
                    ids_placeholder,
                    embedding_weight_file=self._embedding_weight_file,
                use_character_inputs=self._use_character_inputs,
                max_batch_size=self._max_batch_size)
            else:
                with tf.variable_scope('', reuse=True):
                    lm_graph = BidirectionalLanguageModelGraph(
                        self._options,
                        self._weight_file,
                        ids_placeholder,
                        embedding_weight_file=self._embedding_weight_file,
                        use_character_inputs=self._use_character_inputs,
                        max_batch_size=self._max_batch_size
                    )
            ops = self._build_ops(lm_graph)
            self._ops[ids_placeholder] = ops
            self._graphs[ids_placeholder] = lm_graph
            ret = ops

        return ret

    def _build_ops(self, lm_graph):
        with tf.control_dependencies([lm_graph.update_state_op]):
            token_embeddings = lm_graph.embedding
            layers = [tf.concat([token_embeddings, token_embeddings], axis=2)]
            n_lm_layers = len(lm_graph.lstm_outputs['forward'])
            for i in range(n_lm_layers):
                layers.append(tf.concat([lm_graph.lsmt_outputs['forward'][i],
                                         lm_graph.lstm_outputs['backward'][i]],
                                        axis=-1))

            sequence_length_wo_bos_eos = lm_graph.sequence_lengths - 2
            layers_without_bos_eos = []
            for layer in layers:
                layer_wo_bos_eos = layer[:, 1:, :]
                layer_wo_bos_eos = tf.reverse_sequence(layer_wo_bos_eos,
                                                       lm_graph.sequence_lengths - 1,
                                                       seq_axis=1,
                                                       batch_axis=0)
                layer_wo_bos_eos = layer_wo_bos_eos[:, 1:, :]
                layer_wo_bos_eos = tf.reverse_sequence(layer_wo_bos_eos,
                                                       sequence_length_wo_bos_eos,
                                                       seq_axis=1,
                                                       batch_axis=0)
                layers_without_bos_eos.append(layer_wo_bos_eos)

            lm_embeddings = tf.concat([tf.expand_dims(t, axis=1) for t in layers_without_bos_eos],
                                      axis=1)
            mask_wo_bos_eos = tf.cast(lm_graph.mask[:, 1:], 'int32')
            mask_wo_bos_eos = tf.reverse_sequence(mask_wo_bos_eos,
                                                  lm_graph.sequence_lengths - 1,
                                                  seq_axis=1,
                                                  batch_axis=0)
            mask_wo_bos_eos = mask_wo_bos_eos[:, 1:]
            mask_wo_bos_eos = tf.reverse_sequence(mask_wo_bos_eos,
                                                  sequence_length_wo_bos_eos,
                                                  seq_axis=1,
                                                  batch_axis=0)
            mask_wo_bos_eos = tf.cast(mask_wo_bos_eos, 'bool')

        return {
            'lm_embeddings': lm_embeddings,
            'lengths': sequence_length_wo_bos_eos,
            'token_embeddings': lm_graph.embedding,
            'mask': mask_wo_bos_eos
        }


def dump_bilm_embeddings(vocab_file, dataset_file, options_file,
                         weight_file, outfile):
    with open(options_file, 'r') as fin:
        options = json.load(fin)
    options = tf.contrib.training.HParams(**options)
    max_word_length = options.char_cnn['max_characters_per_token']

    vocab = UnicodeCharsVocabulary(vocab_file, max_word_length)
    batcher = Batcher(vocab_file, max_word_length)
    ids_placeholder = tf.placeholder('int32',
                                     shape=(None, None, max_word_length))
    model = BidirectionalLanguageModel(options_file, weight_file)
    ops = model(ids_placeholder)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sentence_id = 0
        with open(dataset_file, 'r') as fin, h5py.File(outfile, 'w') as fout:
            for line in fin:
                sentence = line.strip().split()
                char_ids = batcher.batch_sentences([sentence])
                embeddings = sess.run(
                    ops['lm_embeddings'], feed_dict={ids_placeholder: char_ids}
                )
                ds = fout.create_dataset(
                    '{}'.format(sentence_id),
                    embeddings.shape[1:], dtype='float32',
                    data=embeddings[0, :, :, :]
                )

                sentence_id += 1


def dump_token_embeddings(vocab_file, options_file, weight_file, outfile):
    with open(options_file, 'r') as fin:
        options = json.load(fin)
    options = tf.contrib.training.HParams(**options)
    max_word_length = options.char_cnn['max_characters_per_token']

    vocab = UnicodeCharsVocabulary(vocab_file, max_word_length)
    batcher = Batcher(vocab_file, max_word_length)
    ids_placeholder = tf.placeholder('int32',
                                     shape=(None, None, max_word_length))
    model = BidirectionalLanguageModel(options_file, weight_file)
    embedding_op = model(ids_placeholder)['token_embeddings']
    n_tokens = vocab.size
    embed_dim = int(embedding_op.shape[2])

    embeddings = np.zeros((n_tokens, embed_dim),
                          dtype=DTYPE)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for k in range(n_tokens):
            token = vocab.id_to_word(k)
            char_ids = batcher.batch_sentences([[token]])[0, 1, :].reshape(
                1, 1, -1)
            embeddings[k, :] = sess.run(
                embedding_op, feed_dict={ids_placeholder: char_ids}
            )

    with h5py.File(outfile, 'w') as fout:
        ds = fout.create_dataset(
            'embedding', embeddings.shape, dtype='float32', data=embeddings
        )

