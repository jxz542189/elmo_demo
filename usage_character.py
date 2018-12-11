import tensorflow as tf
import os
from bilm.data_process.batcher import Batcher
from bilm.model.bidirectionallanguagemodel import BidirectionalLanguageModel
from bilm.model.elmo import weight_layers


datadir = '.'
vocab_file = os.path.join(datadir, 'vocab/cn_vocab')
options_file = os.path.join(datadir, 'bilm/save_dir/options.json')
weight_file = os.path.join(datadir, 'bilm/save_dir/weights.hdf5')

batcher = Batcher(vocab_file, 10)

context_character_ids = tf.placeholder('int32', shape=(None, None, 10))
question_character_ids = tf.placeholder('int32', shape=(None, None, 10))

bilm = BidirectionalLanguageModel(options_file, weight_file)

context_embeddings_op = bilm(context_character_ids)
question_embeddings_op = bilm(question_character_ids)

elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
with tf.variable_scope('', reuse=True):
    elmo_question_input = weight_layers('input', question_embeddings_op,
                                        l2_coef=0.0)

elmo_context_output = weight_layers('output', context_embeddings_op,
                                    l2_coef=0.0)
with tf.variable_scope('', reuse=True):
    elmo_question_output = weight_layers('output', question_embeddings_op,
                                         l2_coef=0.0)

raw_context = [
    'Pretrained biLMs compute representations useful for NLP tasks .',
    'They give state of the art performance for many tasks .'
]
tokenized_context = [sentence.split() for sentence in raw_context]
tokenized_question = [
    ['What', 'are', 'biLMs', 'useful', 'for', '?'],
]

with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())

    # Create batches of data.
    context_ids = batcher.batch_sentences(tokenized_context)
    question_ids = batcher.batch_sentences(tokenized_question)

    # Compute ELMo representations (here for the input only, for simplicity).
    elmo_context_input_, elmo_question_input_ = sess.run(
        [elmo_context_input['weighted_op'], elmo_question_input['weighted_op']],
        feed_dict={context_character_ids: context_ids,
                   question_character_ids: question_ids}
    )
    print(elmo_context_input_.shape)
    print(elmo_question_input_.shape)