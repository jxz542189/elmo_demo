import tensorflow as tf
import os
from bilm.data_process.tokenbatcher import TokenBatcher
from bilm.model.elmo import weight_layers
from bilm.model.model import BidirectionalLanguageModel, dump_bilm_embeddings, dump_token_embeddings



raw_context = [
    'Pretrained biLMs compute representations useful for NLP tasks .',
    'They give state of the art performance for many tasks .'
]
tokenized_context = [sentence.split() for sentence in raw_context]
tokenized_question = [
    ['What', 'are', 'biLMs', 'useful', 'for', '?']
]
all_tokens = set(['<S>', '</S>'] + tokenized_question[0])
for context_sentence in tokenized_context:
    for token in context_sentence:
        all_tokens.add(token)
vocab_file = 'vocab/cn_vocab'
datadir = "bilm/save_dir"
options_file = os.path.join(datadir, 'options.json')
weight_file = "/root/PycharmProjects/elmo/bilm/save_dir/weights.hdf5"
token_embedding_file = 'elmo_token_embeddings.hdf5'
dump_token_embeddings(
    vocab_file, options_file, weight_file, outfile = token_embedding_file
)
tf.reset_default_graph()


batcher = TokenBatcher(vocab_file)

context_token_ids = tf.placeholder('int32', shape=(None, None))
question_token_ids = tf.placeholder('int32', shape=(None, None))

bilm = BidirectionalLanguageModel(options_file,
                                  weight_file,
                                  use_character_inputs=False,
                                  embedding_weight_file=token_embedding_file)

context_embeddings_op = bilm(context_token_ids)
question_embeddings_op = bilm(question_token_ids)

elmo_context_input = weight_layers('input', context_embeddings_op,
                                   l2_coef=0.0)
with tf.variable_scope('', reuse=True):
    elmo_question_input = weight_layers('input',
                                        question_embeddings_op,
                                        l2_coef=0.0)

elmo_context_output = weight_layers('output',
                                    context_embeddings_op,
                                    l2_coef=0.0)

with tf.variable_scope('', reuse=True):
    elmo_question_output = weight_layers('output',
                                         question_embeddings_op,
                                         l2_coef=0.0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    context_ids = batcher.batch_sentences(tokenized_context)
    question_ids = batcher.batch_sentences(tokenized_question)

    elmo_context_input_, elmo_question_input_ = sess.run(
        [elmo_context_input['weighted_op'],
         elmo_question_input['weighted_op']],
        feed_dict={context_token_ids:context_ids,
                   question_token_ids:question_ids}
    )
    print(type(elmo_context_input_))
    print(type(elmo_question_input_))
    print(elmo_context_input_.shape)
    print(elmo_question_input_.shape)