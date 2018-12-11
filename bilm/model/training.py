from bilm.data_process.vocabulary import Vocabulary
from bilm.data_process.unicodecharsvocabulary import UnicodeCharsVocabulary
import os, json
import tensorflow as tf
from bilm.model.languagemodel import LanguageModel
from bilm.model.util import print_variable_summary, average_gradients, clip_grads, summary_gradient_updates, _get_feed_dict_from_X
import numpy as np
import time


def train(options, data, n_gpus, tf_save_dir, tf_log_dir,
          restart_ckpt_file=None):
    # if restart_ckpt_file is None:
    #     with open(os.path.join(tf_save_dir, 'options.json'), 'w') as fout:
    #
    #         fout.write(json.dumps(options))

    with tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        lr = options.learning_rate
        opt = tf.train.AdagradOptimizer(learning_rate=lr,
                                          initial_accumulator_value=1.0)
        tower_grads = []
        models = []
        train_perplexity = tf.get_variable('train_perplexity', [],
                                           initializer=tf.constant_initializer(0.0),
                                           trainable=False)
        norm_summaries = []
        for k in range(n_gpus):
            with tf.device('/gpu:%d' % k):
                with tf.variable_scope('lm', reuse=k > 0):
                    model = LanguageModel(options, True)
                    loss = model.total_loss
                    models.append(model)
                    grads = opt.compute_gradients(
                        loss * options.unroll_steps,
                        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                    )
                    tower_grads.append(grads)
                    train_perplexity += loss

        print_variable_summary()
        grads = average_gradients(tower_grads, options.batch_size, options)
        grads, norm_summary_ops = clip_grads(grads, options, True, global_step)
        norm_summaries.extend(norm_summary_ops)

        train_perplexity = tf.exp(train_perplexity / n_gpus)
        perplexity_summary = tf.summary.scalar('train_perplexity', train_perplexity)

        histogram_summaries = [
            tf.summary.histogram('token_embedding', models[0].embedding)
        ]
        lstm_out = tf.get_collection('lstm_output_embeddings')
        histogram_summaries.append(
            tf.summary.histogram('lstm_embedding_0', lstm_out[0])
        )
        if options.bidirectional:
            histogram_summaries.append(tf.summary.histogram('lstm_embedding_1',
                                                            lstm_out[1]))
        train_op = opt.apply_gradients(grads, global_step=global_step)

        for v in tf.global_variables():
            histogram_summaries.append(tf.summary.histogram(
                v.name.replace(":", "_"), v))

        histogram_summaries.extend(summary_gradient_updates(grads, opt, lr))

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        summary_op = tf.summary.merge([perplexity_summary] + norm_summaries)
        hist_summary_op = tf.summary.merge(histogram_summaries)
        init = tf.global_variables_initializer()

    bidirectional = options.bidirectional
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init)

        if restart_ckpt_file is not None:
            loader = tf.train.Saver()
            loader.restore(sess, restart_ckpt_file)

        summary_writer = tf.summary.FileWriter(tf_log_dir, sess.graph)
        batch_size = options.batch_size
        unroll_steps = options.unroll_steps
        n_train_tokens = options.n_train_tokens
        n_tokens_per_batch = batch_size * unroll_steps * n_gpus
        n_batches_per_epoch = int(n_train_tokens / n_tokens_per_batch)
        n_batches_total = options.n_epochs * n_batches_per_epoch
        print("Training for %s epochs and %s batches" % (
            options.n_epochs, n_batches_total))

        init_state_tensors = []
        final_state_tensors = []
        for model in models:
            init_state_tensors.extend(model.init_lstm_state)
            final_state_tensors.extend(model.final_lstm_state)

        char_inputs = True if hasattr(options, 'char_cnn') else False
        if char_inputs:
            max_chars = options.char_cnn['max_characters_per_token']

        if not char_inputs:
            feed_dict = {
                model.token_ids:np.zeros([batch_size, unroll_steps], dtype=np.int64)
                for model in models
            }
        else:
            # print("================================================")
            feed_dict = {
                model.tokens_characters:
                    np.zeros([batch_size, unroll_steps, max_chars],
                             dtype=np.int32)
                for model in models
            }
        if bidirectional:
            if not char_inputs:
                feed_dict.update({
                    model.token_ids_reverse:np.zeros([batch_size, unroll_steps],
                                                     dtype=np.int64)
                    for model in models
                })
            else:
                feed_dict.update({
                    model.tokens_characters_reverse:np.zeros([batch_size, unroll_steps, max_chars],
                                                             dtype=np.int32)
                    for model in models
                })

        init_state_values = sess.run(init_state_tensors, feed_dict=feed_dict)

        t1 = time.time()
        data_gen = data.iter_batches(batch_size * n_gpus, unroll_steps)
        for batch_no, batch in enumerate(data_gen, start=1):
            X = batch
            feed_dict = {t:v for t, v in zip(init_state_tensors, init_state_values)}

            for k in range(n_gpus):
                model = models[k]
                start = k * batch_size
                end = (k + 1) * batch_size

                feed_dict.update(_get_feed_dict_from_X(X, start, end, model,
                                                       char_inputs, bidirectional))

            if batch_no % 1250 != 0:
                ret = sess.run(
                    [train_op, summary_op, train_perplexity] +
                    final_state_tensors,
                    feed_dict=feed_dict
                )
                init_state_values = ret[3:]

            else:
                ret = sess.run([train_op, summary_op, train_perplexity, hist_summary_op] + final_state_tensors,
                               feed_dict=feed_dict)
                init_state_values = ret[4:]

            if batch_no % 1250 == 0:
                summary_writer.add_summary(ret[3], batch_no)
            if batch_no % 100 == 0:
                summary_writer.add_summary(ret[1], batch_no)
                print("Batch %s, train_perplexity=%s" % (batch_no, ret[2]))
                print("Total time: %s" % (time.time() - t1))

            if (batch_no % 1250 == 0) or (batch_no == n_batches_total):
                # save the model
                checkpoint_path = os.path.join(tf_save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)

            if batch_no == n_batches_total:
                # done training!
                break



