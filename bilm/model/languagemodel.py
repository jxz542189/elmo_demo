import tensorflow as tf
from bilm.data_process.bidirectionallmdataset import InvalidNumberOfCharacters
import numpy as np


DTYPE = 'float32'
DTYPE_INT = 'int64'

tf.logging.set_verbosity(tf.logging.INFO)


class LanguageModel(object):
    def __init__(self, options, is_training):
        self.options = options
        self.is_training = is_training
        self.bidirectional = options.bidirectional
        self.char_inputs = True if hasattr(options, 'char_cnn') else False
        # self.char_inputs = options.char_cnn
        self.share_embedding_softmax = True if options.share_embedding_softmax else False
        if self.char_inputs and self.share_embedding_softmax:
            raise ValueError("Sharing softmax and embedding weights requires "
                             "word input")

        self.sample_softmax = options.sample_softmax
        self._build()

    def _build_word_embeddings(self):
        n_tokens_vocab = self.options.n_tokens_vocab
        batch_size = self.options.batch_size
        unroll_steps = self.options.unroll_steps

        projection_dim = self.options.lstm['projection_dim']

        self.token_ids = tf.placeholder(DTYPE_INT, shape=(batch_size, unroll_steps),
                                        name='token_ids')

        with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable("embedding",
                                                     [n_tokens_vocab, projection_dim],
                                                     dtype=DTYPE)
            self.embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                    self.token_ids)

        if self.bidirectional:
            self.token_ids_reverse = tf.placeholder(DTYPE_INT,
                                                    shape=(batch_size, unroll_steps),
                                                    name="token_ids_reverse")
            with tf.device("/cpu:0"):
                self.embedding_reverse = tf.nn.embedding_lookup(self.embedding_weights,
                                                                self.token_ids_reverse)

    def _build_word_char_embeddings(self):
        batch_size = self.options.batch_size
        unroll_steps = self.options.unroll_steps
        projection_dim = self.options.lstm["projection_dim"]

        cnn_options = self.options.char_cnn
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        max_chars = cnn_options["max_characters_per_token"]
        char_embed_dim = cnn_options['embedding']['dim']
        n_chars = cnn_options["n_characters"]
        if self.is_training:
             if n_chars != 261:
                 raise InvalidNumberOfCharacters(
                     "Set n_characters=261 for training see the README.md")
        if cnn_options['activation'] == 'tanh':
            activation = tf.nn.tanh
        elif cnn_options['activation'] == 'relu':
            activation = tf.nn.relu

        self.tokens_characters = tf.placeholder(DTYPE_INT,
                                                shape=(batch_size, unroll_steps,
                                                       max_chars),
                                                name='tokens_characters')
        print("===============tokens_characters=========================")
        print(self.tokens_characters)

        with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable("char_embed",
                                                     [n_chars, char_embed_dim],
                                                     dtype=DTYPE,
                                                     initializer=tf.random_uniform_initializer(-1.0, 1.0))
            self.char_embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                         self.tokens_characters)

            if self.bidirectional:
                self.tokens_characters_reverse = tf.placeholder(DTYPE_INT,
                                                                shape=(batch_size, unroll_steps, max_chars),
                                                                name="tokens_characters_reverse")
                self.char_embedding_reverse = tf.nn.embedding_lookup(self.embedding_weights,
                                                                     self.tokens_characters_reverse)

        def make_convolutions(inp, reuse):
            with tf.variable_scope('CNN', reuse=reuse) as scope:
                convolutions = []
                for i, (width, num) in enumerate(filters):
                    if cnn_options['activation'] == 'relu':
                        w_init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)
                    elif cnn_options['activation'] == 'tanh':
                        w_init = tf.random_normal_initializer(
                            mean=0.0,
                            stddev=np.sqrt(1.0 / (width * char_embed_dim))
                        )
                    w = tf.get_variable(
                        "W_cnn_%s" % i,
                        [1, width, char_embed_dim, num],
                        initializer=w_init,
                        dtype=DTYPE)
                    b = tf.get_variable(
                        "b_cnn_%s" % i, [num], dtype=DTYPE,
                        initializer=tf.constant_initializer(0.0))

                    conv = tf.nn.conv2d(
                            inp, w,
                            strides=[1, 1, 1, 1],
                            padding="VALID") + b
                    # now max pool
                    conv = tf.nn.max_pool(
                            conv, [1, 1, max_chars-width+1, 1],
                            [1, 1, 1, 1], 'VALID')

                    # activation
                    conv = activation(conv)
                    conv = tf.squeeze(conv, squeeze_dims=[2])

                    convolutions.append(conv)

            return tf.concat(convolutions, 2)
        reuse = tf.get_variable_scope().reuse
        embedding = make_convolutions(self.char_embedding, reuse)

        self.token_embedding_layers = [embedding]

        if self.bidirectional:
            embedding_reverse = make_convolutions(self.char_embedding_reverse, True)

        n_highway = cnn_options['n_highway']
        use_highway = n_highway is not None and n_highway > 0
        use_proj = n_filters != projection_dim

        if use_highway or use_proj:
            embedding = tf.reshape(embedding, [-1, n_filters])
            if self.bidirectional:
                embedding_reverse = tf.reshape(embedding_reverse,
                                               [-1, n_filters])

        if use_proj:
            assert n_filters > projection_dim
            with tf.variable_scope('CNN_proj') as scope:
                W_proj_cnn = tf.get_variable("W_proj", [n_filters, projection_dim],
                                             initializer=tf.random_normal_initializer(mean=0.0,
                                                                                      stddev=np.sqrt(1.0 / n_filters)),
                                             dtype=DTYPE)
                b_proj_cnn = tf.get_variable("b_proj",
                                             [projection_dim],
                                             initializer=tf.constant_initializer(0.0),
                                             dtype=DTYPE)

        def high(x, ww_carry, bb_carry, ww_tr, bb_tr):
            carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
            transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
            return carry_gate * transform_gate + (1.0 - carry_gate) * x

        if use_highway:
            highway_dim = n_filters

            for i in range(n_highway):
                with tf.variable_scope('CNN_high_%s' % i) as scope:
                    W_carry = tf.get_variable(
                        'W_carry', [highway_dim, highway_dim],
                        initializer=tf.random_normal_initializer(mean=0.0,
                                                                 stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=DTYPE
                    )
                    b_carry = tf.get_variable('b_carry', [highway_dim],
                                              initializer=tf.constant_initializer(-0.2),
                                              dtype=DTYPE)
                    W_transform = tf.get_variable('W_transform',
                                                  [highway_dim, highway_dim],
                                                  initializer=tf.random_normal_initializer(mean=0.0,
                                                                                           stddev=np.sqrt(1.0 / highway_dim)),
                                                  dtype=DTYPE)
                    b_transform = tf.get_variable('b_transform', [highway_dim],
                                                  initializer=tf.constant_initializer(0.0),
                                                  dtype=DTYPE)
                embedding = high(embedding, W_carry, b_carry,
                                 W_transform, b_transform)
                if self.bidirectional:
                    embedding_reverse = high(embedding_reverse, W_carry, b_carry,
                                             W_transform, b_transform)
                self.token_embedding_layers.append(tf.reshape(embedding, [batch_size, unroll_steps, highway_dim]))

        if use_proj:
            embedding =tf.matmul(embedding, W_proj_cnn) + b_proj_cnn
            if self.bidirectional:
                embedding_reverse = tf.matmul(embedding_reverse, W_proj_cnn) + b_proj_cnn
            self.token_embedding_layers.append(tf.reshape(embedding,
                                                          [batch_size, unroll_steps,
                                                           projection_dim]))
        if use_highway or use_proj:
            shp = [batch_size, unroll_steps, projection_dim]
            embedding = tf.reshape(embedding, shp)
            if self.bidirectional:
                embedding_reverse = tf.reshape(embedding_reverse, shp)

        self.embedding = embedding
        if self.bidirectional:
            self.embedding_reverse = embedding_reverse

    def _build(self):
        n_tokens_vocab = self.options.n_tokens_vocab
        batch_size = self.options.batch_size
        unroll_steps = self.options.unroll_steps

        lstm_dim = self.options.lstm['dim']
        projection_dim = self.options.lstm['projection_dim']
        n_lstm_layers = self.options.lstm.get('n_layers', 1)
        dropout = self.options.dropout
        keep_prob = 1.0 - dropout

        if self.char_inputs:
            self._build_word_char_embeddings()
        else:
            self._build_word_embeddings()

        self.init_lstm_state = []
        self.final_lstm_state = []

        if self.bidirectional:
            lstm_inputs = [self.embedding, self.embedding_reverse]
        else:
            lstm_inputs = [self.embedding]

        cell_clip = self.options.lstm.get('cell_clip')
        proj_clip = self.options.lstm.get('proj_clip')

        use_skip_connections = self.options.lstm.get('use_skip_connections')

        if use_skip_connections:
            print("USING SKIP CONNECTIONS")

        lstm_outputs = []
        for lstm_num, lstm_input in enumerate(lstm_inputs):
            lstm_cells = []
            for i in range(n_lstm_layers):
                if projection_dim < lstm_dim:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_dim, num_proj=projection_dim,
                                                        cell_clip=cell_clip,
                                                        proj_clip=proj_clip)
                else:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_dim,
                                                        cell_clip=cell_clip, proj_clip=proj_clip)
                if use_skip_connections:
                    if i == 0:
                        pass
                    else:
                        lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)

                if self.is_training:
                    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                              input_keep_prob=keep_prob)
                lstm_cells.append(lstm_cell)
            if n_lstm_layers > 1:
                lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
            else:
                lstm_cell = lstm_cells[0]

            with tf.control_dependencies([lstm_input]):
                self.init_lstm_state.append(lstm_cell.zero_state(batch_size, DTYPE))

                if self.bidirectional:
                    with tf.variable_scope('RNN_%s' % lstm_num):
                        _lstm_output_unpacked, final_state = tf.nn.static_rnn(lstm_cell,
                                                                              tf.unstack(lstm_input, axis=1),
                                                                              initial_state=self.init_lstm_state[-1])
                else:
                    _lstm_output_unpacked, final_state = tf.nn.static_rnn(lstm_cell,
                                                                          tf.unstack(lstm_input, axis=1),
                                                                          initial_state=self.init_lstm_state[-1])
                self.final_lstm_state.append(final_state)

            lstm_output_flat = tf.reshape(tf.stack(_lstm_output_unpacked, axis=1),
                                          [-1, projection_dim])
            if self.is_training:
                lstm_output_flat = tf.nn.dropout(lstm_output_flat, keep_prob)
            tf.add_to_collection('lstm_output_embeddings',
                                 _lstm_output_unpacked)
            lstm_outputs.append(lstm_output_flat)

        self._build_loss(lstm_outputs)

    def _build_loss(self, lstm_outputs):
        batch_size = self.options.batch_size
        unroll_steps = self.options.unroll_steps
        n_tokens_vocab = self.options.n_tokens_vocab

        def _get_next_token_placeholders(suffix):
            name = 'next_token_id' + suffix
            id_placeholder = tf.placeholder(DTYPE_INT,
                                            shape=(batch_size, unroll_steps),
                                            name=name)
            return id_placeholder

        self.next_token_id = _get_next_token_placeholders('')
        if self.bidirectional:
            self.next_token_id_reverse = _get_next_token_placeholders('_reverse')

        softmax_dim = self.options.lstm['projection_dim']

        if self.share_embedding_softmax:
            self.softmax_W = self.embedding_weights

        with tf.variable_scope('softmax'), tf.device('/cpu:0'):
            softmax_init = tf.random_normal_initializer(0.0,
                                                        1.0 / np.sqrt(softmax_dim))
            if not self.share_embedding_softmax:
                self.softmax_W = tf.get_variable('W',
                                                 [n_tokens_vocab, softmax_dim],
                                                 dtype=DTYPE,
                                                 initializer=softmax_init)
            self.softmax_b = tf.get_variable('b', [n_tokens_vocab],
                                             dtype=DTYPE,
                                             initializer=tf.constant_initializer(0.0))
        self.individual_losses = []

        if self.bidirectional:
            next_ids = [self.next_token_id, self.next_token_id_reverse]
        else:
            next_ids = [self.next_token_id]

        for id_placeholder, lstm_output_flat in zip(next_ids, lstm_outputs):
            next_token_id_flat = tf.reshape(id_placeholder, [-1, 1])

            with tf.control_dependencies([lstm_output_flat]):
                if self.is_training and self.sample_softmax:
                    losses = tf.nn.sampled_softmax_loss(self.softmax_W, self.softmax_b,
                                                        next_token_id_flat, lstm_output_flat,
                                                        self.options.n_negative_samples_batch,
                                                        self.options.n_tokens_vocab,
                                                        num_true=1)
                else:
                    output_scores = tf.matmul(lstm_output_flat,
                                              tf.transpose(self.softmax_W)
                                              + self.softmax_b)
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=output_scores,
                        labels=tf.squeeze(next_token_id_flat,
                                          squeeze_dims=[1])
                    )

            self.individual_losses.append(tf.reduce_mean(losses))
        if self.bidirectional:
            self.total_loss = 0.5 * (self.individual_losses[0]
                                     + self.individual_losses[1])
        else:
            self.total_loss = self.individual_losses[0]