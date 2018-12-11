from bilm.model.util import _pretrained_initializer
import h5py
import tensorflow as tf
from bilm.data_process.bidirectionallmdataset import InvalidNumberOfCharacters
import numpy as np


DTYPE = 'float32'
DTYPE_INT = 'int64'


class BidirectionalLanguageModelGraph(object):

    def __init__(self, options, weight_file, ids_placeholder,
                 use_character_inputs=True, embedding_weight_file=None,
                 max_batch_size=128):
        self.options = options
        self._max_batch_size = max_batch_size
        self.ids_placeholder = ids_placeholder
        self.use_character_inputs = use_character_inputs

        def custom_getter(getter, name, *args, **kwargs):
            kwargs['trainable'] = False
            kwargs['initializer'] = _pretrained_initializer(
                name, weight_file, embedding_weight_file
            )
            return getter(name, *args, **kwargs)

        if embedding_weight_file is not None:
            with h5py.File(embedding_weight_file, 'r') as fin:
                self._n_tokens_vocab = fin['embedding'].shape[0] + 1
        else:
            self._n_tokens_vocab = None

        with tf.variable_scope('bilm', custom_getter=custom_getter):
            self._build()

    def _build(self):
        if self.use_character_inputs:
            self._build_word_char_embeddings()
        else:
            self._build_word_embeddings()
        self._build_lstms()

    def _build_word_char_embeddings(self):
        projection_dim = self.options.lstm['projection_dim']

        cnn_options = self.options.char_cnn
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        max_chars = cnn_options['max_characters_per_token']
        char_embed_dim = cnn_options['embedding']['dim']
        n_chars = cnn_options['n_characters']
        if n_chars != 262:
            raise InvalidNumberOfCharacters(
                "Set n_characters=262 after training see the README.md"
            )
        if cnn_options['activation'] == 'tanh':
            activation = tf.nn.tanh
        elif cnn_options['activation'] == 'relu':
            activation = tf.nn.relu

        with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable("char_embed",
                                                     [n_chars, char_embed_dim],
                                                     dtype=DTYPE,
                                                     initializer=tf.random_uniform_initializer(-1.0, 1.0))
            self.char_embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                         self.ids_placeholder)

        def make_convolutions(inp):
            with tf.variable_scope('CNN') as scope:
                convolutions = []
                for i, (width, num) in enumerate(filters):
                    if cnn_options['activation'] == 'relu':
                        w_init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)
                    elif cnn_options['activation'] == 'tanh':
                        w_init = tf.random_normal_initializer(mean=0.0,
                                                              stddev=np.sqrt(1.0 / (width * char_embed_dim)))
                    w = tf.get_variable("W_cnn_%s" % i,
                                        [1, width, char_embed_dim, num],
                                        initializer=w_init,
                                        dtype=DTYPE)
                    b = tf.get_variable("b_cnn_%s" % i,
                                        [num],
                                        dtype=DTYPE,
                                        initializer=tf.constant_initializer(0.0))
                    conv = tf.nn.conv2d(inp, w,
                                        strides=[1, 1, 1, 1],
                                        padding="VALID") + b
                    conv = tf.nn.max_pool(conv,
                                          [1, 1, max_chars - width + 1, 1],
                                          [1, 1, 1, 1], "VALID")

                    conv = activation(conv)
                    conv = tf.squeeze(conv, squeeze_dims=[2])
                    convolutions.append(conv)
            return tf.concat(convolutions, 2)

        embedding = make_convolutions(self.char_embedding)
        n_highway = cnn_options.get('n_highway')
        use_highway = n_highway is not None and n_highway > 0
        use_proj = n_filters != projection_dim

        if use_highway or use_proj:
            batch_size_n_tokens = tf.shape(embedding)[0:2]
            embedding = tf.reshape(embedding, [-1, n_filters])

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
                    W_carry = tf.get_variable('W_carry', [highway_dim, highway_dim],
                                              initializer=tf.random_normal_initializer(
                                                  mean=0.0, stddev=np.sqrt(1.0 / highway_dim)
                                              ), dtype=DTYPE)
                    b_carry = tf.get_variable('b_carry',
                                              [highway_dim],
                                              initializer=tf.constant_initializer(-2.0),
                                              dtype=DTYPE)
                    W_transform = tf.get_variable('W_transform',
                                                  [highway_dim, highway_dim],
                                                  initializer=tf.random_normal_initializer(mean=0.0,
                                                                                           stddev=np.sqrt(1.0/ highway_dim)),
                                                  dtype=DTYPE)
                    b_transform = tf.get_variable('b_transform',
                                                  [highway_dim],
                                                  initializer=tf.constant_initializer(0.0),
                                                  dtype=DTYPE)
                embedding = high(embedding, W_carry, b_carry, W_transform, b_transform)

        if use_proj:
            embedding = tf.matmul(embedding, W_proj_cnn) + b_proj_cnn

        if use_highway or use_proj:
            shp = tf.concat([batch_size_n_tokens, [projection_dim]], axis=0)
            embedding = tf.reshape(embedding, shp)

        self.embedding = embedding

    def _build_word_embeddings(self):
        projection_dim = self.options.lstm['projection_dim']

        with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable("embedding",
                                                     [self._n_tokens_vocab,
                                                      projection_dim],
                                                     dtype=DTYPE)
            self.embedding = tf.nn.embedding_lookup(self.embedding, self.ids_placeholder)

    def _build_lstms(self):
        lstm_dim = self.options.lstm['dim']
        projection_dim = self.options.lstm['projection_dim']
        n_lstm_layers = self.options.lstm['n_layers']
        cell_clip = self.options.lstm['cell_clip']
        proj_clip = self.options.lstm['proj_clip']
        use_skip_connections = self.options.lstm['use_skip_connections']
        if use_skip_connections:
            print("USING SKIP CONNECTIONS")
        else:
            print("NOT USING SKIP CONNECTIONS")

        if self.use_character_inputs:
            mask = tf.reduce_any(self.ids_placeholder > 0, axis=2)
        else:
            mask = self.ids_placeholder > 0
        sequence_lengths = tf.reduce_sum(tf.cast(mask, tf.int32),
                                         axis=1)
        batch_size = tf.shape(sequence_lengths)[0]

        self.lstm_outputs = {'forward':[], 'backward':[]}
        self.lstm_state_sizes = {'forward':[], 'backward':[]}
        self.lstm_init_states = {'forward':[], 'backward':[]}
        self.lstm_final_states = {'forward':[], 'backward':[]}

        update_ops = []
        for direction in ['forward', 'backward']:
            if direction == 'forward':
                layer_input = self.embedding
            else:
                layer_input = tf.reverse_sequence(self.embedding,
                                                  sequence_lengths,
                                                  seq_axis=1,
                                                  batch_axis=0)
            for i in range(n_lstm_layers):
                if projection_dim < lstm_dim:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_dim, num_proj=projection_dim,
                                                        cell_clip=cell_clip,
                                                        proj_clip=proj_clip)
                else:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_dim,
                                                        cell_clip=cell_clip,
                                                        proj_clip=proj_clip)
                if use_skip_connections:
                    if i==0:
                        pass
                    else:
                        lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)
                state_size = lstm_cell.state_size

                init_states = [
                    tf.Variable(tf.zeros([self._max_batch_size, dim]),
                                trainable=False)
                    for dim in lstm_cell.state_size
                ]
                batch_init_states = [
                    state[:batch_size, :] for state in init_states
                ]

                if direction == 'forward':
                    i_direction = 0
                else:
                    i_direction = 1
                variable_scope_name = 'RNN_{0}/RNN/MultiRNNCell/Cell{1}'.format(
                    i_direction, i)
                with tf.variable_scope(variable_scope_name):
                    layer_output, final_state = tf.nn.dynamic_rnn(lstm_cell,
                                                                  layer_input,
                                                                  sequence_length=sequence_lengths,
                                                                  initial_state=tf.nn.rnn_cell.LSTMStateTuple(
                                                                      *batch_init_states))
                self.lstm_state_sizes[direction].append(lstm_cell.state_size)
                self.lstm_init_states[direction].append(init_states)
                self.lstm_final_states[direction].append(final_state)
                if direction == 'forward':
                    self.lstm_outputs[direction].append(layer_output)
                else:
                    self.lstm_outputs[direction].append(tf.reverse_sequence(
                            layer_output,
                            sequence_lengths,
                            seq_axis=1,
                            batch_axis=0
                        ))

                with tf.control_dependencies([layer_output]):
                    for i in range(2):
                        new_state = tf.concat([final_state[i][:batch_size, :],
                                               init_states[i][batch_size:, :]], axis=0)
                        state_update_op = tf.assign(init_states[i], new_state)
                        update_ops.append(state_update_op)

                layer_input = layer_output
        self.mask = mask
        self.sequence_lengths = sequence_lengths
        self.update_state_op = tf.group(*update_ops)





