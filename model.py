import tensorflow as tf
from tensorflow.keras import layers, models

from util import ID_TO_CLASS


class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        self.gru_layer = layers.Bidirectional(layers.GRU(hidden_size, return_sequences=True))
        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START

        # Not splitting as eq. 8 in paper.
        # forward, backward = tf.split(rnn_outputs, 2, axis=2)
        # H = tf.add(forward, backward)

        M = tf.nn.tanh(rnn_outputs)
        alpha = tf.nn.softmax(tf.tensordot(M, self.omegas, axes=1), axis=1)
        r = tf.reduce_sum(rnn_outputs * alpha, 1)
        h_star = tf.nn.tanh(r)

        ### TODO(Students) END

        return h_star

    # Ignore the fields position_1 and position_2 here, they are used in the advanced model.
    def call(self, inputs, pos_inputs, position_1, position_2, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START
        mask = tf.cast(inputs != 0, tf.float32)
        rnn_input = tf.concat([word_embed, pos_embed], axis=2)

        # Experiment 0, 2 [2 with shortest path turned off in data.py]
        rnn_outputs = self.gru_layer(rnn_input, mask=mask)

        # Experiment 1, 3 - [1 shortest path turned off in data.py and 3 with it on]
        # rnn_outputs = self.gru_layer(word_embed, mask=mask)

        attention_outputs = self.attn(rnn_outputs)
        logits = self.decoder(attention_outputs)
        ### TODO(Students) END

        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, position_vocab_size: int, position_embed_dim: int,
                 num_filters=128, filter_sizes=[2, 3, 4, 5]):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START
        self.num_classes = len(ID_TO_CLASS)
        self.decoder = layers.Dense(units=self.num_classes)
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))
        self.position_embeddings = tf.Variable(tf.random.normal((position_vocab_size, position_embed_dim)))

        # WORD + POSITION_WRT_E1 + POSITION_WRT_E2
        self.embedding_dim = embed_dim + 2 * position_embed_dim

        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.model_layers = []
        for i, filter_size in enumerate(filter_sizes):
            conv = layers.Conv2D(num_filters, [filter_size, self.embedding_dim],
                                 kernel_initializer=tf.keras.initializers.glorot_normal(),
                                 activation=tf.nn.relu)
            self.model_layers.append(conv)

        ### TODO(Students END

    def call(self, inputs, pos_inputs, position_1, position_2, training):
        ### TODO(Students) START
        sequence_length = tf.shape(inputs)[1]
        word_embed = tf.expand_dims(tf.nn.embedding_lookup(self.embeddings, inputs), -1)
        position_1_embed = tf.expand_dims(tf.nn.embedding_lookup(self.position_embeddings, position_1), -1)
        position_2_embed = tf.expand_dims(tf.nn.embedding_lookup(self.position_embeddings, position_2), -1)

        cnn_input = tf.concat([word_embed, position_1_embed, position_2_embed], axis=2)

        cnn_outputs = []
        for i, layer in enumerate(self.model_layers):
            cnn_output = layer(cnn_input)
            cnn_output = tf.nn.max_pool(cnn_output, ksize=[1, sequence_length - self.filter_sizes[i] + 1, 1, 1],
                                           strides=[1, 1, 1, 1], padding='VALID')
            cnn_outputs.append(cnn_output)

        num_filters_total = self.num_filters * len(self.filter_sizes)

        cnn_outputs = tf.concat(cnn_outputs, 3)
        cnn_outputs = tf.reshape(cnn_outputs, [-1, num_filters_total])
        cnn_outputs = tf.nn.dropout(cnn_outputs, 0.3)

        logits = self.decoder(cnn_outputs)
        ### TODO(Students) END

        return {'logits': logits}
