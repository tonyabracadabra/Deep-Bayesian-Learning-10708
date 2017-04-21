import tensorflow as tf

def dynamic_bilstm(self, encoder_cell, embedded_state):
    with tf.variable_scope("BidirectionalEncoder") as scope:
        ((encoder_fw_outputs,
          encoder_bw_outputs),
         (encoder_fw_state,
          encoder_bw_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                            cell_bw=encoder_cell,
                                            inputs=embedded_state,
                                            time_major=True,
                                            dtype=tf.float32)
            )

        self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

        if isinstance(encoder_fw_state, LSTMStateTuple):

            encoder_state_c = tf.concat(
                (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
            encoder_state_h = tf.concat(
                (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
            self.encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

        elif isinstance(encoder_fw_state, tf.Tensor):
            self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')

        return encoder_state


def embedding_func(embedding_input):
    embedded = tf.nn.embedding_lookup(
        self.embedding_matrix, embedding_input)

    return embedded


def reparameterizing_z(mu, logsigma):
    sigma_std = tf.exp(0.5 * logsigma)
    epsilon = tf.random_normal(tf.shape(sigma_std))
    z = tf.add(mu, tf.multiply(sigma_std, epsilon))

    return z

def variational_encoder(encoder_end_state, h_units_decoder):
    # get mu and sigma from encoder state
    encoder_state_mu = tf.contrib.layers.fully_connected(encoder_end_state.h, self.latent_size)
    encoder_state_logsigma = tf.contrib.layers.fully_connected(encoder_end_state.h, self.latent_size)
    # reparameter to get z
    sample_z = reparameterizing_z(encoder_state_mu, encoder_state_logsigma)
    # get intital state of decoder from z
    encoder_state_h = tf.contrib.layers.fully_connected(sample_z, h_units_decoder)

    encoder_state_c = encoder_end_state_c

    encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

def sampled_softmax(labels, inputs):
    labels = tf.reshape(labels, [-1, 1])  # Add one dimension (nb of true classes, here 1)

    # We need to compute the sampled_softmax_loss using 32bit floats to
    # avoid numerical instabilities.
    localWt     = tf.cast(outputProjection.W_t,             tf.float32)
    localB      = tf.cast(outputProjection.b,               tf.float32)
    localInputs = tf.cast(inputs,                           tf.float32)

    return tf.cast(
        tf.nn.sampled_softmax_loss(
            localWt,  # Should have shape [num_classes, dim]
            localB,
            labels,
            localInputs,
            self.args.softmaxSamples,  # The number of classes to randomly sample per batch
            self.textData.getVocabularySize()),  # The number of classes
        self.dtype)

