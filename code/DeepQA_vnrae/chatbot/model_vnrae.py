# Copyright 2015 Conchylicultor. All Rights Reserved.
# Modifications copyright (C) 2016 Carlos Segura
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Model to predict the next sentence given an input sequence

"""

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import numpy as np
import math
from functools import partial
import tensorflow.contrib.seq2seq as seq2seq

from chatbot.textdata import Batch


class Model:
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        2 LTSM layers
    """

    def __init__(self, args, textData, lookup_matrix):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        print("Model creation...")

        self.textData = textData  # Keep a reference on the dataset
        self.args = args  # Keep track of the parameters of the model
        self.dtype = tf.float32

        self.encoder_inner_cell = LSTMCell(self.args.h_units_words)
        self.encoder_outer_cell = LSTMCell(self.args.h_units_sentences)
        self.decoder_cell = LSTMCell(self.args.h_units_decoder)
        self.lookup_matrix = lookup_matrix

        # (batch_size, n_words) is for inner lstm,
        # after embedding (batch_size, n_words, embedding_size)
        # after lstm, if use return_sequences=True, the output should has shape (batch_size, n_words, h_units_words)

        self._define_placeholders()
        self._init_embedding(lookup_matrix)
        self._define_layers()
        # Construct the graphs
        self._build_network()

    def _define_placeholders(self):
        # shape = (batch_size, n_sentences, n_words)
        # This is the conversation
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None, None])

        # shape = (batch_size, n_words)
        # This is the input sequence to decoder
        self.decoder_inputs = tf.placeholder(tf.int32, [None, None])

        # shape = (batch_size, n_words)
        # This is the target sequence to be predicted
        self.decoder_targets = tf.placeholder(tf.int32, [None, None])

        # shape = (batch_size, n_sentences)
        # number of words in sentences for inner encoder input
        self.encoder_inner_length = tf.placeholder(tf.int32, [None, None])

        # shape = (batch_size)
        # number of sentences for outer encoder input
        self.encoder_outer_length = tf.placeholder(tf.int32, [None])

        # shape = (batch_size)
        # number of words for decoder output
        self.decoder_targets_length = tf.placeholder(tf.int32, [None])

        # (batch_size, n_words_max)
        self.decoder_weights = tf.ones([
            self.args.batch_size,
            5
            # tf.reduce_max(self.decoder_targets_length)
        ], dtype=tf.float32, name="loss_weights")

    def _init_embedding(self, lookup_matrix):
        with tf.variable_scope("embedding") as scope:
            self.embedding_matrix = tf.constant(
                lookup_matrix,
                name="embedding_matrix")

    def _define_layers(self):
        self.inner_lstm = partial(self._dynamic_bilstm, 'inner', self.encoder_inner_cell)
        self.outer_lstm = partial(self._dynamic_bilstm, 'outer', self.encoder_inner_cell)

    def _dynamic_bilstm(self, level, encoder_cell, encoder_inputs, encoder_inputs_length):
        with tf.variable_scope("BidirectionalEncoder") as scope:
            encoder_inputs_embedded = encoder_inputs

            if level == 'inner':
                # (batch_size, n_words, embedding_size)
                encoder_inputs_embedded = tf.nn.embedding_lookup(self.lookup_matrix, encoder_inputs)

            with tf.variable_scope(level) as scope:
                ((encoder_fw_outputs,
                  encoder_bw_outputs),
                 (encoder_fw_state,
                  encoder_bw_state)) = (
                    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                    cell_bw=encoder_cell,
                                                    inputs=encoder_inputs_embedded,
                                                    sequence_length=encoder_inputs_length,
                                                    time_major=True,
                                                    dtype=tf.float64)
                    )

            # (batch_size, n_words, h_units_words)
            
            if level == 'inner':
                encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

                return encoder_outputs

            elif level == 'outer':
                encoder_state_h = tf.concat((encoder_fw_state.h, encoder_bw_state.h), \
                    1, name='bidirectional_concat_h')

                encoder_state_c = tf.concat((encoder_fw_state.c, encoder_bw_state.c), \
                    1, name='bidirectional_concat_c')

                return (encoder_state_h, encoder_state_c)

    def _build_network(self):
        """ Create the computational graph
        """

        self._init_encoder()
        self._init_decoder()
        self._define_loss()

        # Initialize the optimizer
        opt = tf.train.AdamOptimizer(
            learning_rate=self.args.learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08
        )

        self.opt_op = opt.minimize(self.loss)

    def _init_encoder(self):
        '''
        Encoder phase
        '''

        # shape = (n_sentences, batch_size, n_words)
        # Change the order of dimension
        # Each vector in its first dimension can be seen as the input for the inner LSTM
        encoder_inputs_trans = tf.transpose(self.encoder_inputs, [1, 0, 2])
        # (n_sentences, batch_size)
        encoder_inner_length_trans = tf.transpose(self.encoder_inner_length, [1, 0])

        # (n_sentence, batch_size, n_words)

        inner_lstm_outputs = tf.map_fn(lambda x: self.inner_lstm(x[0], x[1]), \
                            (encoder_inputs_trans, encoder_inner_length_trans), \
                            dtype=tf.float64)

        # (batch_size, n_sentence, n_words, h_units_words)
        inner_lstm_outputs_trans = tf.transpose(inner_lstm_outputs, [1, 0, 2, 3])
        # sum out the third dimension for the input of outer lstm
        outer_lstm_input = tf.reduce_sum(inner_lstm_outputs_trans, axis=2)

        #  The encoded state to initialize the dynamic_rnn_decoder
        # encoder_end_state, or the output of the outer lstm
        encoder_end_state = self.outer_lstm(outer_lstm_input, self.encoder_outer_length)

        self.encoder_state = self._init_variational_encoder(encoder_end_state, self.args.h_units_decoder)

        # attention state for the use of apply attention to the decoder [batch_size, n_words, h_units_words]
        self.attention_states = inner_lstm_outputs_trans[:, -1, :, :]


    def _init_decoder(self):
        '''
        Decoder phase
        '''

        self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.lookup_matrix, self.decoder_inputs)

        (attention_keys,
        attention_values,
        attention_score_fn,
        attention_construct_fn) = seq2seq.prepare_attention(
                attention_states=self.attention_states,
                attention_option="bahdanau",
                num_units=self.args.h_units_decoder,
        )

        # attention is added
        decoder_fn_train = seq2seq.attention_decoder_fn_train(
            encoder_state=self.encoder_state,
            attention_keys=attention_keys,
            attention_values=attention_values,
            attention_score_fn=attention_score_fn,
            attention_construct_fn=attention_construct_fn,
            name='attention_decoder'
        )

        self.W_proj = tf.get_variable('weights', [self.args.h_units_decoder, self.textData.getVocabularySize()])
        self.b_proj = tf.get_variable('bias', [self.textData.getVocabularySize()])

        output_fn = lambda output: tf.add(tf.matmul(self.W_proj, output), self.b_proj)

        decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
            output_fn=output_fn,
            encoder_state=self.encoder_state,
            attention_keys=attention_keys,
            attention_values=attention_values,
            attention_score_fn=attention_score_fn,
            attention_construct_fn=attention_construct_fn,
            embeddings=self.lookup_matrix,
            start_of_sequence_id=self.textData.goToken,
            end_of_sequence_id=self.textData.eosToken,
            maximum_length=tf.reduce_max(self.decoder_targets_length) + 3,
            num_decoder_symbols=self.textData.getVocabularySize(),
        )

        print(self.decoder_inputs_embedded)
        print(self.decoder_targets_length)

        (decoder_outputs_train, decoder_state_train, decoder_context_state_train) = seq2seq.dynamic_rnn_decoder(
                cell=self.decoder_cell,
                decoder_fn=decoder_fn_train,
                inputs=self.decoder_inputs_embedded,
                sequence_length=self.decoder_targets_length,
                time_major=False)

        self.decoder_logits_train = output_fn(decoder_outputs_train)
        self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

        (decoder_logits_inference,
            decoder_state_inference,
            decoder_context_state_inference) = (
            seq2seq.dynamic_rnn_decoder(
                cell=self.decoder_cell,
                decoder_fn=decoder_fn_inference,
                time_major=False,
            )
        )

        self.decoder_prediction_inference = tf.argmax(decoder_logits_inference, axis=-1, name='decoder_prediction_inference')

    @staticmethod
    def reparameterizing_z(mu, logsigma):
        sigma_std = tf.exp(0.5 * logsigma)
        epsilon = tf.random_normal(tf.shape(sigma_std), dtype=tf.float64)
        z = tf.add(mu, tf.multiply(sigma_std, epsilon))

        return z

    def _init_variational_encoder(self, encoder_end_state, h_units_decoder):
        # get mu and sigma from encoder state
        encoder_end_state_h, encoder_end_state_c = encoder_end_state

        self.encoder_state_mu = tf.contrib.layers.fully_connected(encoder_end_state_h, self.args.latent_size)
        self.encoder_state_logsigma = tf.contrib.layers.fully_connected(encoder_end_state_h, self.args.latent_size)
        # reparameter to get z
        sample_z = self.reparameterizing_z(self.encoder_state_mu, self.encoder_state_logsigma)
        # get intital state of decoder from z
        encoder_state_h = tf.contrib.layers.fully_connected(sample_z, h_units_decoder)

        encoder_state_c = encoder_end_state_c

        encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

        return encoder_state

    def sampled_softmax(self, labels, inputs):
        labels = tf.reshape(labels, [-1, 1])  # Add one dimension (nb of true classes, here 1)

        # We need to compute the sampled_softmax_loss using 32bit floats to
        # avoid numerical instabilities.
        local_wt = tf.cast(self.W_proj, tf.float32)
        local_b = tf.cast(self.b_proj, tf.float32)
        localInputs = tf.cast(inputs, tf.float32)

        return tf.cast(
            tf.nn.sampled_softmax_loss(
                tf.transpose(local_wt),  # Should have shape [num_classes, dim]
                local_b,
                labels,
                localInputs,
                self.args.softmaxSamples,  # The number of classes to randomly sample per batch
                self.textData.getVocabularySize()),  # The number of classes
            self.dtype)

    def _define_loss(self):

        self.loss_reconstruct = tf.contrib.seq2seq.sequence_loss(
            self.decoder_logits_train,
            self.decoder_targets,
            self.decoder_weights,
            self.textData.getVocabularySize(),
            softmax_loss_function=self.sampled_softmax,  # If None, use default SoftMax
            average_across_timesteps=False
        )

        self.KL = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + self.encoder_state_logsigma 
                                 - tf.pow(self.encoder_state_mu, 2)
                                 - tf.exp(self.encoder_state_logsigma), axis=1))

        self.loss = tf.add(self.KL, tf.reduce_sum(self.loss_reconstruct))

        # Keep track of the cost
        tf.summary.scalar('loss_reconstruct', self.loss_reconstruct)
        tf.summary.scalar('KL', self.KL)
        tf.summary.scalar('loss', self.loss)

    def step(self, batch):
        """ Forward/training step operation.
        Does not perform run on itself but just return the operators to do so. Those have then to be run
        Args:
            batch (Batch): Input data on testing mode, input and target on output mode
        Return:
            (ops), dict: A tuple of the (training, loss) operators or (outputs,) in testing mode with the associated feed dictionary
        """

        # Feed the dictionary
        feedDict = {}
        ops = None

        encoder_inputs = np.array([[[1,3,4,5,-1,-1],[2,3,-1,-1,-1,-1],[2,3,555,1,2,666]],
                          [[999,666,4,-1,-1,-1],[2,3,888,777,-1,-1],[-1,-1,-1,-1,-1,-1]]])

        decoder_inputs = np.array([[-1, 1, 3, 5, 7, 9], [-1, 2, 4, 6, 8, 10]])
        decoder_targets = np.array([[1,3,5,7,9,1],[2,4,6,8,10,2]])

        encoder_inner_length = np.array([[4,2,6],[3,4,0]])
        encoder_outer_length = np.array([3,2])
        decoder_targets_length = np.array([6, 6])

        feedDict[self.encoder_inputs] = encoder_inputs
        feedDict[self.decoder_targets] = decoder_targets
        feedDict[self.encoder_inner_length] = encoder_inner_length
        feedDict[self.encoder_outer_length] = encoder_outer_length
        feedDict[self.decoder_targets_length] = decoder_targets_length
        feedDict[self.encoder_inputs] = decoder_inputs

        ops = (self.opt_op, self.loss)

        '''
        if not self.args.test:  # Training
            feedDict[self.encoder_inputs] = batch.encoder_convs   
            feedDict[self.decoder_targets] = batch.target_seqs
            feedDict[self.encoder_inner_length] = batch.encoder_inner_length
            feedDict[self.encoder_outer_length] = batch.encoder_outer_length
            feedDict[self.target_seqs] = batch.target_seqs

            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoder_inputs[i]]  = batch.encoderSeqs[i]
            for i in range(self.args.maxLengthDeco):
                feedDict[self.decoderInputs[i]]  = batch.decoderSeqs[i]
                feedDict[self.decoderTargets[i]] = batch.targetSeqs[i]
                feedDict[self.decoderWeights[i]] = batch.weights[i]

            ops = (self.optOp, self.lossFct)
        else:  # Testing (batchSize == 1)
            feedDict[self.encoder_inputs] = batch.encoder_convs
            feedDict[self.decoder_targets] = batch.target_seqs

            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]] = batch.encoderSeqs[i]
            feedDict[self.decoderInputs[0]]  = [self.textData.goToken]

            ops = (self.outputs,)
        '''

        # Return one pass operator
        return ops, feedDict
