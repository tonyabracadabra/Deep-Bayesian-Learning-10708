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
from utils import *
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple

from chatbot.textdata import Batch


class ProjectionOp:
    """ Single layer perceptron
    Project input tensor on the output dimension
    """
    def __init__(self, shape, scope=None, dtype=None):
        """
        Args:
            shape: a tuple (input dim, output dim)
            scope (str): encapsulate variables
            dtype: the weights type
        """
        assert len(shape) == 2

        self.scope = scope

        # Projection on the keyboard
        with tf.variable_scope('weights_' + self.scope):
            self.W_t = tf.get_variable(
                'weights',
                shape,
                # initializer=tf.truncated_normal_initializer()  # TODO: Tune value (fct of input size: 1/sqrt(input_dim))
                dtype=dtype
            )
            self.b = tf.get_variable(
                'bias',
                shape[0],
                initializer=tf.constant_initializer(),
                dtype=dtype
            )
            self.W = tf.transpose(self.W_t)

    def getWeights(self):
        """ Convenience method for some tf arguments
        """
        return self.W, self.b

    def __call__(self, X):
        """ Project the output of the decoder into the vocabulary space
        Args:
            X (tf.Tensor): input value
        """
        with tf.name_scope(self.scope):
            return tf.matmul(X, self.W) + self.b


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
        self.decoder_cell = LSTMCell(self.args.h_units_sentences)

        # (batch_size, n_words) is for inner lstm,
        # after embedding (batch_size, n_words, embedding_size)
        # after lstm, if use return_sequences=True, the output should has shape (batch_size, n_words, h_units_words)

        # shape = (batch_size, n_sentences, n_words)
        self.convs_placeholder = tf.placeholder(tf.float32, [None, None, None])

        self._init_embedding(lookup_matrix)
        self._define_layers()
        # Construct the graphs
        self._buildNetwork()

    def _init_embedding(self, lookup_matrix):
        # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        with tf.variable_scope("embedding") as scope:
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            self.embedding_matrix = tf.constant(
                lookup_matrix,
                name="embedding_matrix")

    def _define_layers(self):
        self.Embedding = Lambda(lambda x : self.embedding_func(x))

        dynamic_bilstm_inner = partial(dynamic_bilstm, self.encoder_inner_cell)
        dynamic_bilstm_outer = partial(dynamic_bilstm, self.encoder_outer_cell)

        self.DynamicInnerBiLSTM = Lambda(lambda x : dynamic_bilstm_inner(x))
        self.DynamicOuterBiLSTM = Lambda(lambda x : dynamic_bilstm_outer(x))

    def _buildNetwork(self):
        """ Create the computational graph
        """

        # TODO: Create name_scopes (for better graph visualisation)
        # TODO: Use buckets (better perfs)

        # Parameters of sampled softmax (needed for attention mechanism and a large vocabulary size)
        output_projection = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if 0 < self.args.softmaxSamples < self.textData.getVocabularySize():
            output_projection = ProjectionOp(
                (self.textData.getVocabularySize(), self.args.hiddenSize),
                scope='softmax_projection',
                dtype=self.dtype
        )

        self._init_encoder()
        self._init_decoder()
        self._define_loss()

        # Initialize the optimizer
        opt = tf.train.AdamOptimizer(
            learning_rate=self.args.learningRate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08
        )
        self.opt_op = opt.minimize(self.loss)
        
        # TODO: When the LSTM hidden size is too big, we should project the LSTM output into a smaller space (4086 => 2046): Should speed up
        # training and reduce memory usage. Other solution, use sampling softmax

        # For testing only
        if self.args.test:
            if not output_projection:
                self.outputs = decoderOutputs
            else:
                self.outputs = [output_projection(output) for output in decoderOutputs]

            # TODO: Attach a summary to visualize the output

        # For training only
        else:
            # Finally, we define the loss function
            self.lossFct = tf.contrib.legacy_seq2seq.sequence_loss(
                decoderOutputs,
                self.decoderTargets,
                self.decoderWeights,
                self.textData.getVocabularySize(),
                softmax_loss_function= sampledSoftmax if outputProjection else None  # If None, use default SoftMax
            )
            tf.summary.scalar('loss', self.lossFct)  # Keep track of the cost

            # Initialize the optimizer
            opt = tf.train.AdamOptimizer(
                learning_rate=self.args.learningRate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
            self.optOp = opt.minimize(self.lossFct)


    def _init_encoder(self):
        inner_lstm = Sequential()
        inner_lstm.add(self.Embedding)
        inner_lstm.add(self.DynamicInnerBiLSTM)

        # This defines the outer LSTM
        outer_lstm = Sequential()
        # The embedding size here is the n_units_words from the inner LSTM
        outer_lstm.add(self.DynamicOuterBiLSTM)

        '''
        Below is the nested LSTM
        '''

        '''
        Encoder phase
        '''

        # shape = (n_sentences, batch_size, n_words)
        convs_placeholder_trans = tf.transpose(self.convs_placeholder, [1,0,2])

        # (n_sentence, batch_size, n_words, h_units_words)
        inner_outputs = tf.map_fn(lambda x:inner_lstm(x), convs_placeholder_trans)

        # (batch_size, n_sentence, n_words, h_units_words)
        inner_outputs_trans = tf.transpose(inner_outputs, [1,0,2,3])
        # sum out the third dimension for the input of outer lstm
        outer_lstm_input = tf.reduce_sum(inner_outputs_trans, axis=2)

        #  The encoded state to initialize the dynamic_rnn_decoder
        encoder_end_state = outer_lstm(outer_lstm_input)

        self.encoder_state = variational_encoder(encoder_end_state, self.args.h_units_decoder)

        # attention state for the use of apply attention to the decoder [batch_size, n_words, h_units_words]
        self.attention_states = inner_outputs_trans[:, -1, :, :]

    def _init_decoder(self):
         '''
        Decoder phase
        '''

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

        decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
            output_fn=output_fn,
            encoder_state=self.encoder_state,
            attention_keys=attention_keys,
            attention_values=attention_values,
            attention_score_fn=attention_score_fn,
            attention_construct_fn=attention_construct_fn,
            embeddings=lookup_matrix,
            start_of_sequence_id=EOS,
            end_of_sequence_id=EOS,
            maximum_length = n_words + 3,
            # vocabulary size
            num_decoder_symbols=n_all_words,
        )

        (decoder_outputs_train, decoder_state_train, decoder_context_state_train) = seq2seq.dynamic_rnn_decoder(
                cell=self.decoder_cell,
                decoder_fn=decoder_fn_train,
                inputs=self.decoder_train_inputs_embedded,
                sequence_length=self.decoder_train_length,
                time_major=True,
                scope=scope)

        output_fn = lambda output : tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)

        self.decoder_logits_train = output_fn(self.decoder_outputs_train)
        self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

        scope.reuse_variables()

        (decoder_logits_inference,
            decoder_state_inference,
            decoder_context_state_inference) = (
            seq2seq.dynamic_rnn_decoder(
                cell=self.decoder_cell,
                decoder_fn=decoder_fn_inference,
                time_major=True,
                scope=scope,
            )
        )

        self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1, name='decoder_prediction_inference')


    def _define_loss(self):
        self.loss_reconstruct = tf.contrib.seq2seq.sequence_loss(
            decoderOutputs,
            self.decoderTargets,
            self.decoderWeights,
            self.textData.getVocabularySize(),
            softmax_loss_function= sampled_softmax if output_projection else None  # If None, use default SoftMax
            average_across_timesteps = False
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

        if not self.args.test:  # Training
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]]  = batch.encoderSeqs[i]
            for i in range(self.args.maxLengthDeco):
                feedDict[self.decoderInputs[i]]  = batch.decoderSeqs[i]
                feedDict[self.decoderTargets[i]] = batch.targetSeqs[i]
                feedDict[self.decoderWeights[i]] = batch.weights[i]

            ops = (self.optOp, self.lossFct)
        else:  # Testing (batchSize == 1)
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]]  = batch.encoderSeqs[i]
            feedDict[self.decoderInputs[0]]  = [self.textData.goToken]

            ops = (self.outputs,)

        # Return one pass operator
        return ops, feedDict
