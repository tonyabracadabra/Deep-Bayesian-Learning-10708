from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.models import Sequential
from keras.models.core import Lambda
import tensorflow as tf
import numpy as np
import tensorflow.contrib.seq2seq as seq2seq
from functools import partial


# This defines the inner LSTM
self.vocab_size = 100
self.embedding_size = 30
self.h_units_words = 40
self.h_units_sentences = 50
self.latent_size = 20


batch_size = 32
n_sentences = 5
# how many words in one sentence
n_words = 10

EOS = 1
PAD = 0

# This defines the embedding matrix
lookup_matrix = np.zeros((n_all_words, embedding_size))



def _init_placeholders(self):
    """ Everything is time-major """
    self.encoder_inputs = tf.placeholder(
        shape=(None, None),
        dtype=tf.int32,
        name='encoder_inputs',
    )

    # required for training, not required for testing
    self.decoder_targets = tf.placeholder(
        shape=(None, None),
        dtype=tf.int32,
        name='decoder_targets'
    )

# Uniform(-sqrt(3), sqrt(3)) has variance=1.
with tf.variable_scope("embedding") as scope:
    sqrt3 = math.sqrt(3)
    initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

    self.embedding_matrix = tf.constant(
        lookup_matrix,
        name="embedding_matrix")


def embedding_func(self, embedding_input):
    embedded = tf.nn.embedding_lookup(
        self.embedding_matrix, embedding_input)

    return embedded

self.encoder_inner_cell = LSTMCell(h_units_words)
self.encoder_outer_cell = LSTMCell(h_units_sentences)

Embedding = Lambda(lambda x : self.embedding_func(x))

dynamic_bilstm_inner = partial(dynamic_bilstm, self.encoder_inner_cell)
dynamic_bilstm_outer = partial(dynamic_bilstm, self.encoder_outer_cell)

DynamicInnerBiLSTM = Lambda(lambda x : dynamic_bilstm_inner(x))
DynamicOuterBiLSTM = Lambda(lambda x : dynamic_bilstm_outer(x))

inner_lstm = Sequential()
inner_lstm.add(Embedding)
inner_lstm.add(DynamicInnerBiLSTM)


# This defines the outer LSTM
outer_lstm = Sequential()
# The embedding size here is the n_units_words from the inner LSTM
outer_lstm.add(DynamicOuterBiLSTM)


'''
Below is the nested LSTM
'''

'''
Encoder phase
'''

# (batch_size, n_words) is for inner lstm,
# after embedding (batch_size, n_words, embedding_size)
# after lstm, if use return_sequences=True, the output should has shape (batch_size, n_words, h_units_words)


# shape = (batch_size, n_sentences, n_words)
convs_placeholder = tf.placeholder(tf.float32, [None, None, None])
# shape = (n_sentences, batch_size, n_words)
convs_placeholder_trans = tf.transpose(convs_placeholder, [1,0,2])

# (n_sentence, batch_size, n_words, h_units_words)
inner_outputs = tf.map_fn(lambda x:inner_lstm(x), convs_placeholder_trans)

# (batch_size, n_sentence, n_words, h_units_words)
inner_outputs_trans = tf.transpose(inner_outputs, [1,0,2,3])
# sum out the third dimension for the input of outer lstm
outer_lstm_input = tf.reduce_sum(inner_outputs_trans, axis=2)

#  The encoded state to initialize the dynamic_rnn_decoder
encoder_state = outer_lstm(outer_lstm_input)

def reparameterizing_z(mu, logsigma):
    sigma_std = tf.exp(0.5 * logsigma)
    epsilon = tf.random_normal(tf.shape(sigma_std))
    z = tf.add(mu, tf.multiply(sigma_std, epsilon))
    return z

def variational_encoder(encoder_state):
    # get mu and sigma from encoder state
    encoder_state_mu = tf.contrib.layers.fully_connected(encoder_state.h, self.latent_size)
    encoder_state_logsigma = tf.contrib.layers.fully_connected(encoder_state.h, self.latent_size)
    # reparameter to get z
    sample_z = reparameterizing_z(encoder_state_mu, encoder_state_logsigma)



# attention state for the use of apply attention to the decoder [batch_size, n_words, h_units_words]
attention_states = inner_outputs_trans[:,-1,:,:]


'''
Decoder phase
'''

# Below I basically copied from the tutorial

decoder_hidden_units = 60

(attention_keys,
attention_values,
attention_score_fn,
attention_construct_fn) = seq2seq.prepare_attention(
    attention_states=attention_states,
    attention_option="bahdanau",
    num_units=decoder_hidden_units,
)

# attention is added
decoder_fn_train = seq2seq.attention_decoder_fn_train(
    encoder_state=encoder_state,
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


'''
For Debugging
'''

# conversations variable is used for debugging use
# every time the conversation fed in must be in the same number of sentences
# It contains the input sentences and the sentence to be predicted
conversations = np.zeros((batch_size, n_sentences, n_words))



