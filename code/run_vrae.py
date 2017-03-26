import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seq2seq
from seq2seq.models import AttentionSeq2Seq, SimpleSeq2Seq, Seq2Seq

from keras.layers import Input, Dense, Lambda, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import backend as K
from keras import metrics
import tensorflow as tf
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import pickle

path = 'data/'

with open(path+'word_index_conversion.p','rb') as f:
    word_index_conversion = pickle.load(f)
    
word2index = word_index_conversion['word2index']
index2word = word_index_conversion['index2word']

lookup_matrix = np.load(path+'word2vec.npy')

lookup_matrix = np.zeros((100,20))

sentences = np.load(path+'sentences_as_list.npy')

filtered_sentences = [filter(lambda x:word2index.has_key(x), s) for s in sentences]
# Filter out sentences with length larger than 2
filtered_sentences = filter(lambda x:len(x) > 2, filtered_sentences)

idxes = [[word2index[w] for w in s] for s in filtered_sentences]

# Constants
batch_size = 32
time_steps = 10
embedding_size = lookup_matrix.shape[1]
latent_dim = 2
hidden_dim = 256
epochs = 50
epsilon_std = 1.0
# Length of the sentence
time_steps = 5
epsilon_std = 1
print_freq = 5
max_epochs = 30

padded_idxes = pad_sequences(idxes, maxlen=time_steps, dtype='int32', \
                                 padding='pre', truncating='pre', value=0.)

x_train, x_test = train_test_split(padded_idxes, test_size=0.2, random_state=42)

x_train, x_test = np.zeros((30,time_steps)), np.zeros((10,time_steps))

optimizer, loss, x, y = SimpleSeq2Seq(output_dim=embedding_size, output_length=time_steps, latent_dim=latent_dim, \
                                      batch_size=batch_size, epsilon_std=epsilon_std, lookup_matrix=lookup_matrix, \
                                      input_shape=(time_steps, embedding_size))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    epochs = 0
    while epochs < max_epochs:
        print "Epochs:" + str(epochs)
        i = 0
        while i < len(x_train):
            samples = x_train[i:i+batch_size,:]
            i += batch_size
            _, loss_val = sess.run([optimizer, loss], feed_dict={x:samples, y:samples})
            if i % print_freq == 0:
                print "Loss :" + loss_val
        epochs += 1