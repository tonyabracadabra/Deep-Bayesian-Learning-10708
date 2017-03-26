import gensim
import pickle
import numpy as np

model = gensim.models.Word2Vec.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  

index2word = {i:word for i, word in enumerate(model.index2word)}
word2index = {word:i for i, word in enumerate(model.index2word)}

word_index_conversion = {'index2word':index2word, 'word2index':word2index}

with open("word_index_conversion.p",'wb') as f:
	pickle.dump(word_index_conversion, f)
	
weights = model.syn0

np.save("word2vec.npy", weights)