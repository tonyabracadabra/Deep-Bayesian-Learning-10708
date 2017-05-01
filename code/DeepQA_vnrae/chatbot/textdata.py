# Copyright 2015 Conchylicultor. All Rights Reserved.
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
Loads the dialogue corpus, builds the vocabulary
"""

import numpy as np
import nltk  # For tokenize
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random
import string
from collections import OrderedDict
from collections import deque
from chatbot.corpus.cornelldata import CornellData
from chatbot.corpus.opensubsdata import OpensubsData
from chatbot.corpus.scotusdata import ScotusData
from chatbot.corpus.ubuntudata import UbuntuData
from chatbot.corpus.lightweightdata import LightweightData


class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.encoderSeqs = []
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_targets = []
        self.encoder_inner_length = []
        self.encoder_outer_length = []
        self.decoder_targets_length = []
        self.weights = []


class TextData:
    """Dataset class
    Warning: No vocabulary limit
    """

    availableCorpus = OrderedDict([  # OrderedDict because the first element is the default choice
        ('cornell', CornellData),
        ('opensubs', OpensubsData),
        ('scotus', ScotusData),
        ('ubuntu', UbuntuData),
        ('lightweight', LightweightData),
    ])

    @staticmethod
    def corpusChoices():
        """Return the dataset availables
        Return:
            list<string>: the supported corpus
        """
        return list(TextData.availableCorpus.keys())

    def __init__(self, args):
        """Load all conversations
        Args:
            args: parameters of the model
        """
        # Model parameters
        self.args = args

        # Path variables
        self.corpusDir = os.path.join(self.args.rootDir, 'data', self.args.corpus)
        basePath = self._constructBasePath()
        self.fullSamplesPath = basePath + '_context.pkl'  # Full sentences length/vocab
        self.filteredSamplesPath = basePath + '-lenght{}-filter{}_context.pkl'.format(
            self.args.maxLength,
            self.args.filterVocab,
        )  # Sentences/vocab filtered for this model

        self.padToken = -1  # Padding
        self.goToken = -1  # Start of sequence
        self.eosToken = -1  # End of sequence
        self.unknownToken = -1  # Word dropped from vocabulary

        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]

        self.word2id = {}
        self.id2word = {}  # For a rapid conversion (Warning: If replace dict by list, modify the filtering to avoid linear complexity with del)
        self.idCount = {}  # Useful to filters the words (TODO: Could replace dict by list)

        self.loadCorpus()

        # Plot some stats:
        self._printStats()

        if self.args.playDataset:
            self.playDataset()

    def _printStats(self):
        print('Loaded {}: {} words, {} QA'.format(self.args.corpus, len(self.word2id), len(self.trainingSamples)))

    def _constructBasePath(self):
        """Return the name of the base prefix of the current dataset
        """
        path = os.path.join(self.args.rootDir, 'data/samples/')
        path += 'dataset-{}'.format(self.args.corpus)
        if self.args.datasetTag:
            path += '-' + self.args.datasetTag
        return path

    def makeLighter(self, ratioDataset):
        """Only keep a small fraction of the dataset, given by the ratio
        """
        #if not math.isclose(ratioDataset, 1.0):
        #    self.shuffle()  # Really ?
        #    print('WARNING: Ratio feature not implemented !!!')
        pass

    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.trainingSamples)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def _createBatch(self, samples):
        """Create a single batch from the list of sample. The batch size is automatically defined by the number of
        samples given.
        The inputs should already be inverted. The target should already have <go> and <eos>
        Warning: This function should not make direct calls to args.batch_size !!!
        Args:
            samples (list<Obj>): a list of samples, each sample being on the form [input, target]
        Return:
            Batch: a batch object en
        """
        batch = Batch()
        batch_size = len(samples)
        maxContextLength = 0

        # Create the batch tensor
        for i in range(batch_size):
            # Unpack the sample

            sample = samples[i]
            ##########################################
            # sample := (inputContext, targetWords)
            ##########################################

            # if not self.args.test and self.args.watsonMode:  # Watson mode: invert question and answer
            #     sample = list(reversed(sample))
            # if not self.args.test and self.args.autoEncode:  # Autoencode: use either the question or answer for both input and output
            #     k = random.randint(0, 1)
            #     sample = (sample[k], sample[k])

            # TODO: Why re-processed that at each epoch ? Could precompute that
            # once and reuse those every time. Is not the bottleneck so won't change
            # much ? and if preprocessing, should be compatible with autoEncode & cie.

            context = sample[0]

            contextReversed = []
            contextLength = len(context)
            if contextLength > maxContextLength:
                maxContextLength = contextLength

            # Reverse input in whole context
            nWordsVec = []
            for c in range(contextLength):
                inputSentence = context[c]
                nWords = len(inputSentence)
                assert nWords <= self.args.maxLengthEnco
                nWordsVec.append(nWords)
                # Padding (words)
                inputSentence = [self.padToken] * (self.args.maxLengthEnco  - len(inputSentence)) + inputSentence
                contextReversed.append(list(reversed(inputSentence)))

            batch.encoder_inputs.append(contextReversed)
            batch.encoderSeqs.append(list(reversed(sample[0])))  # Reverse inputs (and not outputs), little trick as defined on the original seq2seq paper
            batch.decoder_inputs.append([self.goToken] + sample[1])  # Add the <go> and <eos> tokens
            batch.decoder_targets.append(sample[1]+[self.eosToken])  # Same as decoder, but shifted to the left (ignore the <go>)

            batch.encoder_inner_length.append(nWordsVec)
            batch.encoder_outer_length.append(contextLength)
            batch.decoder_targets_length.append(len(batch.decoder_inputs[i]))

            # Long sentences should have been filtered during the dataset creation

            # assert len(batch.encoderSeqs[i]) <= self.args.maxLengthEnco
            assert len(batch.decoder_inputs[i]) <= self.args.maxLengthDeco

            # TODO: Should use tf batch function to automatically add padding and batch samples
            # Add padding & define weight
            #batch.encoderSeqs[i]   = [self.padToken] * (self.args.maxLengthEnco  - len(batch.encoderSeqs[i])) + batch.encoderSeqs[i]  # Left padding for the input
            batch.weights.append([1.0] * len(batch.decoder_targets[i]) + [0.0] * (self.args.maxLengthDeco - len(batch.decoder_targets[i])))

        max_len = max(map(len, batch.encoder_inner_length))
        batch.encoder_inner_length = [i + (max_len - len(i)) * [0] for i in batch.encoder_inner_length]

        # dynamical sentence-wise padding
        max_decoder_len = max(batch.decoder_targets_length)
        emptySentence = [self.padToken] * (self.args.maxLengthEnco) # empty sentence
        for i in range(batch_size):
            for j in range(maxContextLength - len(batch.encoder_inputs[i])):
                batch.encoder_inputs[i].append(emptySentence)

            batch.decoder_inputs[i] = batch.decoder_inputs[i] + [self.padToken] * (
                max_decoder_len - len(batch.decoder_inputs[i]))
            batch.decoder_targets[i] = batch.decoder_targets[i] + [self.padToken] * (
                max_decoder_len - len(batch.decoder_targets[i]))

        # Simple hack to reshape the batch
        '''
        encoder_inputsT = []  # Corrected orientation
        encoderSeqsT = []  # Corrected orientation
        for i in range(self.args.maxLengthEnco):
            encoderSeqT = []
            for j in range(batch_size):
                encoderSeqT.append(batch.encoderSeqs[j][i])
            encoderSeqsT.append(encoderSeqT)
        batch.encoderSeqs = encoderSeqsT
        '''

        # decoder_inputsT = []
        # decoder_targetsT = []
        # weightsT = []
        # for i in range(self.args.maxLengthDeco):
        #     decoderSeqT = []
        #     targetSeqT = []
        #     weightT = []
        #     for j in range(batch_size):
        #         decoderSeqT.append(batch.decoder_inputs[j][i])
        #         targetSeqT.append(batch.decoder_targets[j][i])
        #         weightT.append(batch.weights[j][i])
        #     decoder_inputsT.append(decoderSeqT)
        #     decoder_targetsT.append(targetSeqT)
        #     weightsT.append(weightT)
        # batch.decoder_inputs = decoder_inputsT
        # batch.decoder_targets = decoder_targetsT
        # batch.weights = weightsT

        # # Debug
        #self.printBatch(batch)  # Input inverted, padding should be correct
        # print(self.sequence2str(samples[0][0]))
        # print(self.sequence2str(samples[0][1]))  # Check we did not modified the original sample

        return batch

    def getBatches(self):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        self.shuffle()

        batches = []

        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, self.getSampleSize(), self.args.batch_size):
                yield self.trainingSamples[i:min(i + self.args.batch_size, self.getSampleSize())]

        # TODO: Should replace that by generator (better: by tf.queue)

        for samples in genNextSamples():
            batch = self._createBatch(samples)
            batches.append(batch)

        return batches

    def getSampleSize(self):
        """Return the size of the dataset
        Return:
            int: Number of training samples
        """
        return len(self.trainingSamples)

    def getVocabularySize(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.word2id)

    def loadCorpus(self):
        """Load/create the conversations data
        """
        datasetExist = os.path.isfile(self.filteredSamplesPath)
        if not datasetExist:  # First time we load the database: creating all files
            print('Training samples not found. Creating dataset...')

            datasetExist = os.path.isfile(self.fullSamplesPath)  # Try to construct the dataset from the preprocessed entry
            if not datasetExist:
                print('Constructing full dataset...')

                optional = ''
                if self.args.corpus == 'lightweight' and not self.args.datasetTag:
                    raise ValueError('Use the --datasetTag to define the lightweight file to use.')
                else:
                    optional = '/' + self.args.datasetTag  # HACK: Forward the filename

                # Corpus creation
                corpusData = TextData.availableCorpus[self.args.corpus](self.corpusDir + optional)
                self.createFullCorpus(corpusData.getConversations())
                self.saveDataset(self.fullSamplesPath)
            else:
                self.loadDataset(self.fullSamplesPath)
            self._printStats()
            # WTF??
            ###########################################
            print('Filtering words...')
            self.filterFromFull()  # Extract the sub vocabulary for the given maxLength and filterVocab
            ###########################################

            # Saving
            print('Saving dataset...')
            self.saveDataset(self.filteredSamplesPath)  # Saving tf samples
        else:
            self.loadDataset(self.filteredSamplesPath)

        assert self.padToken == 0

    def saveDataset(self, filename):
        """Save samples to file
        Args:
            filename (str): pickle filename
        """

        with open(os.path.join(filename), 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                'word2id': self.word2id,
                'id2word': self.id2word,
                'idCount': self.idCount,
                'trainingSamples': self.trainingSamples
            }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def loadDataset(self, filename):
        """Load samples from file
        Args:
            filename (str): pickle filename
        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word2id = data['word2id']
            self.id2word = data['id2word']
            self.idCount = data.get('idCount', None)
            self.trainingSamples = data['trainingSamples']

            self.padToken = self.word2id['<pad>']
            self.goToken = self.word2id['<go>']
            self.eosToken = self.word2id['<eos>']
            self.unknownToken = self.word2id['<unknown>']  # Restore special words

    def filterFromFull(self):
        """ Load the pre-processed full corpus and filter the vocabulary / sentences
        to match the given model options
        """
        ####################################################################
        # ********************************************
        def mergeSentences(sentences, fromEnd=False):
            """Merge the sentences until the max sentence length is reached
            Also decrement id count for unused sentences.
            Args:
                sentences (list<list<int>>): the list of sentences for the current line
                fromEnd (bool): Define the question on the answer
            Return:
                list<int>: the list of the word ids of the sentence
            """
            # We add sentence by sentence until we reach the maximum length
            merged = []

            # If question: we only keep the last sentences
            # If answer: we only keep the first sentences
            if fromEnd:
                sentences = reversed(sentences)

            for sentence in sentences:
                # If the total length is not too big, we still can add one more sentence
                if len(merged) + len(sentence) <= self.args.maxLength:
                    if fromEnd:  # Append the sentence
                        merged = sentence + merged
                    else:
                        merged = merged + sentence
                else:  # If the sentence is not used, neither are the words
                    for w in sentence:
                        self.idCount[w] -= 1

            # if len(merged) == 0:
            # print('sentences', sentences)
            # print('merged', merged)


            return merged
        # ********************************************

        newSamples = []
        #newSamples = self.trainingSamples.copy()

        # 1st step: Iterate over all words and add filters the sentences
        # according to the sentence lengths
        for inputContext, targetWords in tqdm(self.trainingSamples, desc='Filter sentences:', leave=False):
            newContext = []

            for i in range(len(inputContext)):
                inputWords = inputContext[i]
                inputWords = mergeSentences(inputWords, fromEnd=True)
                newContext.append(inputWords)
            targetWords = mergeSentences(targetWords, fromEnd=False)

            newSamples.append([newContext, targetWords])

        words = []

        # WARNING: DO NOT FILTER THE UNKNOWN TOKEN !!! Only word which has count==0 ?

        # 2nd step: filter the unused words and replace them by the unknown token
        # This is also where we update the correnspondance dictionaries
        specialTokens = {  # TODO: bad HACK to filter the special tokens. Error prone if one day add new special tokens
            self.padToken,
            self.goToken,
            self.eosToken,
            self.unknownToken
        }
        new_mapping = {}  # Map the full words ids to the new one (TODO: Should be a list)
        newId = 0

        '''
        for wordId, count in [(i, self.idCount[i]) for i in range(len(self.idCount))]:  # Iterate in order
            if (count <= self.args.filterVocab and
                wordId not in specialTokens):  # Cadidate to filtering (Warning: don't filter special token)
                new_mapping[wordId] = self.unknownToken
                del self.word2id[self.id2word[wordId]]  # The word isn't used anymore
                del self.id2word[wordId]
            else:  # Update the words ids
                new_mapping[wordId] = newId
                word = self.id2word[wordId]  # The new id has changed, update the dictionaries
                del self.id2word[wordId]  # Will be recreated if newId == wordId
                self.word2id[word] = newId
                self.id2word[newId] = word
                # if newId < 30 or word is 'then' or word is 'how' or word is 'Okay':
                #     print('id {0}: {1}'.format(newId, word))
                # newId += 1
        '''

        # Last step: replace old ids by new ones and filters empty sentences
        '''
        def replace_words(words):
            valid = False  # Filter empty sequences
            for i, w in enumerate(words):
                words[i] = new_mapping[w]
                if words[i] != self.unknownToken:  # Also filter if only contains unknown tokens
                    valid = True
            return valid
        '''
        self.trainingSamples.clear()

        for inputContext, targetWords in tqdm(newSamples, desc='Replace ids:', leave=False):
            #valid = True
            # ******************************************************************************************
            # WARNING: If ONE of the conetxt sentences is filtered, the entire conversation is invalid!!
            # ******************************************************************************************

            # traverse the whole input context
            #for i in range(len(inputContext)):
                #inputWords = inputContext[i]
                #valid &= replace_words(inputWords)

            #valid &= replace_words(targetWords)

            #if True: #valid:
            if len(inputContext) > 0 and len(targetWords) > 0:
                #self.trainingSamples.append([inputWords, targetWords])  # TODO: Could replace list by tuple

                self.trainingSamples.append([inputContext, targetWords])  # TODO: Could replace list by tuple

        self.idCount.clear()  # Not usefull anymore. Free data
        newSamples.clear()
        ###############################################################


    def createFullCorpus(self, conversations):
        """Extract all data from the given vocabulary.
        Save the data on disk. Note that the entire corpus is pre-processed
        without restriction on the sentence lenght or vocab size.
        """
        # Add standard tokens
        self.padToken = self.getWordId('<pad>')  # Padding (Warning: first things to add > id=0 !!)
        self.goToken = self.getWordId('<go>')  # Start of sequence
        self.eosToken = self.getWordId('<eos>')  # End of sequence
        self.unknownToken = self.getWordId('<unknown>')  # Word dropped from vocabulary

        # Preprocessing data
        # conversation is convObj / miniConv
        for conversation in tqdm(conversations, desc='Extract conversations'):
            self.extractConversation(conversation)

        # The dataset will be saved in the same order it has been extracted

    def extractConversation(self, conversation):
        """Extract the sample lines from the conversations
        Args:
            conversation (Obj): a conversation object containing the lines to extract
        """

        maxContextSize = 3

        words = list(map(lambda x: self.extractText(x['text']), conversation['lines']))

        if len(conversation['lines']) <= maxContextSize + 1:
            sub_conversations = [[words[:-1], words[-1]]]
        else:
            sub_conversations = [[words[i:i + maxContextSize], words[i + maxContextSize]]
                                 for i in range(len(words) - maxContextSize - 1)]

        self.trainingSamples.extend(sub_conversations)

        # nLines = len(conversation['lines'])
        #
        # # if nLines <= maxContextSize + 1:
        # targetLine = conversation['lines'][nLines - 1]
        # targetWords = self.extractText(targetLine['text'])
        # inputContext = []
        #
        # # (Tay: Now only iterate for the  context part)
        # # Iterate over all the lines of the conversation (convObj / miniConv)
        # for i in (range(len(conversation['lines']))):
        #     inputLine  = conversation['lines'][i]
        #     inputWords  = self.extractText(inputLine['text'])
        #     inputWords = self.extractText(inputLine['text'])
        #     if inputWords:
        #         inputContext.append(inputWords)
        #
        # # TODO: Need to discuss spec of the context
        # if len(inputContext) > 0 and targetWords:
        #     self.trainingSamples.append([inputContext, targetWords])
        #
        #
        # else:
        #     # moving window
        #     window = deque([], maxContextSize + 1)
        #     for i in range(len(conversation['lines'])):
        #         line = conversation['lines'][i]
        #         words = self.extractText(line['text'])
        #         if words:
        #             window.append(words)
        #             context = list(window)[:-1]
        #             target = list(window)[-1]
        #             self.trainingSamples.append([context, target])

    def extractText(self, line):
        """Extract the words from a sample lines
        Args:
            line (str): a line containing the text to extract
        Return:
            list<list<int>>: the list of sentences of word ids of the sentence
        """
        sentences = []  # List[List[str]]

        # Extract sentences
        sentencesToken = nltk.sent_tokenize(line)

        # We add sentence by sentence until we reach the maximum length
        for i in range(len(sentencesToken)):
            tokens = nltk.word_tokenize(sentencesToken[i])

            tempWords = []
            for token in tokens:
                tempWords.append(self.getWordId(token))  # Create the vocabulary and the training sentences

            sentences.append(tempWords)

        return sentences

    def getWordId(self, word, create=True):
        """Get the id of the word (and add it to the dictionary if not existing). If the word does not exist and
        create is set to False, the function will return the unknownToken value
        Args:
            word (str): word to add
            create (Bool): if True and the word does not exist already, the world will be added
        Return:
            int: the id of the word created
        """
        # Should we Keep only words with more than one occurrence ?

        word = word.lower()  # Ignore case

        # At inference, we simply look up for the word
        if not create:
            wordId = self.word2id.get(word, self.unknownToken)
        # Get the id if the word already exist
        elif word in self.word2id:
            wordId = self.word2id[word]
            self.idCount[wordId] += 1
        # If not, we create a new entry
        else:
            wordId = len(self.word2id)
            self.word2id[word] = wordId
            self.id2word[wordId] = word
            self.idCount[wordId] = 1

        return wordId

    def printBatch(self, batch):
        """Print a complete batch, useful for debugging
        Args:
            batch (Batch): a batch object
        """
        print('----- Print batch -----')
        for i in range(len(batch.encoderSeqs[0])):  # Batch size
            print('Encoder: {}'.format(self.batchSeq2str(batch.encoderSeqs, seqId=i)))
            print('Decoder: {}'.format(self.batchSeq2str(batch.decoder_inputs, seqId=i)))
            print('Targets: {}'.format(self.batchSeq2str(batch.decoder_targets, seqId=i)))
            print('Weights: {}'.format(' '.join([str(weight) for weight in [batchWeight[i] for batchWeight in batch.weights]])))

    def sequence2str(self, sequence, clean=False, reverse=False):
        """Convert a list of integer into a human readable string
        Args:
            sequence (list<int>): the sentence to print
            clean (Bool): if set, remove the <go>, <pad> and <eos> tokens
            reverse (Bool): for the input, option to restore the standard order
        Return:
            str: the sentence
        """

        if not sequence:
            return ''

        if not clean:
            return ' '.join([self.id2word[idx] for idx in sequence])

        sentence = []
        for wordId in sequence:
            if wordId == self.eosToken:  # End of generated sentence
                break
            elif wordId != self.padToken and wordId != self.goToken:
                sentence.append(self.id2word[wordId])

        if reverse:  # Reverse means input so no <eos> (otherwise pb with previous early stop)
            sentence.reverse()

        return self.detokenize(sentence)

    def detokenize(self, tokens):
        """Slightly cleaner version of joining with spaces.
        Args:
            tokens (list<string>): the sentence to print
        Return:
            str: the sentence
        """
        return ''.join([
            ' ' + t if not t.startswith('\'') and
                       t not in string.punctuation
                    else t
            for t in tokens]).strip().capitalize()

    def batchSeq2str(self, batchSeq, seqId=0, **kwargs):
        """Convert a list of integer into a human readable string.
        The difference between the previous function is that on a batch object, the values have been reorganized as
        batch instead of sentence.
        Args:
            batchSeq (list<list<int>>): the sentence(s) to print
            seqId (int): the position of the sequence inside the batch
            kwargs: the formatting options( See sequence2str() )
        Return:
            str: the sentence
        """
        sequence = []
        for i in range(len(batchSeq)):  # Sequence length
            sequence.append(batchSeq[i][seqId])
        return self.sequence2str(sequence, **kwargs)

#########################################################

    def context2enco(self, context):
        if context == '':
            return None

        contextBatch = []
        for i in range(len(context)):
            sentence = context[i]
            tokens = nltk.word_tokenize(sentence)
            if len(tokens) > self.args.maxLength:
                tokens = tokens[:self.args.maxLength]

            wordIds = []
            for token in tokens:
                wordIds.append(self.getWordId(token, create=False))

            contextBatch.append(wordIds)

        batch = self._createBatch([[contextBatch, []]]) # Mono batch, no target output

        return batch

#########################################################

    def sentence2enco(self, sentence):
        """Encode a sequence and return a batch as an input for the model
        Return:
            Batch: a batch object containing the sentence, or none if something went wrong
        """

        if sentence == '':
            return None

        # First step: Divide the sentence in token
        tokens = nltk.word_tokenize(sentence)
        if len(tokens) > self.args.maxLength:
            return None

        # Second step: Convert the token in word ids
        wordIds = []
        for token in tokens:
            wordIds.append(self.getWordId(token, create=False))  # Create the vocabulary and the training sentences

        # Third step: creating the batch (add padding, reverse)
        batch = self._createBatch([[wordIds, []]])  # Mono batch, no target output

        return batch

    def deco2sentence(self, decoderOutputs):
        """Decode the output of the decoder and return a human friendly sentence
        decoderOutputs (list<np.array>):
        """
        sequence = []

        # Choose the words with the highest prediction score
        for out in decoderOutputs:
            sequence.append(np.argmax(out))  # Adding each predicted word ids

        return sequence  # We return the raw sentence. Let the caller do some cleaning eventually

    def playDataset(self):
        """Print a random dialogue from the dataset
        """
        print('Randomly play samples:')
        for i in range(self.args.playDataset):
            idSample = random.randint(0, len(self.trainingSamples) - 1)
            print('Q: {}'.format(self.sequence2str(self.trainingSamples[idSample][0], clean=True)))
            print('A: {}'.format(self.sequence2str(self.trainingSamples[idSample][1], clean=True)))
            print()
        pass


def tqdm_wrap(iterable, *args, **kwargs):
    """Forward an iterable eventually wrapped around a tqdm decorator
    The iterable is only wrapped if the iterable contains enough elements
    Args:
        iterable (list): An iterable object which define the __len__ method
        *args, **kwargs: the tqdm parameters
    Return:
        iter: The iterable eventually decorated
    """
    if len(iterable) > 100:
        return tqdm(iterable, *args, **kwargs)

    return iterable