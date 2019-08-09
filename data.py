import numpy as np
import urllib
import sys
import os
import zipfile
import tarfile
import json
import hashlib
import re
import itertools
from urllib.request import urlretrieve, urlopen
from tqdm import tqdm
# import util
# import tensorflow as tf

# Debug
DISPLAY_GLOVE_COMPARISON = False

# Exploring dataset bAbI
"""
In some tasks in bAbI, the system will encounter words that are not in the GloVe word vectorization. In order for the 
network to be capable of processing these unknown words, we need to maintain a consistent vectorization of those words. 
Common practice is to replace all unknown tokens with a single <UNK> vector, but this isn't always effective. 
Instead, we can use randomization to draw a new vectorization for each unique unknown token.

The first time we run across a new unknown token, we simply draw a new vectorization from the (Gaussian-approximated) 
distribution of the original GloVe vectorizations, and add that vectorization back to the GloVe word map. To gather the 
distribution hyperparameters, Numpy has functions that automatically calculate variance and mean.

fill_unk will take care of giving us a new word vectorization whenever we need one.
"""

class BabiCorpus:
    # def deserialize_glove_vectors():
    # Deserialize GloVe vectors
    def __init__(self, params):
        self.params = params
        
    def deserialize_glove_vector(self):
        self.glove_wordmap = {}
        with open(self.params.glove_vector_file, "r", encoding="utf8") as glove:
            for line in tqdm(glove):
                name, vector = tuple(line.split(" ", 1))
                self.glove_wordmap[name] = np.fromstring(vector, sep=" ")

        if DISPLAY_GLOVE_COMPARISON:
            from matplotlib import pyplot as plt
            plt.title("Glove vector comparison of two words")
            plt.ylabel("Man")
            plt.xlabel("Woman")
            plt.plot(self.glove_wordmap['man'], self.glove_wordmap['woman'], 'ro')
            plt.show()

        wvecs = []
        # Get all the vector values and stack vertically
        for item in self.glove_wordmap.items():
            wvecs.append(item[1])
        s = np.vstack(wvecs)        # convert from list to array
    
        # Gather the distribution hyperparams
        self.params.var = np.var(s, 0)
        self.params.mean = np.mean(s, 0)

    def fill_unk(self, var, mean, unk):
        RS = np.random.RandomState()
        # global glove_wordmap
        self.glove_wordmap[unk] = RS.multivariate_normal(mean, np.diag(var))
        return self.glove_wordmap[unk]
    
    def download_files(self):
        if not os.path.isfile(self.params.glove_vector_file):
            raise ("Check GloVe Path")
    
        if not os.path.isfile(self.params.babi_dataset_zip) and not os.path.isfile(self.params.train_set) \
                and not os.path.isfile(self.params.test_set):
            urlretrieve(self.params.BABI_DATASET_LINK, self.params.babi_dataset_zip)

    @staticmethod
    def unzip_single_file(zip_file_name, output_file_name):
        """
            If the output file is already created, don't recreate
            If the output file does not exist, create it from the zipFile
        """
        if not os.path.isfile(output_file_name):
            with open(output_file_name, 'wb') as out_file:
                with zipfile.ZipFile(zip_file_name) as zipped:
                    for info in zipped.infolist():
                        if output_file_name in info.filename:
                            with zipped.open(info) as requested_file:
                                out_file.write(requested_file.read())
                                return

    @staticmethod
    def targz_unzip_single_file(zip_file_name, output_file_name, interior_relative_path):
        if not os.path.isdir(interior_relative_path):
            os.mkdir(interior_relative_path)

        if not os.path.isfile(os.path.join(interior_relative_path, output_file_name)):
            with tarfile.open(zip_file_name) as unzipped:
                unzipped.extract(interior_relative_path + output_file_name)
    
    def sentence_to_id(self, sentence):
        """
            Turns an input paragraph into an (m,d) matrix, where n is the number of tokens in the sentence and d is the
            number of dimensions each word vector has.
    
            We use a greedy search for words that exist in Stanford's GLoVe word vectorization data set, and if the word
            does not exist, then we fill in the entire word with an unknown, randomly created, new representation.
        """
        tokens = sentence.strip('"(),-').lower().split(" ")
        rows = []
        words = []
    
        # Greedy search for tokens
        for token in tokens:
            i = len(token)
            while len(token) > 0:
                word = token[:i]
                if word in self.glove_wordmap:
                    rows.append(self.glove_wordmap[word])
                    words.append(word)
                    token = token[i:]   # reset while loop if full word is found
                    i = len(token)
                    continue
                else:
                    i = i - 1   # tries if there's a punctuation at the end of the word
                if i == 0:
                    # word OOV
                    rows.append(self.fill_unk(self.params.var, self.params.mean, token))
                    words.append(token)
                    break
        return np.array(rows), words

    def contextualize(self, set_file):
        """
        Read in the dataset of questions and build (question + answer -> context) sets.
        Output is a list of data points, each of which is a 7-element tuple containing:
            The sentences in the context in vectorized form.
            The sentences in the context as a list of string tokens.
            The question in vectorized form.
            The question as a list of string tokens.
            The answer in vectorized form.
            The answer as a list of string tokens.
            A list of numbers for supporting statements, which is currently unused.
        """
        data = []
        context = []
        with open(set_file, 'r', encoding="utf8") as train:
            for line in train:
                l, ine = tuple(line.split(" ", 1))      # maxsplit = 1
                # Split the line numbers from the sentences they refer to
                if l is "1":
                    # New contexts always start with 1, so this is a signal to reset the context
                    context = []
                if "\t" in line:
                    # Tabs are the separator between questions and answers
                    # and are not present in context statements
                    question, answer, support = tuple(ine.split("\t"))
                    data.append((tuple(zip(*context)) +
                                 self.sentence_to_id(question) +
                                 self.sentence_to_id(answer) +
                                 ([int(s) for s in support.split()],)))
                    # Multiple questions may refer to the same context, so we don't reset it
                else:
                    # Context sentence
                    context.append(self.sentence_to_id(ine[:-1]))   # all except last character = '\n'
        return data
    
    def create_train_test_data(self):
        train_data = self.contextualize(self.params.train_set_post)
        test_data = self.contextualize(self.params.test_set_post)
        return train_data, test_data

    @staticmethod
    def finalize(data):
        """
        Prepares data generated by contextualize() for use in the network
        """
        final_data = []
        for cqas in data:   # context, quest, answer, support (not used)
            context_vecs, context_words, quest_vecs, quest_words, ans_vecs, ans_words, suppt = cqas
            lengths = itertools.accumulate(len(c_vec) for c_vec in context_vecs)    # get length of each ctxt vector
            context_vecs = np.concatenate(context_vecs)
            context_words = sum(context_words, [])
    
            # Location markers for the beginnings of new sentences
            sentence_ends = np.array(list(lengths))     # store the range using the length of each ctxt vector
            final_data.append((context_vecs, sentence_ends, quest_vecs, suppt, context_words, cqas, ans_vecs, ans_words))
        return np.array(final_data)

    def create_final_train_test_data(self):
        train_data, test_data = self.create_final_train_test_data()
        final_train_data = BabiCorpus.finalize(train_data)
        final_test_data = self.contextualize(test_data)
        return final_train_data, final_test_data

    @staticmethod
    def prep_batch(batch_data, context_placeholder, input_sentence_endings, query, input_query_lengths,
                   gold_standard, more_data=False):
        """
            Prepare all the preproccessing that needs to be done on a batch-by-batch basis.
        """
        context_vec, sentence_ends, questionvs, spt, context_words, cqas, answervs, _ = zip(*batch_data)
        ends = list(sentence_ends)
        maxend = max(map(len, ends))
        aends = np.zeros((len(ends), maxend))
        for index, i in enumerate(ends):
            for indexj, x in enumerate(i):
                aends[index, indexj] = x-1
        new_ends = np.zeros(aends.shape+(2,))
    
        for index, x in np.ndenumerate(aends):
            new_ends[index+(0,)] = index[0]
            new_ends[index+(1,)] = x
    
        contexts = list(context_vec)
        max_context_length = max([len(x) for x in contexts])
        contextsize = list(np.array(contexts[0]).shape)
        contextsize[0] = max_context_length
        final_contexts = np.zeros([len(contexts)]+contextsize)
    
        contexts = [np.array(x) for x in contexts]
        for i, context in enumerate(contexts):
            final_contexts[i,0:len(context),:] = context
        max_query_length = max(len(x) for x in questionvs)
        querysize = list(np.array(questionvs[0]).shape)
        querysize[:1] = [len(questionvs),max_query_length]
        queries = np.zeros(querysize)
        querylengths = np.array(list(zip(range(len(questionvs)),[len(q)-1 for q in questionvs])))
        questions = [np.array(q) for q in questionvs]
        for i, question in enumerate(questions):
            queries[i,0:len(question),:] = question
        data = {context_placeholder: final_contexts, input_sentence_endings: new_ends,
                                query:queries, input_query_lengths:querylengths, gold_standard: answervs}
        return (data, context_words, cqas) if more_data else data

    def prepare_data(self):
        # TODO: implement better conditions for the files

        if not os.path.isfile(self.params.glove_vector_file):
            raise ("Download and unzip Glove file")
        if not os.path.isfile(self.params.test_set_post):
            """
            self.targz_unzip_single_file(self.params.babi_dataset_zip, self.params.train_set,
                                         self.params.interior_relative_path)
            self.targz_unzip_single_file(self.params.babi_dataset_zip, self.params.test_set,
                                         self.params.interior_relative_path)
            """
            raise ("Download and unzip Babi file")

        self.deserialize_glove_vector()
    
        # Now we can package all the data together needed for each question, including the vectorization of the
        # contexts, questions, and answers.
        train_data = self.contextualize(self.params.train_set_post)
        test_data = self.contextualize(self.params.test_set_post)
    
        final_train_data = []
        final_train_data = BabiCorpus.finalize(train_data)
        final_test_data = BabiCorpus.finalize(test_data)
    
        return final_train_data, final_test_data
