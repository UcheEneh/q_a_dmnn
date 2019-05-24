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

import util
# import tensorflow as tf

# Exploring dataset bAbI

"""
In some tasks in bAbI, the system will encounter words that are not in the GloVe word vectorization. 
In order for the network to be capable of processing these unknown words, we need to maintain a 
consistent vectorization of those words. Common practice is to replace all unknown tokens with a 
single <UNK> vector, but this isn't always effective. Instead, we can use randomization to draw a new 
vectorization for each unique unknown token.

The first time we run across a new unknown token, we simply draw a new vectorization from the 
(Gaussian-approximated) distribution of the original GloVe vectorizations, and add that vectorization 
back to the GloVe word map . To gather the distribution hyperparameters, Numpy has functions that 
automatically calculate variance and mean.

fill_unk will take care of giving us a new word vectorization whenever we need one.
"""

# Deserialize GloVe vectors
glove_wordmap = {}
with open(util.GLOVE_VECTOR_FILE, "r", encoding="utf8") as glove:
    for line in glove:
        name, vector = tuple(line.split(" ", 1))
        glove_wordmap[name] = np.fromstring(vector, sep=" ")

wvecs = []
# Get all the vector values and stack vertically
for item in glove_wordmap.items():
    wvecs.append(item[1])
s = np.vstack(wvecs)

# Gather the distribution hyperparams
var = np.var(s, 0)
mean = np.mean(s, 0)
RS = np.random.RandomState()


def fill_unk(unk):
    global glove_wordmap
    glove_wordmap[unk] = RS.multivariate_normal(mean, np.diag(var))
    return glove_wordmap[unk]




def download_files():
    if not os.path.isfile(util.GLOVE_VECTOR_FILE):
        raise ("Check GloVe Path")

    if not os.path.isfile(util.BABI_DATASET_ZIP) and not os.path.isfile(util.TRAIN_SET) \
            and not os.path.isfile(util.TEST_SET):
        urlretrieve(util.BABI_DATASET_LINK, util.BABI_DATASET_ZIP)


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


def targz_unzip_single_file(zip_file_name, output_file_name, interior_relative_path):
    if not os.path.isfile(output_file_name):
        with tarfile.open(zip_file_name) as unzipped:
            unzipped.extract(interior_relative_path + output_file_name)


def perform_unzip():
    targz_unzip_single_file(util.BABI_DATASET_ZIP, util.TRAIN_SET, "tasks_1-20_v1-2/en/")
    targz_unzip_single_file(util.BABI_DATASET_ZIP, util.TEST_SET, "tasks_1-20_v1-2/en/")


"""
We use a greedy search for words that exist in Stanford's GLoVe word vectorization data set, 
and if the word does not exist, then we fill in the entire word with an unknown, randomly created, 
new representation.
"""

def sentence_to_id(sentence):
    """
        Turns an input paragraph into an (m,d) matrix,
        where n is the number of tokens in the sentence
        and d is the number of dimensions each word vector has.
    """
    tokens = sentence.strip('"(),-').lower().split(" ")
    rows = []
    words = []

    # Greedy search for tokens
    for token in tokens:
        i = len(token)
        while len(token) > 0:
            word = token[:i]
            if word in glove_wordmap:
                rows.append(glove_wordmap[word])
                words.append(word)
                token = token[i:]
                i = len(token)
                continue
            else:
                i = i - 1
            if i == 0:
                # word OOV
                rows.append(fill_unk(token))
                words.append(token)
                break
    return np.array(rows), words


def contextualize(set_file):
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
            l, ine = tuple(line.split(" ", 1))
            # Split the line numbers from the sentences they refer to
            if l is "1":
                # New contexts always start with 1, so this is a signal to reset the context
                context = []
            if "\t" in line:
                # Tabs are the separator between questions and answers
                # and are not present in context statements
                question, answer, support = tuple(ine.split("\t"))
                data.append((tuple(zip(*context)) + sentence_to_id(question) +
                             sentence_to_id(answer) + ([int(s) for s in support.split()],)
                             ))
                # Multiple questions may refer to the same context, so we don't reset it
            else:
                # Context sentence
                context.append(sentence_to_id(ine[:-1]))
    return data


def create_train_test_data():
    train_data = contextualize(util.TRAIN_SET_POST)
    test_data = contextualize(util.TEST_SET_POST)
    return train_data, test_data

def finalize(data):
    """
    Prepares data generated by contextualize() for use in the network
    """
    final_data = []
    for cqas in data:
        contextvs, contextws, questvs, questws, ansvs, answs, suppt = cqas

        lengths = itertools.accumulate(len(c_vec) for c_vec in contextvs)
        context_vec = np.concatenate(contextvs)
        context_words = sum(contextws, [])

        # Location markers for the beginnings of new sentences
        sentence_ends = np.arry(list(lengths))
        final_data.append((context_vec, sentence_ends, questvs, suppt, context_words, cqas, ansvs, answs))
    return np.array(final_data)


def create_final_train_test_data():
    train_data, test_data = create_final_train_test_data()
    final_train_data = finalize(train_data)
    final_test_data = contextualize(test_data)
    return final_train_data, final_test_data

