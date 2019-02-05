from __future__ import absolute_import, print_function, division

import io
import math
import random

import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize


################ Functions for generating data batches ##################
def generate_batch_indices(n, batch_size):
    num = -1
    indices = np.arange(n)
    np.random.shuffle(indices)
    for i in range(0, n, batch_size):
        num += 1
        j = min(i + batch_size, n)
        yield num, indices[i:j]


def get_text_data_generator(texts, labels, batch_size, embedder, verbose=1):
    """
    Return a generator for getting batches of embedded texts with labels.
    """
    def f_generator():                            
        n = len(texts)
        num_batches = int(math.ceil(len(texts) / batch_size))
        for i, indices in generate_batch_indices(n, batch_size):
            if verbose == 1:
                print("On batch {}/{}".format(i+1, num_batches), end='\r')
            # extract batch
            X_batch = [embedder.embed(texts[j]) for j in indices]
            y_batch = [labels[j] for j in indices]
            yield X_batch, y_batch

    return f_generator


################ Functions for loading and writing data ##################
def write_raw_texts(texts, output_path):
    with io.open(output_path, 'w', encoding='utf-8') as f:
        for t in texts:
            for s in t:
                f.write(u"{}\n".format(" ".join(s)))


################ Functions for embedding data ##################
def tokenize_text(text):
    text = sent_tokenize(text)
    text = [word_tokenize(s) for s in text]
    return text


def load_embeddings(file_name, vocabs=None):
    """
    Load embedding vectors from a file. 
    If a vocab set is supplied, only load those that are in there.

    The embed file should be in the following format:
    - First line: number_of_tokens vector_dimension
    - Subsequent lines: token space_separated_floating_points
    """
    fin = io.open(file_name, 'r', encoding='utf-8', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if vocabs is None or tokens[0] in vocabs:
            data[tokens[0]] = map(float, tokens[1:])
    return data, d


class Embedder(object):
    """
    Interface for text embedder.
    """

    def embed(self, text):
        raise NotImplementedError


class Word2VecEmbedder(Embedder):
    """
    Embedder for embed a text document as floating-point vectors (matrix).
    """

    def __init__(self, embed_file, vocabs=None, doclen=100):
        embed_dict, embed_dim = load_embeddings(embed_file, vocabs=vocabs)
        self.embed_dict = embed_dict
        self.embed_dim = embed_dim
        self.doclen = doclen

    
    def embed(self, text):
        vectors = []

        def build_vectors():
            for s in text:
                for w in s:
                    if len(vectors) == self.doclen:
                        return
                    v = self.embed_dict[w] if w in self.embed_dict \
                        else [random.random() for _ in range(self.embed_dim)]
                    vectors.append(v)
            # pad with random vectors if not enough len
            while len(vectors) < self.doclen:
                vectors.append([random.random() for _ in range(self.embed_dim)])

        build_vectors()
        return vectors
