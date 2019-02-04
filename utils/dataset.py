from __future__ import absolute_import, print_function, division

import io
import math

import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize


################ Functions for generating data ##################
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
            X_batch = [embedder(texts[j]) for j in indices]
            y_batch = [labels[j] for j in indices]
        yield X_batch, y_batch

    return f_generator


def write_labeled_texts(texts, labels, output_path):
    return


def read_labeled_texts(input_path):
    return


def write_raw_texts(texts, output_path):
    with io.open(output_path, 'w', encoding='utf-8') as f:
        for t in texts:
            for s in t:
                f.write(u"{}\n".format(" ".join(s)))


def tokenize_text(text):
    text = sent_tokenize(text)
    text = [word_tokenize(s) for s in text]
    return text
