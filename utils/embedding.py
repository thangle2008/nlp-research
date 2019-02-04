from __future__ import absolute_import, print_function, division

import io
import random


def load_embeddings(file_name, vocabs=None):
    """
    Load embedding vectors from a file. 
    If a vocab set is supplied, only load those that are in there.
    """
    fin = io.open(file_name, 'r', encoding='utf-8', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if vocabs is None or tokens[0] in vocabs:
            data[tokens[0]] = map(float, tokens[1:])
    return data, d


def embed_text(text, embed_dict, embed_dim, doclen=100):
    vectors = []
    for s in text:
        for w in s:
            if len(vectors) == doclen:
                break
            v = embed_dict[w] if w in embed_dict \
                else [random.random() for _ in range(embed_dim)]
            vectors.append(v)
        # pad with random vectors
        while len(vectors) < doclen:
            vectors.append([random.random() for _ in range(embed_dim)])
    return vectors
