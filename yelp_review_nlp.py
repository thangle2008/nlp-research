from __future__ import print_function, division, absolute_import

import os, io, json, time
import argparse
import random, itertools, math

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import torch.optim as optim
from deepnn import DeepNet

parser = argparse.ArgumentParser()
parser.add_argument('-embed_file', action='store', required=True) # path to embedding file
parser.add_argument('-seed', type=int, default=42)
# Data configurations
parser.add_argument('-bs', type=int, default=32) # batch size
parser.add_argument('-doclen', type=int, default=100) # maximum number of words for each document
# Optimizer configurations
parser.add_argument('-lr', type=float, default=0.01)
parser.add_argument('-mom', type=float, default=0.9)
parser.add_argument('-epoch', type=int, default=10)


DATA_FILE = './data/yelp/yelp_academic_dataset_review.json'
TEXT_TMP_FILE = './tmp/yelp_review_raw.txt'
NP_RANDOM_MAX = 100

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

# Helper functions
def get_random_state():
    return random.randint(0, NP_RANDOM_MAX)

# Functions for loading dataset
def load_yelp_data(line_limit=50000):
    fin = io.open(DATA_FILE, 'r', encoding='utf-8')
    dataset = []
    line_no = 0
    for line in fin:
        line_no += 1
        data = json.loads(line)
        text, label = data['text'], data['stars']
        dataset.append((text, label))
        if line_no >= line_limit:
            break
    return dataset


def write_yelp_data(fname):
    dataset = load_yelp_data()
    fin = io.open(fname, 'w', encoding='utf-8')
    for data in dataset:
        s = ' '.join(word_tokenize(data['text']))
        fin.write(s.lower() + '\n')


# Text processing functions
def load_embeddings(fname):
    """Load embedding vectors."""
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data, d


def embed_texts(texts, embed_dict, doclen, embed_dim):
    X = []
    for s in texts:
        words = word_tokenize(s)
        vectors = []
        for w in words:
            if len(vectors) == doclen:
                break
            v = embed_dict[w] if w in embed_dict \
                else [random.random() for _ in range(embed_dim)]
            vectors.append(v)
        # pad with random vectors
        while len(vectors) < doclen:
            vectors.append([random.random() for _ in range(embed_dim)])
        X.append(vectors)
    return X


def train_test_split(texts, labels):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                 random_state=get_random_state())
    train_index, test_index = next(sss.split(texts, labels))

    X_train = [texts[i] for i in train_index]
    y_train = [labels[i] for i in train_index]

    X_test = [texts[i] for i in test_index]
    y_test = [labels[i] for i in test_index]

    return (X_train, y_train), (X_test, y_test)


def get_batch_indices(n, batch_size):
    num = -1
    indices = range(n)
    random.shuffle(indices)
    for i in range(0, n, batch_size):
        num += 1
        j = min(i + batch_size, n)
        yield num, indices[i:j]


def extract_batch(X, y, batch_indices, doclen, embed_dim, embed_dict):
    X_batch = [X[b] for b in batch_indices]
    y_batch = [y[b] for b in batch_indices]
    # process texts
    X_batch = embed_texts(X_batch, embed_dict, doclen, embed_dim)
    # convert to tensor
    X_batch = FloatTensor(X_batch).view(-1, 1, doclen, embed_dim)
    y_batch = LongTensor(y_batch)
    return X_batch, y_batch

# Main function
def run(args):
    random.seed(args.seed)

    # load data
    print("Load data...")
    dataset = load_yelp_data()

    texts, labels = zip(*dataset)
    data_train, data_test = train_test_split(texts, labels)

    X_train, y_train = data_train
    X_test, y_test = data_test

    # load embedding vectors
    print("Load embeddings...")
    embed_dict, embed_dim = load_embeddings(args.embed_file)

    num_labels = max(labels) + 1

    # construct model, optimizer, and objective function
    model = DeepNet(num_labels)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
    criterion = nn.CrossEntropyLoss()

    # training loop
    print("Begin training...")
    for e in range(args.epoch):
        # update
        train_loss = 0.0
        niter = 0
        #train_time = time.time()
        total_batches = int(math.ceil(len(X_train) / args.bs))
        for i, batch_indices in get_batch_indices(len(X_train), args.bs):
            print("Train batch {}/{}...".format(i + 1, total_batches), end="\r")
            # extract batch
            X_batch, y_batch = extract_batch(X_train, y_train, batch_indices,
                                             args.doclen, embed_dim, embed_dict)
            # feed forward
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            # back propagation
            loss.backward()
            optimizer.step()
            # accumulate loss
            train_loss += loss.item()
            niter += 1
        train_loss /= niter
        #train_time = time.time() - train_time
        # test
        test_loss = 0.0
        niter = 0
        for i, batch_indices in get_batch_indices(len(X_test), args.bs):
            X_batch, y_batch = extract_batch(X_test, y_test, batch_indices,
                                             args.doclen, embed_dim, embed_dict)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            test_loss += loss.item()
            niter += 1
        test_loss /= niter
        print("Epoch {}/{}: train_loss = {:.6f}, test_loss = {:.6f}".format(
            e + 1, args.epoch, train_loss, test_loss
        ))

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
