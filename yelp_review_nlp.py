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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from deepnn import DeepNet

parser = argparse.ArgumentParser()
# There are 3 modes for this script:
parser.add_argument('-embed_file', action='store', default=None) # path to embedding file
parser.add_argument('-save_embeddings', action='store', default=None)
parser.add_argument('-save_raw_texts', action='store', default=None)
# Data configurations
parser.add_argument('-seed', type=int, default=42)
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
        text, label = data['text'].lower(), data['stars']
        dataset.append((text, label))
        if line_no >= line_limit:
            break
    return dataset


def write_texts(fname, texts):
    fin = io.open(fname, 'w', encoding='utf-8')
    for t in texts:
        s = ' '.join(word_tokenize(t))
        fin.write(s + '\n')


# Functions for loading and saving embeddings
def load_embeddings(fname, vocabs):
    """
    Load embedding vectors. Note that we only load those that are in the vocabs.
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if tokens[0] in vocabs:
            data[tokens[0]] = map(float, tokens[1:])
    return data, d


def save_embeddings(fname, embed_dict, embed_dim):
    """
    Store word embeddings in a file.
    """
    fin = io.open(fname, 'w', encoding='utf-8')
    fin.write(u"{} {}\n".format(len(embed_dict), embed_dim))
    for word, vector in embed_dict.items():
        fin.write(word)
        for x in vector:
            fin.write(u" {}".format(x))
        fin.write(u'\n')


# Text processing functionss
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


# Data processing functions
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


def get_num_corrects(y_pred, y_target):
    _, y_pred = torch.max(y_pred.data, 1)
    return (y_pred == y_target).sum().item()


# Main function
def run(args):
    if not args.embed_file and not args.save_embeddings \
                           and not args.save_raw_texts:
        return

    random.seed(args.seed)

    # load data
    print("Load data...")

    # text data should be in lowercase
    dataset = load_yelp_data()

    texts, labels = zip(*dataset)
    data_train, data_test = train_test_split(texts, labels)

    X_train, y_train = data_train
    X_test, y_test = data_test

    # save raw texts (for training embeddings)
    if args.save_raw_texts:
        print("Save raw texts...") 
        write_texts(args.save_raw_texts, X_train)
        return

    # load embedding vectors
    print("Load embeddings...")
    vocabs = set([w for t in texts for w in word_tokenize(t)])
    embed_dict, embed_dim = load_embeddings(args.embed_file, vocabs)

    # save embeddings (for fast load next time)
    if args.save_embeddings is not None:
        print("Save embeddings...")
        save_embeddings(args.save_embeddings, embed_dict, embed_dim)
        return

    # get number of labels
    num_labels = max(labels) + 1
    print("Number of labels =", num_labels)

    # construct model, optimizer, and objective function
    model = DeepNet(num_labels)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5)
    criterion = nn.CrossEntropyLoss()

    # training loop
    best_val_loss = float("inf")
    print("Begin training...")
    for e in range(args.epoch):
        # update
        train_loss = 0.0
        train_corrects = 0
        total_train_labels = 0
        niter = int(math.ceil(len(X_train) / args.bs))
        for i, batch_indices in get_batch_indices(len(X_train), args.bs):
            print("Train batch {}/{}...".format(i + 1, niter), end="\r")
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
            # accumulate loss and num corrects
            train_loss += loss.item()
            train_corrects += get_num_corrects(y_pred, y_batch)
            total_train_labels += y_batch.size(0)
        train_loss /= niter
        # test
        test_loss = 0.0
        test_corrects = 0
        total_test_labels = 0
        niter = int(math.ceil(len(X_test) / args.bs))
        print("Testing...", end="\r")
        with torch.no_grad():
            for _, batch_indices in get_batch_indices(len(X_test), args.bs):
                X_batch, y_batch = extract_batch(X_test, y_test, batch_indices,
                                                 args.doclen, embed_dim,
                                                 embed_dict)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                # accumuate loss and num corrects
                test_corrects += get_num_corrects(y_pred, y_batch)
                total_test_labels += y_batch.size(0)
                test_loss += loss.item()
            test_loss /= niter
            # decrease learning rate if stuck
            scheduler.step(test_loss)
        print(("Epoch {}/{}: train_loss = {:.6f}, train_acc = {:.2f}%, "
               "test_loss = {:.6f}, test_acc = {:.2f}%").format(
            e + 1, args.epoch,
            train_loss, train_corrects * 100.0 / total_train_labels,
            test_loss, test_corrects * 100.0 / total_test_labels,
        ))

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
