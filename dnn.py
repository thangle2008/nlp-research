from __future__ import absolute_import, print_function, division

import argparse
import random

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from yelp_data import load_yelp_reviews
from model.deepnn import DeepNet, train
from utils.textdata import (
    get_text_data_generator,
    tokenize_text,
    write_raw_texts,
    Word2VecEmbedder,
)


parser = argparse.ArgumentParser()
parser.add_argument('-embed_file', action='store', default=None) # path to embedding file
parser.add_argument('-raw_text_file', action='store', default=None)
# Data configurations
parser.add_argument('--binary', action='store_true')
parser.add_argument('-input_size', type=int, default=100)
parser.add_argument('-seed', type=int, default=42)
parser.add_argument('-bs', type=int, default=32) # batch size
parser.add_argument('-doclen', type=int, default=100) # maximum number of words for each document
# Optimizer configurations
parser.add_argument('-lr', type=float, default=0.01)
parser.add_argument('-mom', type=float, default=0.9)
parser.add_argument('-epoch', type=int, default=10)


# Main function
def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load data
    print("Load data...")

    # text data should be in lowercase
    texts, labels = load_yelp_reviews(line_limit=50000)
    labels = [l-1 for l in labels] # map to index 0

    texts = [tokenize_text(t) for t in texts]

    # split data
    train_t, test_t, train_l, test_l = train_test_split(texts, labels,
        shuffle=True, stratify=labels, test_size=0.2)

    if args.raw_text_file is not None:
        write_raw_texts(train_t, args.raw_text_file)
        return

    # get number of labels
    num_labels = max(labels) - min(labels) + 1
    print("Number of labels =", num_labels)

    # load embedding vectors
    print("Load embeddings...")
    embedder = Word2VecEmbedder(args.embed_file)

    # construct model, optimizer, and objective function
    model = DeepNet(num_labels, input_size=args.input_size)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5)
    criterion = nn.CrossEntropyLoss()

    train_data_generator = get_text_data_generator(train_t, train_l,
            args.bs, embedder, verbose=1)
    eval_data_generator = get_text_data_generator(test_t, test_l,
            args.bs, embedder, verbose=1)

    print("Begin training...")
    train(model, optimizer, criterion, train_data_generator,
            scheduler=scheduler, num_epochs=args.epoch, 
            eval_data_generator=eval_data_generator)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
