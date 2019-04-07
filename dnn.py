from __future__ import absolute_import, print_function, division

import argparse
import random

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import yelp_data as yd
from model.deepnn import DeepNet, train
from utils.textdata import (
    get_text_data_generator,
    tokenize_text,
    Word2VecEmbedder,
)


parser = argparse.ArgumentParser()
parser.add_argument('-embed_file', action='store', required=True) # path to embedding file
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
    train_x, train_y = yd.load_yelp_reviews(yd.TRAIN_FILE)
    test_x, test_y = yd.load_yelp_reviews(yd.TEST_FILE)

    # map labels to start from index 0
    train_y = [l-1 for l in train_y]
    test_y = [l-1 for l in test_y]

    # tokenize texts
    train_x = [tokenize_text(t) for t in train_x]
    test_x = [tokenize_text(t) for t in test_x]

    # get number of labels
    num_labels = 2 if args.binary else 5
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

    train_data_generator = get_text_data_generator(train_x, train_y,
            args.bs, embedder, verbose=1)
    eval_data_generator = get_text_data_generator(test_x, test_y,
            args.bs, embedder, verbose=1)

    print("Begin training...")
    train(model, optimizer, criterion, train_data_generator,
            scheduler=scheduler, num_epochs=args.epoch,
            eval_data_generator=eval_data_generator)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
