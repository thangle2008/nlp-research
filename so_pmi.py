from __future__ import print_function, division, absolute_import

import io
import argparse

import nltk

from yelp_review_nlp import load_yelp_data


def valid_phrase(t1, t2, t3):
    return (t1 == 'JJ' and (t2 == 'NN' or t2 == 'NNS')) or \
           ((t1 == 'RB' or t1 == 'RBR' or t1 == 'RBS') and t2 == 'JJ' and
                   (t3 != 'NN' and t3 != 'NNS')) or \
           (t1 == 'JJ' and t2 == 'JJ' and (t3 != 'NN' and t3 != 'NNS')) or \
           ((t1 == 'NN' or t1 == 'NNS') and t2 == 'JJ' and 
                   (t3 != 'NN' and t3 != 'NNS')) or \
           ((t1 in ['RB', 'RBR', 'RBS']) and (t2 in ['VB', 'VBD', 'VBN', 'VBG']))


def extract_phrases(words):
    res = []
    tagged_words = nltk.pos_tag(words)
    for i in range(0, len(tagged_words) - 2, 3):
        w1, t1 = tagged_words[i]
        w2, t2 = tagged_words[i+1]
        w3, t3 = tagged_words[i+2]
        if valid_phrase(t1, t2, t3):
            res.append((w1, w2))
    return res


def get_pmis(words=None):
    if words is None:
        # default to Brown corpus
        words = nltk.corpus.brown.words(categories=['news', 'reviews'])


def run():
    dataset = load_yelp_data()
    texts, labels = zip(*dataset)
    labels = [l - 1 for l in labels] # map to index 0
