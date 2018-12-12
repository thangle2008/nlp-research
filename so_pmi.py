from __future__ import print_function, division, absolute_import

import io
import argparse
import math

import nltk
from nltk import word_tokenize

from yelp_review_nlp import load_yelp_data


# functions for extracting phrases
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


def tokenize_corpus(texts):
    res = []
    for t in texts:
        res.extend(word_tokenize(t))
    return res


# function for calculating pmi
# TODO: implement a NEAR-op like search engine, impossible to calculate
# without a search algorithm
def pmi(w1, w2, uni_fd, bi_fd):
    # TODO: for now, if either unigram or bigram does not exist,
    # return 0
    if (w1 in uni_fd) and (w2 in uni_fd) and ((w1, w2) in bi_fd):
        num = bi_fd[(w1, w2)] / bi_fd.N()
        denom = (uni_fd[w1] / uni_fd.N()) * (uni_fd[w2] / uni_fd.N())
        return math.log(num / denom, 2) 
    else:
        return 0.0


# main function
def run():
    dataset = load_yelp_data()
    texts, labels = zip(*dataset)
    labels = [1 if l >= 2 else 0 for l in labels]

    # get unigram and bigram frequencies
    print("Get frequencies...")
    corpus_words = tokenize_corpus(texts)
    # TODO: does not work. Need unigram to be phrase (w1, w2) 
    # and bigram to be (phrase, w3). So, basically we need bigram and trigram.
    uni_fd = nltk.FreqDist(corpus_words)
    bi_fd = nltk.FreqDist(nltk.bigrams(corpus_words))

    # classify based on semantic orientation
    print("Classify...")
    num_corrects = 0
    for t, l in zip(texts, labels):
        words = word_tokenize(t)
        phrases = extract_phrases(words)
        so = 0.0
        for p in phrases:
            pmi_pos = pmi(p, 'excellent', uni_fd, bi_fd)
            pmi_neg = pmi(p, 'poor', uni_fd, bi_fd)
            so = pmi_pos - pmi_neg
        predict = 0 if so <= 0.0 else 1
        num_corrects += predict == l
    
    print("accuracy = {:.2f}".format((num_corrects / len(texts)) * 100))

if __name__ == '__main__':
    run()
