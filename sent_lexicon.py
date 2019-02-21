from __future__ import absolute_import, print_function, division

import os

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sklearn.metrics as metrics

from yelp_data import load_yelp_reviews
from utils.statistics import map_range


# TODO: currently, use the pretrained adjective sentiment lexicons
# we should train a review-specific one
SENT_LEXICON_PATH = './data/adjectives/'
lemmatizer = WordNetLemmatizer()


def read_sentiment_lexicons(file_name):
    ret = {}
    with open(file_name, 'r') as f:
        for line in f:
            token, mean, std = line.split()
            ret[token] = (float(mean), float(std))
    return ret


def replace_adverbs_by_adjectives(text):
    ret = []
    for w, t in text:
        if t == "RB":
            w = lemmatizer.lemmatize(w, pos=nltk.corpus.wordnet.ADV)
        ret.append((w, t))
    return ret


def sentiment_score(text, sent_lexicons):
    total = 0.0
    num_sent_words = 0

    for w, _ in text:
        if w in sent_lexicons:
            total += sent_lexicons[w][0]
            num_sent_words += 1

    if num_sent_words == 0:
        return 0.0
    return total / num_sent_words


def run():
    years = range(1850, 2000 + 1, 10)
    sent_lexicons = {}
    for y in years:
        y_path = os.path.join(SENT_LEXICON_PATH, "{}.tsv".format(y))
        sent_lexicons.update(read_sentiment_lexicons(y_path))

    texts, labels = load_yelp_reviews(line_limit=500)
    texts = [replace_adverbs_by_adjectives(nltk.pos_tag(word_tokenize(t))) 
                for t in texts]

    sent_scores = [sentiment_score(t, sent_lexicons) for t in texts]
    predicted_labels = map_range(sent_scores, min(labels), max(labels))
    print(metrics.accuracy_score(labels, predicted_labels))
    print(metrics.confusion_matrix(labels, predicted_labels, labels=range(1, 5+1)))


if __name__ == '__main__':
    run()
