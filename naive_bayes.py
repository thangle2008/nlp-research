from __future__ import absolute_import, print_function, division

import random

import yelp_data as yd
from utils.textdata import tokenize_text
from utils.statistics import map_range

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as metrics


def get_ngram_counts(n, text):
    """
    Return a dictionary counts of all ngrams in the list of sentences.
    """
    ngram_dict = {}
    for s in text:
        for i in range(len(s)):
            if i + n > len(s): break
            ngram = tuple(s[i:i+n])
            ngram_dict[ngram] = ngram_dict.get(ngram, 0) + 1
    return ngram_dict


def evaluate(classifier, data_x, data_y, report=False):
    y_true = data_y
    y_pred = classifier.predict(data_x)
    print("Accuracy =", metrics.accuracy_score(y_true, y_pred))
    if report:
        print(metrics.classification_report(y_true, y_pred))


def run():
    random.seed(42)
    sid = SentimentIntensityAnalyzer()

    print("Load data...")
    train_x, train_y = zip(
        *[(o['text'].strip().lower(), o['stars']) for o in yd.load_yelp_objects(yd.TRAIN_FILE)]
    )
    test_x, test_y = zip(
        *[(o['text'].strip().lower(), o['stars']) for o in yd.load_yelp_objects(yd.TEST_FILE)]
    )
    all_x = train_x + test_x

    # Extract features as dictionaries (it seems bigram works best)
    print("Extract features...")
    train_feat = [get_ngram_counts(2, tokenize_text(t)) for t in train_x]
    test_feat = [get_ngram_counts(2, tokenize_text(t)) for t in test_x]

    # Vectorize
    vec = DictVectorizer()
    vec.fit(train_feat)
    train_feat = vec.transform(train_feat)
    test_feat = vec.transform(test_feat)

    print("Begin training...")
    classifier = MultinomialNB().fit(train_feat, train_y)

    # Evaluating
    print("Evaluating trainset")
    evaluate(classifier, train_feat, train_y)
    print("Evaluating testset")
    evaluate(classifier, test_feat, test_y)


if __name__ == '__main__':
    run()
