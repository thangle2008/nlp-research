from __future__ import absolute_import, print_function, division

import nltk
import random
from sklearn.model_selection import train_test_split

from yelp_data import load_yelp_reviews
from utils.textdata import tokenize_text


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


def run():
    random.seed(42)

    print("Load data...")
    texts, labels = load_yelp_reviews(line_limit=50000)
    X = [get_ngram_counts(1, tokenize_text(t)) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(X, labels, 
            test_size=0.2, shuffle=True, stratify=labels, random_state=42)

    train_set = zip(X_train, y_train)
    test_set = zip(X_test, y_test)

    print("Begin training...")
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    print("Evaluating...")
    print("Train accuracy =", nltk.classify.accuracy(classifier, train_set))
    print("Test accuracy =", nltk.classify.accuracy(classifier, test_set))

if __name__ == '__main__':
    run()
