from __future__ import absolute_import, print_function, division

import os

from yelp_data import load_yelp_reviews
import utils.metrics as metrics


SENT_LEXICON_PATH = './data/adjectives/'


def read_sentiment_lexicons(file_name):
    ret = {}
    with open(file_name, 'r') as f:
        for line in f:
            token, mean, std = line.split()
            ret[token] = (float(mean), float(std))
    return ret


def sentiment_score(words, sent_lexicons):
    ret = sum([sent_lexicons[w].get(w, 0.0) for w in words])
    return ret


def map_range(values, target_min, target_max):
    cur_min = min(values)
    cur_max = max(values)

    def map_value(v):
        # map to (0, 1)
        tmp = (v - cur_min) / (cur_max - cur_min)
        # map to target range
        return target_min + tmp * (target_max - target_min)

    return [map_value(v) for v in values]


def run():
    years = range(1850, 2000 + 1, 10)
    sent_lexicons = {}
    for y in years:
        y_path = os.path.join(SENT_LEXICON_PATH, "{}.tsv".format(y))
        sent_lexicons.update(read_sentiment_lexicons(y_path))

    texts, labels = load_yelp_reviews()

    sent_scores = [sentiment_score(t) for t in texts]
    predicted_labels = map_range(sent_scores, min(labels), max(labels))
    print(metrics.get_accuracy(predicted_labels, labels))


if __name__ == '__main__':
    run()
