from __future__ import absolute_import, print_function, division

import io
import json


DATA_FILE = './data/yelp/review.json'


def load_yelp_reviews(line_limit=50000):
    """
    Load yelp data into texts and corresponding labels, where each text is
    in lowercase.
    """
    fin = io.open(DATA_FILE, 'r', encoding='utf-8')
    texts, labels = [], []
    line_no = 0
    for line in fin:
        if line_no > line_limit:
            break
        line_no += 1
        review = json.loads(line)
        texts.append(review['text'].strip().lower())
        labels.append(int(review['stars']))
    return texts, labels
