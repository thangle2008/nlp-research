from __future__ import absolute_import, print_function, division

import io
import json

from sklearn.model_selection import train_test_split


DATA_FILE = './data/yelp/review.json'
TRAIN_FILE = './data/yelp/review_train.json'
TEST_FILE = './data/yelp/review_test.json'


def load_yelp_reviews(filename, line_limit=50000):
    """
    Load yelp data into texts and corresponding labels, where each text is
    in lowercase.
    """
    objs, labels = load_yelp_objects(filename, line_limit=line_limit)
    texts = [o['text'].strip().lower() for o in objs]
    return texts, labels


def load_yelp_objects(filename, line_limit=50000):
    fin = io.open(filename, 'r', encoding='utf-8')
    objs, labels = [], []
    line_no = 0
    for line in fin:
        if line_no > line_limit:
            break
        line_no += 1
        obj = json.loads(line)
        objs.append(obj)
        labels.append(int(obj['stars']))
    return objs, labels


def write_json_objects(filename, objs):
    with io.open(filename, 'w', encoding='utf-8') as fin:
        for o in objs:
            data = json.dumps(o, ensure_ascii=False, encoding='utf-8')
            fin.write(data + u'\n')


if __name__ == '__main__':
   # split yelp data
   objs, labels = load_yelp_objects(DATA_FILE)
   x_train, x_test, y_train, y_test = train_test_split(objs, labels,
           test_size=0.3, shuffle=True, stratify=labels, random_state=42)
   # train file
   write_json_objects(TRAIN_FILE, x_train)
   write_json_objects(TEST_FILE, x_test)
