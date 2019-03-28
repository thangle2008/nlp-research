from __future__ import absolute_import, print_function, division

import io
import json

from sklearn.model_selection import train_test_split


DATA_FILE = './data/yelp/review.json'
TRAIN_FILE = './data/yelp/review_train.json'
TEST_FILE = './data/yelp/review_test.json'


def load_yelp_objects(filename, line_limit=50000):
    fin = io.open(filename, 'r', encoding='utf-8')
    objs = []
    line_no = 0
    for line in fin:
        if line_no > line_limit:
            break
        line_no += 1
        obj = json.loads(line)
        objs.append(obj)
    return objs


def write_json_objects(filename, objs):
    with io.open(filename, 'w', encoding='utf-8') as fin:
        for o in objs:
            data = json.dumps(o, ensure_ascii=False, encoding='utf-8')
            fin.write(data + u'\n')


if __name__ == '__main__':
   # split yelp data
   objs = load_yelp_objects(DATA_FILE)
   labels = [int(o['stars']) for o in objs]
   x_train, x_test, _, _ = train_test_split(objs, labels,
           test_size=0.3, shuffle=True, stratify=labels, random_state=42)
   # train file
   write_json_objects(TRAIN_FILE, x_train)
   write_json_objects(TEST_FILE, x_test)
