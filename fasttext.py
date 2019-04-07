from __future__ import absolute_import, print_function, division

import io
import argparse
import yelp_data as yd

import sklearn.metrics as metrics


########### Arguments ###############
parser = argparse.ArgumentParser()
parser.add_argument("usage", type=str)
# For evaluating
parser.add_argument("--prediction_file", action="store", required=False,
    help='path to a fasttext file that contains predictions')
parser.add_argument("--test_file", action="store", required=False,
    help='path to a fasttext file that contains true labels')

########### Constants ###############
TRAIN_FILE = './data/yelp/fasttext/fasttext.train.txt'
TEST_FILE = './data/yelp/fasttext/fasttext.test.txt'
LABEL_PREFIX = '__label__'


def write_fasttext_format(input, output):
    """
    Write yelp data to a fasttext-formatted file.
    """
    texts, labels = yd.load_yelp_reviews(input)
    with io.open(output, 'w', encoding='utf-8') as f:
        for txt, labl in zip(texts, labels):
            # replace new line by space
            txt = txt.replace(u'\n', u' ')
            # label
            f.write(unicode(LABEL_PREFIX) + u'{}'.format(labl))
            # a space
            f.write(u' ')
            # text
            f.write(txt)
            # newline
            f.write(u'\n')


def read_labels(filename):
    """
    Read labels from a fasttext-formatted file.
    """
    labels = []
    with io.open(filename, 'r') as f:
        for line in f:
            labl = str(line.split()[0])
            labels.append(labl[len(LABEL_PREFIX):])
    return labels


if __name__ == '__main__':
    args = parser.parse_args()
    if args.usage == 'write':
        write_fasttext_format(yd.TRAIN_FILE, TRAIN_FILE)
        write_fasttext_format(yd.TEST_FILE, TEST_FILE)
    elif args.usage == 'evaluate':
        assert args.prediction_file and args.test_file
        true_labels = read_labels(args.test_file)
        predicted_labels = read_labels(args.prediction_file)
        print(metrics.classification_report(true_labels, predicted_labels))
