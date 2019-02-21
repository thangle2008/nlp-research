from __future__ import absolute_import, print_function, division

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import sklearn.metrics as metrics

from utils.statistics import map_range
from yelp_data import load_yelp_reviews


sid = SentimentIntensityAnalyzer()
texts, labels = load_yelp_reviews(line_limit=50000)

predicted_scores = [sid.polarity_scores(t)['compound'] for t in texts]
predicted_scores = map_range(predicted_scores, min(labels), max(labels))
predicted_labels = [int(round(s)) for s in predicted_scores]

print(metrics.accuracy_score(labels, predicted_labels))
print(metrics.confusion_matrix(labels, predicted_labels, labels=[1, 2, 3, 4, 5]))
print(metrics.classification_report(labels, predicted_labels))
