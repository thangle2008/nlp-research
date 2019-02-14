from __future__ import absolute_import, print_function, division


def get_accuracy(predicted_labels, target_labels):
    assert(len(predicted_labels) == len(target_labels))
    num_corrects = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == target_labels[i]:
            num_corrects += 1
    return num_corrects / len(predicted_labels)
