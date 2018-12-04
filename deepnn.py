from __future__ import print_function, division, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepNet(nn.Module):
    """
    Deep convolutional neural network for text classification.
    """

    def __init__(self, out_features, input_size=100):
        super(DeepNet, self).__init__()
        assert input_size == 100 or input_size == 300
        if input_size == 100:
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(64 * 10 * 10, 1028),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(1028, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, out_features),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(128 * 3 * 15, 1028),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(1028, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, out_features),
            )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
