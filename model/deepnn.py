from __future__ import print_function, division, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor


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


def get_num_corrects(y_pred, y_target):
    _, y_pred = torch.max(y_pred.data, 1)
    return (y_pred == y_target).sum().item()


def convert_to_tensor(X, y, append_color_channel=False):
    # convert to tensor
    X = FloatTensor(X)
    y = LongTensor(y)
    if append_color_channel:
        # since pytorch requires the color channel, we need to append 1
        # nested level 
        X = X.view(-1, 1, X.shape[1], X.shape[2])
    return X, y


def train(model, optimizer, loss_fn, train_data_generator,
          eval_data_generator=None, num_epochs=10, scheduler=None):

    # training loop
    best_val_loss = float("inf")
    for e in range(num_epochs):
        print("Epoch {}/{}".format(e+1, num_epochs))

        # Training
        total = 0
        num_corrects = 0
        losses = []
        for X_batch, y_batch in train_data_generator():
            X_batch, y_batch = convert_to_tensor(X_batch, y_batch, True)
            # feed forward
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # back propagation
            loss.backward()
            optimizer.step()
            # record statistics
            total += len(y_batch)
            num_corrects += get_num_corrects(y_pred, y_batch)
            losses.append(loss.item())

        train_loss = sum(losses) / len(losses)
        train_acc = num_corrects / total

        print("train_loss = {:.3f}, train_acc = {:.3f}".format(train_loss, train_acc))

        # Evaluating
        if eval_data_generator:
            with torch.no_grad():
                total = 0
                num_corrects = 0
                losses = []

                for X_batch, y_batch in eval_data_generator():
                    X_batch, y_batch = convert_to_tensor(X_batch, y_batch, True)
                    # feed forward
                    y_pred = model(X_batch)
                    loss = loss_fn(y_pred, y_batch)
                    # record statistics
                    total += len(y_batch)
                    num_corrects += get_num_corrects(y_pred, y_batch)
                    losses.append(loss.item())

                test_loss = sum(losses) / len(losses)
                test_acc = num_corrects / total

                # decrease learning rate if stuck
                scheduler.step(test_loss)

                print("test_loss = {:.3f}, test_acc = {:.3f}".format(
                    test_loss, test_acc))
