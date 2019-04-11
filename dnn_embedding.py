from __future__ import absolute_import, print_function, division


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk import word_tokenize
import yelp_data as yd


UNK = "<UNK>"
BATCH_SIZE = 32
MAX_DOCLEN = 100


class ConvNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, out_features):
        super(ConvNet, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
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

    def forward(self, x):
        # embed
        x = self.embeddings(x)
        # append color channel
        x = x.view(-1, 1, x.size(1), x.size(2))
        # extract features
        x = self.features(x)
        # flatten and classify
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class TextDataset(Dataset):
    def __init__(self, texts, labels, maxlen=100):
        texts = [word_tokenize(txt) for txt in texts]
        self.data = zip(texts, labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def get_idx_map(data):
    next_id = 0
    word_to_idx = {}
    for txt in data:
        words = word_tokenize(txt)
        for w in words:
            if w not in word_to_idx:
                word_to_idx[w] = next_id
                next_id += 1
    word_to_idx[UNK] = next_id
    return word_to_idx


def convert_to_indices(txt, word_to_idx, maxlen=100):
    indices = []
    for w in word_tokenize(txt)[:maxlen]:
        idx = word_to_idx[w] if w in word_to_idx else word_to_idx[UNK]
        indices.append(idx)
    return indices


def run():
    print("Load data...")
    # load yelp data
    train_x, train_y = yd.load_yelp_reviews(yd.TRAIN_FILE)
    test_x, test_y = yd.load_yelp_reviews(yd.TEST_FILE)
    # construct dataset
    trainset = TextDataset(train_x, train_y, maxlen=MAX_DOCLEN)
    testset = TextDataset(test_x, test_y, maxlen=MAX_DOCLEN)
    # init loaders
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    print("Training...")
    for _, train_batch in enumerate(trainloader):
        print(train_batch[0][0])
        break

    return None


if __name__ == '__main__':
    run()
