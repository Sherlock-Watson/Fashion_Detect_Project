import os

import numpy as np
from torch.utils.data import Dataset
from PIL import Image


NUM_ATTR = 6
num_labels_per_group = [7, 3, 3, 4, 6, 3]


# data loader
class FashionNet_Dataset(Dataset):

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = [[] for _ in range(NUM_ATTR)]
        self.transform = transform

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                # make dummy label for test set
                if 'test' in txt:
                    for i in range(NUM_ATTR):
                        self.labels[i].append(0)
        if 'test' not in txt:
            # train or val set
            with open(txt.replace('.txt', '_attr.txt')) as f:
                for line in f:
                    attrs = line.split()
                    for i in range(NUM_ATTR):
                        self.labels[i].append(int(attrs[i]))

    def __len__(self):
        return len(self.labels[0])

    def __getitem__(self, index):

        path = self.img_path[index]
        label = np.array([self.labels[i][index] for i in range(NUM_ATTR)])
        # print(path)
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform:
            sample = self.transform(sample)

        return sample, label


def encode_label(label):
    new_label = np.zeros(26)
    acc_labels = []
    for i in range(len(label)):
        if i == 0:
            num = 0
        else:
            num = num_labels_per_group[i - 1] + acc_labels[i - 1]
        acc_labels.append(num)
    # print(acc_labels)
    for (index, inner_label) in enumerate(label):
        new_label[acc_labels[index] + inner_label] = 1
    return new_label
