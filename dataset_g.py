import torch
from torch.utils import data

class Dataset_g(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, es, labels):
        'Initialization'
        self.list_IDs = list_IDs
        self.es = es
        self.labels = labels

    def len(self):
        return len(self.list_IDs)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def append(self, other):
        self.list_IDs.extend(other.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        X = torch.LongTensor(ID)
        ecat = self.es[index]
        e = torch.FloatTensor(ecat)
        label = self.labels[index]
        y = torch.LongTensor(label)
        return X, e, y
