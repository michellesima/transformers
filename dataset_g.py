import torch
from torch.utils import data
import numpy as np

class Dataset_g(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, es):
        'Initialization'
        self.list_IDs = list_IDs
        self.es = es

    def len(self):
        return len(self.list_IDs)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def append(self, other):
        self.list_IDs.extend(other.list_IDs)
        self.es = np.append(self.es, other.es)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        X = torch.LongTensor(ID)
        ecat = self.es[index]
        e = torch.FloatTensor(ecat)
        return X, e
