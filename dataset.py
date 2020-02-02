import torch
from torch.utils import data

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, list_len, agen_list):
        'Initialization'
        self.list_IDs = list_IDs
        self.list_len = list_len
        self.agen_list = agen_list

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        X = torch.LongTensor(ID)
        xlen = self.list_len[index]
        agen = self.agen_list[index]
        return X, xlen, agen