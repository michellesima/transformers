import torch

a = torch.FloatTensor([1,2,3])
b = torch.FloatTensor(a.data)
print(b)
