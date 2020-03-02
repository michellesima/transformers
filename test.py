import torch

softmax = torch.nn.Softmax(dim=1)
li = [
    [1, 4, 5],
    [2, 4, 4]
]

tensor = torch.FloatTensor(li)
tensor = softmax(tensor)
print(tensor)
