import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


class MyData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(data):
    data.sort(key=lambda x: len(x), reverse=True)
    seq_len = [s.size(0) for s in data]
    data = pad_sequence(data, batch_first=True).float()
    print(data)
    data = data.unsqueeze(-1)
    data = pack_padded_sequence(data, seq_len, batch_first=True)
    return data


a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([5, 6, 7])
c = torch.tensor([7, 8])
d = torch.tensor([9])
train_x = [a, b, c, d]

data = MyData(train_x)
data_loader = DataLoader(data, batch_size=2, shuffle=False, collate_fn=collate_fn)
batch_x = iter(data_loader).next()

hid = 2
rnn = nn.LSTM(1, hid, 1, batch_first=True)
h0 = torch.rand(1, 2, hid).float()
c0 = torch.rand(1, 2, hid).float()
out, (h1, c1) = rnn(batch_x, (h0, c0))

out_pad, out_len = pad_packed_sequence(out, batch_first=True)

for i in out_pad:
    print(i.detach().numpy())
