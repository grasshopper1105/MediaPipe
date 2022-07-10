import pandas as pd
import numpy as np
import torch
from numpy import float32
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
# from MediaPipe.RNN import RnnModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('D:\\Software\\PyCharm\\AI\\MediaPipe\\data\\000\\P01_01_00_0.csv', header=None, sep=',').iloc[1:, 1:]
df1 = pd.read_csv('D:\\Software\\PyCharm\\AI\\MediaPipe\\data\\000\\P01_01_00_3.csv', header=None, sep=',').iloc[1:, 1:]
df2 = pd.read_csv('D:\\Software\\PyCharm\\AI\\MediaPipe\\data\\000\\P01_01_00_4.csv', header=None, sep=',').iloc[1:, 1:]
# data = np.array(df).astype(float32)
# data1 = np.array(df1).astype(float32)
# data2 = np.array(df2).astype(float32)
data = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]).T.astype(float32)
data1 = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]).T.astype(float32)
data2 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]).T.astype(float32)
data3 = np.array([[1, 2], [1, 2], [1, 2]]).T.astype(float32)

data = torch.from_numpy(data)
data1 = torch.from_numpy(data1)
data2 = torch.from_numpy(data2)
data3 = torch.from_numpy(data3)

train_data = [data, data1, data2, data3]

# a = torch.tensor([1, 2, 3, 4])
# b = torch.tensor([5, 6, 7])
# c = torch.tensor([7, 8])
# d = torch.tensor([9])
# train_data = [a, b, c, d]


# def collate_fn(train_data):
#     train_data.sort(key=lambda data: len(data), reverse=True)
#     data_length = [len(data) for data in train_data]
#     train_data = pad_sequence(train_data, batch_first=True, padding_value=0)
#     return train_data, data_length


def collate_fn(data):
    data.sort(key=lambda x: len(x), reverse=True)
    seq_len = [s.size(0) for s in data]
    data = pad_sequence(data, batch_first=True).float()  # [batch, time_step, feature]
    # data = data.unsqueeze(-1)
    data = pack_padded_sequence(data, seq_len, batch_first=True)
    return data


input_size = 3  # 特征数
# seq_len = 100  # padding后的时间长度
num_class = 2  # 分类数
drop_p = 0.1
batch_size = 2

train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)
batch_x = iter(train_dataloader).next()
# print(batch_x)


class RnnModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, num_classes=500,
                 drop_p=0.0, batch_size=32, batch_first=True):
        super(RnnModel, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # 定义LSTM
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=batch_first)
        # 定义回归层网络，输入的特征维度等于LSTM的输出，输出维度为num_classes
        self.reg = nn.Sequential(
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x, (ht, ct) = self.rnn(x)

        x, out_len = pad_packed_sequence(x, batch_first=True)
        seq_len = x.shape[1]
        print(x.shape)
        x = x.reshape(-1, self.hidden_size)
        print(x.shape)
        x = self.reg(x)
        x = x.reshape(seq_len, batch_size, -1)
        return x


rnn = RnnModel(input_size=input_size, hidden_size=4, num_layers=1, num_classes=num_class,
               batch_size=batch_size, batch_first=True)
out = rnn(batch_x)
print(out.shape)

# rnn = nn.LSTM(input_size=input_size, hidden_size=4, num_layers=1, batch_first=True).to(device)
# h0 = torch.rand(1, 2, 4).float()
# c0 = torch.rand(1, 2, 4).float()
# out, (h1, c1) = rnn(batch_x)

