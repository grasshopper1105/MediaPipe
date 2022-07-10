import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size=256, num_layers=2,
                 num_classes=500, hidden_s1=128, drop_p=0.0, batch_size=32):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.hidden1 = hidden_s1
        self.drop_p = drop_p
        self.batch_size = batch_size

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True, bidirectional=True)
        self.drop = nn.Dropout2d(p=self.drop_p)
        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden1)  # bidirectional=True时×2
        self.fc2 = nn.Linear(self.hidden1, self.num_classes)

    def forward(self, input, hidden=None):
        self.rnn.flatten_parameters()

        out, hidden = self.rnn(input, hidden)
        output = F.selu(self.fc1(out[:, -1, :]))
        output = F.dropout(output, p=self.drop_p)
        output = F.relu(self.fc2(output))
        return output, hidden

    def initHidden(self):
        return torch.randn(self.num_layers, self.batch_size, self.hidden_size)
