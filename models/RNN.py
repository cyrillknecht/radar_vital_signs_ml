import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout=0.2):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
