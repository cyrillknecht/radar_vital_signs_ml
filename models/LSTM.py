import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_features=1, output_features=1, hidden_size=512, num_layers=1, dropout=0.2):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_features, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_features)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2], 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
        x = self.softmax(x)

        return x
