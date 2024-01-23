import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_features=1, output_features=1, hidden_size=512, num_layers=1, dropout=0.2):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_features, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(2, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = x.transpose(2, 1)
        x = self.sigmoid(x)

        return x

