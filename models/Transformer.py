import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout=0.2):
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(input_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.transformer(x)
        x = x[:, :, -1]  # Only take the output from the last time step
        x = self.fc(x)
        return x
