import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_size, output_size, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(output_size, output_size, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(input_size, output_size, padding=padding, kernel_size=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.conv2(y)
        y = self.relu(y)
        y = self.dropout(y)

        res = x if self.downsample is None else self.downsample(x)

        return y + res


class TemporalConvNet(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, output_size, tcn_output_channels, kernel_size=2, input_channels=1, dropout=0.2):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_channels, tcn_output_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(tcn_output_channels[-1], tcn_output_channels[-1] * 2),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(tcn_output_channels[-1] * 2, tcn_output_channels[-1] * 4),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(tcn_output_channels[-1] * 4, output_size))

    def forward(self, x):
        x = self.tcn(x)
        x = x[:, :, -1]
        x = self.fc(x)
        x = x.unsqueeze(1)
        return x
