import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) for regression.
    """

    def __init__(self, input_size,
                 output_size,
                 hidden_sizes=None,
                 dropout_rate=0.2):

        super(MLP, self).__init__()

        # Default hidden sizes
        if hidden_sizes is None:
            hidden_sizes = [4096, 4096, 4096, 4096]

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

        # Dropout regularization
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # Input layer
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)

        # Hidden layers
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
            x = self.dropout(x)

        # Output layer
        x = self.output_layer(x)

        return x
