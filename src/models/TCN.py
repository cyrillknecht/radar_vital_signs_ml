"""
TCN model from https://github.com/locuslab/TCN
 adapted original implementation from paper
  [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling]
(https://arxiv.org/abs/1803.01271)
"""
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    """
    Class for removing the last chomp_size number of elements from the input.
    Inherits from nn.Module.
    """

    def __init__(self, chomp_size):
        """
        Initialize the Chomp1d layer.
        Args:
            chomp_size(int): number of elements to remove from the input
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Removes the last chomp_size number of elements from the input.

        Args:
            x(torch.Tensor): Torch tensor of shape (batch_size, channels, time_steps)

        Returns:
            torch.Tensor: x with the last chomp_size elements removed

        """

        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    A single temporal block of the TCN.

    Consists ofv two dilated causal convolutions with a residual connection.
    Convolutional layers are followed by Chomp1d Layer, a ReLU activation and dropout.

    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        Initialize the TemporalBlock.

        Args:
            n_inputs(int): number of input channels of a single time step
            n_outputs(int):  number of output channels of a single time step
            kernel_size(int):  size of the convolutional kernel
            stride(int): stride of convolution
            dilation(int): dilation of convolution
            padding(int): padding of convolution
            dropout(float): percentage of dropout nodes [0, 1]

        """

        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels=n_inputs,
                                           out_channels=n_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation))
        self.chomp1 = Chomp1d(chomp_size=padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)

        self.conv2 = weight_norm(nn.Conv1d(in_channels=n_outputs,
                                           out_channels=n_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation))
        self.chomp2 = Chomp1d(chomp_size=padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)

        self.net = nn.Sequential(self.conv1,
                                 self.chomp1,
                                 self.relu1,
                                 self.dropout1,
                                 self.conv2,
                                 self.chomp2,
                                 self.relu2
                                 , self.dropout2)
        self.downsample = nn.Conv1d(in_channels=n_inputs,
                                    out_channels=n_outputs,
                                    kernel_size=1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the components of the TemporalBlock.
        """

        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Forward pass of the TemporalBlock.
        Args:
            x(torch.Tensor): Torch tensor of shape (batch_size, channels, time_steps)

        Returns:
            torch.Tensor: Torch tensor of shape (batch_size, channels, time_steps)

        """

        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network.
    Consists of an adaptable stack of TemporalBlocks.
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, no_dilation_layers=3):
        """
        Initialize the TemporalConvNet.

        Args:
            num_inputs(int): number of input channels of a single time step
            num_channels(list): list of number of output channels of each TemporalBlock
            kernel_size(int): size of the convolutional kernel
            dropout(float): percentage of dropout nodes [0, 1]
            no_dilation_layers(int): number of layers with dilation size 1, i.e. no dilation
        """

        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            # Manually shrink receptive field but keep parameters the same
            if i < no_dilation_layers:
                dilation_size = 1
            else:
                dilation_size = 2 ** (i-no_dilation_layers)

            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(n_inputs=in_channels,
                                     n_outputs=out_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the TemporalConvNet.
        Args:
            x(torch.Tensor): Torch tensor of shape (batch_size, channels, time_steps)

        Returns:
            torch.Tensor: Torch tensor of shape (batch_size, channels, time_steps)

        """
        return self.network(x)


class TCN(nn.Module):
    """
    Full TCN model.
    Consists of a TemporalConvNet followed by a linear layer.
    """
    def __init__(self, channel_sizes, input_size=1, output_size=1, kernel_size=3, dropout=0.2, no_dilation_layers=3):
        """
        Initialize the TCN.
        Args:
            channel_sizes(list): list of number of output channels of each TemporalBlock
            input_size(int): number of input channels of a single time step
            output_size(int): number of output channels of a single time step
            kernel_size(int): size of the convolutional kernel
            dropout(float): percentage of dropout nodes [0, 1]
            no_dilation_layers(int): number of layers with dilation size 1, i.e. no dilation

        """
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=input_size,
                                   num_channels=channel_sizes,
                                   kernel_size=kernel_size,
                                   dropout=dropout,
                                   no_dilation_layers=no_dilation_layers)

        self.linear = nn.Linear(in_features=channel_sizes[-1],
                                out_features=output_size)

        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights in the linear layer of the TCN.
        """
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, sigmoid=False):
        """
        Forward pass of the TCN.
        Args:
            x(torch.Tensor): Torch tensor of shape (batch_size, channels, time_steps)
            sigmoid(bool): whether to apply sigmoid activation to the output

        Returns:
            torch.Tensor: Torch tensor of shape (batch_size, channels, time_steps)

        """
        x = self.tcn(x)
        x = x.transpose(1, 2)
        x = self.linear(x)
        x = x.transpose(1, 2)

        if sigmoid:
            x = self.sigmoid(x)

        return x
