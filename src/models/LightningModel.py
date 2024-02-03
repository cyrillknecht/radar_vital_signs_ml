"""
Implementation of  a wrapper class for the models to use with PyTorch Lightning.
"""

import torch
import lightning as pl

from src.models.TCN import TCN
from src.models.LSTM import LSTM
from src.models.GRU import GRU
from src.models.RNN import RNN
from src.models.Losses import get_loss


class LitModel(pl.LightningModule):
    """
    Wrapper class for the models to use with PyTorch Lightning.
    """

    @property
    def device(self):
        """
        Get the device the model is on.
        """

        return self._device

    def __init__(self, model, learning_rate=1e-3, loss_component_weights=None):
        """
        Initiate the Lightning model.
        Args:
            model(torch.nn.Module): model to use
            learning_rate(float): learning rate for the optimizer
            loss_component_weights(dict): weights for the different components of the loss function

        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.loss_component_weights = loss_component_weights

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x(torch.Tensor): input tensor of the model

        Returns:
            torch.Tensor: output tensor of the model

        """

        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        Args:
            batch(torch.Tensor): batch of data of shape (batch_size, time_steps, features)
            batch_idx(int): index of the batch

        Returns:
            torch.Tensor: loss of this step

        """
        loss = self.step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        Args:
            batch(torch.Tensor): batch of data of shape (batch_size, time_steps, features)
            batch_idx(int): index of the batch

        Returns:
            torch.Tensor: loss of this step

        """
        loss = self.step(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.
        Args:
            batch(torch.Tensor): batch of data of shape (batch_size, time_steps, features)
            batch_idx(int): index of the batch

        Returns:
            torch.Tensor: loss of this step

        """

        loss = self.step(batch)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.

        Returns:
            torch.optim.Optimizer: optimizer for the model

        """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def step(self, batch):
        """
        Step for the model.
        Args:
            batch(torch.Tensor): batch of data of shape (batch_size, time_steps, features)

        Returns:
            torch.Tensor: loss of this step, either MSE or CrossEntropy depending on output shape of the model

        """
        radar_signal, ecg_signal = batch
        output = self.model(radar_signal)

        loss = get_loss(y_true=ecg_signal,
                        y_pred=output,
                        device=self.device,
                        component_weights=self.loss_component_weights)

        return loss

    @device.setter
    def device(self, value):
        """
        Set the device the model is on.
        Args:
            value(torch.device): device to set the model on

        """
        self._device = value


def get_model(model_type, input_size, output_size, hidden_size=256, num_layers=2, kernel_size=3, no_dilation_layers=3):
    """
    Get the model for the given type and configuration.

    Args:
        model_type(str): type of model to use [TCN]
        input_size(int): number of features in the input signal in a single time step
        output_size(int): number of features in the output signal in a single time step
        hidden_size(int): size of the hidden layer in the model(# of channels per layer for TCN)
        num_layers(int): number of layers in the model (# Temporal Blocks for TCN)
        kernel_size(int): kernel size for the TCN model

    Returns:
        torch.nn.Module: model to use

    """
    if model_type == "TCN":
        return TCN(channel_sizes=[hidden_size] * num_layers,
                   input_size=input_size,
                   output_size=output_size,
                   kernel_size=kernel_size,
                   dropout=0.2,
                   no_dilation_layers=no_dilation_layers
                   )

    elif model_type == "LSTM":
        return LSTM(input_features=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    output_features=output_size)

    elif model_type == "GRU":
        return GRU(input_features=input_size,
                   hidden_size=hidden_size,
                   num_layers=num_layers,
                   output_features=output_size)

    elif model_type == "RNN":
        return RNN(input_features=input_size,
                   hidden_size=hidden_size,
                   num_layers=num_layers,
                   output_features=output_size)

    else:
        print("Model type not recognized or not yet implemented.")
        exit(1)

