import torch
import torch.nn as nn
import lightning as pl
from models.MLP import MLP
from models.TCN import TCN
from models.LSTM import LSTM
from models.GRU import GRU
from models.RNN import RNN
from models.Transformer import Transformer


class LitModel(pl.LightningModule):
    """
    Wrapper class for the models to use with PyTorch Lightning.
    """

    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        radar_signal, ecg_signal = batch
        output = self.model(radar_signal)
        loss = nn.MSELoss()(output, ecg_signal)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        radar_signal, ecg_signal = batch
        output = self.model(radar_signal)
        loss = nn.MSELoss()(output, ecg_signal)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        radar_signal, ecg_signal = batch
        output = self.model(radar_signal)
        loss = nn.L1Loss()(output, ecg_signal)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def get_model(model_type, input_size, output_size, hidden_size=256, num_layers=2, kernel_size=3):
    """
    Get the model for the given type.

    Args:

        model_type: type of model to get
        input_size: size of input signal in time steps
        output_size: size of output signal in time steps
        hidden_size: size of hidden layers in the model
        num_layers: number of layers in the model
        kernel_size: size of the kernel for the TCN model

    Returns:
            initiated model for the given type

    """
    if model_type == "MLP":
        return MLP(input_size, output_size)
    elif model_type == "TCN":
        return TCN(output_size, [hidden_size]*num_layers, kernel_size)
    elif model_type == "LSTM":
        return LSTM(hidden_size=hidden_size,
                    num_layers=num_layers)
    elif model_type == "GRU":
        return GRU(hidden_size=hidden_size,
                   num_layers=num_layers)
    elif model_type == "RNN":
        return RNN(hidden_size=hidden_size,
                   num_layers=num_layers)
    elif model_type == "Transformer":
        return Transformer(input_size, output_size, hidden_size, num_layers)

    else:
        print("Model type not recognized or not yet implemented.")
        exit(1)
