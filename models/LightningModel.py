import torch
import torch.nn as nn
import lightning as pl
from models.MLP import MLP
from models.TCN import TCN
from models.LSTM import LSTM
from models.GRU import GRU
from models.RNN import RNN
from models.Transformer import Transformer


# TODO: add weight to loss function for class imbalance

class LitModel(pl.LightningModule):
    """
    Wrapper class for the models to use with PyTorch Lightning.
    """

    @property
    def device(self):
        return self._device

    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def step(self, batch):
        radar_signal, ecg_signal = batch
        output = self.model(radar_signal)
        if output.shape[1] == 1:
            loss = nn.MSELoss(reduction='mean')(output, ecg_signal)
        else:
            loss = (nn.CrossEntropyLoss(reduction='mean', weight=torch.tensor([1, 80], device=self.device))
                    (output, ecg_signal))

        return loss

    @device.setter
    def device(self, value):
        self._device = value


def get_model(model_type, input_size, output_size, hidden_size=256, num_layers=2, kernel_size=3, signal_length=2954):
    """
    Get the model for the given type.

    Args:

        model_type: type of model to get
        input_size: size of input signal in time steps
        output_size: size of output signal in time steps
        hidden_size: size of hidden layers in the model
        num_layers: number of layers in the model
        kernel_size: size of the kernel for the TCN model
        signal_length:

    Returns:
            initiated model for the given type

    """
    if model_type == "MLP":
        return MLP(input_size, output_size)
    elif model_type == "TCN":
        return TCN(channel_sizes=[hidden_size] * num_layers,
                   input_size=input_size,
                   output_size=output_size,
                   kernel_size=kernel_size,
                   dropout=0.2
                   )
    elif model_type == "LSTM":
        return LSTM(input_features=input_size,
                    output_features=output_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers)
    elif model_type == "GRU":
        return GRU(input_features=input_size,
                   output_features=output_size,
                   hidden_size=hidden_size,
                   num_layers=num_layers)
    elif model_type == "RNN":
        return RNN(input_features=input_size,
                   output_features=output_size,
                   hidden_size=hidden_size,
                   num_layers=num_layers)
    elif model_type == "Transformer":
        return Transformer(input_size, output_size, hidden_size, num_layers)

    else:
        print("Model type not recognized or not yet implemented.")
        exit(1)
