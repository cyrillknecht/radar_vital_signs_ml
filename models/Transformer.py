import torch
import torch.nn as nn
from lightning import LightningModule


# TODO: positional encoding?

class Transformer(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout=0.2):
        super(Transformer, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.transformer = nn.Transformer(d_model=2048,
                                          nhead=16,
                                          dropout=dropout,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers)
        self.fc2 = nn.Linear(2048, output_size)

    def forward(self, x, target):
        x = self.fc1(x)
        x, _ = self.transformer(x)
        x = self.fc2(x)
        return x


class LitTransformer(LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, x, target):
        return self.model(x, target)

    def training_step(self, batch, batch_idx):
        radar_signal, ecg_signal = batch
        output = self.model(radar_signal, ecg_signal)
        loss = nn.MSELoss()(output, ecg_signal)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        radar_signal, ecg_signal = batch
        output = self.model(radar_signal, ecg_signal)
        loss = nn.MSELoss()(output, ecg_signal)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        radar_signal, ecg_signal = batch
        output = self.model(radar_signal, ecg_signal)
        loss = nn.MSELoss()(output, ecg_signal)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
