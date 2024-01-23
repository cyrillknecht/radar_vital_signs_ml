"""
Implementation of  a wrapper class for the models to use with PyTorch Lightning.
"""

import torch
import torch.nn as nn
import lightning as pl

from models.TCN import TCN
from models.LSTM import LSTM
from models.GRU import GRU
from models.RNN import RNN
from models.MLP import MLP


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

    def __init__(self, model, learning_rate=1e-3):
        """
        Initiate the Lightning model.
        Args:
            model(torch.nn.Module): model to use
            learning_rate(float): learning rate for the optimizer

        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.device = torch.device(object="cuda:0" if torch.cuda.is_available() else "cpu")

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
        if output.shape[1] == 1:
            loss = nn.MSELoss(reduction='mean')(output, ecg_signal)
        else:
            loss = (nn.CrossEntropyLoss(reduction='mean', weight=torch.tensor([1, 80], device=self.device))
                    (output, ecg_signal))

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

    elif model_type == "MLP":
        return MLP(input_size=input_size,
                   hidden_size=hidden_size,
                   num_layers=num_layers,
                   output_size=output_size)

    else:
        print("Model type not recognized or not yet implemented.")
        exit(1)


def peak_position_count_loss(y_true, y_pred, peak_accuracy_weight=0.5, peak_count_weight=0.5, device='cpu'):
    """
    Custom batched loss function for accurate peak position and count in peak detection.

    Args:
    - y_true (Tensor): The true labels (shape [batch_size, 2, sequence_length]).
    - y_pred (Tensor): The predicted logits (shape [batch_size, 2, sequence_length]).
    - peak_accuracy_weight (float): Weighting factor for the peak position accuracy component.
    - peak_count_weight (float): Weighting factor for the peak count accuracy component.

    Returns:
    - Tensor: The computed loss value for the batch.
    """
    batch_size = y_true.shape[0]

    # Binary Cross-Entropy Loss for classification accuracy
    bce_loss = (nn.CrossEntropyLoss(reduction='mean', weight=torch.tensor([1, 80], device=device))(y_pred, y_true))

    # Initialize components for peak position and count accuracy
    position_diff_total = 0.0
    count_diff_total = 0.0
    confidence_total = 0.0

    for i in range(batch_size):
        peaks_pred = y_pred[i, 0, :] - y_pred[i, 1, :]  # if peak, positive; if not peak, negative

        only_peaks = nn.ReLU()(peaks_pred)
        soft_peaks = torch.sigmoid((only_peaks * 1000))  # scale up the peaks to 1
        pred_peak_count = soft_peaks.sum()

        num_peaks_true = torch.sum(y_true[i, 0, :])
        count_diff = torch.abs(num_peaks_true - pred_peak_count)

        # Confidence measure
        confidence = 1 / (count_diff + 1)  # Range:[0, 1/2]

        sequence_length = y_true.shape[2]

        # Generate a positional index tensor
        position_indices = torch.arange(sequence_length, device=device)

        # Weighted sum of positions
        true_weighted_positions = (y_true * position_indices).sum()
        pred_weighted_positions = (soft_peaks * position_indices).sum()

        # Calculate the difference
        position_diff = torch.abs(true_weighted_positions - pred_weighted_positions)

        # Aggregate the components across the batch
        position_diff_total += position_diff
        count_diff_total += count_diff
        confidence_total += confidence

    # Averaging the components across the batch
    position_diff_avg = position_diff_total / batch_size
    count_diff_avg = count_diff_total / batch_size
    confidence_avg = confidence_total / batch_size
    print("Position Difference: ", position_diff_avg)
    print("Count Difference: ", count_diff_avg)
    print("BCE Loss: ", bce_loss)
    print("Confidence: ", confidence_avg)

    # Combining all components
    count_weight = torch.abs(confidence_avg - 1) * peak_count_weight
    position_weight = confidence_avg * peak_accuracy_weight

    combined_loss = (bce_loss +
                     count_weight * count_diff_avg +
                     position_weight * position_diff_avg)

    return combined_loss


def soft_peak_detection_loss(y_true, y_pred, alpha=0.5, soft_threshold=0.5):
    """
    Custom batched loss function with a differentiable approach to peak detection.

    Args:
    - y_true (Tensor): The true values (shape [batch_size, 1, sequence_length]).
    - y_pred (Tensor): The predicted values (shape [batch_size, 1, sequence_length]).
    - alpha (float): Weighting factor for the soft peak detection component of the loss.
    - soft_threshold (float): Soft threshold for detecting peaks.

    Returns:
    - Tensor: The computed loss value for the batch.
    """

    y_true = y_true.squeeze(1)
    y_pred = y_pred.squeeze(1)

    # Mean Squared Error (MSE) Component
    mse_loss = nn.MSELoss()(y_true, y_pred)

    # Soft Peak Detection Component
    # Apply a sigmoid function to soften the thresholding
    y_true_diff = y_true[:, 5:] - y_true[:, :-5]
    y_true_diff_relu = nn.ReLU()(y_true_diff)
    soft_peaks_true = torch.sigmoid((y_true_diff_relu - soft_threshold)*1000)

    y_pred_diff = y_pred[:, 5:] - y_pred[:, :-5]
    y_pred_diff_relu = nn.ReLU()(y_pred_diff)
    soft_peaks_pred = torch.sigmoid((y_pred_diff_relu - soft_threshold)*1000)

    peak_loss = nn.L1Loss()(soft_peaks_pred, soft_peaks_true, reduction='mean')

    # Combining the MSE and soft peak detection losses
    combined_loss = (1 - alpha) * mse_loss + alpha * peak_loss

    return combined_loss
