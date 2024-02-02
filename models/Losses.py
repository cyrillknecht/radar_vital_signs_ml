"""
Loss functions for the models.
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb


def log_loss(loss, name):
    """
    Log the loss to W&B.

    Args:
    - loss (Tensor): The loss value to log.
    - name (str): The name of the loss.

    """

    if wandb.run:
        # convert tensor to float
        loss_logged = loss.item()
        wandb.log({name: loss_logged})


def mse_loss(y_true, y_pred):
    """
    Mean Squared Error (MSE) loss component for regression models.

    Args:
    - y_true (Tensor): The true values (shape [batch_size, 1, sequence_length]).
    - y_pred (Tensor): The predicted values by the LSTM model (shape [batch_size, 1, sequence_length]).

    Returns:
    - Tensor: The computed loss value for the batch.

    """

    # Squeeze the singleton dimension to make the tensor shape [batch_size, sequence_length]
    y_true = y_true.squeeze(1)
    y_pred = y_pred.squeeze(1)

    # Mean Squared Error (MSE) Component
    l2_loss = nn.MSELoss(reduction='mean')(y_pred, y_true)
    log_loss(l2_loss, "MSE Loss")

    return l2_loss


def derivative_loss(y_true, y_pred):
    """
    Custom loss component for penalizing the difference in the derivatives of the predicted and true values.

    Args:
    - y_true (Tensor): The true values (shape [batch_size, 1, sequence_length]).
    - y_pred (Tensor): The predicted values by the LSTM model (shape [batch_size, 1, sequence_length]).

    Returns:
    - Tensor: The computed loss value for the batch.

    """

    # Squeeze the input tensors to make the tensor shape [batch_size, sequence_length]
    y_true = y_true.squeeze(1)
    y_pred = y_pred.squeeze(1)

    # Derivative-based Component
    derivative_pred = torch.diff(y_pred, dim=1)
    derivative_true = torch.diff(y_true, dim=1)

    # Penalize the difference in the derivatives
    grad_loss = nn.L1Loss(reduction='mean')(derivative_pred, derivative_true)
    log_loss(grad_loss, "Derivative Loss")

    return grad_loss


def second_derivative_loss(y_true, y_pred):
    """
    Custom loss component for penalizing the difference in the second derivatives of the predicted and true values.

    Args:
    - y_true (Tensor): The true values (shape [batch_size, 1, sequence_length]).
    - y_pred (Tensor): The predicted values by the LSTM model (shape [batch_size, 1, sequence_length]).

    Returns:
    - Tensor: The computed loss value for the batch.

    """

    # Squeeze the singleton dimension to make the tensor shape [batch_size, sequence_length]
    y_true = y_true.squeeze(1)
    y_pred = y_pred.squeeze(1)

    # Derivative-based Component
    second_derivative_pred = torch.diff(y_pred, n=2, dim=1)
    second_derivative_true = torch.diff(y_true, n=2, dim=1)

    # Penalize the difference in the derivatives
    second_derivative_diff = nn.L1Loss(reduction='mean')(second_derivative_pred, second_derivative_true)
    log_loss(second_derivative_diff, "Second Derivative Loss")

    return second_derivative_diff


def peak_detection_loss(y_true, y_pred, soft_threshold=0.1, filter_width=10):
    """
    Custom batched loss function component with a differentiable approach to peak detection.

    Args:
    - y_true (Tensor): The true values (shape [batch_size, 1, sequence_length]).
    - y_pred (Tensor): The predicted values (shape [batch_size, 1, sequence_length]).
    - soft_threshold (float): Soft threshold for detecting peaks.

    Returns:
    - Tensor: The computed loss value for the batch.

    """

    y_true = y_true.squeeze(1)
    y_pred = y_pred.squeeze(1)

    # Soft Peak Detection Component
    y_true_diff = y_true[:, :-filter_width] - y_true[:, filter_width:]

    y_true_diff_relu = nn.ReLU()(y_true_diff)
    soft_peaks_true = torch.sigmoid((y_true_diff_relu - soft_threshold) * 1000)

    y_pred_diff = y_pred[:, :-filter_width] - y_pred[:, filter_width:]
    y_pred_diff_relu = nn.ReLU()(y_pred_diff)
    soft_peaks_pred = torch.sigmoid((y_pred_diff_relu - soft_threshold) * 1000)

    soft_peaks_true_sum = soft_peaks_true.sum() / filter_width
    soft_peaks_pred_sum = soft_peaks_pred.sum() / filter_width

    peak_loss = nn.L1Loss()(soft_peaks_true_sum, soft_peaks_pred_sum)
    peak_loss = peak_loss / y_true.shape[0]  # scale down the loss by batch size
    peak_loss = peak_loss / 50  # scale down the loss by maximum number of peaks in a sequence

    log_loss(peak_loss, "Peak Detection Loss")

    return peak_loss


def cross_entropy_loss(y_true, y_pred, device='cpu', weight=80):
    """
    Cross Entropy loss component for binary classification models.

    Args:
    - y_true (Tensor): The true labels (shape [batch_size, 2, sequence_length]).
    - y_pred (Tensor): The predicted logits (shape [batch_size, 2, sequence_length]).

    Returns:
    - Tensor: The computed loss value for the batch.

    """
    loss = nn.CrossEntropyLoss(weight=torch.tensor([1, weight], device=device))(y_pred, y_true)
    log_loss(loss, "Cross Entropy Loss")

    return nn.CrossEntropyLoss(weight=torch.tensor([1, weight], device=device))(y_pred, y_true)


def peak_detection_binary(y_true, y_pred, soft_threshold=0.1):
    """
    Custom batched loss function component with a differentiable approach to peak detection for binary classification.
    Scaled to approximately [0, 1].

    Args:
        - y_true (Tensor): The true values (shape [batch_size, 1, sequence_length]).
        - y_pred (Tensor): The predicted values (shape [batch_size, 1, sequence_length]).

    Returns:
        - Tensor: The computed loss value for the batch.

    """

    batch_size = y_true.shape[0]

    # Initialize components for peak position and count accuracy
    count_diff_total = 0.0

    for i in range(batch_size):
        peaks_pred = y_pred[i, 1, :] - y_pred[i, 0, :]  # if peak, positive; if not peak, negative
        only_peaks = nn.ReLU()(peaks_pred)

        soft_peaks = torch.sigmoid((only_peaks - soft_threshold) * 1000)  # scale up the peaks to 1
        pred_peak_count = soft_peaks.sum()  # count the number of peaks

        num_peaks_true = torch.sum(y_true[i, 1, :])
        count_diff = nn.L1Loss()(num_peaks_true, pred_peak_count)

        count_diff_total += count_diff

    count_diff_avg = count_diff_total / batch_size  # average the count difference over the batch
    count_diff_scaled = count_diff_avg / 50  # scale down the loss by maximum number of peaks in a sequence

    log_loss(count_diff_scaled, "Peak Detection Binary Loss")

    return count_diff_scaled


def get_combined_regression_loss(y_true, y_pred, device='cpu', component_weights=None):
    """
    Get the combined regression loss for the given true and predicted values.

    Args:
    - y_true (Tensor): The true labels (shape [batch_size, 2, sequence_length]).
    - y_pred (Tensor): The predicted logits (shape [batch_size, 2, sequence_length]).
    - device (str): The device to use for the loss function.
    - component_weights (dict): Dictionary of weights for the loss components.

    Returns:
    - Tensor: The computed loss value for the batch.

    """
    if component_weights is None:
        component_weights = {'mse': 1.0}

    loss = torch.tensor(0.0, device=device)

    if 'mse' in component_weights and component_weights['mse'] > 0.0:
        loss += component_weights['mse'] * mse_loss(y_true, y_pred)
    if 'derivative' in component_weights and component_weights['derivative'] > 0.0:
        loss += component_weights['derivative'] * derivative_loss(y_true, y_pred)
    if 'second_derivative' in component_weights and component_weights['second_derivative'] > 0.0:
        loss += component_weights['second_derivative'] * second_derivative_loss(y_true, y_pred)
    if 'peak_detection' in component_weights and component_weights['peak_detection'] > 0.0:
        loss += component_weights['peak_detection'] * peak_detection_loss(y_true, y_pred)

    return loss


def get_combined_classification_loss(y_true, y_pred, device='cpu', component_weights=None):
    """
    Get the combined classification loss for the given true and predicted values.

    Args:
    - y_true (Tensor): The true labels (shape [batch_size, 2, sequence_length]).
    - y_pred (Tensor): The predicted logits (shape [batch_size, 2, sequence_length]).
    - device (str): The device to use for the loss function.
    - component_weights (dict): Dictionary of weights for the loss components.

    Returns:
    - Tensor: The computed loss value for the batch.

    """
    if component_weights is None:
        component_weights = {'cross_entropy': 1.0}

    loss = torch.tensor(0.0, device=device)

    if 'cross_entropy' in component_weights and component_weights['cross_entropy'] > 0.0:
        loss += component_weights['cross_entropy'] * cross_entropy_loss(y_true, y_pred, device=device)
    if 'peak_detection_binary' in component_weights and component_weights['peak_detection_binary'] > 0.0:
        loss += component_weights['peak_detection_binary'] * peak_detection_binary(y_true, y_pred)

    return loss


def get_loss(y_true, y_pred, device='cpu', component_weights=None):
    """
    Get the combined loss for the given true and predicted values.

    Args:
    - y_true (Tensor): The true labels (shape [batch_size, 2, sequence_length]).
    - y_pred (Tensor): The predicted logits (shape [batch_size, 2, sequence_length]).
    - device (str): The device to use for the loss function.
    - component_weights (dict): Dictionary of weights for the loss components.

    Returns:
    - Tensor: The computed loss value for the batch.

    """

    if y_true.shape[1] == 1:
        return get_combined_regression_loss(y_true, y_pred, device, component_weights)
    elif y_true.shape[1] == 2:
        return get_combined_classification_loss(y_true, y_pred, device, component_weights)
    else:
        raise ValueError("Invalid shape for y_true: {}".format(y_true.shape))
