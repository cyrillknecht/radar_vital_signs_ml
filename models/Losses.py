"""
Loss functions for the models.
"""

import torch
import torch.nn as nn


def mse_loss(y_true, y_pred):
    """
    Custom batched loss function for sawtooth signal prediction.

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

    return second_derivative_diff


def peak_detection_loss(y_true, y_pred, soft_threshold=0.5):
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
    y_true_diff = y_true[:, 5:] - y_true[:, :-5]
    y_true_diff_relu = nn.ReLU()(y_true_diff)
    soft_peaks_true = torch.sigmoid((y_true_diff_relu - soft_threshold) * 1000)

    y_pred_diff = y_pred[:, 5:] - y_pred[:, :-5]
    y_pred_diff_relu = nn.ReLU()(y_pred_diff)
    soft_peaks_pred = torch.sigmoid((y_pred_diff_relu - soft_threshold) * 1000)

    peak_loss = nn.L1Loss()(soft_peaks_pred, soft_peaks_true)

    return peak_loss


def cross_entropy_loss(y_true, y_pred, device='cpu', weight=80):
    """
    Custom batched loss function for classification accuracy.

    Args:
    - y_true (Tensor): The true labels (shape [batch_size, 2, sequence_length]).
    - y_pred (Tensor): The predicted logits (shape [batch_size, 2, sequence_length]).

    Returns:
    - Tensor: The computed loss value for the batch.

    """

    return nn.CrossEntropyLoss(weight=torch.tensor([1, weight]))(y_pred, y_true)


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

    # Initialize components for peak position and count accuracy
    position_diff_total = 0.0
    count_diff_total = 0.0
    confidence_total = 0.0

    for i in range(batch_size):
        peaks_pred = y_pred[i, 0, :] - y_pred[i, 1, :]  # if peak, positive; if not peak, negative
        only_peaks = nn.ReLU()(peaks_pred)
        pooled_peaks = nn.MaxPool1d(kernel_size=10)(only_peaks.unsqueeze(0))
        pooled_peaks = pooled_peaks.squeeze(0)
        soft_peaks = torch.sigmoid((pooled_peaks * 1000))  # scale up the peaks to 1
        pred_peak_count = soft_peaks.sum()

        num_peaks_true = torch.sum(y_true[i, 0, :])
        count_diff = nn.L1Loss()(num_peaks_true, pred_peak_count)

        # Confidence measure
        confidence = 1 / (count_diff + 1)  # Range:[0, 1/2]

        sequence_length = y_true.shape[2]

        # Generate a positional index tensor
        position_indices = torch.arange(sequence_length, device=device, dtype=torch.float)

        # Weighted sum of positions
        true_weighted_positions = (y_true * position_indices).sum()
        position_indices = position_indices.unsqueeze(0)
        position_indices_pooled = nn.AvgPool1d(kernel_size=10)(position_indices)
        position_indices_pooled = position_indices_pooled.squeeze(0)
        pred_weighted_positions = (soft_peaks * position_indices_pooled).sum()

        # Calculate the difference
        position_diff = nn.L1Loss()(true_weighted_positions, pred_weighted_positions)

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
    print("Confidence: ", confidence_avg)

    # Combining all components
    count_weight = torch.abs(confidence_avg - 1) * peak_count_weight
    position_weight = confidence_avg * peak_accuracy_weight
    combined_loss = count_weight * count_diff_avg + position_weight * position_diff_avg

    return combined_loss


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

    if 'mse' in component_weights:
        loss += component_weights['mse'] * mse_loss(y_true, y_pred)
    if 'derivative' in component_weights:
        loss += component_weights['derivative'] * derivative_loss(y_true, y_pred)
    if 'second_derivative' in component_weights:
        loss += component_weights['second_derivative'] * second_derivative_loss(y_true, y_pred)
    if 'peak_detection' in component_weights:
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

    if 'cross_entropy' in component_weights:
        loss += component_weights['cross_entropy'] * cross_entropy_loss(y_true, y_pred)
    if 'peak_position_count' in component_weights:
        loss += component_weights['peak_position_count'] * peak_position_count_loss(y_true, y_pred)

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
