"""
This file contains the different Deep Learning Models used in the project.
"""
import os

import torch
import torch.nn as nn
import time


# TODO: TCN
# TODO: RNN
# TODO: Transformer
# TODO: Add wandb logging

# Small Example MLP
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_mlp(model, dataset, epochs=10):
    """
    Train the model on the given dataset.

    Args:
        model:
        dataset: dataset to train on
        epochs: number of epochs to train for

    Returns:

    """

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Split dataset into training and validation set
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                               [0.8, 0.2])

    for epoch in range(epochs):
        rec_idx = 0
        for input_signal, target_signal in train_dataset:
            # Training
            print(f"Training on recording {rec_idx + 1}/{len(train_dataset)}.")
            slc_idx = 0
            for input_slice, target_slice in zip(input_signal, target_signal):
                print(f"Slice {slc_idx + 1}/{len(input_signal)}")
                input_slice = torch.from_numpy(input_slice).float()
                input_slice = input_slice.flatten()
                target_slice = torch.from_numpy(target_slice).float()

                optimizer.zero_grad()

                output_slice = model(input_slice)

                loss = criterion(output_slice, target_slice)
                loss.backward()
                optimizer.step()
                slc_idx += 1
            rec_idx += 1

        print(f"Epoch {epoch + 1} Training-Loss: {loss.item():.4f}")

        # Evaluation
        total_loss = 0
        with torch.no_grad():
            rec_idx = 0
            for input_signal, target_signal in val_dataset:
                print(f"Evaluating on recording {rec_idx + 1}/{len(val_dataset)}.")
                slc_idx = 0
                for input_slice, target_slice in zip(input_signal, target_signal):
                    print(f"Slice {slc_idx + 1}/{len(input_signal)}")
                    input_slice = torch.from_numpy(input_slice).float()
                    input_slice = input_slice.flatten()
                    target_slice = torch.from_numpy(target_slice).float()

                    output_slice = model(input_slice)

                    loss = criterion(output_slice, target_slice)
                    total_loss += loss.item()
                    slc_idx += 1
                rec_idx += 1

        print(f"Epoch {epoch + 1} Validation-Loss: {total_loss / len(val_dataset):.4f}")

        # Save model at current epoch
        current_time = time.time()
        model_name = "checkpoints/mlp_" + str(epoch) + ".pt"
        torch.save(model.state_dict(), model_name)

    # Save final model after training is finished
    current_time = time.time()
    model_name = "checkpoints/mlp_" + str(epochs) + str(current_time) + ".pt"
    torch.save(model.state_dict(), model_name)

    # Delete model checkpoint that are not needed anymore
    for i in range(epochs):
        model_name = "checkpoints/mlp_" + str(i) + ".pt"
        os.remove(model_name)

    return


# U-Net
class UNet(nn.Module):
    # TODO: Dimensions do not make sense yet (output is 2D)
    # TODO: Needs to be adapted for newly added slices
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder (contracting path)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Middle (bottleneck)
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder (expansive path)
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder(x)
        # Middle
        middle = self.middle(enc1)
        # Decoder
        dec1 = self.decoder(middle)
        return dec1


def train_unet(model, train_dataset, epochs=10):
    """
    Train the model on the given dataset.

    Args:
        model:
        train_dataset: dataset to train on
        epochs: number of epochs to train for

    Returns:

    """

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        idx = 0
        for input_signal, target_signal in train_dataset:
            print(f"Training on datapoint {idx} of {len(train_dataset)}")
            input_signal = torch.from_numpy(input_signal).float()
            target_signal = torch.from_numpy(target_signal).float()
            input_signal = input_signal.permute(2, 0, 1)
            target_signal = target_signal.unsqueeze(0).unsqueeze(0)
            target_signal = target_signal.permute(2, 0, 1)

            optimizer.zero_grad()

            output_signal = model(input_signal)

            loss = criterion(output_signal, target_signal)
            loss.backward()
            optimizer.step()
            idx += 1

        print(f"Epoch {epoch + 1} Loss: {loss.item():.4f}")

        # Save model
        model_name = "checkpoints/unet_" + str(epoch) + ".pt"
        torch.save(model.state_dict(), model_name)

    return


# General model-unspecific functions
def train_model(model_type, model, train_dataset, epochs=10):
    """
    Train the model on the given dataset.

    Args:
        model_type: type of model to train
        model: model to train
        train_dataset: dataset to train on
        epochs: number of epochs to train for

    Returns:

    """
    if model_type == "MLP":
        train_mlp(model, train_dataset, epochs)
    elif model_type == "UNet":
        train_unet(model, train_dataset, epochs)
    else:
        print("Model type not recognized or not yet implemented.")
        exit(1)

    return


def get_model(model_type, input_size, output_size):
    """
    Get the model for the given type.

    Args:
        model_type: type of model to get
        input_size: size of input
        output_size: size of output

    Returns:
        model

    """
    if model_type == "MLP":
        return MLP(input_size, output_size)
    elif model_type == "UNet":
        return UNet(input_size, output_size)
    else:
        print("Model type not recognized or not yet implemented.")
        exit(1)


def get_inference(model, input_signal):
    """
    Get the inference for the given input signal.

    Args:
        model: model to get inference from
        input_signal: input signal to get inference from

    Returns:
        inference

    """
    model.eval()
    input_signal = torch.from_numpy(input_signal).float()
    input_signal = input_signal.flatten()
    inference = model(input_signal)
    inference = inference.detach().numpy()
    return inference
