"""
This file implements a custom PyTorch Dataset class for the ECG-Radar dataset.
The dataset is loaded from HDF5 files directly into memory.

When run as a script, this file will plot the first batch of the training set(s) to verify that the data is loaded
correctly (only works for single-input-feature radar signals).

"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset class for the ECG-Radar dataset.
    """

    def __init__(self, data_dir, data_files):
        """
        Initialize the dataset by loading the data from the HDF5 files into memory.
        Args:
            data_dir(str): Path to the directory containing the HDF5 files from working directory
            data_files(list): List of HDF5 files to load. First file should contain the ECG data, second file should
                contain the radar data

        """

        super().__init__()

        self.data_dir = data_dir
        self.data_files = data_files

        self.ecg_data = torch.from_numpy(
            h5py.File(os.path.join(data_dir, self.data_files[0] + '.h5'), 'r')['dataset'][:].astype(np.float32))
        self.radar_data = torch.from_numpy(
            h5py.File(os.path.join(data_dir, self.data_files[1] + '.h5'), 'r')['dataset'][:].astype(np.float32))

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        Returns:
            int: Number of samples in the dataset

        """

        return len(self.ecg_data)

    def __getitem__(self, idx):
        """
        Returns the data sample at the given index.
        Args:
            idx(int): Index of the sample to return

        Returns:
            torch.Tensor: Radar data sample of shape
            torch.Tensor: ECG data sample

        """

        radar_tensor = self.radar_data[idx]
        ecg_tensor = self.ecg_data[idx]

        return radar_tensor, ecg_tensor


def get_data_loaders(batch_size=8, data_dir="dataset_processed"):
    """
    Returns DataLoader instances for the training, validation, and test sets.
    Files have to be in the working directory and be named:
        - ecg_train.h5
        - radar_train.h5
        - ecg_val.h5
        - radar_val.h5
        - ecg_test.h5
        - radar_test.h5
    to be loaded correctly.

    Args:
        batch_size(int): Batch size for the DataLoader instances
        data_dir(str): Path to the directory containing the HDF5 files from working directory

    Returns:
        DataLoader: Training set
        DataLoader: Validation set
        DataLoader: Test set

    """

    train_dataset = CustomDataset(data_dir, ['ecg_train', 'radar_train'])
    val_dataset = CustomDataset(data_dir, ['ecg_val', 'radar_val'])
    test_dataset = CustomDataset(data_dir, ['ecg_test', 'radar_test'])

    print("Loaded {} training samples.".format(len(train_dataset)))
    print("Loaded {} validation samples.".format(len(val_dataset)))
    print("Loaded {} test samples.".format(len(test_dataset)))

    # Create DataLoader instances for training and validation sets
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=15)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=15)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=15)

    return train_loader, val_loader, test_loader


def test_data_loaders():
    """
    Test function to verify that the data is loaded correctly.
    Plots the first batch of the training set(s) to verify that the data is loaded correctly (only works for
    single-input-feature radar signals).

    """

    train_ldr, val_ldr, test_ldr = get_data_loaders(data_dir="../../dataset_processed")
    print("Number of batches in training set:", len(train_ldr))
    print("Number of batches in validation set:", len(val_ldr))
    print("Number of batches in test set:", len(test_ldr))

    first_batch = next(iter(train_ldr))
    features, labels = first_batch

    frame_time = 0.01015455
    t_signal = np.array(list(range(len(features[0, 0, :])))) * frame_time

    print("Radar signal batch shape:", features.shape)
    print("ECG signal batch shape:", labels.shape)
    if features.shape[1] != 1:
        print("Multi-dimensional radar signal detected. Can not be plotted with this test function.")
        exit(0)

    # Plot the first radar signal
    plt.plot(features[0, 0, :], label='radar')
    plt.title("Preprocessed Radar Signal")
    plt.show()

    # Plot the first ECG signal
    plt.plot(labels[0, 0, :], label='ecg')
    plt.legend()
    plt.title("Target ECG Signal")
    plt.show()

    # Plot the first radar and ECG signal together
    plt.plot(t_signal, labels[0, 0, :].numpy(), label='target')
    plt.plot(t_signal, features[0, 0, :].numpy(), label='radar signal')
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Input and Target Signal Comparison")
    plt.show()


if __name__ == "__main__":
    test_data_loaders()
