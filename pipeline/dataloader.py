import os
import pandas as pd
import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data_dir, data_files=None):
        if data_files is None:
            self.data_files = ["full_dataset_ecg.csv", "full_dataset_radar.csv"]
        self.data_dir = data_dir
        self.ecg_data = torch.from_numpy(
            pd.read_csv(os.path.join(data_dir, self.data_files[0]), header=None).values.astype(np.float32))
        self.radar_data = torch.from_numpy(
            pd.read_csv(os.path.join(data_dir, self.data_files[1]), header=None).values.astype(np.float32))

    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, idx):
        ecg_tensor = self.ecg_data[idx]
        radar_tensor = self.radar_data[idx]

        ecg_tensor = ecg_tensor.unsqueeze(0)
        radar_tensor = radar_tensor.unsqueeze(0)

        return radar_tensor, ecg_tensor


def get_data_loaders(batch_size=8, data_dir="dataset_processed"):
    custom_dataset = CustomDataset(data_dir)

    # Ensure that the dataset is not empty
    if len(custom_dataset) == 0:
        print("Error: The dataset is empty.")
        return

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(custom_dataset))
    val_size = int(0.1 * len(custom_dataset))
    test_size = len(custom_dataset) - train_size - val_size

    # Ensure that the split sizes are positive
    if train_size <= 0 or val_size <= 0:
        print("Error: Invalid split sizes. Adjust the split ratio or dataset size.")
        return

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(custom_dataset,
                                                                             [train_size, val_size, test_size])

    print("Train Dataset Length:", len(train_dataset))
    # Create DataLoader instances for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data loader by running this script
    print("Testing data loader...")
    train_ldr, val_ldr, test_ldr = get_data_loaders()
    print("Number of batches in training set:", len(train_ldr))
    print("Number of batches in validation set:", len(val_ldr))
    print("Number of batches in test set:", len(test_ldr))

    first_batch = next(iter(train_ldr))
    features, labels = first_batch
    print("Radar signal batch shape:", features.shape)
    print("ECG signal batch shape:", labels.shape)

    # Plot the first radar signal
    import matplotlib.pyplot as plt

    plt.plot(features[0, 0, :], label='radar')
    plt.title("Preprocessed Radar Signal")
    plt.show()
    # Plot the first ECG signal
    plt.plot(labels[0, 0, :], label='ecg')
    plt.legend()
    plt.title("Target ECG Signal")
    plt.show()

    frame_time = 0.01015455
    t_signal = np.array(list(range(len(features[0, 0, :])))) * frame_time

    plt.plot(t_signal, labels[0, 0, :].numpy(), label='target')
    plt.plot(t_signal, features[0, 0, :].numpy(), label='radar signal')
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Input and Target Signal Comparison")
    plt.show()
