"""
Script to run inference on a previously trained model.
Saves the results in csv files.
"""

import lightning as pl
from models.LightningModel import LitModel, get_model
import torch
import numpy as np
import os
import pandas as pd
from omegaconf import DictConfig
import hydra
from pipeline.dataloader import get_data_loaders
import csv
import h5py


def store_data_h5(data, target_dir, filename):
    """
    Stores the given data in an HDF5 file.

    Args:
        data(np.array): Data to store
        target_dir(str): Directory to store the file in
        filename(str): Name of the file to store the data in

    """

    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, filename + '.h5')

    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset('dataset', data=data)


def inference(cfg):
    """
    Runs inference on a previously trained model.
    Args:
        cfg(DictConfig): Hydra config object containing all necessary parameters

    Returns:
        np.array: Array containing all the predicted signals for the test set

    """

    print("Starting inference...")
    # delete old results file if it exists
    if os.path.exists(os.path.join(cfg.dirs.data_dir, "results.csv")):
        os.remove(os.path.join(cfg.data_dir, "results.csv"))

    print("Loading dataset...")
    _, _, dataset = get_data_loaders(cfg.training.batch_size, cfg.dirs.data_dir)

    print(f"Loaded {len(dataset) * cfg.training.batch_size} recording slices for testing.")

    # Load the model
    checkpoint = torch.load(cfg.inference.model_path, map_location=torch.device('cpu'))

    # Weird bug where the model is saved with "model." prefix but can only be loaded without it
    new_state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}

    model_type = cfg.model
    model = get_model(model_type=cfg.model,
                      input_size=cfg.models[model_type].input_size,
                      output_size=cfg.models[model_type].output_size,
                      hidden_size=cfg.models[model_type].hidden_size,
                      num_layers=cfg.models[model_type].num_layers,
                      kernel_size=cfg.models[model_type].kernel_size,
                      no_dilation_layers=cfg.models[model_type].no_dilation_layers)

    litModel = LitModel(model=model, learning_rate=cfg.training.learning_rate)

    litModel.model.load_state_dict(new_state_dict)

    print("Loaded  model from checkpoint.")

    print("Running inference...")
    trainer = pl.Trainer(enable_model_summary=True,
                         deterministic=True,
                         fast_dev_run=False,
                         default_root_dir=cfg.dirs.save_dir,
                         )

    result_signal_list = []
    for batch in dataset:
        features, targets = batch
        for feature, _ in zip(features, targets):
            sample = torch.unsqueeze(torch.unsqueeze(feature, 0), 0)
            result = trainer.predict(litModel, sample)
            result_signal = result[0].squeeze(0).detach().numpy()
            result_signal_list.append(result_signal)

    # Save the result h5
    result_signal_list = np.array(result_signal_list)
    store_data_h5(result_signal_list, cfg.dirs.data_dir, "results")

    print("Inference done.")

    return result_signal_list


@hydra.main(version_base="1.2", config_path="../configs", config_name="config")
def inference_hydra(cfg: DictConfig):
    """
    Hydra wrapper for running the inference function as a script.
    Args:
        cfg(DictConfig): Hydra config object containing all necessary parameters

    """

    hydra.output_subdir = None  # Prevent hydra from creating a new folder for each run
    inference(cfg)


if __name__ == "__main__":
    inference_hydra()
