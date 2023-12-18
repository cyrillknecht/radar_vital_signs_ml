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


def write_to_csv(signal, filename):
    """
    Append a signal as a new row to a csv file.
    Args:
        signal: signal to append
        filename: filename of the csv file

    """
    with open(filename, "a") as file:
        writer = csv.writer(file)
        writer.writerow(signal)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_inference(cfg: DictConfig):
    """
    Run inference on a test dataset.
    """

    # Load the dataset
    print("Loading dataset...")
    dataset = get_data_loaders(cfg.batch_size, cfg.data_dir, test=True)

    print(f"Loaded {len(dataset) * cfg.batch_size} recording slices for testing.")

    # Load the model
    checkpoint = torch.load(cfg.model_path, map_location=torch.device('cpu'))

    # Weird bug where the model is saved with "model." prefix but can only be loaded without it
    new_state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}

    model = get_model(model_type=cfg.model, input_size=cfg.input_size, output_size=cfg.output_size,
                      hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, kernel_size=cfg.kernel_size)

    litModel = LitModel(model=model, learning_rate=cfg.learning_rate)

    litModel.model.load_state_dict(new_state_dict)

    print("Loaded  model from checkpoint.")

    # Run inference on the test dataset
    print("Running inference...")
    trainer = pl.Trainer(enable_model_summary=True,
                         deterministic=True,
                         fast_dev_run=False,
                         default_root_dir=cfg.save_dir,
                         )

    for batch in dataset:
        features, targets = batch
        for feature, _ in zip(features, targets):
            sample = torch.unsqueeze(torch.unsqueeze(feature, 0), 0)
            result = trainer.predict(litModel, sample)
            result_signal = result[0][0][0].detach().numpy()

            # Save the result
            write_to_csv(result_signal, os.path.join(cfg.data_dir, "results.csv"))

    print("Inference done.")


if __name__ == "__main__":
    # Load the dataset

    # Run inference
    run_inference()
