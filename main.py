"""
Main runner file for training and evaluating the model.
"""

# TODO: Remove API key before pushing to GitLab
# TODO: Input data visualization
# TODO: Inference function, function to load model and predict on new data
# TODO: check TCN implementation
# TODO: Check preprocessing

import hydra

import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from models.LightningModel import get_model, LitModel
from pipeline.dataloader import get_data_loaders
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


@hydra.main(version_base=None, config_path="configs", config_name="config")
def training_loop(cfg: DictConfig):
    hydra.output_subdir = None
    print("Loading dataset...")
    train_dataset, test_dataset, val_dataset = get_data_loaders(cfg.batch_size, cfg.data_dir)
    print(f"Loaded {len(train_dataset)*cfg.batch_size} recording slices for training.")
    print(f"Loaded {len(val_dataset)*cfg.batch_size} recording slices for validation.")
    print(f"Loaded {len(test_dataset)*cfg.batch_size} recording slices for testing.")

    print("Loading model...")
    model = get_model(model_type=cfg.model, input_size=cfg.input_size, output_size=cfg.output_size,
                      hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, kernel_size=cfg.kernel_size)
    litModel = LitModel(model=model, learning_rate=cfg.learning_rate)
    print("Model loaded.")

    if not cfg.dev_mode:  # If not in dev mode, use wandb logger
        print("Loading Logger...")
        wandb.login(key=cfg.api_key)
        wandbLogger = WandbLogger(project=cfg.project_name, name=cfg.model, save_dir=cfg.save_dir)
        print("Logger loaded.")

        print("Training model...")
        trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                             enable_model_summary=True,
                             logger=wandbLogger,
                             callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')],
                             deterministic=True,
                             fast_dev_run=False,
                             )
    else:
        print("Training model...")
        trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                             enable_model_summary=True,
                             deterministic=True,
                             fast_dev_run=True
                             )

    trainer.fit(litModel, train_dataset, val_dataset)
    print("Model trained.")

    print("Evaluating model...")
    trainer.test(litModel, test_dataset, verbose=True)
    sample = next(iter(test_dataset))
    feature, target = sample
    sample = torch.unsqueeze(feature, 0)
    result = trainer.predict(litModel, sample)

    # plot the result against the target, time steps are on the x-axis
    ecg_samplingrate = 130
    t_signal_ecg = np.array(list(range(len(target[0, 0, :])))) * (1 / ecg_samplingrate)

    plt.plot(t_signal_ecg, target[0, 0, :].numpy(), label='target')
    plt.plot(t_signal_ecg, result[0][0, 0, :].numpy(), label='prediction')
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("ECG Prediction")

    fig = plt.gcf()

    if not cfg.dev_mode:
        print("Logging results...")
        # Log the result to wandb
        wandb.log({"result": fig})
        print("Results logged.")

    plt.show()

    print("Model evaluated.")

    print("Saving model...")


if __name__ == "__main__":
    training_loop()
