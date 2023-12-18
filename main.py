"""
Main runner file for training and evaluating the model.
"""

# TODO: function to load model and predict on new data
# TODO: Add visualization of validation results

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
from lightning.pytorch.callbacks import ModelCheckpoint


@hydra.main(version_base=None, config_path="configs", config_name="config")
def training_loop(cfg: DictConfig):
    if not cfg.api_key and not cfg.dev_mode and not cfg.test_only:
        print("No wandb API key provided. Please provide a key in the config file.")
        return

    hydra.output_subdir = None
    print("Loading dataset...")
    train_dataset, val_dataset = get_data_loaders(cfg.batch_size, cfg.data_dir)
    print(f"Loaded {len(train_dataset) * cfg.batch_size} recording slices for training.")
    print(f"Loaded {len(val_dataset) * cfg.batch_size} recording slices for validation.")
    test_dataset = get_data_loaders(cfg.batch_size, cfg.data_dir, test=True)
    print("Dataset loaded.")

    print("Loading model...")
    if cfg.model_path:

        checkpoint = torch.load(cfg.model_path, map_location=torch.device('cpu'))

        # Weird bug where the model is saved with "model." prefix but can only be loaded without it
        new_state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}

        model = get_model(model_type=cfg.model, input_size=cfg.input_size, output_size=cfg.output_size,
                          hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, kernel_size=cfg.kernel_size)

        litModel = LitModel(model=model, learning_rate=cfg.learning_rate)

        litModel.model.load_state_dict(new_state_dict)

        print("Loaded  model from checkpoint.")

    else:
        model = get_model(model_type=cfg.model, input_size=cfg.input_size, output_size=cfg.output_size,
                          hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, kernel_size=cfg.kernel_size)
        litModel = LitModel(model=model, learning_rate=cfg.learning_rate)
        print("New model loaded.")

    if not cfg.dev_mode:  # If not in dev mode or just testing, use wandb logger
        print("Loading Logger...")
        wandb.login(key=cfg.api_key)
        wandbLogger = WandbLogger(project=cfg.project_name, name=cfg.model, save_dir=cfg.save_dir)
        print("Logger loaded.")

        # Create the ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.save_dir,
            filename='model-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',  # Specify the metric to monitor for saving checkpoints
            mode='min',  # 'min' or 'max' depending on whether you want to minimize or maximize the monitored metric
            save_top_k=5,  # Save the top 3 models based on the monitored metric
        )

        trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                             enable_model_summary=True,
                             logger=wandbLogger,
                             deterministic=True,
                             fast_dev_run=False,
                             default_root_dir=cfg.save_dir,
                             callbacks=[EarlyStopping(monitor='val_loss', patience=5),
                                        checkpoint_callback]
                             )
    else:
        trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                             enable_model_summary=True,
                             deterministic=True,
                             fast_dev_run=True
                             )

    if not cfg.test_only:
        print("Training model...")
        trainer.fit(litModel, train_dataset, val_dataset)
        print("Model trained.")

    print("Evaluating model...")
    trainer.test(litModel, test_dataset, verbose=True)

    for batch in test_dataset:
        features, targets = batch
        for feature, target in zip(features, targets):
            sample = torch.unsqueeze(torch.unsqueeze(feature, 0), 0)
            result = trainer.predict(litModel, sample)

            # plot the result against the target, time steps are on the x-axis
            frame_time = 0.01015455
            t_signal = np.array(list(range(len(target[0, :])))) * frame_time

            plt.plot(t_signal, target[0, :].numpy(), label='target')
            plt.plot(t_signal, result[0][0, 0, :].numpy(), label='prediction')
            plt.legend()
            plt.xlabel("Time [s]")
            plt.ylabel("Normalized Amplitude")
            plt.title("ECG Prediction")

            fig = plt.gcf()

            if not cfg.dev_mode:
                print("Logging results...")
                # Log the result to wandb
                wandb.log({"result": fig})
                print("Results logged.")

    print("Model evaluated.")


if __name__ == "__main__":
    training_loop()
