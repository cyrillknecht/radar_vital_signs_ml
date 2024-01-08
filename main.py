"""
Main runner file for training a model.
"""

import hydra
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
import pprint

from models.LightningModel import get_model, LitModel
from pipeline.dataloader import get_data_loaders


@hydra.main(version_base=None, config_path="configs", config_name="config")
def training_loop(cfg: DictConfig):
    if not cfg.api_key and not cfg.dev_mode:
        print("No wandb API key provided. Please provide a key in the config file to train.")
        return

    if not cfg.dev_mode:  # If not in dev mode or , use wandb logger and save checkpoints
        print("Loading Logger...")
        wandb.login(key=cfg.api_key)
        logger = WandbLogger(project=cfg.project_name, name=cfg.model, save_dir=cfg.save_dir)
        print("Logger loaded.")

    else:
        logger = None

    # Print configuration
    print(OmegaConf.to_yaml(cfg))

    hydra.output_subdir = None  # Prevent hydra from creating a new folder for each run

    print("Loading dataset...")
    train_dataset, val_dataset = get_data_loaders(cfg.batch_size, cfg.data_dir)
    test_dataset = get_data_loaders(cfg.batch_size, cfg.data_dir, test=True)
    print(f"Loaded {len(train_dataset) * cfg.batch_size} recording slices for training.")
    print(f"Loaded {len(val_dataset) * cfg.batch_size} recording slices for validation.")
    print(f"Loaded {len(test_dataset) * cfg.batch_size} recording slices for testing.")

    print("Loading model...")
    if cfg.model_path:

        checkpoint = torch.load(cfg.model_path, map_location=torch.device('cpu'))

        # Weird bug where the model is saved with "model." prefix but can only be loaded without it
        new_state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}

        model = get_model(model_type=cfg.model,
                          input_size=cfg.input_size,
                          output_size=cfg.output_size,
                          hidden_size=cfg.hidden_size,
                          num_layers=cfg.num_layers,
                          kernel_size=cfg.kernel_size,
                          signal_length=cfg.signal_length)

        litModel = LitModel(model=model, learning_rate=cfg.learning_rate)

        litModel.model.load_state_dict(new_state_dict)

        print("Loaded model from checkpoint for further training.")

    else:
        model = get_model(model_type=cfg.model, input_size=cfg.input_size, output_size=cfg.output_size,
                          hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, kernel_size=cfg.kernel_size)
        litModel = LitModel(model=model, learning_rate=cfg.learning_rate)
        print("New model loaded.")

    # Create the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.save_dir,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
    )

    trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                         enable_model_summary=True,
                         logger=logger,
                         deterministic=True,
                         fast_dev_run=cfg.dev_mode,
                         default_root_dir=cfg.save_dir,
                         callbacks=[EarlyStopping(monitor='val_loss', patience=5),
                                    checkpoint_callback]
                         )

    print("Training model...")
    trainer.fit(litModel, train_dataset, val_dataset)
    print("Model trained.")

    print("Evaluating model...")
    trainer.test(litModel, test_dataset, verbose=True)

    if cfg.dev_mode:
        print("Training finished. Running in dev mode, so not logging results.")
        return

    print("Generating plotted results...")
    for batch in test_dataset:
        features, targets = batch
        for feature, target in zip(features, targets):
            sample = torch.unsqueeze(torch.unsqueeze(feature, 0), 0)
            result = trainer.predict(litModel, sample)

            fig = get_plot(result, target, cfg.frame_time)

            # Log the result to wandb
            wandb.log({"result": fig})

    print("Result plots logged to WandB.")

    print("Training finished.")

    return


def get_plot(result, target, frame_time):
    if len(target.shape) == 1:
        print("Debug: had to unsqueeze target")
        target = torch.unsqueeze(target, 0)

    if target.shape[0] > 1:  # If classification
        target = target.argmax(axis=0)
        target = torch.unsqueeze(target, 0)

        result = result[0].argmax(axis=0)
        result = torch.unsqueeze(result, 0)

    t_signal = np.array(list(range(len(target[0, :])))) * frame_time

    plt.plot(t_signal, target[0, :].numpy(), label='target')
    plt.plot(t_signal, result[0, 0, :].numpy(), label='prediction')
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Normalized Amplitude")
    plt.title("ECG Prediction")

    fig = plt.gcf()

    return fig


if __name__ == "__main__":
    training_loop()
