"""
Main runner file for training and evaluating the model.
"""

# TODO: Implement TCN instead of MLP
# TODO: Ask because of different shapes of ecg and radar_data even after resampling
# TODO: Training Evaluation Split, Inference
# TODO: Conf file

import os.path
import time

import torch
import matplotlib.pyplot as plt
from models import get_model, train_model, get_inference
from preprocessing import load_dataset

if __name__ == "__main__":

    # Configuration
    NUM_SUBJECTS = 2
    NUM_RECORDINGS = 3
    EPOCHS = 2
    INPUT_SIZE = 980 * 128 * 3
    OUTPUT_SIZE = 1300
    MODEL_TYPE = "MLP"
    MODEL_PATH = ""

    subject_list = [i for i in range(0, NUM_SUBJECTS)]
    recording_list = [i for i in range(0, NUM_RECORDINGS)]

    start = time.time()

    print("Loading dataset...")
    dataset = load_dataset(subject_list, recording_list)
    print(f"Loaded {len(dataset)} recordings")

    print("Loading model...")
    model = get_model(MODEL_TYPE, INPUT_SIZE, OUTPUT_SIZE)

    print("Model architecture summary:")
    print(model)

    if MODEL_PATH:
        print("Loading previously trained model...")
        model.load_state_dict(torch.load(MODEL_PATH))

    print("Training model...")
    train_model(MODEL_TYPE, model, dataset, EPOCHS)

    end = time.time()

    print("Training finished...")
    print(f"Training time: {round(end - start)} seconds")

    print("Plotting example prediction...")
    input_signal = dataset[0][0][0]
    target_signal = dataset[0][1][0]
    prediction = get_inference(model, input_signal)
    plt.plot(prediction)
    plt.plot(target_signal)
    plt.legend(["Prediction", "Target"])
    plt.title("Example Prediction")
    plt.show()
