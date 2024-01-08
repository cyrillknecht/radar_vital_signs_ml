"""
Original and predicted signals are analyzed and compared
using a peak detection algorithm.
The metrics used for comparison are:
Peak Count Error,
Average Absolute Peak Position Error

"""

import numpy as np
import pandas as pd
import h5py
import torch
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

from helpers.fmcw_utils import HR_calc_ecg


def smooth_signal(signal, window_size):
    """
    Smoothens a signal using a moving average.

    Parameters:
        signal (array-like): The input signal.
        window_size (int): The size of the moving average window.

    Returns:
        array-like: Smoothed signal.
    """
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='valid')


def peak_detection(signal, prominence=0.7):
    """
    Find peaks in the given ECG and radar signals.

    Args:
        signal: ECG or radar signal
        prominence: prominence for peak detection
        plot: whether to plot the signals

    Returns:
        peak idxs for ECG and radar signals

    """

    # flatten signal
    # signal = smooth_signal(signal, 3)

    # peak detection algorithm
    peaks, _ = find_peaks(signal, prominence=prominence, distance=10)

    return peaks


def peak_count_error(ecg_peaks, radar_peaks):
    """
    Calculate the peak count error between the given ECG and radar signals.

    Args:
        ecg_peaks: peak idxs for the ECG signal
        radar_peaks: peak idxs for the radar signal

    Returns:
        peak count error

    """

    # Calculate the peak count error
    error = abs(len(ecg_peaks) - len(radar_peaks))

    if len(ecg_peaks) < len(radar_peaks):
        print("Warning: More peaks detected in radar signal than in ECG signal.")

    return error


def average_absolute_peak_position_error(ecg_peaks, radar_peaks):
    """
    Calculate the average absolute peak position error between the given ECG and radar signals.
    Only takes predicted peaks into account. Missing peaks are ignored.

    Args:
        ecg_peaks: peak idxs for the ECG signal
        radar_peaks: peak idxs for the radar signal

    Returns:
        average absolute peak position error

    """

    if len(ecg_peaks) == 0 or len(radar_peaks) == 0:
        print("Error: No peaks detected.")
        return -1
    # Assuming true_peaks and detected_peaks are sorted arrays of peak positions
    positional_errors = []

    for predicted_peak in radar_peaks:
        # Find the closest detected peak
        closest_detected_peak = min(ecg_peaks, key=lambda x: abs(x - predicted_peak))
        # Calculate the absolute positional error
        error = abs(predicted_peak - closest_detected_peak)
        positional_errors.append(error)

    if len(radar_peaks) > len(ecg_peaks):
        to_delete = len(radar_peaks) - len(ecg_peaks)
        # sort positional errors
        positional_errors.sort()
        # delete the to_delete largest positional errors since they have no corresponding peak in the ecg signal
        positional_errors = positional_errors[:-to_delete]

    # Calculate the average absolute positional error
    average_positional_error = sum(positional_errors) / len(positional_errors)

    return average_positional_error


def analyze_signal(predicted_ecg_signal, original_ecg_signal, plot=False, prominence=0.7):
    """
    Analyze the results of the model.

    Args:
        predicted_ecg_signal: predicted ECG signal
        original_ecg_signal: original ECG signal
        plot: whether to plot the signals
        prominence:

    """
    if len(original_ecg_signal.shape) == 2: # If classification
        original_ecg_signal = original_ecg_signal.argmax(axis=0)

    hr, ecg_peaks, filtered, ecg_info = HR_calc_ecg(original_ecg_signal, mode=1, safety_check=False)
    predicted_ecg_peaks = peak_detection(predicted_ecg_signal, prominence)
    print("ecg_peaks:", len(ecg_peaks))
    print("predicted_ecg_peaks:", len(predicted_ecg_peaks))

    error_count = peak_count_error(ecg_peaks, predicted_ecg_peaks)
    print("error_count:", error_count)

    avg_abs_peak_pos_error = average_absolute_peak_position_error(ecg_peaks, predicted_ecg_peaks)

    # convert to milliseconds
    avg_abs_peak_pos_error *= 30 * 1000 / len(original_ecg_signal)

    if plot:
        plt.plot(predicted_ecg_signal)
        plt.plot(original_ecg_signal)
        plt.plot(ecg_peaks, original_ecg_signal[ecg_peaks], "x")
        plt.plot(predicted_ecg_peaks, predicted_ecg_signal[predicted_ecg_peaks], "o")
        plt.legend(["Predicted ECG", "Original ECG", "Original Peaks", "Predicted Peaks"])
        plt.title(f"Peak Count Error {round(error_count, 2)}  and "
                  f" Avg. Peak Pos. Error {round(avg_abs_peak_pos_error, 2)}ms")
        plt.show()

    return error_count, avg_abs_peak_pos_error


def compare_signals(predicted_ecg_signal, processed_radar_signal, original_ecg_signal, plot=False, prominence=0.7):
    """
    Compare the original and predicted signal.

    Args:
        prominence:
        predicted_ecg_signal: predicted ECG signal
        processed_radar_signal: processed radar signal
        original_ecg_signal: original ECG signal
        plot: whether to plot the signals

    Returns:
        peak count error for the predicted ECG signal
        average absolute peak position error for the predicted ECG signal
        peak count error for the processed radar signal
        average absolute peak position error for the processed radar signal

    """
    print("Analyzing predicted ECG signal...")
    error_count_prediction, pos_error_prediction = analyze_signal(predicted_ecg_signal, original_ecg_signal, plot,
                                                                  prominence)
    print("Analyzing processed radar signal...")
    error_count, pos_error = analyze_signal(processed_radar_signal, original_ecg_signal, plot, prominence)

    return error_count_prediction, pos_error_prediction, error_count, pos_error


def compare_signal_lists(predicted_ecg_signal_list,
                         processed_radar_signal_list,
                         original_ecg_signal_list,
                         plot=False,
                         prominence=0.7):
    """
    Compare the original and predicted signal.

    Args:
        prominence:
        predicted_ecg_signal_list: list of predicted ECG signals
        processed_radar_signal_list: list of processed radar signals
        original_ecg_signal_list: list of original ECG signals
        plot: whether to plot the signals

    Returns:
        peak count error for the predicted ECG signal
        average absolute peak position error for the predicted ECG signal
        peak count error for the processed radar signal
        average absolute peak position error for the processed radar signal

    """

    avg_error_count_prediction = 0
    avg_pos_error_prediction = 0
    avg_error_count = 0
    avg_pos_error = 0
    counter = 0

    # Check that inference was done correctly
    if len(predicted_ecg_signal_list) != len(processed_radar_signal_list) or len(predicted_ecg_signal_list) != len(
            original_ecg_signal_list):
        print("Error: The lists are not the same length.")
        return -1, -1, -1, -1

    print(f"Comparing {len(predicted_ecg_signal_list)} signals...")

    for predicted_ecg_signal, processed_radar_signal, original_ecg_signal in zip(predicted_ecg_signal_list,
                                                                                 processed_radar_signal_list,
                                                                                 original_ecg_signal_list):
        error_count_prediction, pos_error_prediction, error_count, pos_error = compare_signals(predicted_ecg_signal,
                                                                                               processed_radar_signal,
                                                                                               original_ecg_signal,
                                                                                               plot,
                                                                                               prominence)
        avg_error_count_prediction += error_count_prediction
        avg_pos_error_prediction += pos_error_prediction
        avg_error_count += error_count
        avg_pos_error += pos_error

        if error_count_prediction == 0:
            counter += 1

    avg_error_count_prediction /= len(predicted_ecg_signal_list)
    avg_pos_error_prediction /= len(predicted_ecg_signal_list)
    avg_error_count /= len(predicted_ecg_signal_list)
    avg_pos_error /= len(predicted_ecg_signal_list)

    return avg_error_count_prediction, avg_pos_error_prediction, avg_error_count, avg_pos_error


if __name__ == "__main__":
    # Load the data
    data_dir = "dataset_processed"
    plot = False
    prominence = 0.2
    """if plot:
        ecg_signal_list = pd.read_csv(os.path.join(data_dir, "ecg_test.csv"), header=None).values[-1:, :]
        radar_signal_list = pd.read_csv(os.path.join(data_dir, "radar_test.csv"), header=None).values[-1:, :]
        predicted_ecg_signal_list = pd.read_csv(os.path.join(data_dir, "results.csv"), header=None).values[-1:, :]
    else:
        ecg_signal_list = pd.read_csv(os.path.join(data_dir, "ecg_test.csv"), header=None).values
        radar_signal_list = pd.read_csv(os.path.join(data_dir, "radar_test.csv"), header=None).values
        predicted_ecg_signal_list = pd.read_csv(os.path.join(data_dir, "results.csv"), header=None).values
"""
    # Load the data from h5 files
    ecg_signal_list = torch.from_numpy(
        h5py.File(os.path.join(data_dir, "ecg_test.h5"), 'r')['dataset'][:].astype(np.float32)).squeeze(1)
    radar_signal_list = torch.from_numpy(
        h5py.File(os.path.join(data_dir, "radar_test_1d.h5"), 'r')['dataset'][:].astype(np.float32)).squeeze(1)
    predicted_ecg_signal_list = torch.from_numpy(
        h5py.File(os.path.join(data_dir, "results.h5"), 'r')['dataset'][:].astype(np.float32)).squeeze(1)

    # Compare the signals
    errors = compare_signal_lists(predicted_ecg_signal_list,
                                  radar_signal_list,
                                  ecg_signal_list,
                                  plot,
                                  prominence)

    # Print the results
    print("Average Peak Count Error for Predicted ECG Signals:", round(errors[0], 2))
    print("Average Absolute Peak Position Error for Predicted ECG Signals[ms]:", round(errors[1], 2))
    print("Average Peak Count Error for Processed Radar Signals:", round(errors[2], 2))
    print("Average Absolute Peak Position Error for Processed Radar Signals[ms]:", round(errors[3], 2))
