"""
Original and predicted signals are analyzed and compared
using a peak detection algorithm.
The metrics used for comparison are:
Peak Count Error,
Average Absolute Peak Position Error

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

from helpers.fmcw_utils import HR_calc_ecg


def peak_detection(signal, threshold=0.8, plot=False):
    """
    Find peaks in the given ECG and radar signals.

    Args:
        signal: ECG or radar signal
        threshold: threshold for peak detection
        plot: whether to plot the signals

    Returns:
        peak idxs for ECG and radar signals

    """

    # peak detection algorithm
    peaks, _ = find_peaks(signal, height=0.1, prominence=0.2)

    # Plot the signal and the peaks
    if plot:
        plt.plot(signal)
        plt.plot(peaks, signal[peaks], "x")
        plt.plot(np.zeros_like(signal), "--", color="gray")
        plt.show()

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

    return error


def average_absolute_peak_position_error(ecg_peaks, radar_peaks):
    """
    Calculate the average absolute peak position error between the given ECG and radar signals.

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

    for true_peak in ecg_peaks:
        # Find the closest detected peak
        closest_detected_peak = min(radar_peaks, key=lambda x: abs(x - true_peak))
        # Calculate the absolute positional error
        error = abs(true_peak - closest_detected_peak)
        positional_errors.append(error)

    # Calculate the average absolute positional error
    average_positional_error = sum(positional_errors) / len(positional_errors)

    return average_positional_error


def analyze_signal(predicted_ecg_signal, original_ecg_signal, plot=False):
    """
    Analyze the results of the model.

    Args:
        predicted_ecg_signal: predicted ECG signal
        original_ecg_signal: original ECG signal
        plot: whether to plot the signals

    """

    hr, ecg_peaks, filtered, ecg_info = HR_calc_ecg(original_ecg_signal, mode=1, safety_check=False)
    predicted_ecg_peaks = peak_detection(predicted_ecg_signal, threshold=0.8, plot=False)
    print("hr:", hr)
    print("ecg_peaks:", len(ecg_peaks))
    print("predicted_hr:", len(predicted_ecg_peaks) * 2)
    print("predicted_ecg_peaks:", len(predicted_ecg_peaks))

    if plot:
        plt.plot(predicted_ecg_signal)
        plt.plot(ecg_peaks, np.ones_like(ecg_peaks), "x")
        plt.plot(predicted_ecg_peaks, np.ones_like(predicted_ecg_peaks), "o")
        plt.show()

    error_count = peak_count_error(ecg_peaks, predicted_ecg_peaks)

    avg_abs_peak_pos_error = average_absolute_peak_position_error(ecg_peaks, predicted_ecg_peaks)

    return error_count, avg_abs_peak_pos_error


def compare_signals(predicted_ecg_signal, processed_radar_signal, original_ecg_signal, plot=False):
    """
    Compare the original and predicted signal.

    Args:
        predicted_ecg_signal: predicted ECG signal
        processed_radar_signal: processed radar signal
        original_ecg_signal: original ECG signal

    Returns:
        peak count error for the predicted ECG signal
        average absolute peak position error for the predicted ECG signal
        peak count error for the processed radar signal
        average absolute peak position error for the processed radar signal

    """
    print("Analyzing predicted ECG signal...")
    error_count_prediction, pos_error_prediction = analyze_signal(predicted_ecg_signal, original_ecg_signal, plot)
    print("Analyzing processed radar signal...")
    error_count, pos_error = analyze_signal(processed_radar_signal, original_ecg_signal, plot)

    return error_count_prediction, pos_error_prediction, error_count, pos_error


def compare_signal_lists(predicted_ecg_signal_list, processed_radar_signal_list, original_ecg_signal_list, plot=False):
    """
    Compare the original and predicted signal.

    Args:
        predicted_ecg_signal_list: list of predicted ECG signals
        processed_radar_signal_list: list of processed radar signals
        original_ecg_signal_list: list of original ECG signals
        cap: number of signals to compare

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

    for predicted_ecg_signal, processed_radar_signal, original_ecg_signal in zip(predicted_ecg_signal_list,
                                                                                 processed_radar_signal_list,
                                                                                 original_ecg_signal_list):

        error_count_prediction, pos_error_prediction, error_count, pos_error = compare_signals(predicted_ecg_signal,
                                                                                               processed_radar_signal,
                                                                                               original_ecg_signal,
                                                                                               plot)
        avg_error_count_prediction += error_count_prediction
        avg_pos_error_prediction += pos_error_prediction
        avg_error_count += error_count
        avg_pos_error += pos_error

    avg_error_count_prediction /= len(predicted_ecg_signal_list)
    avg_pos_error_prediction /= len(predicted_ecg_signal_list)
    avg_error_count /= len(predicted_ecg_signal_list)
    avg_pos_error /= len(predicted_ecg_signal_list)

    return avg_error_count_prediction, avg_pos_error_prediction, avg_error_count, avg_pos_error


if __name__ == "__main__":
    # Load the data
    data_dir = "dataset_processed"
    ecg_signal_list = pd.read_csv(os.path.join(data_dir, "ecg_test.csv"), header=None).values
    radar_signal_list = pd.read_csv(os.path.join(data_dir, "radar_test.csv"), header=None).values
    predicted_ecg_signal_list = pd.read_csv(os.path.join(data_dir, "results.csv"), header=None).values

    # Compare the signals
    errors = compare_signal_lists(predicted_ecg_signal_list,
                                  radar_signal_list,
                                  ecg_signal_list,
                                  plot=True)

    print("Average Peak Count Error for Predicted ECG Signals:", errors[0])
    print("Average Absolute Peak Position Error for Predicted ECG Signals:", errors[1])
    print("Average Peak Count Error for Processed Radar Signals:", errors[2])
    print("Average Absolute Peak Position Error for Processed Radar Signals:", errors[3])
