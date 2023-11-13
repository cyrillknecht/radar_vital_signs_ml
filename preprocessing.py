"""
Helper functions for preprocessing the data.

Data gets loaded and sliced into chunks of 10 seconds.
ECG data is simplified into a sawtooth signal keeping the original peaks.
"""

from helpers.fmcw_utils import HR_calc_ecg
import numpy as np
from reader import read_radar_data_npz, read_ecg
import os.path


def get_signals(subj, recording):
    """
    Get the radar data from its files and ecg signal for a given subject and recording.
    Args:
        subj(int): subject number
        recording(int): recording number

    Returns:
        tuple of radar data and ecg signal for the given subject and recording

    """
    radar_data, _ = read_radar_data_npz(os.path.join("dataset", str(subj), f"recording_{recording}_60GHz.npz"))

    ecg, _ = read_ecg(os.path.join("dataset", str(subj), f"recording_{recording}_ecg.txt"))

    return radar_data, ecg


def preprocess_target_signal(ecg_signal, ecg_samplingrate=130):
    """
    Generate the simplified target signal from the ECG signal.
    Args:
        ecg_signal(np.array): Loaded ECG signal
        ecg_samplingrate(int): Sampling rate of the ECG signal
    Returns:
        simplified version of the ECG signal
    """

    simplified_signal = get_sawtooth_signal(ecg_signal)

    simplified_signal_sliced = slice_signal(simplified_signal, sampling_rate=ecg_samplingrate)

    return simplified_signal_sliced


def get_sawtooth_signal(signal):
    """
    Generate a sawtooth signal from the input_signal signal.
    Args:
        signal(np.array): Loaded ECG signal

    Returns:
        sawtooth signal generated from signal with original peaks

    """
    # use the existing processing for peaks and filtered resampled signal
    _, peaks, filtered, _ = HR_calc_ecg(signal, mode=1, safety_check=False)

    # Create a new array of zeros
    new_signal = np.zeros_like(filtered)

    # Set the values at peak indices to their original values
    new_signal[peaks] = filtered[peaks]

    # Linearly interpolate between the peaks
    for i in range(len(peaks) - 1):
        start_idx, end_idx = peaks[i], peaks[i + 1]
        start_value, end_value = filtered[start_idx], filtered[end_idx]

        # Linear interpolation
        interpolation = np.linspace(0, end_value, end_idx - start_idx + 1)

        # Update the new signal
        new_signal[start_idx:end_idx + 1] = interpolation

    return new_signal


def get_gaussian_signal(signal):
    """
    Generate a gaussian signal from the input_signal signal.
    Args:
        signal:

    Returns:
        gaussian signal

    """
    # TODO: Work in Progress: Implement gaussian interpolation between peaks

    # use the existing processing for peaks and filtered resampled signal
    _, peaks, filtered, _ = HR_calc_ecg(signal, mode=1, safety_check=False)

    # Create a new array of zeros
    new_signal = np.zeros_like(filtered)

    # Set the values at peak indices to their original values
    new_signal[peaks] = filtered[peaks]

    return new_signal


def preprocess_input_signal(radar_data, frame_time=0.01015455):
    """
    Generate the input_signal signal from the raw radar data.
    Args:
        frame_time:
        radar_data: raw radar data 5D array of shape [num_frames,shape_group_repetitions,shape_repetitions,number of samples,num_rx_antennas]
    Returns:
        input_signal signal: 3D array of shape [num_frames, num_samples, num_rx_antennas]

    """

    # get rid of the shape group repetitions as we only have one
    radar_data = radar_data[:, 0]

    # Phase accumulation by taking the mean of the shape repetitions
    radar_data = np.mean(radar_data, axis=1)
    radar_data_sliced = slice_signal(radar_data, sampling_rate=round(1 / frame_time))

    return radar_data_sliced


def slice_signal(signal, sampling_rate, slice_length=10, start_idx=10):
    """
    Slice the signal into chunks of the given length in seconds.
    Args:
        signal: original signal
        sampling_rate(int): sampling rate of the original signal
        slice_length(int): length of the slices in seconds
        start_idx(int): defines the start of the first slice in seconds

    Returns:
        list of signal chunks

    """
    slices = []
    for i in range(start_idx, len(signal) - slice_length * sampling_rate, slice_length):
        slc = signal[i * sampling_rate:(i + slice_length) * sampling_rate]
        if len(slc) == slice_length * sampling_rate:
            slices.append(slc)

    return slices


def load_datapoint(subject, recording):
    """
    Load the data for a given subject and recording.

    Args:
        subject(int): subject number
        recording(int): recording number

    Returns:
        Tuple of input_signal signal and target signal chunk lists for given subject and recording

    """
    radar_data, ecg_signal = get_signals(subject, recording)
    input_data = preprocess_input_signal(radar_data)
    target_data = preprocess_target_signal(ecg_signal)

    return input_data, target_data


def load_dataset(subjects, recordings):
    """
    Load the dataset for the given subjects and recordings.

    Args:
        subjects: list of subjects to load
        recordings: list of recordings to load

    Returns:
        List of tuples of input_signal signal and target signal chunk lists

    """
    dataset = []
    for subject in subjects:
        for recording in recordings:
            dataset.append(load_datapoint(subject, recording))

    return dataset
