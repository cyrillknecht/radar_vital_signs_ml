"""
Script to preprocess the original radar and ecg data and save it in a csv file.
Data is processed in 30-second windows which are represented by a row in the csv files.
Radar Data is processed using processing algorithms provided by supervisors.
"""
import datetime
import os.path
import csv

import numpy as np
import scipy.signal

from helpers.envelopes import peak_envelopes
from helpers.fmcw_utils import range_fft, HR_calc_ecg, extract_phase_multibin, find_max_bin, butt_filt


def read_ecg(filename):
    """Reader function for the ecg data

    Args:
      filename: filename of ecg-data

    Returns:
      ecg-input_signal,the start-time of the input_signal

    """
    with open(filename, "r") as ecg_file:
        start_time = ecg_file.readline().strip("\n")
        start_time = datetime.datetime.strptime(start_time, '%H:%M:%S.%f')
        ecg_np = np.loadtxt(ecg_file)

    return ecg_np, start_time,


def read_radar_data_npz(filename, which_shape=1):
    """Read Radar Data,

    Args:
      filename: filename of the radar data
      which_shape:  which shape to read (Default value = 1)

    Returns:
        the data of the radar in the format
         [num_frames,shape_group_repetitions,shape_repetitions,number of samples,num_rx_antennas]
    """

    radar_file = np.load(filename, allow_pickle=True)
    radar_data = radar_file[f"data_shape{which_shape}"]
    config = radar_file["config"].item()
    start_time = radar_file["start_time"][0]
    other = [radar_file["comment"].item()]
    return radar_data, (start_time, config, other)


def get_sawtooth_signal(input_signal):
    """
    Generate a sawtooth signal from the input_signal signal.
    Args:
        input_signal(np.array): Loaded ECG signal

    Returns:
        sawtooth signal generated from signal with original peaks

    """
    # downsample the signal
    resampled_signal = scipy.signal.resample(input_signal, 2954)

    # use the existing processing for peaks and filtered resampled input_signal
    _, peaks, filtered, _ = HR_calc_ecg(resampled_signal, mode=1, safety_check=False)

    # Create a new array of zeros
    new_signal = np.zeros_like(filtered)

    # Set the values at peak indices to their original values
    new_signal[peaks] = 1.0

    # Linearly interpolate between the peaks
    for i in range(len(peaks) - 1):
        start_idx, end_idx = peaks[i], peaks[i + 1]

        # Linear interpolation
        interpolation = np.linspace(0, 1, end_idx - start_idx + 1)

        # Update the new input_signal
        new_signal[start_idx:end_idx + 1] = interpolation

    return new_signal


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


def get_signals(subj, recording):
    """
    Get the radar data from its files and ecg input_signal for a given subject and recording.
    Args:
        subj(int): subject number
        recording(int): recording number

    Returns:
        tuple of radar data and ecg input_signal for the given subject and recording

    """

    radar_data, radar_info = read_radar_data_npz(os.path.join("dataset", str(subj), f"recording_{recording}_60GHz.npz"))

    ecg, _ = read_ecg(os.path.join("dataset", str(subj), f"recording_{recording}_ecg.txt"))

    return radar_data, radar_info, ecg


def preprocess_data(subj_list, rec_list, target_dir, slice_start_time=10, slice_duration=30, slice_stride=5):
    # delete old directory if it exists
    if os.path.exists(target_dir):
        import shutil

        shutil.rmtree(target_dir)
    # Create a folder for the dataset
    os.makedirs(target_dir, exist_ok=True)

    for subject in subj_list:
        # Create a folder for the subject if it doesn't exist
        subject_folder = os.path.join(target_dir, str(subject))
        os.makedirs(subject_folder, exist_ok=True)

        for recording in rec_list:
            ecg_samplingrate = 130

            radar_data, radar_info, ecg = get_signals(subject, recording)

            # get rid of the shape group repetitions as we only have one
            radar_data = radar_data[:, 0]
            # Phase accumulation by taking the mean of the shape repetitions
            radar_data = np.mean(radar_data, axis=1)

            # get range ffts of first chirp of antenna 0
            range_ffts = np.apply_along_axis(range_fft, 1, radar_data, windowing=False, mean_removal=False)

            frame_time = radar_info[1]["frame_time"]

            # Process data in 30-second windows
            for window_start in range(int(slice_start_time // frame_time),
                                      len(range_ffts) - int(slice_duration // frame_time),
                                      int(slice_stride // frame_time)):
                window_end = window_start + int(slice_duration // frame_time)

                # Extract the current 30-second window
                current_range_ffts = range_ffts[window_start:window_end, ...]

                # get the bin where the person is for the current window
                index, all_bins = find_max_bin(current_range_ffts[..., -1:], mode=1, min_index=8,
                                               window=int(4 // frame_time), step=int(1 // frame_time))
                print(
                    f"Person {subject} in recording {recording} is in bin {index} for the window starting"
                    f" at {int(window_start * frame_time)} seconds"
                    f"and ending at {int(window_end * frame_time)} seconds.")

                print(f"frame_time: {frame_time}")

                # phase extraction with multibin
                multibin_phase = extract_phase_multibin(current_range_ffts[:, index - 1:index + 2, -1], alpha=0.995, )

                signal = butt_filt(multibin_phase, 10, 22, 1 / frame_time)
                hf_signal = peak_envelopes(signal)

                # Normalize signal to range [0, 1]
                hf_signal = (hf_signal - np.min(hf_signal)) / (np.max(hf_signal) - np.min(hf_signal))

                # Write hf_signal slice on new line in csv file
                write_to_csv(hf_signal, os.path.join(subject_folder, f"recording_{recording}_radar.csv"))
                write_to_csv(hf_signal, os.path.join(target_dir, "full_dataset_radar.csv"))

                print(f"hf_signal shape: {hf_signal.shape}")

                # Get ecg input_signal slice
                ecg_slice = ecg[int(window_start * frame_time) * ecg_samplingrate:int(
                    window_end * frame_time) * ecg_samplingrate]
                # process ecg_slice
                ecg_slice = get_sawtooth_signal(ecg_slice)

                write_to_csv(ecg_slice, os.path.join(subject_folder, f"recording_{recording}_ecg.csv"))
                write_to_csv(ecg_slice, os.path.join(target_dir, "full_dataset_ecg.csv"))

                print(f"ecg_slice shape: {ecg_slice.shape}")


if __name__ == "__main__":
    TARGET_DIR = "dataset_processed"

    # Process all subjects and recordings
    subject_list = [i for i in range(0, 25)]
    recording_list = [i for i in range(0, 3)]

    preprocess_data(subject_list, recording_list, TARGET_DIR)
