"""
Preprocessing of the radar data and ecg input_signal.
All data is sliced into windows of adjustable length and stride.
The radar data is processed by extracting the phase of the range ffts and further processing it.
Then either a one-dimensional or a multi-dimensional signal is saved.

The ecg target signal is simplified to a either a sawtooth signal or a binary one-hot encoded signal.
"""
import datetime
import os.path
import csv
import time
import h5py
import hydra
import torch
import shutil
import numpy as np
import scipy.signal
import warnings
from omegaconf import DictConfig

from helpers.envelopes import peak_envelopes
from helpers.fmcw_utils import range_fft, HR_calc_ecg, extract_phase_multibin, find_max_bin, butt_filt


def process_phase_signal(signal, frame_time, apply_peak_envelopes=True):
    """
    Process the phase signal by filtering, extracting the peak envelopes and normalizing it to [0,1].
    Args:
        signal(np.array): phase signal
        frame_time(float): frame time of the radar data
        apply_peak_envelopes(bool): whether to apply peak envelopes to the signal

    Returns:
        np.array: processed, normalized signal


    """
    hf_signal = butt_filt(signal, 10, 22, 1 / frame_time)
    if apply_peak_envelopes:
        hf_signal = peak_envelopes(hf_signal)
    if np.max(hf_signal) - np.min(hf_signal) == 0:
        hf_signal = np.zeros_like(hf_signal)
    else:
        hf_signal = (hf_signal - np.min(hf_signal)) / (np.max(hf_signal) - np.min(hf_signal))

    return hf_signal


def phase_extraction(current_range_ffts, index, frame_time, multiDim=False, apply_peak_envelopes=True, use_magnitude=True):
    """
    Extract the phase from the range fft and process further
    Args:
        current_range_ffts(np.array): range ffts of the current window shape
        index(int): index of the bin where the person is
        frame_time(float): frame time of the radar data
        multiDim(bool): whether to generate a one-dimensional or a multi-dimensional signal.
        One-dimensional signals are generated by only extracting the phase of the bin where the person is most likely.
        Multi-dimensional signals are generated by extracting the phases of all bins and their magnitudes.
        apply_peak_envelopes(bool): whether to apply peak envelopes to the signal
        use_magnitude(bool): whether to use the magnitude of the range ffts as additional input

    Returns: np.array: processed, normalized signal of shape [1, signal_length] for one-dimensional signals or of
    shape [130, signal_length] for multi-dimensional signals

    """

    if not multiDim:
        multibin_phase = extract_phase_multibin(current_range_ffts[:, index - 1:index + 2, -1], alpha=0.995)
        hf_signal = process_phase_signal(multibin_phase, frame_time, apply_peak_envelopes=apply_peak_envelopes)
        hf_signal = np.expand_dims(hf_signal, axis=0)
        return hf_signal

    multidim_hf_signal = []
    for i in range(current_range_ffts.shape[1]):
        # Magnitude extraction
        magnitude = np.abs(current_range_ffts[:, i, -1])

        # Multibin phase extraction
        if i - 1 < 0 or i + 2 > current_range_ffts.shape[1]:
            phase = np.unwrap(np.angle(current_range_ffts[:, i, -1]))
        else:
            phase = extract_phase_multibin(current_range_ffts[:, i - 1:i + 2, -1], alpha=0.995)

        # Normal phase extraction
        # phase = np.unwrap(np.angle(current_range_ffts[:, i, -1]))

        hf_signal_multi = process_phase_signal(phase, frame_time,apply_peak_envelopes=apply_peak_envelopes)
        multidim_hf_signal.append(hf_signal_multi)
        if use_magnitude:
            multidim_hf_signal.append(magnitude)

    return np.array(multidim_hf_signal)


def delete_old_data(target_dir):
    """
    Delete old data in the target directory
    Args:
        target_dir(str): target directory

    """

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)


def store_data_h5(data, target_dir, filename):
    """
    Store data in a h5 file.

    Args:
        data(np.array): data to store
        target_dir(str): target directory
        filename(str): filename of the h5 file

    """
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, filename + '.h5')

    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset('dataset', data=data)


def read_ecg(filename):
    """Reader function for the ecg data

    Args:
      filename(str): filename of ecg-data

    Returns:
        the data of the ecg

    """

    with open(filename, "r") as ecg_file:
        start_time = ecg_file.readline().strip("\n")
        start_time = datetime.datetime.strptime(start_time, '%H:%M:%S.%f')
        ecg_np = np.loadtxt(ecg_file)

    return ecg_np, start_time,


def read_radar_data_npz(filename, which_shape=1):
    """Read Radar Data from npz file.

    Args:
      filename(str): filename of radar-data
      which_shape(int=1): which shape to use

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


def get_sawtooth_signal(input_signal, signal_length=2954):
    """
    Generate a sawtooth signal from the input_signal signal.

    Args:
        input_signal(np.array): Loaded ECG signal
        signal_length(int): length of the output signal

    Returns:
        sawtooth signal generated from signal with original peaks

    """

    # downsample the signal
    resampled_signal = scipy.signal.resample(input_signal, signal_length)

    # use the existing processing for peaks and filtered resampled input_signal
    _, peaks, filtered, _ = HR_calc_ecg(resampled_signal, mode=1, safety_check=False)

    # Create a new array of zeros
    new_signal = np.zeros_like(filtered)

    # Set the values at peak indices to 1
    new_signal[peaks] = 1.0

    # Linearly interpolate between the peaks
    for i in range(len(peaks) - 1):
        start_idx, end_idx = peaks[i], peaks[i + 1]

        # Linear interpolation
        interpolation = np.linspace(0, 1, end_idx - start_idx + 1)

        # Update the new input_signal
        new_signal[start_idx:end_idx + 1] = interpolation

    # Un-squeeze the signal
    new_signal = np.expand_dims(new_signal, axis=0)

    return new_signal


def get_binary_signal(input_signal, signal_length=2954):
    """
    Generate a binary signal from the input_signal signal.

    Args:
        input_signal(np.array): Loaded ECG signal
        signal_length(int): length of the output signal in time steps

    Returns:
        np.array: binary signal generated from signal with original peaks one-hot encoded

    """

    # downsample the signal
    resampled_signal = scipy.signal.resample(input_signal, signal_length)

    # use the existing processing for peaks and filtered resampled input_signal
    _, peaks, filtered, _ = HR_calc_ecg(resampled_signal, mode=1, safety_check=False)

    # Create a new array of zeros
    new_signal = np.zeros_like(filtered)

    # Set the values at peak indices to their original values
    new_signal[peaks] = 1

    # One hot encode the signal
    new_signal_tensor = torch.from_numpy(new_signal).to(torch.int64)
    new_signal_one_hot = torch.nn.functional.one_hot(new_signal_tensor, num_classes=2).numpy()
    new_signal_one_hot = np.transpose(new_signal_one_hot)

    return new_signal_one_hot


def write_to_csv(signal, filename):
    """
    Append a signal as a new row to a csv file.

    Args:
        signal(np.array): signal to append
        filename(str): filename of the csv file

    """
    with open(filename, "a") as file:
        writer = csv.writer(file)
        writer.writerow(signal)


def get_signals(subj, recording, data_dir="dataset"):
    """
    Get the radar data from its files and ecg input_signal for a given subject and recording.
    Args:
        subj(int): subject number
        recording(int): recording number
        data_dir(str): directory of the data

    Returns:
        tuple of radar data, information about the radar date and ecg input_signal for the given subject and recording

    """

    radar_data, radar_info = read_radar_data_npz(os.path.join(data_dir, str(subj), f"recording_{recording}_60GHz.npz"))

    ecg, _ = read_ecg(os.path.join(data_dir, str(subj), f"recording_{recording}_ecg.txt"))

    return radar_data, radar_info, ecg


def preprocess_data(subj_list,
                    rec_list,
                    multi_dim=False,
                    mode="sawtooth",
                    slice_start_time=10,
                    slice_duration=30,
                    slice_stride=5,
                    data_dir="dataset",
                    apply_peak_envelopes=True,
                    use_magnitude=True):
    """
    Preprocess the data for the given subjects and recordings.

    Args:
        subj_list(list): list of subjects to preprocess
        rec_list(list): list of recordings to preprocess
        multi_dim(bool): whether to generate a one-dimensional or a multi-dimensional signal.
        mode(str): whether to generate a sawtooth or a binary classification signal [sawtooth, binary classification]
        slice_start_time(int): start time of the first slice in s
        slice_duration(int): duration of the slices in s
        slice_stride(int): stride of the slices in s
        data_dir(str): directory of the data
        apply_peak_envelopes(bool): whether to apply peak envelopes to the signal
        use_magnitude(bool): whether to use the magnitude of the range ffts as additional input

    Returns:
        tuple[List, List]: tuple of radar data and ecg input_signal lists for the given subjects and recordings

    """

    radar_data_storage = []
    ecg_data_storage = []
    for subject in subj_list:
        for recording in rec_list:
            ecg_samplingrate = 130

            radar_data, radar_info, ecg = get_signals(subject, recording, data_dir=data_dir)

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

                hf_signal = phase_extraction(current_range_ffts, index, frame_time, multi_dim, apply_peak_envelopes,use_magnitude)

                radar_data_storage.append(hf_signal)

                # Get ecg input_signal slice
                start_idx = int(window_start * frame_time) * ecg_samplingrate
                end_idx = int(window_end * frame_time) * ecg_samplingrate
                ecg_slice = ecg[start_idx:end_idx]

                # process ecg_slice
                if mode == "sawtooth":
                    ecg_slice = get_sawtooth_signal(ecg_slice, signal_length=window_end - window_start)
                elif mode == "binary classification":
                    ecg_slice = get_binary_signal(ecg_slice, signal_length=window_end - window_start)

                ecg_data_storage.append(ecg_slice)

    print(f"Radar data storage shape: {np.array(radar_data_storage).shape}")
    print(f"ECG data storage shape: {np.array(ecg_data_storage).shape}")

    return radar_data_storage, ecg_data_storage


def preprocess(target_dir,
               train_subjects,
               val_subjects,
               test_subjects,
               multi_dim=True,
               mode="sawtooth",
               files=None,
               data_dir="dataset",
               slice_duration=30,
               apply_peak_envelopes=True,
               use_magnitude=True):
    """
    Preprocess the data for the given subjects and recordings and store it in h5 files.

    Args:
        target_dir(str): target directory
        train_subjects(list): list of subjects to use for training
        val_subjects(list): list of subjects to use for validation
        test_subjects(list): list of subjects to use for testing
        multi_dim(bool): whether to generate a one-dimensional or a multi-dimensional signal.
        mode(str): whether to generate a sawtooth or a binary classification signal [sawtooth, binary classification]
        files(dict): dictionary of filenames for the h5 files
        data_dir(str): directory of the data
        slice_duration(int): duration of the slices
        apply_peak_envelopes(bool): whether to apply peak envelopes to the signal
        use_magnitude(bool): whether to use the magnitude of the range ffts as additional input

    Returns:
        tuple[List, List, List, List, List, List, List]:
        tuple of preprocessed datasets for training, validation and testing
        [radar_train, ecg_train, radar_val, ecg_val, radar_test, ecg_test, one_dim_radar_test]

    """

    print("Starting preprocessing...")

    # Suppress warnings from scipy
    warnings.filterwarnings("ignore", category=UserWarning, module='scipy.interpolate._fitpack2')
    recordings = [i for i in range(0, 3)]

    if not files:
        files = {"train_data_file": "radar_train",
                 "train_gt_file": "ecg_train",
                 "val_data_file": "radar_val",
                 "val_gt_file": "ecg_val",
                 "test_data_file": "radar_test",
                 "test_gt_file": "ecg_test",
                 "one_dim_test_data_file": "radar_test_1d"}

    # Measure time
    start_time = time.time()

    # Delete old data
    delete_old_data(target_dir)

    # Preprocess data
    radar_train, ecg_train = preprocess_data(subj_list=train_subjects,
                                             rec_list=recordings,
                                             multi_dim=multi_dim,
                                             mode=mode,
                                             data_dir=data_dir,
                                             slice_duration=slice_duration,
                                             apply_peak_envelopes=apply_peak_envelopes)

    radar_val, ecg_val = preprocess_data(subj_list=val_subjects,
                                         rec_list=recordings,
                                         multi_dim=multi_dim,
                                         mode=mode,
                                         data_dir=data_dir,
                                         slice_duration=slice_duration,
                                         apply_peak_envelopes=apply_peak_envelopes)

    radar_test, ecg_test = preprocess_data(subj_list=test_subjects,
                                           rec_list=recordings,
                                           multi_dim=multi_dim,
                                           mode=mode,
                                           data_dir=data_dir,
                                           slice_duration=30,
                                           apply_peak_envelopes=apply_peak_envelopes)

    # We always need one-dimensional data for testing
    one_dim_radar_test, _ = preprocess_data(subj_list=test_subjects,
                                            rec_list=recordings,
                                            multi_dim=False,
                                            mode=mode,
                                            data_dir=data_dir,
                                            slice_duration=30,
                                            apply_peak_envelopes=apply_peak_envelopes)

    # Store data
    store_data_h5(data=radar_train,
                  target_dir=target_dir,
                  filename=files["train_data_file"])

    store_data_h5(data=ecg_train,
                  target_dir=target_dir,
                  filename=files["train_gt_file"])

    store_data_h5(data=radar_val,
                  target_dir=target_dir,
                  filename=files["val_data_file"])

    store_data_h5(data=ecg_val,
                  target_dir=target_dir,
                  filename=files["val_gt_file"])

    store_data_h5(data=radar_test,
                  target_dir=target_dir,
                  filename=files["test_data_file"])

    store_data_h5(data=ecg_test,
                  target_dir=target_dir,
                  filename=files["test_gt_file"])

    store_data_h5(data=one_dim_radar_test,
                  target_dir=target_dir,
                  filename=files["one_dim_test_data_file"])

    end_time = time.time()
    print(f"Preprocessing took {round(end_time - start_time)} seconds.")

    return radar_train, ecg_train, radar_val, ecg_val, radar_test, ecg_test, one_dim_radar_test


@hydra.main(version_base="1.2", config_path="../configs", config_name="config")
def preprocessing_hydra(cfg: DictConfig):
    """
    Hydra wrapper for preprocessing.
    Args:
        cfg(DictConfig): Hydra config object containing all necessary parameters

    """

    hydra.output_subdir = None  # Prevent hydra from creating a new folder for each run

    LEFT_OUT_SUBJECT = 1
    TRAIN_SUBJECTS = [i for i in range(1, 25) if i != LEFT_OUT_SUBJECT]
    VAL_SUBJECTS = [0]
    TEST_SUBJECTS = [LEFT_OUT_SUBJECT]

    preprocess(target_dir="../" + cfg.dirs.data_dir,
               train_subjects=TRAIN_SUBJECTS,
               val_subjects=VAL_SUBJECTS,
               test_subjects=TEST_SUBJECTS,
               multi_dim=cfg.preprocessing.multi_dim,
               mode=cfg.preprocessing.mode,
               data_dir="../" + cfg.dirs.unprocessed_data_dir,
               slice_duration=cfg.preprocessing.slice_duration,
               apply_peak_envelopes=cfg.preprocessing.apply_peak_envelopes,
               use_magnitude=cfg.preprocessing.use_magnitude)


if __name__ == "__main__":
    preprocessing_hydra()



