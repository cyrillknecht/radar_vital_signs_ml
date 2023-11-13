import datetime
import os.path

import numpy as np
from helpers.envelopes import peak_envelopes
from helpers.fmcw_utils import range_fft, HR_calc_ecg, extract_phase_multibin, find_max_bin, butt_filt
from helpers.plotting_helpers import plot_ecg_lines

import matplotlib.pyplot as plt


def read_ecg(filename):
    """Reader function for the ecg data

    Args:
      filename: filename of ecg-data

    Returns:
      ecg-signal,the start-time of the signal

    """
    with open(filename, "r") as file:
        start_time_ecg = file.readline().strip("\n")
        start_time_ecg = datetime.datetime.strptime(start_time_ecg, '%H:%M:%S.%f')
        ecg = np.loadtxt(file)

    return ecg, start_time_ecg,


def read_acc(filename):
    """Reader function for the acc data

    Args:
      filename: filename of ecg-data

    Returns:
      acc signals

    """
    with open(filename, "r") as file:
        start_time = file.readline().strip("\n")
        start_time = datetime.datetime.strptime(start_time, '%H:%M:%S.%f')
        acc_signals = np.loadtxt(file)

    return acc_signals, start_time_acc


def read_radar_data_npz(filename, which_shape=1):
    """Read Radar Data,

    Args:
      filename: filename of the radar data
      which_shape:  which shape to read (Default value = 1)

    Returns:
        the data of the radar in the format [num_frames,shape_group_repetitions,shape_repetitions,number of samples,num_rx_antennas]
    """

    file = np.load(filename, allow_pickle=True)
    radar_data = file[f"data_shape{which_shape}"]
    config = file["config"].item()
    start_time = file["start_time"][0]
    other = [file["comment"].item()]
    return radar_data, (start_time, config, other)


if __name__ == "__main__":
    subject = "1"
    recording = "2"
    ecg_samplingrate = 130

    ecg, start_time_ecg, = read_ecg(os.path.join("dataset", str(subject), f"recording_{recording}_ecg.txt"))
    acc, start_time_acc = read_acc(os.path.join("dataset", str(subject), f"recording_{recording}_acc.txt"))

    radar_60, info_60, = read_radar_data_npz(os.path.join("dataset", str(subject), f"recording_{recording}_60GHz.npz"))
    # radar_24, info_24,=read_radar_data_npz(os.path.join("dataset",str(subject),f"recording_{recording}_24GHz.npz"))
    # radar_120, info_120,=read_radar_data_npz(os.path.join("dataset",str(subject),f"recording_{recording}_120GHz.npz"))

    data = radar_60

    # get rid of the shape group repetitions as we only have one
    data = data[:, 0]
    # Phase accumulation by taking the mean of the shape repetitions
    data = np.mean(data, axis=1)

    # plot range ffts of first chirp of antenna 0
    range_ffts = np.apply_along_axis(range_fft, 1, data, windowing=False, mean_removal=False)
    fig, ax = plt.subplots()
    ax.plot(np.abs(range_ffts[0, 1:, 0]))
    plt.show()

    frame_time = info_60[1]["frame_time"]

    # extract the first 30 seconds
    range_ffts = range_ffts[:int(30 // frame_time), ...]
    ecg_part = ecg[:int(30 * ecg_samplingrate)]

    hr, peaks, filtered, ecg_info = HR_calc_ecg(ecg_part, mode=1, safety_check=False)

    # get the bin where the person is for the first 30 seconds
    index, all_bins = find_max_bin(range_ffts[..., -1:], mode=1, min_index=8,
                                   window=int(4 // frame_time), step=int(1 // frame_time))
    print(f"Person is in bin {index} for the first 30 seconds")

    # normal phase extraction
    normal_phase = np.unwrap(np.angle(range_ffts[:, index, -1]))

    # phase extraction with multibin
    multibin_phase = extract_phase_multibin(range_ffts[:, index - 1:index + 2, -1], alpha=0.995, )

    plt.plot(normal_phase)
    plt.plot(multibin_phase)
    plt.show()

    signal = butt_filt(multibin_phase, 10, 22, 1 / frame_time)
    hf_signal = peak_envelopes(signal)

    t_signal = np.array(list(range(len(signal)))) * frame_time

    t_signal_ecg = np.array(list(range(len(ecg_part)))) * (1 / ecg_samplingrate)

    # Debug prints
    print(f"Shape of hf signal: {hf_signal.shape}")
    print(f"Shape of t_signal: {t_signal.shape}")
    print(f"Shape of ecg: {t_signal_ecg.shape}")
    print(f"Shape of t_signal_ecg: {t_signal_ecg.shape}")

    fig2, ax = plt.subplots()
    ax.plot(t_signal, hf_signal)
    plot_ecg_lines(t_signal_ecg[peaks], ax)
    plt.show()
