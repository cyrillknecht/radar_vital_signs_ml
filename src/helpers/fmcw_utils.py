import datetime
import os
from collections import Counter
from sys import platform

import circle_fit as cf
import heartpy as hp
import neurokit2
import numpy as np
import scipy
import scipy.signal as signal
from pandas import read_csv

from src.helpers.envelopes import peak_envelopes

########################################################################################################################
# This file includes all the gennal functions that are used in the project
########################################################################################################################
# CONSTANTS
c = 299792458
DISTANCE_ANTENNAS = 0.00235
calibration_matrix_2 = np.array([[1.00000000e+00 + 2.57907340e-18j, -1.94808388e-01 + 6.87288555e-02j],
                               [3.46682389e-18 - 9.45316020e-18j, -2.05643097e-01 + 8.90573508e-01j]])
calibrationn_matrix_3 = np.array([[-4.42781222e-01 + 2.90620574e-01j, 8.84513109e-17 + 1.44711746e-16j,
                                 -4.42781222e-01 + 2.90620574e-01j],
                                [-1.28144984e-01 + 4.00408546e-02j, 1.00000000e+00 - 2.18808886e-17j,
                                 -1.28144984e-01 + 4.00408546e-02j],
                                [-1.10201839e-01 + 4.10861259e-01j, -2.20923674e-16 + 3.80978620e-17j,
                                 -1.10201839e-01 + 4.10861259e-01j]])


def range_fft(series, windowing=False, padding=None, mean_removal=False):
    """Range FFT to apply to all chirps

    Args:
      series: Data of one chirp
      windowing: If the Hann window should be applied first (Default value = False)
      padding: If input_signal should be padded first (Default value = None)
      mean_removal:  (Default value = False)

    Returns:
        FFT of input_signal with the preprocessing applied
    """
    windowed = series
    if mean_removal:
        windowed = windowed - np.average(windowed)
    if windowing:
        hann_w = signal.get_window("hann", len(series))
        windowed = series * hann_w
    if padding is not None:
        windowed = np.pad(windowed, (0, padding), 'constant')
    return np.fft.rfft(windowed)


def butt_filt(signal_to_filter, lf, hf, fps, order=4):
    """Apply simple butterworth filter (between lf and hf) to the input_signal

    Args:
      signal_to_filter (array): input_signal to filter
      lf (float): low frequency
      hf (float): high frequency
      fps (float): frames per second
      order:  (Default value = 4)

    Returns:
        filtered input_signal
    """
    filter_coeff = signal.butter(order, Wn=[lf, hf], btype='bandpass', fs=fps, analog=False, output='sos')
    filtered = signal.sosfiltfilt(filter_coeff, signal_to_filter)
    return filtered


def low_filt(signal_to_filter, lf, fps):
    """Apply simple lowpass filter to the input_signal

    Args:
      signal_to_filter (array): input_signal to filter
      lf (float): filter frequency
      fps (float): frames per second

    Returns:
        filtered input_signal
    """
    filter_coeff = signal.butter(4, Wn=lf, btype='lowpass', fs=fps, analog=False, output='sos')
    filtered = signal.sosfiltfilt(filter_coeff, signal_to_filter)
    return filtered


def high_filt(signal_to_filter, hf, fps):
    """Apply simple highpass filter to the input_signal

    Args:
      signal_to_filter (array): input_signal to filter
      hf (float): filter frequency
      fps (float): frames per second

    Returns:
        filtered input_signal

    """
    filter_coeff = signal.butter(4, Wn=hf, btype='highpass', fs=fps, analog=False, output='sos')
    filtered = signal.sosfiltfilt(filter_coeff, signal_to_filter)
    return filtered


def dacm_phase(iq, start_angle=None):
    """DACM Phase unwrapping (Alternative to arctan unwrapping)

    Args:
      iq: complex input_signal
      start_angle: return: DACM phase unwrapped input_signal (Default value = None)

    Returns:
      DACM phase unwrapped input_signal

    """
    I_signal = np.real(iq)
    Q_signal = np.imag(iq)
    if start_angle == None:
        start_angle = np.angle(iq[0])
    phases = [start_angle]
    for i in range(1, len(iq)):
        phase_for_i = (I_signal[i] * (Q_signal[i] - Q_signal[i - 1]) - Q_signal[i] * (
                I_signal[i] - I_signal[i - 1])) / (I_signal[i] ** 2 + Q_signal[i] ** 2)
        if phase_for_i > np.pi:
            phase_for_i = np.pi * np.sign(phase_for_i)
        phases.append(phase_for_i + phases[-1])
    return phases


def dc_offset(signal, mode=0):
    """calculate DC-Offset of the input_signal

    Args:
      signal: complex input_signal
      mode: With which mode to calculate the DC offset (0: Circle fit, 1: Mean) (Default value = 0)

    Returns:
      The DC offset corrected complex input_signal, the complex DC offset [real,imag]

    """
    if mode == 0:
        iq_window_cf = np.array([np.real(signal), np.imag(signal)]).swapaxes(0, 1)
        middle_re, middle_imag, r, residu = cf.least_squares_circle(iq_window_cf)
    elif mode == 1:
        middle_re, middle_imag = np.mean(np.array([np.real(signal), np.imag(signal)]), axis=1)
    else:
        raise ValueError("Invalid Mode")

    signal_corrected = signal - middle_re - middle_imag * 1j
    return signal_corrected, [middle_re, middle_imag]


def find_max_bin(range_ffts, mode=0, min_index=2, max_index=-1, which_antenna=None, window=None, step=50):
    """

    Args:
      range_ffts: Range FFTs (complex values) to search for peaks (shape: [num_chirps, num_range_bins,num_rcvrs])
      mode: Which mode to use (0: find just maximum peak, 1: find bin with maximum variance) (Default value = 0)
      min_index: minimum range-index to search for peak (Default value = 2)
      max_index: maximum range-index to search for peak (Default value = -1)
      which_antenna: If only one rx channel should be used (None for all/ 0,1,2 for one specific) (Default value = None)
      window: window size to look for the best bin in a windowed fashion (None for no window) (Default value = None)
      step: step size for windowed approach (Default value = 50)

    Returns:
      If no window is supplied it only returns one bin, otherwise it returns the most frequent bin and the bin per window as a list

    """

    if range_ffts.shape[-1] == 1:
        which_antenna = 0

    # If no window is supplied we take the whole input_signal length as the window
    if window is None:
        window = range_ffts.shape[0]
    else:
        # Safety statement to ensure window is not too long
        if window > range_ffts.shape[0]:
            window = range_ffts.shape[0]

    # List to store the chosen bins for each window
    all_chosen_bins = []

    while True:
        # Check if enough data is available, otherwise stop loop and return values
        if range_ffts.shape[0] < window:
            chosen_bin = max(all_chosen_bins, key=all_chosen_bins.count)
            return chosen_bin, all_chosen_bins

        # extract window and move one step further
        range_ffts_window = range_ffts[:window]
        range_ffts = range_ffts[step:]

        if mode == 0:
            # ust find bin with maximum peak
            max_bins = np.argmax(np.abs(range_ffts_window[:, min_index:max_index, :]), axis=1) + min_index
            if which_antenna is not None:
                current_chosen_bin = max(list(max_bins[:, which_antenna].flatten()),
                                         key=list(max_bins[:, which_antenna].flatten()).count)
            else:
                current_chosen_bin = max(list(max_bins.flatten()), key=list(max_bins.flatten()).count)
        if mode == 1:
            max_bin_per_antenna = np.argmax(np.var(range_ffts_window[:, min_index:max_index, :], axis=0),
                                            axis=0) + min_index
            if which_antenna is not None:
                current_chosen_bin = max_bin_per_antenna[which_antenna]
            else:
                current_chosen_bin = max(list(max_bin_per_antenna), key=list(max_bin_per_antenna).count)

        all_chosen_bins.append(current_chosen_bin)


def extract_hr_gaussian_comb(signal, chirp_interval, hr_to_check=list(range(48, 95, 1)), peak_signal=False):
    """Extract HR from a input_signal with the correlation of different gaussian combs (OLD method that assumes very small HRV)

    Args:
      signal: Signal to analyze (either the raw displacement or the peak input_signal. If peak input_signal peak_signal param must be set True)
      chirp_interval: Chirp frequency interval
      hr_to_check: Which HRs to check (Default value = list(range(48)
      peak_signal: If the input_signal is already the peak input_signal or just the displacement (Default value = False)
      95: 
      1)): 

    Returns:
      detected HR

    """

    if not peak_signal:
        # Generate the peak input_signal first
        HF_filtered = butt_filt(signal, 8, 20, 1 / chirp_interval)
        envelope = peak_envelopes(HF_filtered)
        peak_signal = envelope

    else:
        peak_signal = signal

    # List of max correlations
    maxes = []
    # List of correlations
    corrs = []

    for i in hr_to_check:
        # Create a gaussian pulse with the desired width
        window = scipy.signal.gaussian(int(round((60 / i) / chirp_interval)), std=3)

        # Calculate the number of times this pulse fits in the input_signal and take two less to have some room to correlate it
        number_of_windows = len(peak_signal) // len(window) - 2
        window_comb = list(window) * number_of_windows

        # Devide it by the sum of the comb to have a fair comparison as the combs have different lengths
        window_comb = window_comb / np.sum(window_comb)

        cor = np.correlate(window_comb, peak_signal, mode="valid")
        corrs.append(cor)
        maxes.append(np.max(cor))
    current_hr = hr_to_check[np.argmax(maxes)]
    return current_hr


def detect_outliers(data, mode=1, max_val=100000, threshold=3):
    """Detect outliers in a given data set

    Args:
          data: data to analyze
          mode: mode to use (0: z-score, 1: 90/10 quantile) (Default value = 1)
          max_val: maximum value to consider (Default value = 100000)
          threshold: threshold for z-score (Default value = 3)

    Returns:
        indices of outliers

        """
    if mode == 1:
        mean = np.mean(data)
        std = np.std(data)
        outliers = []
        for i, y in enumerate(data):
            z_score = (y - mean) / std
            if np.abs(z_score) > threshold:
                outliers.append(i)
    else:
        reduced_data = data[data < max_val]
        data_09 = np.quantile(reduced_data, 0.9)
        data_01 = np.quantile(reduced_data, 0.1)
        outlier_b = np.where(data > data_09)[0]
        outlier_s = np.where(data < data_01)[0]
        outliers = list(outlier_s) + list(outlier_b)

    return outliers


def extract_hr_peaks(signal, chirp_interval, simple=False, fft_mode=True):
    """OLD FUNCTION. DO NOT USE but stays her for future ideas

    Args:
      signal: 
      chirp_interval: 
      simple:  (Default value = False)
      fft_mode:  (Default value = True)

    Returns:

    """
    signal_padded = np.pad(signal, (0, len(signal)), 'constant')
    frequencies = np.fft.rfftfreq(len(signal_padded), chirp_interval)
    frequencies_reduced = frequencies[frequencies > 0.65]
    fft_signal = np.abs(np.fft.rfft(signal_padded))[frequencies > 0.65]
    peaks, properties = scipy.signal.find_peaks(fft_signal, height=0, distance=5)
    order = np.argsort(properties["peak_heights"])
    ordered_peaks = peaks[order]
    ordered_frequencies = frequencies_reduced[ordered_peaks[::-1]]

    # if math.isclose(ordered_frequencies[0] / 2, ordered_frequencies[1], abs_tol=0.05):
    #     hr_fft = ordered_frequencies[1] * 60
    #
    # else:
    #     hr_fft = ordered_frequencies[0] * 60

    #
    peaks, properties = scipy.signal.find_peaks(signal, distance=(60 / 110) / chirp_interval, prominence=0.01,
                                                height=0.005)
    mean = np.mean(properties["peak_heights"])
    std = np.std(properties["peak_heights"])

    # peaks, properties = scipy.input_signal.find_peaks(input_signal, distance=(60 / 110) / chirp_interval, prominence=0.01,
    #                                             height=(mean-std,mean+3*std))
    all_diffs = np.diff(peaks)
    all_diffs = np.round(all_diffs / (0.02 / chirp_interval)).astype(int)
    if simple:
        outliers = detect_outliers(all_diffs, mode=1, threshold=1)
        # print("outliers:",outliers)
        all_diffs = np.delete(all_diffs, outliers)
        return 60 / (all_diffs * 0.02), std

    # all_diffs=np.round(60 / (all_diffs * chirp_interval)).astype(int)
    if len(all_diffs) > 0:
        # print(all_diffs)
        min_diff = np.min(all_diffs)
        max_diff = np.max(all_diffs)

        occurences = np.array([0] * (max_diff - min_diff + 1))
        counter = Counter(all_diffs)
        occurences[np.array(list(counter.keys())) - min_diff] = list(counter.values())
        convolution = np.convolve(occurences, np.ones(5), mode='valid')
        middle = np.argmax(convolution)
        maxes = np.where(convolution == convolution[middle])[0]

        if len(maxes) > 1:
            middle = np.argmax(np.convolve(convolution, np.ones(3), mode='valid')) + 1

        # Should actually be +2 but as the peaks are always overestimating the HR we do + 1 more
        middle = middle + min_diff + 3
        # most_occuring=max(list(np.round(all_diffs / 4)), key=list(np.round(all_diffs / 4)).count)
        # upper_limit= np.round((most_occuring+1.5)*4)
        # lower_limit=np.round((most_occuring-1.5)*4)

        lower_limit = middle - 6
        upper_limit = middle + 6
        distance_mask = (all_diffs >= lower_limit) & (all_diffs <= upper_limit)
        diffs_hist_old = all_diffs[distance_mask]

        distance_mask = np.pad(distance_mask, pad_width=1, constant_values=False)
        peak_mask = [distance_mask[i] or distance_mask[i + 1] for i in range(len(distance_mask) - 1)]

        new_diffs = np.diff(peaks[peak_mask])
        new_diffs = np.round(new_diffs / (0.02 / chirp_interval)).astype(int)

        skipped_peaks = np.diff(np.where(peak_mask)[0])
        new_diffs = new_diffs / skipped_peaks

        lower_limit = middle - 6
        upper_limit = middle + 6
        distance_mask = (new_diffs >= lower_limit) & (new_diffs <= upper_limit)

        diffs_hist = new_diffs[distance_mask]
        if len(diffs_hist) > 1:
            confidence = (len(all_diffs) / len(diffs_hist) - 1) * 3
        else:
            confidence = 0
        # outliers=detect_outlier_simple(all_diffs,max_val=(60/40)/chirp_interval)

        # diffs_simple = np.delete(all_diffs, outliers)
        if False:
            outliers = detect_outlier(all_diffs, threshold=1)
            diffs = np.delete(all_diffs, outliers)
            new_peaks = np.delete(peaks, outliers[np.where(np.diff(outliers) == 1)[0]] + 1)
            fraction = np.diff(new_peaks) / np.mean(diffs)
            rounded_fraction = np.round(fraction)
            use_diff = np.abs(rounded_fraction - fraction) < 0.30
            new_diffs = np.diff(new_peaks)[use_diff] / rounded_fraction[use_diff]

            peak_outliers = peaks[outliers[np.where(np.diff(outliers) == 1)[0]] + 1]
            hr_med = 60 / (diffs * chirp_interval)
            hr_med_2 = 60 / (new_diffs * chirp_interval)

        diffs_hist = diffs_hist * 0.02 / chirp_interval
        diffs_hist_old = diffs_hist_old * 0.02 / chirp_interval
        hr_org = 60 / (np.mean(diffs_hist) * chirp_interval)

        # hr_org=diffs_hist
        # print(occurences, np.mean(hr_org))

        # print(np.mean(hr_med),cv(hr_med),np.mean(hr_med_2),cv(hr_med_2),np.mean(hr_org),cv(hr_org))
        hr_org = 60 / (diffs_hist * chirp_interval)

        hr_index = find_nearest(ordered_frequencies[:3] * 60, np.mean(hr_org))
        # return [ordered_frequencies[hr_index]*60],0
        return hr_org, confidence

    else:
        return [], 0


def extract_phase_multibin(input_fft_data, alpha=0.99, apply_dc_offset=False, first_angle=0, prev_value=None,
                           normalize=False):
    """Method to extract Phase from multiple range bins (Method 3 from Paper: Vital Sign Detection Using Short-range mm-Wave Pulsed Radar)
    It calculates the angle difference between consecutive chirps of multiple range bins and then accumulates them with a weighting of alpha
    
    TODO: During bin change there is now a 0 but could be handled better

    Args:
      input_fft_data: Complex range fft data in the form [num_chirps,num_range_bins]
      alpha: high pass filter factor (The higher the smoother the input_signal) (Default value = 0.99)
      apply_dc_offset: If DC-offset should be applied to each range bin (Default value = False)
      first_angle:  With which angle to start (Default value = 0), needed if bin was changing
      prev_value:  with which complex value to start (Default value = None) needed if bin was changing
      normalize:  normalize the power of each bin (Default value = False)

    Returns:

    """
    input_data = input_fft_data.copy()
    if prev_value is not None:
        fft_data = np.insert(input_data, 0, prev_value, 0)
    else:
        fft_data = input_data
    if apply_dc_offset:
        for i in range(fft_data.shape[1]):
            fft_data[:, i] = dc_offset(fft_data[:, i])[0]
    conj = fft_data[1:, :] * np.conjugate(fft_data[:-1, :])

    if normalize:
        conj = conj / abs(fft_data[:-1, :])
    conj_sum = np.sum(conj, axis=1)

    angles_adapted = [first_angle]

    angles = np.angle(conj_sum)

    for i in range(len(angles)):
        angles_adapted.append(angles_adapted[-1] * alpha + angles[i])

    if prev_value is not None:
        return np.array(angles_adapted)[1:]
    else:
        return np.array(angles_adapted)


def find_nearest(array, value):
    """Simple function to find nearest index of a vlaue in an array

    Args:
      array: array to search in
      value: value to search for

    Returns:
        index of the nearest value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def read_ecg(filename):
    """Reader function for the ecg data

    Args:
      filename: filename of ecg-data

    Returns:
      ecg-input_signal, the index of the ecg peaks, the start-time of the input_signal, additional info

    """
    with open(filename, "r") as file:
        start_time_ecg = file.readline().strip("\n")
        start_time_ecg = datetime.datetime.strptime(start_time_ecg, '%H:%M:%S.%f')
        ecg = np.loadtxt(file)

    # check if polar belt was worn incorrectly
    ecg, inverted = neurokit2.ecg_invert(ecg, sampling_rate=130)
    if inverted:
        print("Inverted Polar input_signal")

    # calculat HR ground truth from ecg and get the R-peaks
    hr, peaks, _, info = HR_calc_ecg(ecg, mode=1, safety_check=False, all=True)
    peaks = peaks.astype(int)
    return ecg, peaks, start_time_ecg, info


def read_acc(filename):
    """Reader function for the acc data

    Args:
      filename: filename of ecg-data

    Returns:
      acc signals

    """
    if os.path.exists(filename):
        with open(filename, "r") as file:
            start_time_acc = file.readline().strip("\n")
            start_time_acc = datetime.datetime.strptime(start_time_acc, '%H:%M:%S.%f')
            acc = np.loadtxt(file)

        return acc, start_time_acc
    else:
        print("No acc file")
        return [], datetime.datetime.now()


def read_radar_data_csv_npz(filename, mean=True, which_shape=1):
    """Read Radar Data, supports different file formats as it has changed over time
        newest one is npz with data field for each shape, as well as config and start_time
    Args:
      filename: filename of the radar data
      which_shape:  which shape to read (Default value = 1)

    Returns:
        the data of the radar in the format [num_frames,num_shape_repetitions,number of samples,num_rx_antennas]
    """
    ending = os.path.splitext(filename)[1]
    if ending == ".npz":
        file = np.load(filename, allow_pickle=True)
        # one format where only data was stored in npz and the additional info was extracted from file name
        if "data" in file.files:
            data = file["data"]
            info = filename.split("__")[1].split(".npz")[0].split("_")

        # new format where data is stored in different fields and additional info is stored in npz
        else:
            data = file[f"data_shape{which_shape}"]
            config = file["config"].item()
            start_time = file["start_time"][0]
            other = [file["comment"].item()]
            return data, (start_time, config, other)
    # very old format with csv. could probably be discared
    else:
        data = read_csv(filename, compression="gzip", index_col=0).to_numpy(dtype=np.float32)
        info = filename.split("__")[1].split("_")

    repetitions = int(info[0])
    samples = int(info[1])
    start_time = info[-1]
    start_time = float(start_time)
    start_time = datetime.datetime.fromtimestamp(start_time)
    data = data.reshape(-1, int(repetitions), samples, 3)

    return data, (start_time, None, None)


def calc_beamforming(range_ffts, radar):
    """Calculate the beamforming in the middle for the given data (only for 60GHz radar)
    
    @todo: Add 120GHz radar beamforming

    Args:
      range_ffts: the range ffts in the form ([...,rcvs])
      angle: not implemented yet
      radar: which radar frequency

    Returns:
      returns all the range fft that was given as input put adds a new dimension for the beamforming

    """
    if radar == 60:
        multiplier_0 = 1 / np.exp(1j * 2 * np.pi * 1 * 0.5 * np.sin(np.deg2rad(-45)))
        multiplier_1 = 1 / np.exp(1j * 2 * np.pi * 1 * 0.5 * np.sin(np.deg2rad(20)))
        multiplier_2 = 1

    # elif radar == 120:
    #    multiplier_0 = np.exp(1j * 2 * np.pi * 1 * 0.5 * np.sin(np.deg2rad(-40)))
    #    multiplier_1 = 1
    #    multiplier_2 = np.exp(1j * 2 * np.pi * 1 * 0.5 * np.sin(np.deg2rad(-30)))
    else:
        multiplier_0 = 1
        multiplier_1 = 0
        multiplier_2 = 0

    multipliers = np.array([multiplier_0, multiplier_1, multiplier_2])
    range_ffts_calibrated = range_ffts * multipliers
    calibrated_infineon =np.conj(calibrationn_matrix_3)@np.rollaxis(range_ffts,-1,1)
    calibrated_infineon=np.rollaxis(calibrated_infineon,-1,1)

    range_ffts_dbf_infineon= (calibrated_infineon[...,0]+calibrated_infineon[...,1]+calibrated_infineon[...,2])/3
    range_ffts_dbf = (range_ffts_calibrated[..., 0] + range_ffts_calibrated[..., 1] + range_ffts_calibrated[..., 2]) / 3

    range_ffts_calibrated = np.append(range_ffts_calibrated, range_ffts_dbf_infineon[..., None], -1)
    range_ffts_calibrated = np.append(range_ffts_calibrated, range_ffts_dbf[..., None], -1)
    return range_ffts_calibrated





def HR_calc_ecg(ecg_data, mode=1, safety_check=True, all=False):
    """
    Calculate the HR from an ECG input_signal, is based on the HeartPy library
    Args:
      ecg_data: The ecg data from which the HR is estimated
      mode: Which mode to use for the HR calculation (mode 0: number of peaks per minute, mode 1: median distance between peaks) (Default value = 1)
      safety_check:  (Default value = True)


    Returns:
      Hr estimate, the indexes of the ecg peaks, the filtered ECG input_signal, additional measures from the HeartPy library

    """
    resample_factor = 6

    # recomended to filter the input_signal before processing
    ecg_filtered = hp.filter_signal(ecg_data, cutoff=0.05, sample_rate=130, filtertype='notch')

    # sometimes seems to help but leave for now
    ecg_filtered_enhanced = hp.enhance_ecg_peaks(ecg_filtered, sample_rate=130, iterations=3)

    # resample to make further analysis better
    ecg_resampled = scipy.signal.resample(ecg_filtered, len(ecg_data) * resample_factor)

    try:
        ecg_processed, measures = hp.process(hp.scale_data(ecg_resampled), 130 * resample_factor, windowsize=0.75)
        peaks = ecg_processed["peaklist"]
        peaks = np.round(np.array(peaks) / resample_factor).astype(int)
        peaks = peaks[ecg_processed["binary_peaklist"].astype(bool)]

        return measures["bpm"], peaks[peaks < len(ecg_filtered)], ecg_filtered, (ecg_processed, measures)
    except hp.exceptions.BadSignalWarning:
        return np.nan, np.array([], dtype=int), len(ecg_filtered) * [0], (None, None)


def estimate_radar_HR(displacment_signal, chirp_time, mode):
    """Calculates the heart rate from the phase input_signal with different methods

    Args:
      displacment_signal: unwrapped phase input_signal
      chirp_time: time between chirps
      mode: 0: Banpass Method, 1: High Frequency Method

    Returns:
      estimated HR

    """

    # prepare the frequencies off the FFTs
    # we do zero padding to get a better resolution (8 times)
    frequencies = np.fft.rfftfreq(len(displacment_signal) * 8, chirp_time)
    frequencies_mask = (frequencies > 0.75) & (frequencies < 2.1)
    frequencies_reduced = frequencies[frequencies_mask]

    if mode == 0:
        temp_signal = butt_filt(displacment_signal, 0.75, 2, 1 / chirp_time)
        phase_padded = np.pad(temp_signal, (0, 7 * len(temp_signal)), 'constant')
        fft_signal = np.fft.rfft(phase_padded)[frequencies_mask]
        freq_est = frequencies_reduced[np.argmax(np.abs(fft_signal))] * 60

    elif mode == 1:

        HF_filtered = butt_filt(displacment_signal, 10, 22, 1 / chirp_time)

        if True:
            # for now peak envelops seems to work best
            # signal_part_hf, _, _ = hyperbolicEqn(np.abs(HF_filtered), chirp_time, n=4, C=2.0 * 10 * 0.85, K=100, )
            signal_part_hf = peak_envelopes(HF_filtered)
            # signal_part_hf= shannon_envelope(HF_filtered)

            temp_signal = butt_filt(signal_part_hf, 0.75, 2, 1 / chirp_time)

            phase_padded = np.pad(temp_signal, (0, 7 * len(temp_signal)), 'constant')
            fft_signal = np.fft.rfft(phase_padded)[frequencies_mask]
            freq_est = frequencies_reduced[np.argmax(np.abs(fft_signal))] * 60
            temp_signal = signal_part_hf

    return freq_est, (fft_signal, temp_signal)


def get_first_port():
    """Get the first port of the computer

    Args:

    Returns:
      the name of the first port

    """
    if platform == "linux" or platform == "linux2":
        ports = os.listdir("/dev/")
        ports = [port for port in ports if "ACM" in port]
        com_port_name = f"/dev/{ports[0]}"
    elif platform == "darwin":
        com_port_name = Communication.get_port_list()[0]
    else:
        com_port_name = Communication.get_port_list()[0]
    return com_port_name


def gaussian(x, amplitude, mean, stddev):
    """Create a gaussian input_signal

    Args:
        x: time vector
        amplitude: amplitude of the gaussian
        mean: mean of the gaussian
        stddev: standard deviation of the gaussian

    Returns:
        gaussian input_signal
    """
    return amplitude * np.exp(-((x - mean) / 4 / stddev) ** 2)


def create_comb_from_peaks(peaks, t_signal, width=1.0):
    """Create a comb input_signal from the detected peaks
        The comb should do a small gaussian around the peak

    Args:
      peaks: peaks to create the comb from
      t_signal: t_signal of the comb
      width:  (Default value = 1.0)

    Returns:
        the generated comb input_signal
    """
    comb = np.zeros_like(t_signal, dtype=float)
    for p in peaks:
        comb += gaussian(t_signal, 0.1, p, width)
    return comb


def find_best_correleation_ecg_and_radar(t_signal_ecg, peaks_ecg, t_signal_radar, signal_radar):
    """Find the best correlation between the ecg and radar input_signal
    
    Args:
      t_signal_ecg: time vector of the ecg input_signal
      peaks_ecg: peaks of the ecg input_signal
      t_signal_radar: time vector of the radar input_signal
      signal_radar: radar input_signal
    l
    Returns:
      the best correlation
    """

    max_shift = 80
    t_signal_ecg_peaks = t_signal_ecg[peaks_ecg]

    # find the smalest and biggest t of both time signals
    t_min = np.max([t_signal_ecg[0], t_signal_radar[0]])
    t_max = np.min([t_signal_ecg[-1], t_signal_radar[-1]])

    # cut the signals to the same time range
    t_signal_ecg_peaks = t_signal_ecg_peaks[(t_signal_ecg_peaks >= t_min) & (t_signal_ecg_peaks <= t_max)]
    t_signal_ecg = t_signal_ecg[(t_signal_ecg >= t_min) & (t_signal_ecg <= t_max)]
    signal_radar = signal_radar[(t_signal_radar >= t_min) & (t_signal_radar <= t_max)]
    t_signal_radar = t_signal_radar[(t_signal_radar >= t_min) & (t_signal_radar <= t_max)]

    comb = create_comb_from_peaks(t_signal_ecg_peaks, t_signal_ecg, 0.03)

    # resample the comb to the radar input_signal
    comb_for_radar = scipy.signal.resample(comb, len(t_signal_radar))[max_shift:-max_shift]

    corelation = scipy.signal.correlate(comb_for_radar, signal_radar, mode="valid")

    # find maximas of corelation
    maximas = scipy.signal.argrelextrema(corelation, np.greater)[0] - max_shift
    best_shift_2 = maximas[np.argmin(np.abs(maximas))]
    return best_shift_2



if __name__ == '__main__':
    pass
