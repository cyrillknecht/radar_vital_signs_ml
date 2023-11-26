from helpers.DBF import DBF
from helpers.fmcw_utils import c, DISTANCE_ANTENNAS, range_fft
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_ecg_lines(peaks, ax, color="k", colors=None):
    """Plot vertical lines at the detected peaks

    Args:
        peaks (list): list of peak indices
        ax (matplotlib.axes): axes to plot on
        color (str, optional): color of the lines. Defaults to "k".
        colors (list, optional): list of colors for each line. Defaults to None, in which case all lines are the same color.

    Returns:
        None
    """

    if colors is None:
        colors = [color] * len(peaks)

    for i, x in enumerate(peaks):
        if i == 0:
            ax.axvline(x=x, color=colors[i], alpha=0.3, label="ECG")
        else:
            ax.axvline(x=x, color=colors[i], alpha=0.3, )


def plot_values_in_complex_plane(signal, ax, indexes=None):
    """Plot complex input_signal in complex plane where points change colors over time

    Args:
      signal: complex input_signal (usually one bin of the range FFT over time)
      ax (matplotlib.axes): axis to plot on
      indexes:  Indexes which have special indications (used for heartbeats) (Default value = None)

    Returns:

    """
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(signal)))
    ax.scatter(np.real(signal), np.imag(signal), marker=".",
               color=colors)
    for i, color in enumerate(colors):
        ax.plot(np.real(signal[i:i + 2]), np.imag(signal[i:i + 2]), color=color)
    if indexes is not None:
        ax.scatter(np.real(signal[indexes]), np.imag(signal[indexes]), marker="x", color="black")


def plot_angle(data, bandwidth, num_beans, max_angle_degree, ax, mode="e", min_range=0, max_range=1, ):
    """Plot range angle map and calculate the biggest reflection angle and range between min_range and max_range

    Args:
      data: Radardata of the form [3,num_of_chirps, samples_per_chirp]
      bandwidth: list of two values: start and end frequency of Chirp bandwidth
      num_beans: number of beams, which defines the angle resolution
      max_angle_degree: Maximum degree to beamform to
      ax: Matplotlib axes to put image in
      mode: Either "e" for E-Plane (Antennas 1&2) or "h" for H-Plane (Antennas 0&2) (Default value = "e")
      min_range: Minimum Range to plot and search for biggest reflection (in m) (Default value = 0)
      max_range: Maximum Range to plot and search for biggest reflection (in m) (Default value = 1)

    Returns:

    """
    wavelength = c / bandwidth[0]
    B = abs(bandwidth[1] - bandwidth[0])
    bin_size = c / (2 * B)

    min_bin = int(min_range // bin_size)
    max_bin = int(max_range // bin_size)
    range_bins = tuple(range(min_bin, max_bin, 1))

    dbf = DBF.DBF(2, num_beams=num_beans, max_angle_degrees=max_angle_degree,
                  d_by_lambda=DISTANCE_ANTENNAS / wavelength)

    if mode == "e":
        range_fft_data = np.apply_along_axis(range_fft, -1, data[[1, 2], :, :], windowing=True)
    elif mode == "h":
        range_fft_data = np.apply_along_axis(range_fft, -1, data[[2, 0], :, :], windowing=True)
    else:
        raise ValueError("Invalid Mode")

    range_fft_data = np.moveaxis(range_fft_data, 0, -1)
    range_fft_data = range_fft_data.swapaxes(0, 1)
    rd_beam_formed = dbf.run(range_fft_data)

    beam_range_energy = np.zeros((rd_beam_formed.shape[0], rd_beam_formed.shape[2]))

    for i_beam in range(num_beans):
        doppler_i = rd_beam_formed[:, :, i_beam]
        beam_range_energy[:, i_beam] += np.linalg.norm(doppler_i, axis=1) / np.sqrt(num_beans)

    beam_range_energy_range = beam_range_energy[range_bins, :]
    max_angle = np.argmax(beam_range_energy_range) % num_beans
    max_distance = int(np.argmax(beam_range_energy_range) // num_beans)
    angle = (max_angle - num_beans // 2) * (2 * max_angle_degree / num_beans)
    ax.imshow(
        beam_range_energy_range,
        cmap='viridis',
        extent=(-max_angle_degree,
                max_angle_degree,
                range_bins[0] * bin_size,
                range_bins[-1] * bin_size),
        origin='lower',
        aspect="auto")
    ax.set_title(f"{mode}-plane: {angle}° {(range_bins[0] + max_distance) * bin_size:.2f}m")
    ax.set_xlabel("angle (degrees)")
    ax.set_ylabel("distance (m)")
    return angle
