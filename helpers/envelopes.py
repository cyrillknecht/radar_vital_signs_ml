import matplotlib.pyplot as plt
import numpy as np
import scipy


def hyperbolicEqn(F, dT, M=1, C=2.0 * 10 * 0.707, K=100, n=4, x0=0.0, v0=0.0):
    """
    See Paper: A cardiac sound characteristic waveform method for in-home heart disorder monitoring with electric stethoscope
    """
    if n == 0:
        a = 0.5
        b = 0.5
    elif n == 1:
        a = 0.5
        b = 1.0 / 3.0
    elif n == 2:
        a = 0.5
        b = 0.0
    elif n == 3:
        a = 3.0 / 2.0
        b = 8.0 / 5.0
    elif n == 4:
        a = 3.0 / 2.0
        b = 2.0
    invM = 1.0 / M
    disp = np.zeros(len(F))
    vel = np.zeros(len(F))
    accl = np.zeros(len(F))
    disp[0] = x0
    vel[0] = v0
    accl[0] = invM * (F[0] - K * x0 - C * v0);
    for i in range(1, len(F)):
        if i == len(F):
            dF = -F[-1]
        elif i > len(F):
            dF = 0.0
        else:
            dF = F[i] - F[i - 1]
        KK = (2.0 / (b * dT ** 2)) * M + (2.0 * a / (b * dT)) * C + K
        FF = dF + ((2.0 / (b * dT)) * M + (2.0 * a / b) * C) * vel[i - 1] + ((1.0 / b) * M + dT * (1.0 - a / b) * C) * \
             accl[i - 1]
        dU = FF / KK
        disp[i] = disp[i - 1] + dU
        vel[i] = vel[i - 1] + dT * (1.0 - a / b) * accl[i - 1] + (2.0 * a / (b * dT)) * dU - (2.0 * a / b) * vel[i - 1]
        accl[i] = accl[i - 1] + (2 / (b * dT ** 2)) * dU - (2 / (b * dT)) * vel[i - 1] - (1.0 / b) * accl[i - 1]
    return disp, vel, accl


def peak_envelopes(s, dmin=1, dmax=1, split=False, both=False):
    """
    Envelope based on the peaks of the input_signal. d
    Input :
    s: 1d-array, data input_signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks to filter peaks
    split: bool, optional, Use if data is not centered around 0
    both: If both the lower envelope and higher envelope should be returned, otherwise sum of both
    Output :
    """
    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        s_mid = np.mean(s)
        lmin = lmin[s[lmin] < s_mid]
        lmax = lmax[s[lmax] > s_mid]

    # Filter for peaks in window of length dmin/dmax
    lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]

    # Safety for input_signal without peaks as interpolation would fail otherwise
    if len(lmax) == 0 and len(lmin) == 0:
        h_signal = s
        l_signal = s
    else:
        h_signal = np.interp(list(range(len(s))), lmax, s[lmax])
        l_signal = np.interp(list(range(len(s))), lmin, s[lmin])
    total = (np.abs(l_signal) + np.abs(h_signal)) / 2
    if both:
        return h_signal, l_signal
    else:
        return total


def shannon_envelope(s,window_length=15):
    """
    Envelope based on shannon energy
    """
    # Diff
    dn = (np.append(s[1:], 0) - s)

    dtn = dn / (np.max(abs(dn)))
    # Shannon energy
    sn = -(dtn ** 2) * np.log10(dtn ** 2)

    sn_mean = np.convolve(sn, np.ones(window_length) / window_length, mode="same")

    return sn_mean


def hilbert_envelope(s):
    """
    envelope from hilbert transform
    """
    return np.abs(scipy.signal.hilbert(s))


if __name__ == '__main__':
    signal = np.loadtxt('signal_for_envelopes.txt', dtype=np.float64)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(signal)
    ax2 = fig.add_subplot(212)
    hilbert = hilbert_envelope(signal)
    shannon = shannon_envelope(signal)
    peak_envelope = peak_envelopes(signal)

    # Model input_signal as single degree-of-freedom and solve differential equation
    # Paper: A cardiac sound characteristic waveform method for in-home heart disorder monitoring with electric stethoscope
    hyperbolic, _, _ = hyperbolicEqn(np.abs(signal), 1 / 100)
    plt.plot((shannon - np.mean(shannon)) / np.std(shannon), label="shannon")
    plt.plot((hilbert - np.mean(hilbert)) / np.std(hilbert), label="hilbert")
    plt.plot((peak_envelope - np.mean(peak_envelope)) / np.std(peak_envelope), label="peak_envelope")
    plt.plot((hyperbolic - np.mean(hyperbolic)) / np.std(hyperbolic), label="hyperbolic")

    plt.legend()
