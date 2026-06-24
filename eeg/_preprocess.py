"""EEG preprocessing — Butterworth filter with scipy."""

import numpy as np
from scipy.signal import butter, filtfilt


def eeg_preprocess(data, fs, lowcut=0.5, highcut=45.0, notch=None, order=2):
    """Butterworth bandpass + optional notch filter using zero-phase filtfilt.

    Parameters
    ----------
    data : ndarray, (n_chans, n_samples) or (n_samples,)
        Raw EEG in µV.
    fs : float
        Sampling rate in Hz.
    lowcut : float or None
        High-pass cutoff in Hz. None to skip.
    highcut : float or None
        Low-pass cutoff in Hz. None to skip.
    notch : float or None
        Notch frequency in Hz (e.g. 50 or 60). None to skip.
    order : int
        Butterworth filter order.

    Returns
    -------
    filtered : ndarray, same shape as data
        Filtered EEG (zero-phase, sample count unchanged).
    """
    data = np.asarray(data, dtype=np.float64)
    squeeze = data.ndim == 1
    if squeeze:
        data = data[None, :]

    nyq = fs / 2.0

    if lowcut is not None:
        b, a = butter(order, lowcut / nyq, btype='high')
        data = filtfilt(b, a, data, axis=-1)

    if highcut is not None:
        b, a = butter(order, highcut / nyq, btype='low')
        data = filtfilt(b, a, data, axis=-1)

    if notch is not None:
        b, a = butter(order, [notch * 0.8 / nyq, notch * 1.2 / nyq], btype='bandstop')
        data = filtfilt(b, a, data, axis=-1)

    if squeeze:
        data = data[0]
    return data
