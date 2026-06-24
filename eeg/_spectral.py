"""EEG spectral analysis — Welch PSD and band power integration."""

import numpy as np

# Default frequency bands (clinical standard)
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# BSR thresholds in µV
BSR_THRESHOLDS = (5, 10, 20)


def eeg_psd(epochs, fs, window='hamming'):
    """Compute Welch PSD averaged across epochs.

    Parameters
    ----------
    epochs : ndarray, (n_epochs, n_chans, epoch_samples)
    fs : float
        Sampling rate in Hz.
    window : str or tuple
        Window type for scipy.signal.get_window.

    Returns
    -------
    psd : ndarray, (n_chans, n_freqs)
        Power spectral density in µV²/Hz.
    freqs : ndarray, (n_freqs,)
        Frequency axis in Hz.
    """
    from scipy.signal import get_window

    n_epochs, n_chans, n_samples = epochs.shape
    win = get_window(window, n_samples)
    U = np.sum(win ** 2)  # window correction factor

    # FFT across all epochs
    fft_vals = np.fft.rfft(epochs * win, axis=-1)  # (n_epochs, n_chans, n_freqs)
    psd = np.mean(np.abs(fft_vals) ** 2, axis=0)  # average across epochs
    # Single-sided PSD
    psd = psd * 2.0 / (U * fs)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / fs)

    return psd, freqs


def eeg_band_power(psd, freqs, bands=None):
    """Integrate PSD over frequency bands (mean power per band).

    Parameters
    ----------
    psd : ndarray, (..., n_freqs)
    freqs : ndarray, (n_freqs,)
    bands : dict or None
        {name: (low, high)}. None defaults to BANDS.

    Returns
    -------
    band_power : ndarray, (..., n_bands)
        Mean power in µV² per band.
    band_names : list of str
    """
    if bands is None:
        bands = BANDS

    band_names = list(bands.keys())
    result = np.zeros(psd.shape[:-1] + (len(band_names),))

    for i, (name, (lo, hi)) in enumerate(bands.items()):
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            result[..., i] = np.mean(psd[..., mask], axis=-1)

    return result, band_names
