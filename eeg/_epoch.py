"""EEG epoching, cleaning, and bad-channel detection."""

import numpy as np


def eeg_epoch(data, epoch_len_s, step_s=None, fs=None, t_start=0, detrend=True):
    """Slice continuous EEG into fixed-length epochs with absolute start times.

    Parameters
    ----------
    data : ndarray, (n_chans, n_samples) or (n_samples,)
    epoch_len_s : float
        Epoch duration in seconds.
    step_s : float or None
        Step between epoch starts. Default = epoch_len_s (no overlap).
    fs : float
        Sampling rate in Hz.
    t_start : float
        Absolute start time of the first sample (seconds).
    detrend : bool
        If True, subtract the channel mean from each epoch.

    Returns
    -------
    epochs : ndarray, (n_epochs, n_chans, epoch_samples)
    epoch_times : ndarray, (n_epochs,)
        Absolute start time of each epoch in seconds.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]

    epoch_samples = int(round(epoch_len_s * fs))
    if step_s is None:
        step_samples = epoch_samples
    else:
        step_samples = int(round(step_s * fs))

    n_samples = data.shape[-1]
    n_epochs = max(0, (n_samples - epoch_samples) // step_samples + 1)

    if n_epochs == 0:
        return np.empty((0, data.shape[0], epoch_samples)), np.empty((0,))

    idx = np.arange(epoch_samples)[None, :] + np.arange(n_epochs)[:, None] * step_samples
    epochs = data[:, idx].transpose(1, 0, 2)  # (n_epochs, n_chans, epoch_samples)

    if detrend:
        epochs = epochs - epochs.mean(axis=-1, keepdims=True)

    epoch_times = np.arange(n_epochs) * step_samples / fs + t_start
    return epochs, epoch_times


def eeg_clean_epochs(epochs, epoch_times=None, min_amp=1.0, max_amp=300.0):
    """Reject epochs where any channel's peak-to-peak amplitude is out of range.

    Parameters
    ----------
    epochs : ndarray, (n_epochs, n_chans, epoch_samples)
    epoch_times : ndarray or None
        Epoch start times. If provided, rejected times are removed in sync.
    min_amp, max_amp : float
        Allowed peak-to-peak range in µV.

    Returns
    -------
    clean_epochs : ndarray, (n_valid, n_chans, epoch_samples)
    clean_times : ndarray or None
    """
    ptp = np.ptp(epochs, axis=-1)  # (n_epochs, n_chans)
    keep = np.all((ptp >= min_amp) & (ptp <= max_amp), axis=1)

    clean_epochs = epochs[keep]
    if epoch_times is not None:
        return clean_epochs, epoch_times[keep]
    return clean_epochs, None


def eeg_bad_channels(epochs, ratio=0.5):
    """Detect bad channels by comparing effective epoch counts across channels.

    A channel where n_valid_epochs < ratio * mean_valid_epochs is flagged.

    Parameters
    ----------
    epochs : ndarray, (n_epochs, n_chans, epoch_samples)
    ratio : float
        Threshold ratio relative to the mean across channels.

    Returns
    -------
    bad_channels : list of int
    good_channels : list of int
    """
    ptp = np.ptp(epochs, axis=-1)
    n_valid = np.sum((ptp >= 1.0) & (ptp <= 300.0), axis=0)  # per-channel
    threshold = ratio * np.mean(n_valid)
    good = np.where(n_valid >= threshold)[0]
    bad = np.where(n_valid < threshold)[0]
    return bad.tolist(), good.tolist()
