"""EEG nonlinear features — Approximate Entropy and Burst Suppression Ratio."""

import numpy as np


def eeg_apen(epochs, m=2, r_factor=0.2):
    """Approximate Entropy (ApEn) per channel, averaged across epochs.

    Uses Chebyshev (max-norm) distance. Self-implemented, no EntropyHub dependency.

    Parameters
    ----------
    epochs : ndarray, (n_epochs, n_chans, epoch_samples)
    m : int
        Template length (default 2).
    r_factor : float
        Tolerance factor multiplied by std of the epoch.

    Returns
    -------
    apen : ndarray, (n_channels,)
        ApEn per channel, averaged across epochs.
    """
    n_epochs, n_chans, n_samples = epochs.shape
    apen_all = np.zeros((n_epochs, n_chans))

    for ep in range(n_epochs):
        for ch in range(n_chans):
            x = epochs[ep, ch]
            r = r_factor * np.std(x)
            if r == 0:
                apen_all[ep, ch] = 0.0
                continue
            apen_all[ep, ch] = _apen_1d(x, m, r)

    return np.mean(apen_all, axis=0)


def _apen_1d(x, m, r):
    """ApEn for a single 1D signal."""
    N = len(x)

    def _phi(mm):
        templates = np.array([x[i:i + mm] for i in range(N - mm + 1)])
        count = 0
        total = len(templates)
        for i in range(total):
            dist = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(dist < r) - 1  # exclude self-match
        if count == 0:
            return 0.0
        return np.log(count / (total * (total - 1)))

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    return max(phi_m - phi_m1, 0.0)


def eeg_bsr(epochs, thresholds=(5, 10, 20)):
    """Burst Suppression Ratio: percentage of epochs below threshold amplitude.

    Parameters
    ----------
    epochs : ndarray, (n_epochs, n_chans, epoch_samples)
    thresholds : tuple of float
        Amplitude thresholds in µV.

    Returns
    -------
    bsr : ndarray, (n_channels, len(thresholds))
        Percentage [0, 100] of suppressed epochs per threshold.
    """
    n_epochs, n_chans = epochs.shape[:2]
    thresholds = np.asarray(thresholds)
    ptp = np.ptp(epochs, axis=-1)  # (n_epochs, n_chans)
    bsr = np.zeros((n_chans, len(thresholds)))

    for i, thr in enumerate(thresholds):
        bsr[:, i] = np.sum(ptp < thr, axis=0) / n_epochs * 100.0

    return bsr
