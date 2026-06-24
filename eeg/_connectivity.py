"""EEG functional connectivity — magnitude-squared coherence."""

import numpy as np


def eeg_cpsd(epochs, fs):
    """Compute cross power spectral density between all channel pairs.

    Parameters
    ----------
    epochs : ndarray, (n_epochs, n_chans, epoch_samples)
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    cpsd : ndarray, (n_chans, n_chans, n_freqs)
    freqs : ndarray, (n_freqs,)
    """
    n_epochs, n_chans, n_samples = epochs.shape
    n_freqs = n_samples // 2 + 1

    # Hamming window
    win = np.hamming(n_samples)
    U = np.sum(win ** 2)

    cpsd = np.zeros((n_chans, n_chans, n_freqs))
    for ep in range(n_epochs):
        x = epochs[ep] * win  # (n_chans, n_samples)
        fft = np.fft.rfft(x, axis=-1)
        for i in range(n_chans):
            for j in range(i, n_chans):
                val = fft[i] * np.conj(fft[j]) * 2.0 / (U * fs)
                cpsd[i, j] += val.real

    cpsd /= n_epochs
    # Fill lower triangle
    for i in range(n_chans):
        for j in range(i + 1, n_chans):
            cpsd[j, i] = cpsd[i, j]

    freqs = np.fft.rfftfreq(n_samples, 1.0 / fs)
    return cpsd, freqs


def eeg_coherence(epochs, fs, bands=None):
    """Magnitude-squared coherence between all channel pairs, per band."""
    from neurocardiokit.eeg._spectral import BANDS

    if bands is None:
        bands = BANDS

    cpsd, freqs = eeg_cpsd(epochs, fs)
    return _cpsd_to_coherence(cpsd, freqs, bands)


def _cpsd_to_coherence(cpsd, freqs, bands):
    """Convert CPSD to magnitude-squared coherence per band.

    cpsd : (n_chans, n_chans, n_freqs)
    Returns coh : (n_chans, n_chans, n_bands), band_names
    """
    n_chans = cpsd.shape[0]
    band_names = list(bands.keys())
    coh = np.zeros((n_chans, n_chans, len(band_names)))

    for bi, (lo, hi) in enumerate(bands.values()):
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            continue
        for i in range(n_chans):
            for j in range(i, n_chans):
                num = np.abs(cpsd[i, j, mask].mean())
                den_i = np.abs(cpsd[i, i, mask].mean())
                den_j = np.abs(cpsd[j, j, mask].mean())
                den = np.sqrt(den_i * den_j)
                val = (num / den) ** 2 if den > 0 else 0.0
                coh[i, j, bi] = val
                coh[j, i, bi] = val

    return coh, band_names
