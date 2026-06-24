"""neurocardiokit.eeg — General-purpose EEG analysis library.

Public API:
    eeg_analyze(data, fs, ...) → dict    — main analysis entry point
    eeg_flatten(result) → dict            — flatten to single-level dict for CSV/SQL
    BANDS                                 — default frequency band definitions

Internal modules (importable for step-by-step use):
    eeg_preprocess, eeg_epoch, eeg_clean_epochs, eeg_bad_channels,
    eeg_psd, eeg_band_power, eeg_coherence, eeg_apen, eeg_bsr,
    eeg_split_segments
"""

import numpy as np

from neurocardiokit.eeg._preprocess import eeg_preprocess
from neurocardiokit.eeg._epoch import eeg_epoch, eeg_clean_epochs, eeg_bad_channels
from neurocardiokit.eeg._spectral import eeg_psd, eeg_band_power, BANDS, BSR_THRESHOLDS
from neurocardiokit.eeg._connectivity import eeg_cpsd, eeg_coherence, _cpsd_to_coherence
from neurocardiokit.eeg._nonlinear import eeg_apen, eeg_bsr
from neurocardiokit.eeg._utils import eeg_split_segments, eeg_flatten

# All feature classes that may appear in feature_matrix
_FEATURE_CLASSES = [
    'band_power', 'band_power_rel', 'bsr', 'apen', 'coherence',
]


def eeg_analyze(
    data,
    fs,
    t_start=0,
    channel_names=None,
    segment_len=30,
    segment_overlap=None,
    epoch_len=2,
    epoch_overlap=None,
    bands=None,
    preprocess=True,
    clean=True,
    min_amp=1.0,
    max_amp=300.0,
    feat_include=None,
):
    """End-to-end EEG segment analysis.

    Pipeline: preprocess → split segments → epoch → clean → PSD/bands/coherence/ApEn/BSR
              → concatenate into (n_segments, n_features) feature matrix.

    Parameters
    ----------
    data : ndarray, (n_chans, n_samples) or (n_samples,)
        Raw EEG in µV.
    fs : float
        Sampling rate in Hz.
    t_start : float
        Absolute start time of the first sample (seconds).
    channel_names : list of str or None
        Channel names, passed through to result.
    segment_len : float
        Segment length in seconds (default 30, AASM standard).
    segment_overlap : float or None
        Segment step in seconds. None = no overlap (step = segment_len).
    epoch_len : float
        Epoch length within each segment (default 2s).
    epoch_overlap : float or None
        Epoch step in seconds. None = no overlap (step = epoch_len).
    bands : dict or None
        {name: (low, high)}. None uses clinical-standard BANDS.
    preprocess : bool
        If True, apply 0.5-45 Hz bandpass before analysis.
    clean : bool
        If True, reject artifact epochs.
    min_amp, max_amp : float
        Allowed peak-to-peak range in µV for epoch cleaning.
    feat_include : list of str or None
        Feature classes to compute. None = all classes.
        Choices: 'band_power', 'band_power_rel', 'coherence', 'apen', 'bsr'.

    Returns
    -------
    result : dict
        Top-level keys: fs, n_channels, channel_names, epoch_len_s,
        epoch_overlap_s, t_start, global (None), segments.
        segments contains: segment_len_s, segment_overlap_s, n_segments,
        segment_t_starts, n_valid_epochs, feature_matrix, feat_names.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    n_chans = data.shape[0]

    if bands is None:
        bands = BANDS

    # Step 1: preprocess
    if preprocess:
        data = eeg_preprocess(data, fs, lowcut=0.5, highcut=45.0)

    # Step 2: split into segments
    segments = eeg_split_segments(data, fs, segment_len, step_s=segment_overlap, t_start=t_start)
    n_segments = len(segments)

    # Determine which feature classes are active
    active = _resolve_active_classes(feat_include)
    feat_names = _build_feat_names(active, n_chans, bands, BSR_THRESHOLDS, channel_names)
    n_features = len(feat_names)

    # Allocate output
    feature_matrix = np.zeros((n_segments, n_features), dtype=np.float64)
    segment_t_starts = np.zeros(n_segments)
    n_valid_epochs = np.zeros(n_segments, dtype=int)

    # Step 3: per-segment analysis
    for seg_idx, (seg_data, seg_t) in enumerate(segments):
        segment_t_starts[seg_idx] = seg_t

        # Epoch
        epochs, _ = eeg_epoch(seg_data, epoch_len, step_s=epoch_overlap, fs=fs, detrend=True)

        if len(epochs) == 0:
            continue

        # Clean
        if clean:
            epochs, _ = eeg_clean_epochs(epochs, min_amp=min_amp, max_amp=max_amp)
        n_valid_epochs[seg_idx] = len(epochs)

        if len(epochs) == 0:
            continue

        # Compute all features and fill the row
        row = _compute_segment_features(active, epochs, fs, bands, BSR_THRESHOLDS)
        feature_matrix[seg_idx] = row

    # Assemble result
    result = {
        'fs': fs,
        'n_channels': n_chans,
        'channel_names': channel_names,
        'epoch_len_s': epoch_len,
        'epoch_overlap_s': epoch_overlap,
        't_start': t_start,

        'global': None,

        'segments': {
            'segment_len_s': segment_len,
            'segment_overlap_s': segment_overlap,
            'n_segments': n_segments,
            'segment_t_starts': segment_t_starts,
            'n_valid_epochs': n_valid_epochs,
            'feature_matrix': feature_matrix,
            'feat_names': feat_names,
        },
    }
    return result


def _resolve_active_classes(feat_include):
    """Determine which feature classes go into feature_matrix."""
    if feat_include is None:
        return list(_FEATURE_CLASSES)
    return [c for c in feat_include if c in _FEATURE_CLASSES]


def _build_feat_names(active, n_chans, bands, bsr_thresholds, channel_names):
    """Build the list of feature column names."""
    band_names = list(bands.keys())
    names = []
    for cls in active:
        if cls == 'band_power':
            for ch in range(n_chans):
                for bn in band_names:
                    names.append(f'bp_{bn}_ch{ch}')
        elif cls == 'band_power_rel':
            for ch in range(n_chans):
                for bn in band_names:
                    names.append(f'bpr_{bn}_ch{ch}')
        elif cls == 'bsr':
            for ch in range(n_chans):
                for thr in bsr_thresholds:
                    names.append(f'bsr_{thr}uv_ch{ch}')
        elif cls == 'apen':
            for ch in range(n_chans):
                names.append(f'apen_ch{ch}')
        elif cls == 'coherence':
            for i in range(n_chans):
                for j in range(i + 1, n_chans):
                    for bn in band_names:
                        names.append(f'coh_{bn}_ch{i}_ch{j}')
    return names


def _compute_segment_features(active, epochs, fs, bands, bsr_thresholds):
    """Compute all features for one segment, return a 1D row array."""
    need_psd = {'band_power', 'band_power_rel'} & set(active)
    need_coh = 'coherence' in active

    # Compute PSD.  When coherence is also needed, compute CPSD once and
    # derive PSD from its diagonal — avoids a second FFT pass.
    if need_psd and need_coh:
        cpsd, freqs = eeg_cpsd(epochs, fs)
        psd = cpsd.diagonal(axis1=0, axis2=1).T  # (n_chans, n_freqs)
    elif need_psd:
        psd, freqs = eeg_psd(epochs, fs)

    if need_psd:
        bp, _ = eeg_band_power(psd, freqs, bands)
        total_power = np.sum(bp, axis=-1, keepdims=True)
        bpr = np.zeros_like(bp)
        mask = total_power[:, 0] > 0
        bpr[mask] = bp[mask] / total_power[mask] * 100.0

    parts = []
    for cls in active:
        if cls == 'band_power':
            parts.append(bp.ravel())
        elif cls == 'band_power_rel':
            parts.append(bpr.ravel())
        elif cls == 'bsr':
            bsr_val = eeg_bsr(epochs, bsr_thresholds)
            parts.append(bsr_val.ravel())
        elif cls == 'apen':
            parts.append(eeg_apen(epochs))
        elif cls == 'coherence':
            if not (need_psd and need_coh):
                cpsd, freqs = eeg_cpsd(epochs, fs)
            coh, _ = _cpsd_to_coherence(cpsd, freqs, bands)
            n_chans = coh.shape[0]
            for i in range(n_chans):
                for j in range(i + 1, n_chans):
                    parts.append(coh[i, j, :])

    return np.concatenate(parts) if parts else np.array([])
