"""EEG utilities — segment splitting and flat dict export."""

import numpy as np


def eeg_split_segments(data, fs, segment_len_s=30, step_s=None, t_start=0):
    """Split continuous EEG into equal-length segments by absolute time.

    Parameters
    ----------
    data : ndarray, (n_chans, n_samples) or (n_samples,)
    fs : float
        Sampling rate in Hz.
    segment_len_s : float
        Segment length in seconds.
    step_s : float or None
        Step between segment starts in seconds. None = no overlap (step = segment_len_s).
    t_start : float
        Absolute start time of the first sample.

    Returns
    -------
    segments : list of (segment_data, segment_t_start)
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]

    seg_samples = int(round(segment_len_s * fs))
    n_total = data.shape[-1]
    if n_total < seg_samples:
        return [(data, t_start)]

    if step_s is None:
        step_samples = seg_samples
    else:
        step_samples = int(round(step_s * fs))

    n_segments = (n_total - seg_samples) // step_samples + 1

    segments = []
    for i in range(n_segments):
        start = i * step_samples
        seg_data = data[:, start:start + seg_samples]
        seg_t = t_start + start / fs
        segments.append((seg_data, seg_t))

    return segments


def eeg_flatten(result):
    """Flatten eeg_analyze result to a single-level dict of scalars.

    Each key follows the pattern: {feat_name}_seg{seg}

    Parameters
    ----------
    result : dict
        Output from eeg_analyze().

    Returns
    -------
    flat : dict
        Mapping from feature_segment keys to scalar values.
    """
    flat = {}
    seg = result.get('segments')
    if seg is None:
        return flat
    X = seg.get('feature_matrix')
    names = seg.get('feat_names')
    if X is None or names is None:
        return flat

    n_segments, n_features = X.shape
    for i in range(n_segments):
        for fi, name in enumerate(names):
            key = f'{name}_seg{i}'
            flat[key] = float(X[i, fi])

    return flat
