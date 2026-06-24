"""Windowing and cross-segment summary utilities."""
import numpy as np


def split_windows(nn, nn_time=None, segment_len_s=300, step_s=None,
                  min_beats=10, total_s=None):
    """
    Split a full-length NN sequence into fixed-duration windows.

    Window boundaries are defined by absolute time (nn_time) when available;
    otherwise fall back to cumulative sum of NN intervals.

    Args:
        nn: 1D array of NN intervals (ms).
        nn_time: 1D array of absolute start times (s), same length as nn.
                 If None, use cumsum(nn)/1000.
        segment_len_s: Window length in seconds (default 300 = 5 min).
        step_s: Step between consecutive windows in seconds.
                None (default) = contiguous (step = segment_len_s).
        min_beats: Minimum number of beats per window. Windows with fewer
                   are returned as empty arrays when total_s is set.
        total_s: Total duration in seconds to generate windows for.
                 If provided, windows are generated from 0 to total_s
                 regardless of nn_time range. Windows with < min_beats
                 become empty lists (not dropped).

    Returns:
        List of 1D numpy arrays, each containing the NN intervals of one window.
    """
    if len(nn) == 0:
        return []

    if step_s is None:
        step_s = segment_len_s

    if step_s <= 0:
        raise ValueError("step must be positive")

    if nn_time is not None and len(nn_time) == len(nn):
        time_s = np.asarray(nn_time, dtype=float)
    else:
        time_s = np.cumsum(nn) / 1000.0

    if total_s is None:
        total_s = time_s[-1] + nn[-1] / 1000.0

    windows = []
    start_s = 0.0
    while start_s + segment_len_s <= total_s:
        end_s = start_s + segment_len_s
        mask = (time_s >= start_s) & (time_s < end_s)
        seg = nn[mask]
        if len(seg) >= min_beats or total_s is not None:
            windows.append(seg if len(seg) >= min_beats else np.array([], dtype=float))
        start_s += step_s

    return windows


def summarize(feature_list):
    """
    Compute mean, std, min, max across a list of per-window feature dicts.

    Args:
        feature_list: list of dicts, each with the same set of numeric keys.

    Returns:
        Flat dict: {key}_mean, {key}_std, {key}_min, {key}_max for each key.
    """
    if not feature_list:
        return {}

    keys = [k for k in feature_list[0] if isinstance(feature_list[0][k], (int, float))]
    result = {}
    for key in keys:
        vals = [d[key] for d in feature_list if key in d]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        if not vals:
            continue
        result[f'{key}_mean'] = float(np.mean(vals))
        result[f'{key}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        result[f'{key}_min'] = float(np.min(vals))
        result[f'{key}_max'] = float(np.max(vals))
    return result


def flatten_result(result):
    """
    Flatten the nested dict from hrv_analyze into a single-level dict.

    Useful for CSV export, feature vectors, or database insertion.

    Naming convention:
        global_{feat}: value
        seg{i}_{feat}: value

    Args:
        result: Dict returned by hrv_analyze.

    Returns:
        Flat dict with string keys and scalar values.
    """
    flat = {}

    # Metadata
    flat['n_beats'] = int(result.get('n_beats', 0))
    flat['duration_min'] = float(result.get('duration_min', 0.0))

    # Global features
    gl = result.get('global')
    if gl is not None:
        X = gl.get('feature_matrix')
        names = gl.get('feat_names')
        if X is not None and names is not None and X.shape[0] > 0:
            for fi, name in enumerate(names):
                flat[f'global_{name}'] = float(X[0, fi])

    # Segment features
    seg = result.get('segments')
    if seg is not None:
        flat['n_segments'] = int(seg.get('n_segments', 0))
        flat['segment_len_s'] = float(seg.get('segment_len_s', 0))
        X = seg.get('feature_matrix')
        names = seg.get('feat_names')
        if X is not None and names is not None:
            for i in range(X.shape[0]):
                for fi, name in enumerate(names):
                    flat[f'seg{i}_{name}'] = float(X[i, fi])

    return flat
