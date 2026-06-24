"""
HRV Analysis Toolbox — neurocardiokit/hrv/

Single unified entry point:
    hrv_analyze — global + segment HRV features from R-peak positions or NN intervals

Built on top of hrvanalysis (v1.0.5) for single-segment feature extraction.
Output structure mirrors neurocardiokit.eeg: global + segments, each with feature_matrix + feat_names.
"""
import numpy as np

from neurocardiokit.hrv._preprocess import rpos_to_nn
from neurocardiokit.hrv._time_domain import time_domain, cross_segment_features
from neurocardiokit.hrv._freq_domain import freq_domain
from neurocardiokit.hrv._nonlinear import nonlinear as _nonlinear_features
from neurocardiokit.hrv._utils import split_windows, flatten_result

_FEATURE_CLASSES = ['time', 'freq', 'nonlinear']

# Ordered feature names per class — defines column order in feature_matrix
_TIME_NAMES = [
    'mean_nn', 'sdnn', 'rmssd', 'sdsd', 'nn50', 'pnn50', 'nn20', 'pnn20',
    'mean_hr', 'min_hr', 'max_hr', 'std_hr', 'cvnn', 'cvsd', 'median_nn', 'range_nn',
]
_FREQ_NAMES = [
    'total_power', 'vlf', 'lf', 'hf', 'lf_hf_ratio', 'lfnu', 'hfnu',
]
_NONLINEAR_NAMES = [
    'sd1', 'sd2', 'sd_ratio', 'sampen', 'csi', 'cvi', 'modified_csi',
    'apen', 'dfa_alpha1', 'dfa_alpha2',
]

_CLASS_NAMES = {
    'time': _TIME_NAMES,
    'freq': _FREQ_NAMES,
    'nonlinear': _NONLINEAR_NAMES,
}


def _resolve_active_classes(feat_include):
    if feat_include is None:
        return list(_FEATURE_CLASSES)
    return [c for c in feat_include if c in _FEATURE_CLASSES]


def _build_feat_names(active):
    """Build ordered list of feature column names for the active classes."""
    names = []
    for cls in _FEATURE_CLASSES:
        if cls in active:
            names.extend(_CLASS_NAMES[cls])
    return names


def _dict_to_row(feat_dict, names):
    """Convert a feature dict to a 1D array in the given name order."""
    return np.array([feat_dict.get(n, np.nan) for n in names], dtype=float)


def _compute_global_matrix(nn, fs_nn, active, feat_names):
    """Compute global features as a (1, n_features) matrix."""
    row_parts = []
    for cls in _FEATURE_CLASSES:
        if cls not in active:
            continue
        if cls == 'time':
            d = time_domain(nn)
            row_parts.append(_dict_to_row(d, _CLASS_NAMES[cls]))
        elif cls == 'freq':
            d = freq_domain(nn, fs_nn=fs_nn)
            row_parts.append(_dict_to_row(d, _CLASS_NAMES[cls]))
        elif cls == 'nonlinear':
            d = _nonlinear_features(nn)
            row_parts.append(_dict_to_row(d, _CLASS_NAMES[cls]))

    if row_parts:
        return np.concatenate(row_parts)[None, :]  # (1, n_features)
    return np.empty((1, 0))


def _compute_segments_result(nn, nn_time, fs_nn, segment_len, segment_overlap,
                              active, feat_names, total_s=None):
    """Compute per-segment features → feature_matrix + metadata."""
    step_s = segment_len - segment_overlap if segment_overlap else None
    windows = split_windows(nn, nn_time=nn_time, segment_len_s=segment_len,
                            step_s=step_s, total_s=total_s)

    n_segments = len(windows)
    n_features = len(feat_names)
    feature_matrix = np.full((n_segments, n_features), np.nan, dtype=float)

    # Build segment_t_starts
    segment_t_starts = np.zeros(n_segments)
    start_s = 0.0
    eff_step = step_s if step_s is not None else segment_len
    for seg_idx in range(n_segments):
        segment_t_starts[seg_idx] = start_s
        start_s += eff_step

    # Fill per-window rows; empty windows stay NaN
    for i, seg in enumerate(windows):
        if len(seg) < 2:
            continue
        row_parts = []
        for cls in _FEATURE_CLASSES:
            if cls not in active:
                continue
            if cls == 'time':
                d = time_domain(seg)
                row_parts.append(_dict_to_row(d, _CLASS_NAMES[cls]))
            elif cls == 'freq':
                d = freq_domain(seg, fs_nn=fs_nn)
                row_parts.append(_dict_to_row(d, _CLASS_NAMES[cls]))
            elif cls == 'nonlinear':
                d = _nonlinear_features(seg)
                row_parts.append(_dict_to_row(d, _CLASS_NAMES[cls]))
        if row_parts:
            feature_matrix[i] = np.concatenate(row_parts)

    return {
        'segment_len_s': segment_len,
        'segment_overlap_s': segment_overlap,
        'n_segments': n_segments,
        'segment_t_starts': segment_t_starts,
        'feature_matrix': feature_matrix,
        'feat_names': feat_names,
    }


def hrv_analyze(
    rpos=None,
    nn=None,
    anntype=None,
    fs_ecg=250,
    t_start=0,
    nn_time=None,
    fs_nn=4,
    segment_len=300,
    segment_overlap=None,
    total_s=None,
    feat_include=None,
):
    """
    Unified HRV analysis — global and/or segment-based features.

    Accepts either R-peak positions (rpos) or pre-computed NN intervals (nn).

    Args:
        rpos: 1D array of R-peak sample indices. Mutually exclusive with nn.
        nn: 1D array of pre-computed NN intervals (ms). Mutually exclusive with rpos.
        anntype: 1D array of beat type labels (1 = normal). Only with rpos.
        fs_ecg: ECG sampling rate in Hz. Only with rpos.
        t_start: Absolute start time of first R-peak in seconds.
        nn_time: 1D array of absolute start times (s) per NN. Only with nn.
        fs_nn: Resampling frequency for frequency-domain analysis (Hz).
        segment_len: Segment length in seconds. None = global only.
        segment_overlap: Overlap between consecutive segments in seconds.
                         None = contiguous (no overlap).
        total_s: Total recording duration in seconds. When set, segments span
                 exactly [0, total_s) regardless of NN coverage. Windows with
                 insufficient beats are filled with NaN.
        feat_include: List of feature classes to compute. None = all.
                      Choices: 'time', 'freq', 'nonlinear'.

    Returns:
        Dict with keys: nn, nn_time, n_beats, duration_min, global, segments.
        global and segments each contain feature_matrix + feat_names.
    """
    if rpos is not None and nn is not None:
        raise ValueError("Provide either rpos or nn, not both.")
    if rpos is None and nn is None:
        raise ValueError("Either rpos or nn must be provided.")

    active = _resolve_active_classes(feat_include)
    feat_names = _build_feat_names(active)

    if rpos is not None:
        nn, nn_time = rpos_to_nn(rpos, anntype=anntype, fs=fs_ecg, t_start=t_start)
    else:
        nn = np.asarray(nn, dtype=float)
        if nn_time is not None:
            nn_time = np.asarray(nn_time, dtype=float)

    result = {
        'nn': nn,
        'nn_time': nn_time if nn_time is not None else np.array([]),
        'n_beats': len(nn),
        'duration_min': float(np.sum(nn) / 60000.0) if len(nn) > 0 else 0.0,
    }

    # Global — always computed
    if len(nn) >= 2:
        result['global'] = {
            'feature_matrix': _compute_global_matrix(nn, fs_nn, active, feat_names),
            'feat_names': feat_names,
        }
    else:
        result['global'] = {
            'feature_matrix': np.empty((1, len(feat_names))),
            'feat_names': feat_names,
        }

    # Segments
    if segment_len is not None:
        if len(nn) >= 2:
            result['segments'] = _compute_segments_result(
                nn, nn_time, fs_nn, segment_len, segment_overlap,
                active, feat_names, total_s=total_s,
            )
        else:
            n_features = len(feat_names)
            result['segments'] = {
                'segment_len_s': segment_len,
                'segment_overlap_s': segment_overlap,
                'n_segments': 0,
                'segment_t_starts': np.array([]),
                'feature_matrix': np.empty((0, n_features)),
                'feat_names': feat_names,
            }

    return result


__all__ = ['hrv_analyze', 'flatten_result',
           'rpos_to_nn', 'time_domain', 'freq_domain', 'nonlinear',
           'split_windows']
