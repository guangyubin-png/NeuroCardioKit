"""Time-domain HRV features — thin wrapper around hrvanalysis."""
import numpy as np
from hrvanalysis.extract_features import get_time_domain_features


def time_domain(nn):
    """
    Compute time-domain features for a single NN segment.

    Wraps hrvanalysis.get_time_domain_features() with project-consistent key names.

    Args:
        nn: 1D array of NN intervals (ms).

    Returns:
        Dict with keys: mean_nn, sdnn, rmssd, sdsd, nn50, pnn50, nn20, pnn20,
        mean_hr, min_hr, max_hr, std_hr, cvnn, cvsd, median_nn, range_nn.
    """
    if len(nn) < 2:
        return {}
    raw = get_time_domain_features(nn.tolist())

    return {
        'mean_nn': raw['mean_nni'],
        'sdnn': raw['sdnn'],
        'rmssd': raw['rmssd'],
        'sdsd': raw['sdsd'],
        'nn50': raw['nni_50'],
        'pnn50': raw['pnni_50'],
        'nn20': raw['nni_20'],
        'pnn20': raw['pnni_20'],
        'mean_hr': raw['mean_hr'],
        'min_hr': raw['min_hr'],
        'max_hr': raw['max_hr'],
        'std_hr': raw['std_hr'],
        'cvnn': raw['cvnni'],
        'cvsd': raw['cvsd'],
        'median_nn': raw['median_nni'],
        'range_nn': raw['range_nni'],
    }


def cross_segment_features(segment_nn_list):
    """
    Compute cross-segment time-domain features: SDANN and SDNNI.

    These features only exist when you have multiple 5-min windows — they
    capture the variability of heart rate across the recording.

    Args:
        segment_nn_list: List of 1D numpy arrays, each a window's NN intervals.

    Returns:
        Dict with keys: sdann, sdnn_i, plus global mean_hr and cvnn.
    """
    if len(segment_nn_list) < 2:
        return {}

    # Per-window mean and std
    means = np.array([np.mean(seg) for seg in segment_nn_list])
    stds = np.array([np.std(seg, ddof=1) for seg in segment_nn_list])

    # Concatenate for global stats
    all_nn = np.concatenate(segment_nn_list)
    sdnn_global = np.std(all_nn, ddof=1)
    mean_nn_global = np.mean(all_nn)

    return {
        'sdann': float(np.std(means, ddof=1)),
        'sdnn_i': float(np.mean(stds)),
        'mean_hr': float(60000.0 / mean_nn_global),
        'cvnn': float(sdnn_global / mean_nn_global),
    }
