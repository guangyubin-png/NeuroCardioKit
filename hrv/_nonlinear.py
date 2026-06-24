"""Nonlinear HRV features — Poincare, entropy (hrvanalysis) + DFA + ApEn (self-implemented)."""
import numpy as np
from hrvanalysis.extract_features import (
    get_poincare_plot_features,
    get_sampen,
    get_csi_cvi_features,
)


def poincare(nn):
    """Poincaré plot features: sd1, sd2, sd_ratio."""
    if len(nn) < 2:
        return {}
    raw = get_poincare_plot_features(nn.tolist())
    return {
        'sd1': raw['sd1'],
        'sd2': raw['sd2'],
        'sd_ratio': raw['ratio_sd2_sd1'],
    }


def sampen(nn):
    """Sample entropy (wraps hrvanalysis, which uses nolds)."""
    if len(nn) < 4:
        return {}
    raw = get_sampen(nn.tolist())
    return {'sampen': raw['sampen']}


def csi_cvi(nn):
    """Cardiac Sympathetic Index and Cardiac Vagal Index."""
    if len(nn) < 2:
        return {}
    raw = get_csi_cvi_features(nn.tolist())
    return {
        'csi': raw['csi'],
        'cvi': raw['cvi'],
        'modified_csi': raw['Modified_csi'],
    }


# ---- Self-implemented nonlinear features ----


def apen(nn, m=2, r=None):
    """
    Approximate Entropy.

    Args:
        nn: 1D array of NN intervals.
        m: Embedding dimension.
        r: Tolerance (default: 0.2 * std(nn)).

    Returns:
        Dict with key 'apen'.
    """
    N = len(nn)
    if N < m + 2:
        return {}

    if r is None:
        r = 0.2 * np.std(nn)

    def _phi(mm):
        templates = np.array([nn[i:i + mm] for i in range(N - mm + 1)])
        count = 0
        for i in range(len(templates)):
            dist = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(dist <= r) - 1  # exclude self-match
        return count / (N - mm + 1)

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    if phi_m == 0 or phi_m1 == 0:
        return {'apen': 0.0}

    return {'apen': float(np.log(phi_m) - np.log(phi_m1))}


def dfa(nn, scales=None):
    """
    Detrended Fluctuation Analysis.

    Returns alpha1 (short-term, 4-16 beats) and alpha2 (long-term, 16-64 beats).

    Args:
        nn: 1D array of NN intervals.
        scales: List of (min, max) tuples defining scale ranges.
                Default: [(4, 16), (16, 64)].

    Returns:
        Dict: dfa_alpha1, dfa_alpha2.
    """
    N = len(nn)
    if N < 64:
        return {}

    if scales is None:
        scales = [(4, 16), (16, 64)]

    # Integrated signal
    y = np.cumsum(nn - np.mean(nn))

    def _fluctuation(scale_min, scale_max):
        ns = np.arange(scale_min, min(scale_max + 1, N // 2 + 1))
        fluct = np.zeros(len(ns))
        for idx, n in enumerate(ns):
            n_segments = N // n
            rms_sum = 0.0
            for k in range(n_segments):
                seg = y[k * n:(k + 1) * n]
                x = np.arange(len(seg))
                fit = np.polyfit(x, seg, 1)
                trend = np.polyval(fit, x)
                rms_sum += np.mean((seg - trend) ** 2)
            fluct[idx] = np.sqrt(rms_sum / n_segments)
        # log-log slope
        log_n = np.log10(ns)
        log_f = np.log10(fluct)
        slope = np.polyfit(log_n, log_f, 1)[0]
        return float(slope)

    result = {}
    for idx, (smin, smax) in enumerate(scales, 1):
        result[f'dfa_alpha{idx}'] = _fluctuation(smin, smax)

    return result


def nonlinear(nn):
    """All nonlinear features combined."""
    result = {}
    result.update(poincare(nn))
    result.update(sampen(nn))
    result.update(csi_cvi(nn))
    result.update(apen(nn))
    result.update(dfa(nn))
    return result
