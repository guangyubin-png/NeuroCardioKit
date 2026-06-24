"""Frequency-domain HRV features — thin wrapper around hrvanalysis."""
from hrvanalysis.extract_features import get_frequency_domain_features


def freq_domain(nn, fs_nn=4, method='welch'):
    """
    Compute frequency-domain features for a single NN segment.

    Wraps hrvanalysis.get_frequency_domain_features().
    Bands follow Task Force 1996: VLF 0.003-0.04, LF 0.04-0.15, HF 0.15-0.4 Hz.

    Args:
        nn: 1D array of NN intervals (ms).
        fs_nn: Resampling frequency for NN interpolation (Hz). Default 4.
        method: PSD method, 'welch' or 'lomb'.

    Returns:
        Dict: total_power, vlf, lf, hf, lf_hf_ratio, lfnu, hfnu.
    """
    if len(nn) < 3:
        return {}
    raw = get_frequency_domain_features(
        nn.tolist(),
        method=method,
        sampling_frequency=fs_nn,
    )
    return {
        'total_power': raw['total_power'],
        'vlf': raw['vlf'],
        'lf': raw['lf'],
        'hf': raw['hf'],
        'lf_hf_ratio': raw['lf_hf_ratio'],
        'lfnu': raw['lfnu'],
        'hfnu': raw['hfnu'],
    }
