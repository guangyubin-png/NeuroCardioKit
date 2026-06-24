"""R-peak positions → clean NN interval sequence + absolute timestamps."""
import numpy as np
from hrvanalysis.preprocessing import get_nn_intervals


def rpos_to_nn(rpos, anntype=None, fs=250, t_start=0, low_rri=300, high_rri=2000):
    """
    Convert R-peak sample positions to clean NN intervals (ms) with absolute time.

    Steps:
      1. rpos → RR (ms) = (rpos[i] - rpos[i-1]) * 1000 / fs,
         absolute time = rpos[i-1] / fs + t_start
      2. [if anntype] keep only intervals between two normal beats
      3. hrvanalysis.get_nn_intervals: remove_outliers → interpolate
         → remove_ectopic_beats(malik) → interpolate
         (length preserved — outliers are interpolated, not dropped)

    Args:
        rpos: 1D array of R-peak sample indices (N+).
        anntype: 1D array of beat type labels (1 = normal). None to skip pre-filter.
        fs: ECG sampling rate in Hz.
        t_start: Absolute start time of the first R-peak in seconds.
        low_rri: Lower bound of plausible RR interval (ms).
        high_rri: Upper bound of plausible RR interval (ms).

    Returns:
        (nn, nn_time) tuple:
            nn:      1D float array of clean NN intervals (ms), length M.
            nn_time: 1D float array of absolute start times (s), length M.
                     nn_time[i] is the absolute instant of the R-peak at the start
                     of interval nn[i].
    """
    if len(rpos) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    rr_ms = np.diff(rpos).astype(float) * 1000.0 / fs
    # Absolute start time of each RR interval (the R-peak that starts the interval)
    time_rr = rpos[:-1].astype(float) / fs + t_start

    if anntype is not None:
        mask = (anntype[:-1] == 1) & (anntype[1:] == 1)
        rr_ms = rr_ms[mask]
        time_rr = time_rr[mask]

    if len(rr_ms) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # get_nn_intervals interpolates outliers → length preserved
    nn_list = get_nn_intervals(
        rr_ms.tolist(),
        low_rri=low_rri,
        high_rri=high_rri,
        ectopic_beats_removal_method='malik',
        verbose=False,
    )

    nn = np.array(nn_list, dtype=float)
    # Safeguard: drop any remaining NaN (should not happen after interpolation)
    valid = ~np.isnan(nn)
    return nn[valid], time_rr[valid]
