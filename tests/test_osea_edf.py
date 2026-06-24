"""
neurocardiokit.osea — EDF-based integration test.

Reads an ECG channel from a sleep EDF, runs OSEA QRS detection,
and reports beat counts / heart rate.

Usage:
    python tests\test_osea_edf.py
"""

import os
import sys
import numpy as np

from neurocardiokit.osea import OseaEngine, ANNOTYPE_LABELS


EDF_PATH = r'E:\DataBase\physionet2026\sub-I0002150001401_ses-2.edf'
ECG_CHANNEL_NAME = 'EKG'  # common label in sleep EDF; adjust if needed
EDF_FS = 500.0
OSEA_FS = 200.0
OSEA_ADC_PER_MV = 200    # 200 ADC units = 1 mV
READ_DURATION_S = 60      # read 1 minute
READ_OFFSET_S = 600       # skip first 10 min


def _find_channel_index(f, label):
    for i in range(f.signals_in_file):
        if f.getLabel(i) == label:
            return i
    return -1


def _read_ecg_from_edf(edf_path, ecg_label, offset_sec, duration_sec, fs_edf):
    """Read an ECG channel from EDF, return (signal_uv, fs_edf) or None."""
    import pyedflib

    f = pyedflib.EdfReader(edf_path)
    try:
        ch_idx = _find_channel_index(f, ecg_label)
        if ch_idx < 0:
            # list available channels for diagnostics
            labels = [f.getLabel(i) for i in range(f.signals_in_file)]
            print(f"Channel '{ecg_label}' not found. Available: {labels}")
            return None

        n = int(duration_sec * fs_edf)
        offset = int(offset_sec * fs_edf)
        sig = f.readSignal(ch_idx, start=offset, n=n).astype(np.float64)
        return sig
    finally:
        f.close()


def _resample_ecg(sig, fs_src, fs_dst):
    """Resample ECG from fs_src to fs_dst using linear interpolation."""
    from scipy import interpolate
    n_src = len(sig)
    n_dst = int(n_src * fs_dst / fs_src)
    t_src = np.arange(n_src) / fs_src
    t_dst = np.arange(n_dst) / fs_dst
    f = interpolate.interp1d(t_src, sig, kind='linear', copy=False,
                             bounds_error=False, fill_value='extrapolate')
    return f(t_dst)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_osea_from_edf_detects_beats():
    sig_uv = _read_ecg_from_edf(EDF_PATH, ECG_CHANNEL_NAME,
                                READ_OFFSET_S, READ_DURATION_S, EDF_FS)
    if sig_uv is None:
        raise OSError(f"ECG channel not available in {EDF_PATH}")

    # Convert µV → OSEA ADC units (200 units/mV), plus center around 0
    sig_adc = sig_uv * (OSEA_ADC_PER_MV / 1000.0)  # µV → mV → ADC
    sig_adc -= np.mean(sig_adc)                     # baseline ~0

    # Resample 500 Hz → 200 Hz
    sig_200hz = _resample_ecg(sig_adc, EDF_FS, OSEA_FS)
    sig_int = sig_200hz.astype(int)

    engine = OseaEngine()
    results = engine.process_array(sig_int)

    assert len(results) > 0, "No beats detected in ECG signal"
    bpm = len(results) / (READ_DURATION_S / 60.0)
    print(f"  Detected {len(results)} beats, ~{bpm:.1f} BPM")


def test_osea_from_edf_beat_types():
    sig_uv = _read_ecg_from_edf(EDF_PATH, ECG_CHANNEL_NAME,
                                READ_OFFSET_S, READ_DURATION_S, EDF_FS)
    if sig_uv is None:
        raise OSError(f"ECG channel not available in {EDF_PATH}")

    sig_adc = sig_uv * (OSEA_ADC_PER_MV / 1000.0)
    sig_adc -= np.mean(sig_adc)
    sig_200hz = _resample_ecg(sig_adc, EDF_FS, OSEA_FS)

    engine = OseaEngine()
    results = engine.process_array(sig_200hz.astype(int))

    type_counts = {}
    for r in results:
        label = r['type_label']
        type_counts[label] = type_counts.get(label, 0) + 1

    print(f"  Beat types: {type_counts}")

    normal = type_counts.get('NORMAL', 0)
    assert normal > len(results) * 0.3, \
        f"Expected >30% NORMAL beats, got {normal}/{len(results)}"


def test_osea_from_edf_resample_quality():
    """Verify resample preserves ~same BPM within tolerance."""
    sig_uv = _read_ecg_from_edf(EDF_PATH, ECG_CHANNEL_NAME,
                                READ_OFFSET_S, READ_DURATION_S, EDF_FS)
    if sig_uv is None:
        raise OSError(f"ECG channel not available in {EDF_PATH}")

    sig_adc = sig_uv * (OSEA_ADC_PER_MV / 1000.0)
    sig_adc -= np.mean(sig_adc)

    # 500 Hz detection
    engine_500 = OseaEngine()
    r_500 = engine_500.process_array(sig_adc.astype(int))

    # 200 Hz detection
    sig_200hz = _resample_ecg(sig_adc, EDF_FS, OSEA_FS)
    engine_200 = OseaEngine()
    r_200 = engine_200.process_array(sig_200hz.astype(int))

    assert len(r_200) > 0
    ratio = len(r_200) / max(len(r_500), 1)
    print(f"  Beats: {len(r_500)} @500Hz vs {len(r_200)} @200Hz (ratio={ratio:.2f})")
    assert 0.5 <= ratio <= 1.5, \
        f"Resample changed beat count too much: {len(r_500)} → {len(r_200)}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import traceback

    # Quick check before running
    if not os.path.exists(EDF_PATH):
        print(f"EDF file not found: {EDF_PATH}")
        print("Update EDF_PATH in this file to point to a valid sleep EDF.")
        sys.exit(1)

    try:
        import pyedflib
        import scipy
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]

    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f'  PASS  {fn.__name__}')
            passed += 1
        except Exception:
            print(f'  FAIL  {fn.__name__}')
            traceback.print_exc()
            failed += 1

    print(f'\n{passed} passed, {failed} failed out of {len(tests)} tests')
    sys.exit(0 if failed == 0 else 1)
