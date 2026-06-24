"""
neurocardiokit.osea — QRS detection integration tests.

Uses synthetic ECG to avoid external file dependencies.
Run:
    python tests/test_osea.py
"""
import numpy as np
import sys
import os

from neurocardiokit.osea import OseaEngine, ANNOTYPE_LABELS


def _synthetic_ecg(duration_sec=10, bpm=70, noise=8, seed=42):
    """Generate synthetic ECG (200 Hz, 200 ADC units/mV)."""
    rng = np.random.RandomState(seed)
    fs = 200
    n = int(duration_sec * fs)
    signal = np.zeros(n, dtype=np.float64)
    beat_interval = int(fs * 60.0 / bpm)
    t = beat_interval

    while t < n - 50:
        # P wave
        for i in range(-30, -15):
            idx = t + i
            if 0 <= idx < n:
                signal[idx] += 20 * np.sin(np.pi * (i + 30) / 15)
        # QRS
        for i in range(-6, 12):
            idx = t + i
            if 0 <= idx < n:
                signal[idx] += 180 * np.sin(np.pi * (i + 6) / 18)
        # T wave
        for i in range(18, 50):
            idx = t + i
            if 0 <= idx < n:
                signal[idx] += 50 * np.sin(np.pi * (i - 18) / 32)
        t += beat_interval

    signal += rng.randn(n) * noise
    return signal.astype(int)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_osea_detects_beats():
    ecg = _synthetic_ecg(duration_sec=30, bpm=70)
    engine = OseaEngine()
    results = engine.process_array(ecg)
    expected = 30 * 70 // 60
    assert 0.5 * expected <= len(results) <= 1.5 * expected, \
        f"Expected ~{expected} beats, got {len(results)}"


def test_osea_beat_types():
    ecg = _synthetic_ecg(duration_sec=20, bpm=70)
    engine = OseaEngine()
    results = engine.process_array(ecg)
    normal = sum(1 for r in results if r['type_label'] == 'NORMAL')
    assert normal > len(results) * 0.5, \
        f"Expected >50% NORMAL, got {normal}/{len(results)}"


def test_osea_result_keys():
    ecg = _synthetic_ecg(duration_sec=15, bpm=70)
    engine = OseaEngine()
    results = engine.process_array(ecg)
    assert len(results) > 0
    r = results[0]
    for key in ['index', 'delay', 'type', 'type_label', 'match']:
        assert key in r, f"Missing key: {key}"


def test_osea_reset():
    engine = OseaEngine()
    ecg = _synthetic_ecg(duration_sec=5, bpm=80)
    results1 = engine.process_array(ecg)
    results2 = engine.process_array(ecg)
    # Results should be similar after reset
    assert abs(len(results1) - len(results2)) <= 2, \
        f"Reset produces different counts: {len(results1)} vs {len(results2)}"


def test_osea_qrs_only():
    ecg = _synthetic_ecg(duration_sec=15, bpm=70)
    engine = OseaEngine()
    results = engine.process_array_qrs_only(ecg)
    assert len(results) >= 1, f"QRS-only mode: only {len(results)} detected"


def test_osea_detect_per_sample():
    ecg = _synthetic_ecg(duration_sec=15, bpm=70)
    engine = OseaEngine()
    engine.reset()
    detections = []
    for idx, sample in enumerate(ecg):
        delay = engine.detect(int(sample))
        if delay > 0:
            detections.append((idx, delay))
    assert engine.detection_count == len(detections)
    assert len(detections) > 0


def test_osea_properties():
    ecg = _synthetic_ecg(duration_sec=10, bpm=70)
    engine = OseaEngine()
    engine.process_array(ecg)
    assert engine.detection_count > 0
    assert engine.beat_type in ANNOTYPE_LABELS
    assert isinstance(engine.beat_type_label, str)
    assert len(engine.detections) == engine.detection_count


def test_osea_empty_array():
    engine = OseaEngine()
    results = engine.process_array([])
    assert results == []


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import traceback

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
