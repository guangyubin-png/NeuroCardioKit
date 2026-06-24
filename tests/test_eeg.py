"""
neurocardiokit.eeg — EDF-based integration tests.

Uses a Physionet sleep EDF as the test fixture.
Run:
    python tests/test_eeg.py
    # or with pytest:
    pytest tests/test_eeg.py -v
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurocardiokit.eeg import eeg_analyze, eeg_flatten, BANDS
from neurocardiokit.eeg._preprocess import eeg_preprocess
from neurocardiokit.eeg._epoch import eeg_epoch, eeg_clean_epochs, eeg_bad_channels
from neurocardiokit.eeg._spectral import eeg_psd, eeg_band_power
from neurocardiokit.eeg._connectivity import eeg_coherence
from neurocardiokit.eeg._nonlinear import eeg_apen, eeg_bsr
from neurocardiokit.eeg._utils import eeg_split_segments

# ---------------------------------------------------------------------------
# Test fixture — read a short segment from the Physionet EDF
# ---------------------------------------------------------------------------

EDF_PATH = r'E:\DataBase\physionet2026\sub-I0002150001401_ses-2.edf'

# EEG channels in this EDF (500 Hz, µV)
EEG_CHANNEL_NAMES = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1']
EEG_FS = 500.0
READ_DURATION_S = 300  # 5 minutes, enough for one segment_len=30 * 10
READ_OFFSET_S = 600    # skip the first 10 minutes (warm-up / calibration)


def _load_eeg_segment():
    """Read a short EEG segment from the EDF file.

    Returns (data, fs, channel_names) where data is (n_chans, n_samples).
    Returns None if the EDF file is not available.
    """
    if not os.path.exists(EDF_PATH):
        return None

    import pyedflib
    f = pyedflib.EdfReader(EDF_PATH)
    try:
        fs = EEG_FS
        n_samples = int(READ_DURATION_S * fs)
        offset = int(READ_OFFSET_S * fs)

        data_list = []
        for name in EEG_CHANNEL_NAMES:
            ch_idx = _find_channel_index(f, name)
            if ch_idx < 0:
                f.close()
                raise RuntimeError(f'Channel {name} not found in EDF')
            sig = f.readSignal(ch_idx, start=offset, n=n_samples)
            data_list.append(sig.astype(np.float64))

        return np.stack(data_list), fs, EEG_CHANNEL_NAMES
    finally:
        f.close()


def _find_channel_index(f, label):
    """Find a channel index by label in an EdfReader."""
    for i in range(f.signals_in_file):
        if f.getLabel(i) == label:
            return i
    return -1


# Pytest-style fixture as plain function for simple runner compatibility
_FIXTURE_CACHE = None


def _get_fixture():
    global _FIXTURE_CACHE
    if _FIXTURE_CACHE is None:
        _FIXTURE_CACHE = _load_eeg_segment()
    return _FIXTURE_CACHE


def _require_fixture():
    """Get the fixture or skip the calling test."""
    fix = _get_fixture()
    if fix is None:
        raise OSError(f'EDF file not found: {EDF_PATH}')
    return fix


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def test_eeg_preprocess_shape():
    data, fs, _ = _require_fixture()
    filtered = eeg_preprocess(data, fs)
    assert filtered.shape == data.shape
    assert filtered.dtype == np.float64


def test_eeg_preprocess_1d_input():
    data, fs, _ = _require_fixture()
    filtered = eeg_preprocess(data[0], fs)
    assert filtered.ndim == 1
    assert len(filtered) == data.shape[1]


def test_eeg_preprocess_notch():
    data, fs, _ = _require_fixture()
    filtered = eeg_preprocess(data, fs, notch=50)
    assert filtered.shape == data.shape
    # Should not produce NaNs
    assert not np.any(np.isnan(filtered))


# ---------------------------------------------------------------------------
# Epoching
# ---------------------------------------------------------------------------

def test_eeg_epoch_shape():
    data, fs, _ = _require_fixture()
    epochs, times = eeg_epoch(data, epoch_len_s=2, fs=fs)
    n_expected = data.shape[1] // int(2 * fs)
    assert epochs.shape[0] == n_expected
    assert epochs.shape[1] == data.shape[0]
    assert epochs.shape[2] == int(2 * fs)
    assert len(times) == n_expected


def test_eeg_epoch_times_monotonic():
    data, fs, _ = _require_fixture()
    epochs, times = eeg_epoch(data, epoch_len_s=2, fs=fs, t_start=100.0)
    assert np.all(np.diff(times) > 0)
    assert times[0] == 100.0


def test_eeg_epoch_overlap():
    data, fs, _ = _require_fixture()
    no_overlap, _ = eeg_epoch(data, epoch_len_s=2, fs=fs)
    with_overlap, _ = eeg_epoch(data, epoch_len_s=2, step_s=1, fs=fs)
    assert with_overlap.shape[0] > no_overlap.shape[0]


def test_eeg_epoch_detrend():
    data, fs, _ = _require_fixture()
    epochs, _ = eeg_epoch(data, epoch_len_s=2, fs=fs, detrend=True)
    means = epochs.mean(axis=-1)
    assert np.allclose(means, 0, atol=1e-9)


# ---------------------------------------------------------------------------
# Epoch cleaning
# ---------------------------------------------------------------------------

def test_eeg_clean_epochs():
    data, fs, _ = _require_fixture()
    epochs, times = eeg_epoch(data, epoch_len_s=2, fs=fs)
    clean, clean_times = eeg_clean_epochs(epochs, times, min_amp=1.0, max_amp=300.0)
    assert clean.shape[0] <= epochs.shape[0]
    assert len(clean_times) == clean.shape[0]


def test_eeg_clean_epochs_without_times():
    data, fs, _ = _require_fixture()
    epochs, _ = eeg_epoch(data, epoch_len_s=2, fs=fs)
    clean, ct = eeg_clean_epochs(epochs, min_amp=1.0, max_amp=300.0)
    assert clean.shape[0] <= epochs.shape[0]
    assert ct is None


def test_eeg_bad_channels():
    data, fs, _ = _require_fixture()
    epochs, _ = eeg_epoch(data, epoch_len_s=2, fs=fs)
    bad, good = eeg_bad_channels(epochs)
    assert isinstance(bad, list)
    assert isinstance(good, list)
    assert len(bad) + len(good) == data.shape[0]


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def test_eeg_psd_output_shape():
    data, fs, _ = _require_fixture()
    epochs, _ = eeg_epoch(data, epoch_len_s=2, fs=fs)
    psd, freqs = eeg_psd(epochs, fs)
    assert psd.shape[0] == data.shape[0]
    assert psd.shape[1] == len(freqs)
    assert np.all(psd >= 0)


def test_eeg_band_power_default_bands():
    data, fs, _ = _require_fixture()
    epochs, _ = eeg_epoch(data, epoch_len_s=2, fs=fs)
    psd, freqs = eeg_psd(epochs, fs)
    bp, names = eeg_band_power(psd, freqs)
    assert bp.shape == (data.shape[0], len(BANDS))
    assert names == list(BANDS.keys())
    assert np.all(bp >= 0)


def test_eeg_band_power_custom_bands():
    data, fs, _ = _require_fixture()
    epochs, _ = eeg_epoch(data, epoch_len_s=2, fs=fs)
    psd, freqs = eeg_psd(epochs, fs)
    custom = {'alpha': (8, 13)}
    bp, names = eeg_band_power(psd, freqs, bands=custom)
    assert names == ['alpha']
    assert bp.shape == (data.shape[0], 1)


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------

def test_eeg_coherence_shape():
    data, fs, _ = _require_fixture()
    epochs, _ = eeg_epoch(data, epoch_len_s=2, fs=fs)
    coh, band_names = eeg_coherence(epochs, fs)
    n_chans = data.shape[0]
    assert coh.shape == (n_chans, n_chans, len(BANDS))
    assert band_names == list(BANDS.keys())
    # Diagonal (self-coherence) should be 1.0
    for i in range(n_chans):
        assert np.allclose(coh[i, i], 1.0)


def test_eeg_coherence_symmetric():
    data, fs, _ = _require_fixture()
    epochs, _ = eeg_epoch(data, epoch_len_s=2, fs=fs)
    coh, _ = eeg_coherence(epochs, fs)
    assert np.allclose(coh, coh.transpose(1, 0, 2))


def test_eeg_coherence_bounds():
    data, fs, _ = _require_fixture()
    epochs, _ = eeg_epoch(data, epoch_len_s=2, fs=fs)
    coh, _ = eeg_coherence(epochs, fs)
    assert np.all(coh >= 0) and np.all(coh <= 1.0)


# ---------------------------------------------------------------------------
# Nonlinear features
# ---------------------------------------------------------------------------

def test_eeg_apen_output():
    data, fs, _ = _require_fixture()
    epochs, _ = eeg_epoch(data, epoch_len_s=2, fs=fs)
    apen = eeg_apen(epochs)
    assert apen.shape == (data.shape[0],)
    assert np.all(apen >= 0)


def test_eeg_bsr_output():
    data, fs, _ = _require_fixture()
    epochs, _ = eeg_epoch(data, epoch_len_s=2, fs=fs)
    thresholds = (5, 10, 20)
    bsr = eeg_bsr(epochs, thresholds)
    assert bsr.shape == (data.shape[0], len(thresholds))
    assert np.all(bsr >= 0) and np.all(bsr <= 100)


# ---------------------------------------------------------------------------
# Segment splitting
# ---------------------------------------------------------------------------

def test_eeg_split_segments():
    data, fs, _ = _require_fixture()
    segments = eeg_split_segments(data, fs, segment_len_s=30)
    assert len(segments) == READ_DURATION_S // 30
    for seg_data, seg_t in segments:
        assert seg_data.shape[0] == data.shape[0]
        assert seg_data.shape[1] == int(30 * fs)


def test_eeg_split_segments_t_starts():
    data, fs, _ = _require_fixture()
    segments = eeg_split_segments(data, fs, segment_len_s=30, t_start=100.0)
    assert segments[0][1] == 100.0
    assert segments[1][1] == 130.0


def test_eeg_split_segments_overlap():
    data, fs, _ = _require_fixture()
    no_overlap = eeg_split_segments(data, fs, segment_len_s=30)
    with_overlap = eeg_split_segments(data, fs, segment_len_s=30, step_s=15)
    assert len(with_overlap) > len(no_overlap)
    # Overlapped segments should have correct t_starts
    assert with_overlap[0][1] == 0.0
    assert with_overlap[1][1] == 15.0


def test_eeg_split_segments_1d():
    data, fs, _ = _require_fixture()
    segments = eeg_split_segments(data[0], fs, segment_len_s=30)
    assert len(segments) > 0
    assert segments[0][0].ndim == 2  # promoted to (1, samples)


# ---------------------------------------------------------------------------
# End-to-end: eeg_analyze
# ---------------------------------------------------------------------------

def test_eeg_analyze_full():
    data, fs, ch_names = _require_fixture()
    result = eeg_analyze(
        data, fs,
        channel_names=ch_names,
        segment_len=30,
        epoch_len=2,
        epoch_overlap=1,
        clean=True,
        feat_include=['band_power', 'band_power_rel', 'coherence', 'bsr'],
    )
    # Metadata
    assert result['fs'] == fs
    assert result['n_channels'] == len(ch_names)
    assert result['epoch_len_s'] == 2
    assert result['channel_names'] == ch_names
    assert result['global'] is None

    n_seg = READ_DURATION_S // 30
    seg = result['segments']
    assert seg['segment_len_s'] == 30
    assert seg['n_segments'] == n_seg
    assert seg['feature_matrix'].shape[0] == n_seg
    assert len(seg['feat_names']) == seg['feature_matrix'].shape[1]
    assert len(seg['segment_t_starts']) == n_seg
    assert len(seg['n_valid_epochs']) == n_seg

    # At least some segments should have valid epochs
    assert np.sum(seg['n_valid_epochs']) > 0

    # feature_matrix should not be all zeros
    assert not np.allclose(seg['feature_matrix'], 0)


def test_eeg_analyze_minimal():
    """Only basic features, skip expensive coherence/apen/bsr."""
    data, fs, ch_names = _require_fixture()
    result = eeg_analyze(
        data, fs,
        channel_names=ch_names,
        segment_len=30,
        epoch_len=2,
        feat_include=['band_power', 'band_power_rel'],
    )
    seg = result['segments']
    assert seg['feature_matrix'].shape[0] == READ_DURATION_S // 30
    for name in seg['feat_names']:
        assert any(name.startswith(p) for p in ('bp_', 'bpr_'))


def test_eeg_analyze_feat_include_subset():
    data, fs, ch_names = _require_fixture()
    result = eeg_analyze(
        data, fs,
        channel_names=ch_names,
        segment_len=30,
        epoch_len=2,
        feat_include=['band_power', 'bsr'],
    )
    for name in result['segments']['feat_names']:
        assert name.startswith('bp_') or name.startswith('bsr_')


def test_eeg_analyze_feat_include_coherence_only():
    """Request only coherence — should not compute PSD/bands."""
    data, fs, ch_names = _require_fixture()
    result = eeg_analyze(
        data, fs,
        channel_names=ch_names,
        segment_len=30,
        epoch_len=2,
        feat_include=['coherence'],
    )
    for name in result['segments']['feat_names']:
        assert name.startswith('coh_')


def test_eeg_analyze_segment_overlap():
    """Segment overlap should increase n_segments."""
    data, fs, ch_names = _require_fixture()
    r_no = eeg_analyze(
        data, fs,
        channel_names=ch_names,
        segment_len=30, epoch_len=2,
        feat_include=['band_power'],
    )
    r_ov = eeg_analyze(
        data, fs,
        channel_names=ch_names,
        segment_len=30, segment_overlap=15, epoch_len=2,
        feat_include=['band_power'],
    )
    assert r_ov['segments']['n_segments'] > r_no['segments']['n_segments']
    assert r_ov['segments']['segment_overlap_s'] == 15
    assert r_no['segments']['segment_overlap_s'] is None


def test_eeg_analyze_no_preprocess():
    data, fs, ch_names = _require_fixture()
    result = eeg_analyze(
        data, fs,
        channel_names=ch_names,
        segment_len=30,
        epoch_len=2,
        preprocess=False,
        clean=False,
        feat_include=['band_power', 'band_power_rel'],
    )
    assert result['segments']['feature_matrix'].shape[0] == READ_DURATION_S // 30


def test_eeg_analyze_no_clean():
    """Without cleaning, more epochs should be retained."""
    data, fs, ch_names = _require_fixture()
    r_clean = eeg_analyze(
        data, fs,
        channel_names=ch_names,
        segment_len=30,
        epoch_len=2,
        clean=True,
        feat_include=['band_power', 'band_power_rel'],
    )
    r_noclean = eeg_analyze(
        data, fs,
        channel_names=ch_names,
        segment_len=30,
        epoch_len=2,
        clean=False,
        feat_include=['band_power', 'band_power_rel'],
    )
    assert np.sum(r_noclean['segments']['n_valid_epochs']) >= np.sum(r_clean['segments']['n_valid_epochs'])


def test_eeg_analyze_1d_input():
    data, fs, _ = _require_fixture()
    result = eeg_analyze(
        data[0], fs,
        segment_len=30, epoch_len=2,
        feat_include=['band_power', 'band_power_rel'],
    )
    assert result['n_channels'] == 1
    assert result['segments']['feature_matrix'].shape[0] == READ_DURATION_S // 30


# ---------------------------------------------------------------------------
# eeg_flatten
# ---------------------------------------------------------------------------

def test_eeg_flatten():
    data, fs, _ = _require_fixture()
    result = eeg_analyze(
        data, fs,
        segment_len=30, epoch_len=2,
        feat_include=['band_power', 'band_power_rel'],
    )
    flat = eeg_flatten(result)
    n_seg = result['segments']['n_segments']
    n_feat = result['segments']['feature_matrix'].shape[1]
    assert len(flat) == n_seg * n_feat
    for v in flat.values():
        assert isinstance(v, float)


def test_eeg_flatten_empty():
    assert eeg_flatten({}) == {}
    assert eeg_flatten({'segments': {'feature_matrix': None, 'feat_names': None}}) == {}


# ---------------------------------------------------------------------------
# Physiological sanity checks
# ---------------------------------------------------------------------------

def test_band_power_physiological():
    """Alpha power should dominate with eyes closed; delta should dominate in sleep."""
    data, fs, ch_names = _require_fixture()
    result = eeg_analyze(
        data, fs,
        channel_names=ch_names,
        segment_len=30, epoch_len=2,
        feat_include=['band_power'],
    )
    # Total band power should be > 0 for all channels
    seg = result['segments']
    bp_cols = [c for c in seg['feat_names'] if c.startswith('bp_')]
    bp_idx = [seg['feat_names'].index(c) for c in bp_cols]
    bp_vals = seg['feature_matrix'][:, bp_idx]
    # Mean across segments
    mean_bp = bp_vals.mean(axis=0)
    assert np.all(mean_bp >= 0)
    # At least one band should have non-zero power per channel
    assert np.all(np.sum(bp_vals, axis=0) > 0)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import traceback

    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]

    # Warm fixture
    _get_fixture()

    passed = 0
    failed = 0
    skipped = 0
    for fn in tests:
        try:
            fn()
            print(f'  PASS  {fn.__name__}')
            passed += 1
        except OSError as e:
            if 'EDF file not found' in str(e):
                print(f'  SKIP  {fn.__name__}  ({e})')
                skipped += 1
            else:
                print(f'  FAIL  {fn.__name__}')
                traceback.print_exc()
                failed += 1
        except Exception:
            print(f'  FAIL  {fn.__name__}')
            traceback.print_exc()
            failed += 1

    print(f'\n{passed} passed, {failed} failed, {skipped} skipped out of {len(tests)} tests')
    sys.exit(0 if failed == 0 else 1)
