"""
HRV Toolbox test suite.

Run:
    python tests/test_hrv.py
    # or with pytest:
    pytest tests/test_hrv.py -v
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurocardiokit.hrv import hrv_analyze, flatten_result
from neurocardiokit.hrv._preprocess import rpos_to_nn
from neurocardiokit.hrv._utils import split_windows, summarize
from neurocardiokit.hrv._time_domain import time_domain, cross_segment_features
from neurocardiokit.hrv._freq_domain import freq_domain
from neurocardiokit.hrv._nonlinear import nonlinear, poincare, sampen, apen, dfa, csi_cvi

FS = 250  # standard ECG sampling rate


# ---- helpers ----

def _make_stable_rpos(n_beats, bpm=60, fs=FS):
    """Constant RR intervals: no HRV at all."""
    rr_samples = int(fs * 60.0 / bpm)
    rpos = np.arange(n_beats) * rr_samples
    anntype = np.ones(n_beats, dtype=int)
    return rpos, anntype


def _make_varying_rpos(n_beats, bpm=60, fs=FS, seed=42):
    """Sinusoidal + random RR variation mimicking respiratory sinus arrhythmia."""
    np.random.seed(seed)
    rr_base = int(fs * 60.0 / bpm)
    rr = rr_base + (np.sin(np.linspace(0, 10 * np.pi, n_beats)) * 20).astype(int)
    rr += np.random.randint(-8, 8, n_beats)
    rpos = np.cumsum(rr).astype(int)
    anntype = np.ones(n_beats, dtype=int)
    return rpos, anntype


def _make_ectopic_rpos(n_beats, bpm=60, fs=FS, seed=42):
    """Mild sinus variation + a few ectopic beats."""
    rpos, anntype = _make_varying_rpos(n_beats, bpm, fs, seed)
    for idx in [50, 150, 300, 500, 750]:
        if idx < n_beats:
            anntype[idx] = 5  # PVC marker
    return rpos, anntype


def _feat_val(feature_matrix, feat_names, name):
    """Extract a single feature value from a matrix by name."""
    idx = feat_names.index(name)
    return float(feature_matrix[0, idx])


# ---- preprocessing tests ----

def test_rpos_to_nn_basic():
    rpos, anntype = _make_stable_rpos(200)
    nn, nn_time = rpos_to_nn(rpos, anntype, fs=FS)
    assert len(nn) > 0
    assert len(nn) == len(nn_time)
    assert np.all(nn > 300) and np.all(nn < 2000)


def test_rpos_to_nn_without_anntype():
    rpos, anntype = _make_stable_rpos(200)
    nn, nn_time = rpos_to_nn(rpos, anntype=None, fs=FS)
    assert len(nn) > 0
    assert len(nn) == len(nn_time)


def test_rpos_to_nn_ectopic_filtering():
    rpos, anntype = _make_ectopic_rpos(500)
    nn_with, _ = rpos_to_nn(rpos, anntype, fs=FS)
    nn_without, _ = rpos_to_nn(rpos, anntype=None, fs=FS)
    assert len(nn_with) > 0
    assert len(nn_without) > 0
    # anntype pre-filter should remove some intervals
    assert len(nn_with) <= len(nn_without)


def test_rpos_to_nn_t_start():
    """Verify t_start shifts absolute time correctly."""
    rpos, anntype = _make_stable_rpos(200)
    _, t0 = rpos_to_nn(rpos, anntype, fs=FS, t_start=0)
    _, t10 = rpos_to_nn(rpos, anntype, fs=FS, t_start=10)
    assert np.allclose(t10 - t0, 10.0)


def test_rpos_to_nn_time_monotonic():
    """nn_time should be strictly increasing."""
    rpos, anntype = _make_varying_rpos(500)
    _, nn_time = rpos_to_nn(rpos, anntype, fs=FS)
    assert np.all(np.diff(nn_time) > 0)


def test_rpos_to_nn_time_aligns_with_rpos():
    """nn_time[0] should be close to rpos[0]/fs."""
    rpos, anntype = _make_varying_rpos(500)
    nn, nn_time = rpos_to_nn(rpos, anntype, fs=FS)
    expected_start = rpos[0] / FS
    assert abs(nn_time[0] - expected_start) < 0.01


def test_rpos_to_nn_empty():
    nn, nn_time = rpos_to_nn(np.array([], dtype=int), fs=FS)
    assert len(nn) == 0
    assert len(nn_time) == 0

    nn, nn_time = rpos_to_nn(np.array([100], dtype=int), fs=FS)
    assert len(nn) == 0
    assert len(nn_time) == 0


# ---- time domain tests ----

def test_time_domain():
    rpos, anntype = _make_varying_rpos(500)
    nn, _ = rpos_to_nn(rpos, anntype, fs=FS)
    td = time_domain(nn)
    assert 'sdnn' in td
    assert 'rmssd' in td
    assert td['sdnn'] > 0
    assert td['rmssd'] > 0
    assert 0 <= td['pnn50'] <= 100


def test_time_domain_short_input():
    td = time_domain(np.array([800, 820]))
    assert len(td) > 0


def test_cross_segment_features():
    rpos, anntype = _make_varying_rpos(1800)
    nn, nn_time = rpos_to_nn(rpos, anntype, fs=FS)
    windows = split_windows(nn, nn_time=nn_time, segment_len_s=300)
    cs = cross_segment_features(windows)
    assert cs['sdann'] >= 0
    assert cs['sdnn_i'] >= 0
    assert 30 < cs['mean_hr'] < 200


def test_cross_segment_features_too_few():
    cs = cross_segment_features([])
    assert cs == {}
    cs = cross_segment_features([np.array([800, 820, 810])])
    assert cs == {}


# ---- frequency domain tests ----

def test_freq_domain():
    rpos, anntype = _make_varying_rpos(1000)
    nn, _ = rpos_to_nn(rpos, anntype, fs=FS)
    fd = freq_domain(nn)
    assert 'lf' in fd
    assert 'hf' in fd
    assert 'lf_hf_ratio' in fd
    assert fd['total_power'] > 0


def test_freq_domain_lomb():
    rpos, anntype = _make_varying_rpos(500)
    nn, _ = rpos_to_nn(rpos, anntype, fs=FS)
    fd = freq_domain(nn, method='lomb')
    assert fd['total_power'] > 0


def test_freq_domain_short_input():
    fd = freq_domain(np.array([800, 820, 810]))
    for k in ('lf', 'hf', 'lf_hf_ratio'):
        assert k in fd


# ---- nonlinear tests ----

def test_poincare():
    rpos, anntype = _make_varying_rpos(500)
    nn, _ = rpos_to_nn(rpos, anntype, fs=FS)
    p = poincare(nn)
    assert p['sd1'] > 0
    assert p['sd2'] > 0
    assert p['sd_ratio'] > 0


def test_sampen():
    rpos, anntype = _make_varying_rpos(300)
    nn, _ = rpos_to_nn(rpos, anntype, fs=FS)
    s = sampen(nn)
    assert 'sampen' in s


def test_csi_cvi():
    rpos, anntype = _make_varying_rpos(200)
    nn, _ = rpos_to_nn(rpos, anntype, fs=FS)
    c = csi_cvi(nn)
    assert 'csi' in c and 'cvi' in c


def test_apen():
    rpos, anntype = _make_varying_rpos(300)
    nn, _ = rpos_to_nn(rpos, anntype, fs=FS)
    a = apen(nn)
    assert a['apen'] > 0


def test_apen_short():
    a = apen(np.array([800, 820, 810]))
    assert a == {}


def test_dfa():
    rpos, anntype = _make_varying_rpos(1000)
    nn, _ = rpos_to_nn(rpos, anntype, fs=FS)
    d = dfa(nn)
    assert 'dfa_alpha1' in d
    assert 0 < d['dfa_alpha1'] < 2.5


def test_dfa_short():
    d = dfa(np.array([800, 820, 810] * 10))
    assert d == {}


def test_nonlinear_combined():
    rpos, anntype = _make_varying_rpos(500)
    nn, _ = rpos_to_nn(rpos, anntype, fs=FS)
    nl = nonlinear(nn)
    assert 'sd1' in nl and 'sampen' in nl and 'dfa_alpha1' in nl
    assert 'apen' in nl and 'csi' in nl


# ---- utils tests ----

def test_split_windows():
    rpos, anntype = _make_varying_rpos(1800)
    nn, nn_time = rpos_to_nn(rpos, anntype, fs=FS)
    windows = split_windows(nn, nn_time=nn_time, segment_len_s=300)
    assert len(windows) >= 4
    for w in windows:
        assert len(w) >= 10
        assert np.all(w > 0)


def test_split_windows_fallback():
    """Without nn_time, split_windows falls back to cumsum."""
    rpos, anntype = _make_varying_rpos(1800)
    nn, _ = rpos_to_nn(rpos, anntype, fs=FS)
    windows = split_windows(nn, segment_len_s=300)  # no nn_time
    assert len(windows) >= 4


def test_split_windows_overlap():
    rpos, anntype = _make_varying_rpos(1800)
    nn, nn_time = rpos_to_nn(rpos, anntype, fs=FS)
    no_overlap = split_windows(nn, nn_time=nn_time, segment_len_s=300)
    with_overlap = split_windows(nn, nn_time=nn_time, segment_len_s=300, step_s=180)
    assert len(with_overlap) > len(no_overlap)


def test_split_windows_empty():
    assert split_windows(np.array([], dtype=float)) == []


def test_summarize():
    feat_list = [
        {'a': 1.0, 'b': 2.0, 'c': 'skip'},
        {'a': 3.0, 'b': 4.0, 'c': 'skip'},
        {'a': 5.0, 'b': 6.0, 'c': 'skip'},
    ]
    s = summarize(feat_list)
    assert s['a_mean'] == 3.0
    assert s['b_min'] == 2.0
    assert s['b_max'] == 6.0
    assert 'c_mean' not in s


def test_summarize_empty():
    assert summarize([]) == {}
    assert summarize([{}]) == {}


# ---- integration tests (global only) ----

def test_hrv_analyze_global():
    rpos, anntype = _make_varying_rpos(1000)
    nn, _ = rpos_to_nn(rpos, anntype, fs=FS)
    r = hrv_analyze(nn=nn, segment_len=None)
    gl = r['global']
    X = gl['feature_matrix']
    names = gl['feat_names']
    assert X.shape[0] == 1
    assert X.shape[1] == 33  # 16 + 7 + 10
    assert 'sdnn' in names
    assert 'lf_hf_ratio' in names
    assert 'dfa_alpha1' in names
    assert _feat_val(X, names, 'sdnn') > 0


def test_hrv_analyze_global_feat_include():
    rpos, anntype = _make_varying_rpos(1000)
    nn, _ = rpos_to_nn(rpos, anntype, fs=FS)
    r = hrv_analyze(nn=nn, segment_len=None, feat_include=['time'])
    names = r['global']['feat_names']
    assert 'sdnn' in names
    assert 'lf' not in names
    assert 'sampen' not in names
    assert r['global']['feature_matrix'].shape[1] == 16


def test_hrv_analyze_global_feat_include_empty():
    """Empty feat_include should compute nothing."""
    rpos, anntype = _make_varying_rpos(1000)
    nn, _ = rpos_to_nn(rpos, anntype, fs=FS)
    r = hrv_analyze(nn=nn, segment_len=None, feat_include=[])
    assert r['global']['feature_matrix'].shape[1] == 0
    assert r['global']['feat_names'] == []


# ---- integration tests (segment) ----

def test_hrv_analyze_segment():
    rpos, anntype = _make_varying_rpos(1800)
    nn, nn_time = rpos_to_nn(rpos, anntype, fs=FS)
    r = hrv_analyze(nn=nn, nn_time=nn_time, segment_len=300)
    seg = r['segments']
    assert seg['segment_len_s'] == 300
    assert seg['segment_overlap_s'] is None
    assert seg['n_segments'] >= 4
    assert seg['feature_matrix'].shape[0] == seg['n_segments']
    assert seg['feature_matrix'].shape[1] == 33
    assert len(seg['segment_t_starts']) == seg['n_segments']
    # All rows should have non-NaN values for time domain
    assert not np.any(np.isnan(seg['feature_matrix'][:, :16]))


def test_hrv_analyze_segment_feat_include():
    rpos, anntype = _make_varying_rpos(1800)
    nn, nn_time = rpos_to_nn(rpos, anntype, fs=FS)
    r = hrv_analyze(nn=nn, nn_time=nn_time, segment_len=300,
                    feat_include=['time'])
    seg = r['segments']
    names = seg['feat_names']
    assert 'sdnn' in names
    assert 'lf' not in names
    assert 'sampen' not in names
    assert seg['feature_matrix'].shape[1] == 16


def test_hrv_analyze_segment_overlap():
    rpos, anntype = _make_varying_rpos(1800)
    nn, nn_time = rpos_to_nn(rpos, anntype, fs=FS)
    r_no = hrv_analyze(nn=nn, nn_time=nn_time, segment_len=300)
    r_ov = hrv_analyze(nn=nn, nn_time=nn_time, segment_len=300,
                       segment_overlap=120)
    assert r_ov['segments']['n_segments'] > r_no['segments']['n_segments']
    assert r_ov['segments']['segment_overlap_s'] == 120


def test_hrv_analyze_segment_also_has_global():
    """Even with segment_len, global should still be present."""
    rpos, anntype = _make_varying_rpos(1000)
    nn, _ = rpos_to_nn(rpos, anntype, fs=FS)
    r = hrv_analyze(nn=nn, segment_len=300)
    assert r['global']['feature_matrix'].shape[0] == 1
    assert r['global']['feat_names'] == r['segments']['feat_names']


# ---- integration tests (from rpos) ----

def test_hrv_analyze_from_rpos():
    rpos, anntype = _make_varying_rpos(1800)
    r = hrv_analyze(rpos=rpos, anntype=anntype, fs_ecg=FS, segment_len=300)
    assert r['n_beats'] > 1700
    assert len(r['nn']) == len(r['nn_time'])
    assert 25 < r['duration_min'] < 35
    assert r['global']['feature_matrix'].shape[1] == 33
    assert r['segments']['n_segments'] >= 4


def test_hrv_analyze_from_rpos_with_ectopics():
    rpos, anntype = _make_ectopic_rpos(1800)
    r = hrv_analyze(rpos=rpos, anntype=anntype, fs_ecg=FS, segment_len=300)
    assert r['n_beats'] > 0
    assert len(r['nn']) == len(r['nn_time'])
    names = r['global']['feat_names']
    assert _feat_val(r['global']['feature_matrix'], names, 'sdnn') > 0


def test_hrv_analyze_from_rpos_no_anntype():
    rpos, _ = _make_varying_rpos(1000)
    r = hrv_analyze(rpos=rpos, anntype=None, fs_ecg=FS, segment_len=300)
    assert r['n_beats'] > 0
    assert len(r['nn']) == len(r['nn_time'])
    names = r['global']['feat_names']
    assert _feat_val(r['global']['feature_matrix'], names, 'sdnn') > 0


def test_hrv_analyze_from_rpos_t_start():
    """nn_time should reflect t_start offset."""
    rpos, anntype = _make_varying_rpos(500)
    r0 = hrv_analyze(rpos=rpos, anntype=anntype, fs_ecg=FS, t_start=0)
    r100 = hrv_analyze(rpos=rpos, anntype=anntype, fs_ecg=FS, t_start=100)
    assert np.allclose(r100['nn_time'] - r0['nn_time'], 100.0)
    # feature values should be identical (NN intervals don't change)
    names = r0['global']['feat_names']
    v0 = _feat_val(r0['global']['feature_matrix'], names, 'sdnn')
    v100 = _feat_val(r100['global']['feature_matrix'], names, 'sdnn')
    assert v0 == v100


def test_hrv_analyze_from_rpos_empty():
    r = hrv_analyze(rpos=np.array([], dtype=int), fs_ecg=FS)
    assert r['n_beats'] == 0
    assert len(r['nn_time']) == 0
    assert r['global']['feature_matrix'].shape[0] == 1


def test_hrv_analyze_from_rpos_feat_include():
    rpos, anntype = _make_varying_rpos(1800)
    r = hrv_analyze(rpos=rpos, anntype=anntype, fs_ecg=FS, segment_len=300,
                    feat_include=['freq', 'nonlinear'])
    names = r['global']['feat_names']
    assert 'sdnn' not in names
    assert 'lf' in names
    assert 'sampen' in names


# ---- regression: features must be within physiological bounds ----

def test_physiological_bounds():
    rpos, anntype = _make_varying_rpos(3000)
    r = hrv_analyze(rpos=rpos, anntype=anntype, fs_ecg=FS)
    X = r['global']['feature_matrix']
    names = r['global']['feat_names']

    assert 300 < _feat_val(X, names, 'mean_nn') < 2000
    assert 0 < _feat_val(X, names, 'sdnn') < 300
    assert 0 < _feat_val(X, names, 'rmssd') < 300
    assert 0 <= _feat_val(X, names, 'pnn50') <= 100
    assert 30 < _feat_val(X, names, 'mean_hr') < 200
    assert 0.01 < _feat_val(X, names, 'cvnn') < 0.5

    assert _feat_val(X, names, 'total_power') > 0
    assert _feat_val(X, names, 'lf_hf_ratio') > 0

    assert 0 < _feat_val(X, names, 'sd1') < 200
    assert _feat_val(X, names, 'sd2') > _feat_val(X, names, 'sd1')
    assert 0 < _feat_val(X, names, 'dfa_alpha1') < 2.5


# ---- flatten_result tests ----

def test_flatten_result():
    rpos, anntype = _make_varying_rpos(1800)
    r = hrv_analyze(rpos=rpos, anntype=anntype, fs_ecg=FS, segment_len=300)
    flat = flatten_result(r)
    assert isinstance(flat, dict)
    # Metadata
    assert 'n_beats' in flat
    assert 'duration_min' in flat
    assert 'n_segments' in flat
    # Global features
    assert 'global_sdnn' in flat
    assert 'global_lf_hf_ratio' in flat
    assert 'global_sd1' in flat
    # Segment features
    seg_keys = [k for k in flat if k.startswith('seg0_')]
    assert len(seg_keys) > 0
    # All values should be scalar
    for v in flat.values():
        assert isinstance(v, (int, float))


def test_flatten_result_global_only():
    """flatten_result with segment_len=None should have no segment keys."""
    rpos, anntype = _make_varying_rpos(1000)
    nn, _ = rpos_to_nn(rpos, anntype, fs=FS)
    r = hrv_analyze(nn=nn, segment_len=None)
    flat = flatten_result(r)
    assert 'global_sdnn' in flat
    assert 'n_segments' not in flat


def test_flatten_result_empty():
    """flatten_result should handle empty result gracefully."""
    empty = {'n_beats': 0, 'duration_min': 0.0,
             'global': {'feature_matrix': None, 'feat_names': None}}
    flat = flatten_result(empty)
    assert flat['n_beats'] == 0


# ---- runner ----

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
