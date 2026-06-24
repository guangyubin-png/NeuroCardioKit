"""
Analyze entire sleep EDF: OSEA QRS detection → HRV + EEG → joint.

Usage:
    python scripts/analyze_ecg_eeg.py
"""

import os
import time
import numpy as np

from neurocardiokit.osea import OseaEngine, ANNOTYPE_LABELS
from neurocardiokit.hrv import hrv_analyze
from neurocardiokit.eeg import eeg_analyze

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EDF_PATH = r'E:\DataBase\physionet2026\sub-I0002150001401_ses-2.edf'
EDF_FS = 500.0

ECG_CHANNEL = 'EKG'
EEG_CHANNELS = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1']

SEGMENT_LEN = 30            # 30s segments for joint EEG-HRV alignment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_channel_index(f, label):
    for i in range(f.signals_in_file):
        if f.getLabel(i) == label:
            return i
    return -1


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ecg_with_rpos(ecg_uv, fs, rpos, anntype, save_path,
                       start_sec=0, duration_sec=60):
    """Plot a short ECG segment with R-peak annotations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _LABELS = {1: "NORMAL", 5: "PVC", 13: "UNKNOWN"}

    t_orig = np.arange(len(ecg_uv)) / fs
    mask = (t_orig >= start_sec) & (t_orig < start_sec + duration_sec)
    t_plot = t_orig[mask]
    sig_plot = ecg_uv[mask]

    r_times = rpos / fs
    in_win = (r_times >= start_sec) & (r_times < start_sec + duration_sec)
    normal = (anntype == 1)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t_plot, sig_plot, color='#1a1a2e', linewidth=0.6, label='ECG')

    # NORMAL
    nw = in_win & normal
    if nw.any():
        r_t = r_times[nw]
        idx = np.clip(np.searchsorted(t_plot, r_t), 0, len(sig_plot) - 1)
        ax.plot(r_t, sig_plot[idx], 'o', color='#2ecc71', markersize=6,
                markerfacecolor='none', markeredgewidth=1.2,
                label=f'NORMAL ({nw.sum()})')

    # OTHER
    ow = in_win & ~normal
    if ow.any():
        codes = np.unique(anntype[ow])
        lbls = set(_LABELS.get(c, f'CODE_{c}') for c in codes)
        for t_sec in r_times[ow]:
            ax.axvline(t_sec, color='red', alpha=0.4, linewidth=0.8)
        ax.plot([], [], 'x', color='red', markersize=5,
                label=f'OTHER ({ow.sum()}: {",".join(lbls)})')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ECG (µV)')
    ax.set_title(f'OSEA QRS Detection — middle of recording')
    ax.legend(loc='upper right', framealpha=0.8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_trends(eeg_res, hrv_res, nn, nn_time, total_h, save_path):
    """Plot EEG + HRV + BPM trends over the entire recording."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    seg_t = eeg_res['segments']['segment_t_starts'] / 3600.0  # → hours
    X_eeg = eeg_res['segments']['feature_matrix']
    X_hrv = hrv_res['segments']['feature_matrix']
    eeg_names = eeg_res['segments']['feat_names']
    hrv_names = hrv_res['segments']['feat_names']
    n_hrv = X_hrv.shape[0]

    # BPM per segment
    seg_t_hrv = seg_t[:n_hrv]
    bpm = np.full(n_hrv, np.nan)
    if nn_time is not None and len(nn_time) > 0:
        for i in range(n_hrv):
            t0 = seg_t_hrv[i] * 3600.0
            t1 = t0 + SEGMENT_LEN
            m = (nn_time >= t0) & (nn_time < t1)
            if m.sum() > 0:
                bpm[i] = 60.0 / (nn[m].mean() / 1000.0)

    # Features to plot
    plots = [
        ('EEG δ power (F3-M2)', X_eeg[:, eeg_names.index('bp_delta_ch0')], '#3498db'),
        ('EEG α power (F3-M2)', X_eeg[:, eeg_names.index('bp_alpha_ch0')], '#3498db'),
        ('HR (BPM)', bpm, '#e67e22'),
        ('HRV SDNN (ms)', X_hrv[:, hrv_names.index('sdnn')], '#e74c3c'),
        ('HRV LF/HF ratio', X_hrv[:, hrv_names.index('lf_hf_ratio')], '#e74c3c'),
    ]

    fig, axes = plt.subplots(len(plots), 1, figsize=(18, 14), sharex=True)

    for ax, (label, data, color) in zip(axes, plots):
        t = seg_t if data is bpm else (seg_t[:len(data)] if len(data) < len(seg_t) else seg_t)
        ax.plot(t[:len(data)], data, '-', color=color, linewidth=0.8, alpha=0.9)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, total_h)

    axes[-1].set_xlabel('Time (hours)', fontsize=11)
    fig.suptitle(f'Feature Trends — {total_h:.1f}h Recording', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import pyedflib
    script_dir = os.path.dirname(os.path.abspath(__file__))
    t_start = time.time()

    # ---- Metadata ----
    f = pyedflib.EdfReader(EDF_PATH)
    all_labels = [f.getLabel(i) for i in range(f.signals_in_file)]
    total_dur = f.getFileDuration()  # seconds
    f.close()
    print(f"EDF: {os.path.basename(EDF_PATH)}")
    print(f"  Duration:  {total_dur / 3600:.1f} h ({total_dur:.0f} s)")
    print(f"  Segments:  {int(total_dur / SEGMENT_LEN)} × {SEGMENT_LEN}s")
    print(f"  Channels:  {all_labels}")

    # ---- Read ECG + OSEA ----
    print(f"\n--- ECG (OSEA) ---")
    t0 = time.time()
    f = pyedflib.EdfReader(EDF_PATH)
    ek_ch = all_labels.index(ECG_CHANNEL)
    ecg_uv = f.readSignal(ek_ch).astype(np.float64)
    f.close()
    print(f"  Read:       {len(ecg_uv):,} samples ({time.time() - t0:.1f}s)")

    t0 = time.time()
    engine = OseaEngine()
    rpos, anntype = engine.detect_rpeaks(ecg_uv / 1000.0, EDF_FS)
    dt_osea = time.time() - t0

    n_total = len(rpos)
    n_normal = int((anntype == 1).sum())
    type_counts = {}
    for code in np.unique(anntype):
        type_counts[ANNOTYPE_LABELS.get(code, f'CODE_{code}')] = int((anntype == code).sum())

    print(f"  Total beats:{n_total}")
    print(f"  NORMAL:     {n_normal}")
    print(f"  Types:      {type_counts}")
    print(f"  BPM:        ~{n_normal / (total_dur / 60.0):.0f}")
    print(f"  Time:       {dt_osea:.1f}s ({len(ecg_uv) / dt_osea / 1e6:.1f}M samples/s)")

    # ---- Plot ECG snippet (middle ~60s) ----
    mid = total_dur / 2 - 30
    plot_ecg_with_rpos(ecg_uv, EDF_FS, rpos, anntype,
                       save_path=os.path.join(script_dir, 'ecg_rpos.png'),
                       start_sec=mid, duration_sec=60)

    # ---- Read EEG ----
    print(f"\n--- EEG ---")
    t0 = time.time()
    f = pyedflib.EdfReader(EDF_PATH)
    eeg_data = []
    for name in EEG_CHANNELS:
        ch = all_labels.index(name)
        eeg_data.append(f.readSignal(ch).astype(np.float64))
    eeg_data = np.stack(eeg_data)
    f.close()
    print(f"  Read:       {eeg_data.shape} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    eeg_result = eeg_analyze(
        eeg_data, fs=EDF_FS,
        channel_names=EEG_CHANNELS,
        segment_len=SEGMENT_LEN,
        feat_include=['band_power', 'band_power_rel'],
    )
    dt_eeg = time.time() - t0
    X_eeg = eeg_result['segments']['feature_matrix']
    print(f"  Segments:   {eeg_result['segments']['n_segments']}")
    print(f"  Features:   {X_eeg.shape[1]}")
    print(f"  Time:       {dt_eeg:.1f}s")

    # ---- HRV ----
    print(f"\n--- HRV ---")
    t0 = time.time()
    hrv_result = hrv_analyze(
        rpos=rpos,
        anntype=anntype,
        fs_ecg=EDF_FS,
        segment_len=SEGMENT_LEN,
        total_s=total_dur,
        feat_include=['time', 'freq'],
    )
    dt_hrv = time.time() - t0
    X_hrv = hrv_result['segments']['feature_matrix']
    print(f"  NN count:   {len(hrv_result['nn'])}")
    print(f"  Segments:   {hrv_result['segments']['n_segments']}")
    print(f"  Features:   {X_hrv.shape[1]}")
    print(f"  Time:       {dt_hrv:.1f}s")

    # ---- Joint ----
    n_joint = min(X_eeg.shape[0], X_hrv.shape[0])
    X_joint = np.hstack([X_eeg[:n_joint], X_hrv[:n_joint]])
    print(f"\nJoint:   {X_joint.shape} ({n_joint} seg × {X_joint.shape[1]} feat)")
    print(f"Total time: {time.time() - t_start:.0f}s")

    # ---- Plot trends ----
    plot_trends(eeg_result, hrv_result,
                hrv_result['nn'], hrv_result.get('nn_time'),
                total_dur / 3600.0,
                save_path=os.path.join(script_dir, 'features_trend.png'))

    print(f"\nDone. Plots saved to {script_dir}/")


if __name__ == '__main__':
    main()
