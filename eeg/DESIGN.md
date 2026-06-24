# EEG 分析库 — neurocardiokit/eeg/ — 设计

## 1. 背景与目标

本库构建一个纯 numpy/scipy 的通用 EEG 分析库，**独立于 Qt、MNE、特定硬件**。

**核心入口只有一个：`eeg_analyze`**。分段 → epoch → 清洗 → 频谱/相干/BSR/ApEn → 拼接为 `(n_segments, n_features)` 特征矩阵。

> EEG 不做"全程全局平均"分析——8 小时睡眠的平均频谱会把觉醒期 alpha、N3 期 delta、REM 期 theta 搅在一起。分段分析才是正确粒度。

## 2. 能力边界

| 模块 | 功能 |
|------|------|
| 预处理 | Butterworth 带通/陷波滤波 |
| 分段 | 固定时长切分，可配 overlap，保留 epoch 绝对起始时间 |
| 清洗 | 振幅阈值剔除伪迹 epoch，时间同步剔除 |
| 坏通道 | 基于有效 epoch 占比识别 |
| 频谱 | Welch PSD，5 频带功率积分 |
| 相干 | 幅值平方相干 (MSC) |
| 非线性 | ApEn（自实现）、BSR |
| 编排 | `eeg_analyze()` — 端到端分析入口 |
| 扁平化 | `eeg_flatten()` — 展平为单层 dict |

**不提供**：全局平均频谱、实时流处理、C DLL WSI、MNE 流程 (ICA/CSP)、设备通信、可视化、深度学习特征。

## 3. 目录结构

```
neurocardiokit/eeg/
├── __init__.py          # 公开 API：eeg_analyze, eeg_flatten, BANDS
├── _preprocess.py       # eeg_preprocess()
├── _epoch.py            # eeg_epoch(), eeg_clean_epochs(), eeg_bad_channels()
├── _spectral.py         # eeg_psd(), eeg_band_power()
├── _connectivity.py     # eeg_coherence()
├── _nonlinear.py        # eeg_apen(), eeg_bsr()
└── _utils.py            # eeg_flatten(), eeg_split_segments()
```

核心依赖仅 numpy + scipy。公开只暴露 `__init__.py` 中的 2 个函数。

## 4. 数据模型

### 4.1 双层窗口

```
全程 EEG (n_chans, n_samples)
  └─ segment_len=30s 大窗 × N 段  (AASM 睡眠分期标准)
       └─ epoch_len=2s 子窗 × M 个  (频谱估计基本单元)
```

- **30s 大窗**：AASM 临床金标准，够长包含完整睡眠波形，够短捕捉阶段转换
- **2s 子窗**：EEG 非平稳信号，2s 内可近似平稳；256Hz 时 512 点 → FFT 分辨率 0.5 Hz，恰好区分 delta/theta

### 4.2 默认频带与 BSR 阈值

```python
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}
BSR_THRESHOLDS = (5, 10, 20)  # µV — 深度/中度/轻度抑制
```

频带可通过参数覆盖。delta 下界取 0.5 Hz（临床标准）而非 PhysioNet 的 2.0 Hz。

### 4.3 输出结构 — 与 HRV 对齐

```python
result = {
    # Recording-level metadata
    'fs': 256, 'n_channels': 19,
    'channel_names': ['Fp1', 'Fp2', ...] or None,
    'epoch_len_s': 2.0, 'epoch_overlap_s': None,
    't_start': 0.0,

    # Global — EEG 不做全局分析，留空与 HRV 结构对齐
    'global': None,

    # Segments
    'segments': {
        'segment_len_s': 30.0,
        'segment_overlap_s': None,
        'n_segments': 20,
        'segment_t_starts': (n_segments,) array,
        'n_valid_epochs': (n_segments,) array,
        'feature_matrix': (n_segments, n_features) float64,
        'feat_names': list of str,
    },
}
```

与 `neurocardiokit.hrv.hrv_analyze` 输出结构一致：顶层 `global` + `segments`，分段特征在 `segments.feature_matrix` 中。

| 特征类 | 展开列数 | 命名模板 |
|--------|---------|---------|
| band_power | n_ch × 5 | `bp_{band}_ch{ch}` |
| band_power_rel | n_ch × 5 | `bpr_{band}_ch{ch}` |
| bsr | n_ch × 3 | `bsr_{thr}uv_ch{ch}` |
| apen | n_ch | `apen_ch{ch}` |
| coherence | C(n_ch,2) × 5 | `coh_{band}_ch{a}_ch{b}` (a<b) |

## 5. 公开 API

```python
from neurocardiokit.eeg import eeg_analyze, eeg_flatten
```

### `eeg_analyze(data, fs, ...)` → dict

```python
def eeg_analyze(
    data,              # (n_chans, n_samples) — 原始 EEG, µV
    fs,                # 采样率 Hz
    t_start=0,         # 数据绝对起始时间 (s)
    channel_names=None,# 通道名 list，透传到结果
    segment_len=30,    # 段长 (s)
    segment_overlap=None,# 段步长 (s), None=不重叠
    epoch_len=2,       # epoch 时长 (s)
    epoch_overlap=None,# epoch 步长 (s), None=不重叠
    bands=None,        # 频带定义, None=用 BANDS
    clean=True,        # 是否清洗 epoch
    min_amp=1.0,       # 清洗振幅下界 µV
    max_amp=300.0,     # 清洗振幅上界 µV
    feat_include=None, # 要计算的特征类列表, None=全部
):
```

管线：预处理 (0.5-45Hz) → 按 `segment_len` 切段 → 每段 epoch → 清洗 → PSD/频带/相干/ApEn/BSR → 拼接 feature_matrix。

`feat_include` 同时控制计算和输出——未列出的特征类不会计算，节省时间。可选值：`band_power`, `band_power_rel`, `coherence`, `apen`, `bsr`。

### `eeg_flatten(result)` → dict

将 `segments.feature_matrix` 展平为 `{feat_name}_seg{seg}: value` 的单层标量 dict，用于 CSV/SQL 导出。ML 直接用 `feature_matrix`。

## 6. 使用示例

### 6.1 基本用法

```python
from neurocardiokit.eeg import eeg_analyze

result = eeg_analyze(data, fs=256, segment_len=30, epoch_len=2,
                     channel_names=['Fp1','Fp2','F3','F4','C3','C4','P3','P4',
                                    'O1','O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz'])
# result['segments']['feature_matrix'].shape → (960, 285)
# result['segments']['feat_names'][:3] → ['bp_delta_ch0', 'bp_theta_ch0', 'bp_alpha_ch0']
```

### 6.2 绘图 — 通过 feat_names 定位

```python
seg = result['segments']
X, names, t = seg['feature_matrix'], seg['feat_names'], seg['segment_t_starts']

def col(name): return names.index(name)

# Fp1 (ch0) 全频带
for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
    plt.plot(t, X[:, col(f'bp_{band}_ch0')], label=band)

# BSR (10µV) 各通道
bsr_cols = [col(f'bsr_10uv_ch{ch}') for ch in range(result['n_channels'])]
plt.plot(t, X[:, bsr_cols])

# 相邻段差分 — 检测频谱突变
delta = np.diff(X, axis=0)
```

### 6.3 喂给 LSTM / sklearn

```python
seg = result['segments']
X, names = seg['feature_matrix'], seg['feat_names']

# sklearn
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier().fit(X, y)
importances = dict(zip(names, rf.feature_importances_))

# PyTorch LSTM: X → sliding window → (batch, seq_len, n_features)
import torch
model = torch.nn.LSTM(X.shape[1], 128, batch_first=True)

# 精简特征: 只取频带功率
result2 = eeg_analyze(data, fs=256, segment_len=30,
                      feat_include=['band_power', 'band_power_rel'])
```

### 6.4 EEG-HRV 联合分析

```python
from neurocardiokit.eeg import eeg_analyze
from neurocardiokit.hrv import hrv_analyze

eeg_result = eeg_analyze(eeg_data, fs=256, segment_len=30, feat_include=['band_power'])
hrv_result = hrv_analyze(rpos=rpos, fs_ecg=250, segment_len=30)

# Same global/segments structure — align and stack
X = np.hstack([eeg_result['segments']['feature_matrix'],
               hrv_result['segments']['feature_matrix']])
```

### 6.5 分步使用内部函数

```python
from neurocardiokit.eeg import eeg_preprocess, eeg_epoch, eeg_clean_epochs, eeg_psd, eeg_band_power

filtered = eeg_preprocess(data, fs=256)
epochs, epoch_times = eeg_epoch(filtered, epoch_len_s=2, fs=256, t_start=100.0)
clean, clean_times = eeg_clean_epochs(epochs, epoch_times=epoch_times)
psd, freqs = eeg_psd(clean, fs=256)
bp, names = eeg_band_power(psd, freqs)
```

## 7. 设计决策

| 决策 | 理由 |
|------|------|
| 不做全局平均频谱 | 不同脑状态的频谱混合后无临床意义 |
| 唯一入口 `eeg_analyze` | 端到端，数据时长 < segment_len 时自动退化为单段 |
| 不依赖 MNE / EntropyHub | 仅需 scipy 滤波 + numpy FFT；ApEn 自实现 |
| 频带可配置 | 默认临床标准 (0.5-45Hz)，可通过 `bands=` 覆盖 |
| 分段默认 30s | AASM 睡眠分期标准帧长 |
| `feat_include` 统一控制特征计算 | 同时控制是否计算+是否输出，未列出的特征类不会浪费计算时间 |
| `global` + `segments` 统一结构 | 与 HRV 对齐，`global` 为 None 占位，联合分析时结构一致 |
| 输出 `(n_segments, n_features)` + `feat_names` | sklearn/PyTorch 原生格式，列名用于特征重要性/SHAP |
| `channel_names` 透传 | 不参与计算，仅标注，与 MNE `raw.info['ch_names']` 兼容 |

## 8. 依赖与测试

**依赖**：numpy, scipy。不依赖 MNE、EntropyHub、PyQt6。

**测试** (`tests/test_eeg.py`)：预处理 shape 不变、epoch 数量/时间正确、清洗同步剔除、PSD 满足 Parseval、相干自相干=1、ApEn/BSR 边界、feature_matrix shape/命名正确、空数据不报错。
