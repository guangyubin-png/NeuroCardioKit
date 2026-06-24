# HRV Analysis Toolbox — Design

## 1. 目标

构建 `neurocardiokit/hrv/` HRV 分析工具箱，与 `neurocardiokit/eeg/` 对齐——**统一入口 `hrv_analyze`、`feat_include` 控制特征类、`feature_matrix` + `feat_names` 输出、`segment_len`/`segment_overlap` 秒为单位分段**。

| 分析模式 | segment_len | 输出                                    | 用途                                 |
| -------- | ----------- | --------------------------------------- | ------------------------------------ |
| Global   | None        | global.feature_matrix (1, n)            | 24h 动态心电、整段静息 ECG           |
| Segment  | 300 (5min)  | global + segments.feature_matrix (m, n) | 短时 HRV、趋势分析、EEG-HRV 联合分析 |

## 2. 底层依赖

以 `hrvanalysis` v1.0.5（PyPI 包 `hrv-analysis`）为特征计算引擎。它提供**单段 NN → 单次特征**的原子函数，不做分段或编排。

| 函数                              | 输出字段                                                                           |
| --------------------------------- | ---------------------------------------------------------------------------------- |
| `get_time_domain_features()`      | mean_nni, sdnn, rmssd, sdsd, nni_50, pnni_50, nni_20, pnni_20, mean_hr, cvnni, ... |
| `get_frequency_domain_features()` | lf, hf, lf_hf_ratio, lfnu, hfnu, total_power, vlf                                  |
| `get_poincare_plot_features()`    | sd1, sd2, ratio_sd2_sd1                                                            |
| `get_sampen()`                    | sampen                                                                             |
| `get_csi_cvi_features()`          | csi, cvi, Modified_csi                                                             |

预处理一站式：`get_nn_intervals(rr_list)` = remove_outliers → interpolate → remove_ectopic_beats(malik) → interpolate。

**hrvanalysis 不提供的**（我们自己写）：rpos→RR 转换、分窗、DFA、ApEn、统一编排 API、feature_matrix 拼接。

## 3. 目录结构

```
neurocardiokit/hrv/
├── __init__.py          # hrv_analyze（统一入口）, flatten_result
├── _preprocess.py       # rpos_to_nn() — R波→NN清洗
├── _time_domain.py      # time_domain() + cross_segment_features()
├── _freq_domain.py      # freq_domain() — Welch/Lomb
├── _nonlinear.py        # nonlinear() — Poincaré/熵/DFA/ApEn
└── _utils.py            # split_windows(), flatten_result()
```

硬依赖：`hrvanalysis` + `numpy`。不依赖 PyQt6。

## 4. 数据流

```
rpos + fs_ecg
    │
    ▼
_preprocess.rpos_to_nn()  →  nn (ms) + nn_time (s, 绝对时间)
    │
    ▼
hrv_analyze(nn=..., segment_len=300)
    │
    ├── global  ─── 全程 time/freq/nonlinear → feature_matrix (1, n)
    │
    └── segments ─── split_windows(nn, nn_time, segment_len_s, step_s)
         ├── 每窗: time / freq / nonlinear → 各窗 dict 列表
         └── 拼接 → feature_matrix (n_segments, n) + segment_t_starts
```

分段以 `nn_time` 绝对时间为准，不用 `cumsum(nn)`（插值后的 NN 会累积偏移）。

**EEG-HRV 联合分析**：两个模块使用相同的 `segment_len`（秒）和 `segment_overlap`（秒）参数，同一次记录切出 EEG 频谱特征和 HRV 时/频/非线性特征后按 `segment_t_starts` 对齐，`np.hstack([eeg_matrix, hrv_matrix])` 即可。

## 5. 公开 API

```python
from neurocardiokit.hrv import hrv_analyze, flatten_result
```

### `hrv_analyze(rpos=None, nn=None, ...)` → dict

```python
def hrv_analyze(
    rpos=None,              # R-peak sample indices
    nn=None,                # pre-computed NN intervals (ms)
    anntype=None,           # beat type labels, 1=normal (only with rpos)
    fs_ecg=250,             # ECG sampling rate Hz (only with rpos)
    t_start=0,              # absolute start time of first sample (s)
    nn_time=None,           # absolute time per NN (s), only with nn
    fs_nn=4,                # NN resampling freq for freq-domain (Hz)
    segment_len=300,        # segment length in seconds. None = global only
    segment_overlap=None,   # overlap between segments in seconds. None = contiguous
    feat_include=None,      # ['time', 'freq', 'nonlinear']. None = all
):
```

两种输入互斥（`rpos` 或 `nn` 二选一）。返回结构：

```python
{
    # Metadata
    'nn': np.ndarray,               # clean NN intervals (ms)
    'nn_time': np.ndarray,          # absolute time per NN (s)
    'n_beats': int,
    'duration_min': float,

    # Global — always present
    'global': {
        'feature_matrix': np.ndarray,   # (1, n_features) float64
        'feat_names': list of str,
    },

    # Segments — only when segment_len is not None
    'segments': {
        'segment_len_s': 300.0,
        'segment_overlap_s': None,
        'n_segments': int,
        'segment_t_starts': np.ndarray,   # (n_segments,)
        'feature_matrix': np.ndarray,     # (n_segments, n_features) float64
        'feat_names': list of str,
    },
}
```

### 特征列（33 列）

| 类别      | 列数 | 列名                                                                                                                   |
| --------- | ---- | ---------------------------------------------------------------------------------------------------------------------- |
| time      | 16   | mean_nn, sdnn, rmssd, sdsd, nn50, pnn50, nn20, pnn20, mean_hr, min_hr, max_hr, std_hr, cvnn, cvsd, median_nn, range_nn |
| freq      | 7    | total_power, vlf, lf, hf, lf_hf_ratio, lfnu, hfnu                                                                      |
| nonlinear | 10   | sd1, sd2, sd_ratio, sampen, csi, cvi, modified_csi, apen, dfa_alpha1, dfa_alpha2                                       |

`global.feature_matrix` 和 `segments.feature_matrix` 列相同，`feat_include` 控制计算哪些类别。

### `flatten_result(result)` → dict

将 global 和 segments 的 `feature_matrix` 展平为单层标量 dict：

```
global_{feat}: value
seg{i}_{feat}: value
```

用于 CSV/SQL 导出。ML 直接用 `feature_matrix`。

### Step-by-step 函数（仍可单独导入）

| 函数                     | 用途                      |
| ------------------------ | ------------------------- |
| `rpos_to_nn(rpos, ...)`  | R 波位置 → 清洗后 NN 间期 |
| `time_domain(nn)`        | 单段时域特征 dict         |
| `freq_domain(nn, ...)`   | 单段频域特征 dict         |
| `nonlinear(nn)`          | 单段非线性特征 dict       |
| `split_windows(nn, ...)` | NN → 按时间窗切分         |
| `flatten_result(result)` | 嵌套 dict → 单层标量 dict |

## 6. 使用示例

### 6.1 基本用法

```python
from neurocardiokit.hrv import hrv_analyze

# Global analysis only
result = hrv_analyze(nn=nn, segment_len=None)
X_global = result['global']['feature_matrix']  # (1, 33)

# 5-min segment analysis
result = hrv_analyze(rpos=rpos, fs_ecg=250, segment_len=300)
X_seg = result['segments']['feature_matrix']   # (n_segments, 33)

# Feature selection by name
names = result['segments']['feat_names']
sdnn_idx = names.index('sdnn')
lf_hf_idx = names.index('lf_hf_ratio')
```

### 6.2 EEG-HRV 联合分析

```python
from neurocardiokit.eeg import eeg_analyze
from neurocardiokit.hrv import hrv_analyze

# Same segment_len, same segment_overlap → aligned segment_t_starts
eeg_result = eeg_analyze(eeg_data, fs=256, segment_len=30, feat_include=['band_power'])
hrv_result = hrv_analyze(rpos=rpos, fs_ecg=250, segment_len=30)

# Align and stack — both have identical global/segments structure
X = np.hstack([eeg_result['segments']['feature_matrix'],
               hrv_result['segments']['feature_matrix']])
all_names = (eeg_result['segments']['feat_names'] +
             hrv_result['segments']['feat_names'])
```

### 6.3 喂给 sklearn / PyTorch

```python
# Global features as single sample
X, names = result['global']['feature_matrix'], result['global']['feat_names']

# Segment features as time series
X, names = result['segments']['feature_matrix'], result['segments']['feat_names']
# sklearn
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier().fit(X, y)
# PyTorch LSTM
import torch
model = torch.nn.LSTM(X.shape[1], 128, batch_first=True)
```

## 7. 设计决策

| #   | 决策                                                          | 理由                                                        |
| --- | ------------------------------------------------------------- | ----------------------------------------------------------- |
| 1   | **hrvanalysis 硬依赖**                                        | Welch PSD、Poincaré、样本熵等自实现无增益                   |
| 2   | **`segment_len` 秒为单位，默认 300（5min）**                  | 与 `eeg_analyze` 对齐，支持 EEG-HRV 联合分析                |
| 3   | **`segment_overlap` 秒为单位**                                | 同上，与 EEG 参数一致                                       |
| 4   | **分段基于绝对时间而非 beat 数**                              | 累积 NN 有偏移，`nn_time` 为权威时间源                      |
| 5   | **唯一入口 `hrv_analyze`**                                    | 与 `eeg_analyze` 对齐。接受 R 波或 NN，分步函数仍可导入     |
| 6   | **`feat_include` 控制特征类**                                 | 未列出的不计算，节省时间。与 EEG 对齐                       |
| 7   | **global + segments 都输出 `feature_matrix` + `feat_names`**  | 与 EEG 对齐，sklearn/PyTorch 原生格式，同名字段便于联合分析 |
| 8   | **global 始终存在，segments 仅在 segment_len 非 None 时存在** | 全局特征是 HRV 临床标准，短时分段是附加分析                 |
| 9   | **DFA / ApEn 自实现**                                         | hrvanalysis 不提供，算法简单                                |
| 10  | **每窗最少 10 beats**                                         | 低于 10 个搏动的统计量不可靠                                |
| 11  | **`anntype` 可选**                                            | 形态学预筛和 Malik 规则互补                                 |
| 12  | **`flatten_result()` 扁平化**                                 | feature_matrix → 单层标量，便于 CSV/SQL 导出                |
