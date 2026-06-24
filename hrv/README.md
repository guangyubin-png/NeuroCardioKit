# neurocardiokit/hrv — HRV Analysis Toolbox

心率变异性（HRV）分析工具箱，基于 [hrv-analysis](https://pypi.org/project/hrv-analysis/) 单段特征提取，向上提供分窗、汇总、编排等能力。

## Quickstart

```python
from neurocardiokit.hrv import hrv_global, hrv_segment, hrv_from_rpos

# 1. 已有干净 NN 序列（ms），只算全局特征
result = hrv_global(nn_intervals)
print(result['time']['sdnn'], result['freq']['lf_hf_ratio'])

# 2. 已有 NN 序列，要 5 分钟分段分析
seg = hrv_segment(nn_intervals, window_min=5)
print(seg['summary']['cross']['sdann'])  # 跨窗 SDANN

# 3. 从 R 波位置开始，端到端
rpos = ...      # R 波样本索引 (来自 EcgAnalyzer / qrsdet)
anntype = ...   # 搏动类型 (1=正常)，可选
result = hrv_from_rpos(rpos, anntype, fs_ecg=250, window_min=5)
```

## API

### `hrv_global(nn, fs_nn=4)` → dict

全程特征（不对 NN 做分段）。

| 键 | 内容 |
|----|------|
| `time` | mean_nn, sdnn, rmssd, pnn50, nn50, mean_hr, cvnn, ... |
| `freq` | total_power, vlf, lf, hf, lf_hf_ratio, lfnu, hfnu |
| `nonlinear` | sd1, sd2, sd_ratio, sampen, csi, cvi, apen, dfa_alpha1, dfa_alpha2 |

### `hrv_segment(nn, fs_nn=4, window_min=5, overlap=0)` → dict

按固定时长切窗，每窗算特征，再出跨窗汇总。

| 键 | 内容 |
|----|------|
| `window_min` | 窗长（分钟） |
| `n_valid` | 有效窗数（每窗 ≥ 10 beats） |
| `time` | `[{窗1特征}, {窗2特征}, ...]` |
| `freq` | 同上 |
| `nonlinear` | 同上 |
| `summary.time` | 时域各字段的 mean/std/min/max |
| `summary.cross` | **跨窗独有**：sdann, sdnn_i, mean_hr, cvnn |

### `hrv_from_rpos(rpos, anntype=None, fs_ecg=250, window_min=5)` → dict

R 波 → NN 清洗 → Global + Segment 全流程。

## 预处理逻辑

```
rpos → RR(ms) → [anntype预筛] → hrvanalysis.get_nn_intervals()
                                      ├── remove_outliers (300-2000ms)
                                      ├── interpolate (linear)
                                      ├── remove_ectopic_beats (Malik)
                                      └── interpolate (linear)
```

两层互补：anntype 用 QRS 波形形态筛，Malik 用 RR 间期变化筛。

## 模块结构

| 文件 | 职责 |
|------|------|
| `__init__.py` | 公开 API |
| `_preprocess.py` | R 波 → NN 清洗 |
| `_time_domain.py` | 时域特征 + 跨窗 SDANN/SDNNI |
| `_freq_domain.py` | 频域特征（Welch/Lomb） |
| `_nonlinear.py` | Poincaré / 熵 / DFA |
| `_utils.py` | 分窗、汇总 |

## 依赖

- `numpy`, `scipy`
- `hrv-analysis` (v1.0.5, 特征计算引擎)
- `pandas` (hrv-analysis 间接依赖)
- `nolds` (hrv-analysis 间接依赖，用于样本熵)
