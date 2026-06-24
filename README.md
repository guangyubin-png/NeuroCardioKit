# NeuroCardioKit

EEG + ECG/HRV 联合分析工具箱，面向睡眠数据的脑心轴研究。

统一接口，输出 sklearn/PyTorch 原生的 `(n_samples, n_features)` 特征矩阵。

## 安装

```bash
# 本地开发安装（含 OSEA C++ 自动编译）
git clone https://github.com/guangyubin-png/NeuroCardioKit.git
cd NeuroCardioKit
pip install -e .

# 或直接 pip 安装
pip install git+https://github.com/guangyubin-png/NeuroCardioKit.git
```

`pip install -e .` 会自动编译 OSEA C++ 源码生成共享库（Windows `.dll` / Linux `.so` / macOS `.dylib`）。

## 快速开始

### ECG QRS 检测 (OSEA)

```python
from neurocardiokit.osea import OseaEngine

engine = OseaEngine()

# 高级 API：输入 mV 信号 + 采样率，直接输出原始采样率下的 R 波位置和搏动类型
ecg_mv = ...          # 任意采样率的 ECG 信号 (mV)
rpos, anntype = engine.detect_rpeaks(ecg_mv, fs=500)
# rpos:    R 波样本索引（已延时校正）
# anntype: 搏动类型 (1=NORMAL, 5=PVC, ...)
```

### HRV 分析

```python
from neurocardiokit.hrv import hrv_analyze

# 从 R 波位置出发
result = hrv_analyze(
    rpos=rpos, anntype=anntype, fs_ecg=500,
    segment_len=30,                    # 30s 分段（与 EEG 对齐）
    total_s=total_duration,            # 强制固定时间窗，空窗填 NaN
    feat_include=['time', 'freq'],     # 可选：'time', 'freq', 'nonlinear'
)

X_hrv = result['segments']['feature_matrix']  # (n_segments, 23)
```

### EEG 分析

```python
from neurocardiokit.eeg import eeg_analyze

result = eeg_analyze(
    eeg_data, fs=500,
    channel_names=['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1'],
    segment_len=30,
    feat_include=['band_power', 'band_power_rel'],
)

X_eeg = result['segments']['feature_matrix']  # (n_segments, 60)
```

### 脑心轴联合分析

```python
import numpy as np
from neurocardiokit.osea import OseaEngine
from neurocardiokit.eeg import eeg_analyze
from neurocardiokit.hrv import hrv_analyze

# 1. QRS 检测
engine = OseaEngine()
rpos, anntype = engine.detect_rpeaks(ecg_mv, fs=500)

# 2. EEG 分析
eeg_result = eeg_analyze(eeg_data, fs=500, channel_names=ch_names,
                         segment_len=30,
                         feat_include=['band_power', 'band_power_rel'])

# 3. HRV 分析（total_s 确保与 EEG 严格对齐，空窗填 NaN）
hrv_result = hrv_analyze(rpos=rpos, anntype=anntype, fs_ecg=500,
                         segment_len=30, total_s=total_dur,
                         feat_include=['time', 'freq'])

# 4. 联合特征矩阵 — 同一行 = 同一时间段
X_joint = np.hstack([
    eeg_result['segments']['feature_matrix'],
    hrv_result['segments']['feature_matrix']
])
# X_joint.shape: (n_segments, n_eeg_features + n_hrv_features)
```

完整示例见 [scripts/analyze_ecg_eeg.py](scripts/analyze_ecg_eeg.py)。

## 输出结构

EEG 和 HRV 的 `*_analyze()` 返回统一结构：

```python
{
    'global': {                           # 全程特征
        'feature_matrix': ndarray,        # (1, n_features)
        'feat_names': list of str,
    },
    'segments': {                         # 分段特征
        'segment_len_s': float,
        'n_segments': int,
        'segment_t_starts': ndarray,      # 每段起始时间 (s)
        'feature_matrix': ndarray,        # (n_segments, n_features)
        'feat_names': list of str,
    },
}
```

## 模块

| 模块                  | 说明                              |
| --------------------- | --------------------------------- |
| `neurocardiokit.osea` | ECG QRS 检测与搏动分类 (C++ 引擎) |
| `neurocardiokit.hrv`  | HRV 时域、频域、非线性特征        |
| `neurocardiokit.eeg`  | EEG 频谱、连接性、非线性特征      |

## 可选特征类

| HRV         | 特征数 | 内容                           |
| ----------- | ------ | ------------------------------ |
| `time`      | 16     | SDNN, RMSSD, pNN50, mean_hr 等 |
| `freq`      | 7      | VLF, LF, HF, LF/HF ratio 等    |
| `nonlinear` | 10     | Poincaré, 样本熵, DFA 等       |

| EEG              | 特征数 | 内容                     |
| ---------------- | ------ | ------------------------ |
| `band_power`     | 5/通道 | δ/θ/α/β/γ 绝对功率 (µV²) |
| `band_power_rel` | 5/通道 | δ/θ/α/β/γ 相对功率 (%)   |
| `coherence`      | —      | 通道间幅值平方相干       |
| `apen`           | 1/通道 | 近似熵                   |
| `bsr`            | 3/通道 | 爆发抑制比 (5/10/20 µV)  |

## 依赖

- Python >= 3.8
- numpy, scipy, hrv-analysis

## License

MIT
