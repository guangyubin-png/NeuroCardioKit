# NeuroCardioKit

EEG + ECG/HRV 联合分析工具箱，面向脑心轴研究。

## 安装

```bash
pip install git+https://github.com/guangyubin-png/NeuroCardioKit.git
```

OSEA QRS 检测引擎（C++）首次导入时自动从 Release 下载预编译库，无需安装编译器。

## 快速开始

### EEG 分析

```python
from neurocardiokit.eeg import eeg_analyze, eeg_flatten

result = eeg_analyze(data, fs=256, segment_len=30, epoch_len=2,
                     feat_include=['band_power', 'band_power_rel', 'coherence', 'apen', 'bsr'])

X = result['segments']['feature_matrix']  # (n_segments, n_features)
```

### HRV 分析

```python
from neurocardiokit.hrv import hrv_analyze

result = hrv_analyze(rpos=r_positions, fs_ecg=250, segment_len=300,
                     feat_include=['time', 'freq', 'nonlinear'])

X = result['segments']['feature_matrix']  # (n_segments, 33)
```

### ECG QRS 检测

```python
from neurocardiokit.osea import OseaEngine

engine = OseaEngine()
results = engine.process_array(ecg_samples)  # 200 Hz, 200 ADC units/mV
r_positions = [r['index'] for r in results if r['type_label'] == 'NORMAL']
```

### 脑心轴联合分析

```python
eeg_result = eeg_analyze(eeg_data, fs=256, segment_len=30)
hrv_result = hrv_analyze(rpos=rpos, fs_ecg=250, segment_len=30)

import numpy as np
X_joint = np.hstack([
    eeg_result['segments']['feature_matrix'],
    hrv_result['segments']['feature_matrix']
])
```

## 模块

| 模块 | 说明 |
|------|------|
| `neurocardiokit.eeg` | EEG 频谱、连接性、非线性特征 |
| `neurocardiokit.hrv` | HRV 时域、频域、非线性特征 |
| `neurocardiokit.osea` | ECG QRS 检测与搏动分类 |

## 依赖

- Python >= 3.8
- numpy, scipy, hrv-analysis

## License

MIT
