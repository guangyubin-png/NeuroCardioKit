# NeuroCardioKit — 脑心轴分析工具箱

EEG + ECG/HRV 联合分析工具箱，面向睡眠数据的脑心轴研究和疾病分类。
统一接口设计，输出 sklearn/PyTorch 原生的 `(n_samples, n_features)` 特征矩阵。

## 项目结构

```
NeuroCardioKit/
├── setup.py                 # pip install 入口 (package_dir 映射 neurocardiokit)
├── pyproject.toml           # 构建系统声明
├── __init__.py              # neurocardiokit 包根
├── eeg/                    # EEG 分析：频谱、连接性、非线性
│   ├── __init__.py         # eeg_analyze() 统一入口 + eeg_flatten()
│   ├── _preprocess.py      # Butterworth 带通/陷波滤波 (0.5-45Hz)
│   ├── _epoch.py           # 切分 epoch + 伪迹清洗 + 坏通道检测
│   ├── _spectral.py        # Welch PSD + 5频带功率积分
│   ├── _connectivity.py    # 幅值平方相干 (MSC)
│   ├── _nonlinear.py       # ApEn 近似熵 + BSR 爆发抑制比
│   └── _utils.py           # 分段拆分 + 扁平化导出
├── hrv/                    # HRV 分析：时域、频域、非线性
│   ├── __init__.py         # hrv_analyze() 统一入口 + flatten_result()
│   ├── _preprocess.py      # R波位置 → 清洗后 NN 间期
│   ├── _time_domain.py     # 时域特征 (SDNN, RMSSD, pNN50 等 16 项)
│   ├── _freq_domain.py     # 频域特征 (VLF/LF/HF, LF/HF ratio 等 7 项)
│   ├── _nonlinear.py       # 非线性特征 (Poincaré, 熵, DFA 等 10 项)
│   └── _utils.py           # 时间窗切分 + 汇总 + 扁平化
├── osea/                   # ECG QRS 检测与搏动分类 (C++ 源码发布)
│   ├── _osea_bridge.py     # OseaEngine — ctypes 封装 Hamilton-Tompkins 算法
│   ├── build_osea.py       # 编译 osea.dll (支持 MSVC/MinGW/Conda)
│   ├── osea_bridge.cpp     # C++ bridge 源码
│   ├── *.cpp / *.h         # OSEA C++ 核心算法源码
│   └── test_osea.py        # 合成 ECG 测试
├── tests/
│   ├── test_eeg.py         # EEG 模块测试 (需 PhysioNet EDF 文件)
│   └── test_hrv.py         # HRV 模块测试 (合成数据)
├── eeg/DESIGN.md           # EEG 模块设计文档
├── hrv/DESIGN.md           # HRV 模块设计文档
└── hrv/README.md           # HRV 旧版 README
```

## 快速开始

### 1. 环境与依赖

```bash
# 1. 激活 conda 环境
conda activate ecg_ai
#   conda 路径: D:\anaconda3\envs\ecg_ai\python  (ecg_ai, Python 3.10)

# 2. 克隆仓库
git clone <repo-url>
cd NeuroCardioKit

# 3. 一键安装（自动编译 OSEA C++ 源码）
pip install -e .
```

`pip install -e .` 会：
- 以 `neurocardiokit` 包名注册到当前 Python 环境
- 自动调用 `osea/build_osea.py` 编译 C++ 源码生成 `osea.dll`
- 以开发模式安装，修改源码后无需重新安装

安装完成后，所有模块可直接导入：

```python
from neurocardiokit.eeg import eeg_analyze, eeg_flatten
from neurocardiokit.hrv import hrv_analyze, flatten_result
from neurocardiokit.osea import OseaEngine
```

### 2. EEG 分析

```python
from neurocardiokit.eeg import eeg_analyze, eeg_flatten

# data: (n_channels, n_samples) ndarray, 单位 µV
result = eeg_analyze(
    data, fs=256,
    segment_len=30,       # AASM 标准 30s 分段
    epoch_len=2,          # 2s 子窗
    channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4'],
    feat_include=['band_power', 'band_power_rel', 'coherence', 'apen', 'bsr'],
    clean=True,            # 自动剔除振幅异常 epoch
)

# 特征矩阵: (n_segments, n_features)
X = result['segments']['feature_matrix']
feat_names = result['segments']['feat_names']
# X.shape 示例: (960, 285) — 19通道×5频带×3类特征

# 导出为 CSV 友好的扁平字典
flat = eeg_flatten(result)  # {'bp_delta_ch0_seg0': 12.3, ...}
```

### 3. HRV 分析

```python
from neurocardiokit.hrv import hrv_analyze, flatten_result

# 方式一：从 R 波位置开始
result = hrv_analyze(
    rpos=r_positions,        # R 波样本索引
    anntype=beat_types,      # 搏动类型 (1=正常, 可选)
    fs_ecg=250,              # ECG 采样率
    segment_len=300,         # 5min 分段
)

# 方式二：从已清洗的 NN 间期开始
result = hrv_analyze(
    nn=nn_intervals,         # NN 间期 (ms)
    nn_time=time_stamps,     # 绝对时间 (可选)
    segment_len=300,
    feat_include=['time', 'freq', 'nonlinear'],  # 可选
)

# 全局特征
X_global = result['global']['feature_matrix']   # (1, 33)
# 分段特征
X_seg = result['segments']['feature_matrix']    # (n_segments, 33)
```

### 4. ECG QRS 检测 (OSEA)

OSEA 是 C++ 源码发布的 QRS 检测引擎。`pip install -e .` 会根据平台自动选择编译器并编译：

| 平台      | 默认编译器    | 编译器缺失时自动处理                              |
|-----------|-------------|------------------------------------------------|
| Windows   | MinGW       | conda 环境下自动 `conda install m2w64-gcc`       |
| Linux     | g++         | conda 环境下自动 `conda install gcc`；否则提示 apt/yum |
| macOS     | clang++     | conda 环境下自动 `conda install clang`；否则提示 xcode-select |

```python
from neurocardiokit.osea import OseaEngine

engine = OseaEngine()
results = engine.process_array(ecg_samples)  # 200 Hz, 200 ADC units/mV

# results: [{'index': 样本点, 'delay': 延迟, 'type': 类型, 'type_label': 'NORMAL'}, ...]
r_positions = [r['index'] for r in results if r['type_label'] == 'NORMAL']
```

> 编译器缺失时，`pip install` 会**自动通过 conda 安装**编译器（需在 conda 环境下）。非 conda 环境则打印安装指引。

### 5. EEG-HRV 联合分析（脑心轴核心用法）

```python
from neurocardiokit.eeg import eeg_analyze
from neurocardiokit.hrv import hrv_analyze
import numpy as np

# 使用相同的 segment_len 保证时间对齐
eeg_result = eeg_analyze(eeg_data, fs=256, segment_len=30,
                         feat_include=['band_power', 'band_power_rel'])
hrv_result = hrv_analyze(rpos=rpos, fs_ecg=250, segment_len=30)

# 拼接特征矩阵 — 同一行对应同一时间段
X_joint = np.hstack([
    eeg_result['segments']['feature_matrix'],
    hrv_result['segments']['feature_matrix']
])
# X_joint.shape: (n_segments, n_eeg_features + n_hrv_features)
```

## 核心设计

### 统一输出结构

EEG 和 HRV 的 `*_analyze()` 返回相同的结构：

```python
{
    'global': {                          # 全程特征
        'feature_matrix': ndarray,       # (1, n_features) 或 None (EEG)
        'feat_names': list of str,
    },
    'segments': {                        # 分段时间窗特征
        'segment_len_s': float,
        'segment_overlap_s': float,
        'n_segments': int,
        'segment_t_starts': ndarray,     # 每段绝对起始时间
        'feature_matrix': ndarray,       # (n_segments, n_features)
        'feat_names': list of str,
    },
}
```

### feature_matrix — 机器学习就绪

- 每行 = 一个时间段
- 每列 = 一个特征（通过 `feat_names` 索引）
- 直接喂给 sklearn：`RandomForestClassifier().fit(X, y)`
- 直接喂给 PyTorch：`nn.LSTM(n_features, 128, batch_first=True)`

### feat_include — 按需计算

| EEG 可选特征类    | 说明                    |
|------------------|------------------------|
| `band_power`     | 绝对频带功率 (µV²)       |
| `band_power_rel` | 相对频带功率 (%)         |
| `coherence`      | 通道间幅值平方相干        |
| `apen`           | 近似熵                   |
| `bsr`            | 爆发抑制比 (5/10/20 µV) |

| HRV 可选特征类  | 列数 | 说明                  |
|---------------|------|----------------------|
| `time`        | 16   | SDNN, RMSSD, pNN50 等 |
| `freq`        | 7    | VLF, LF, HF, LF/HF 等 |
| `nonlinear`   | 10   | Poincaré, 熵, DFA 等  |

### 数据模型对照

| 参数           | EEG 默认  | HRV 默认   | 含义        |
|---------------|-----------|------------|-------------|
| segment_len   | 30s       | 300s (5min)| 分段时间窗   |
| epoch_len     | 2s        | —          | 子窗 (PSD 估计) |

## 机器学习工作流示例

```python
# 1. 加载数据
import pyedflib
import numpy as np

f = pyedflib.EdfReader('sleep_recording.edf')
eeg_data = np.array([f.readSignal(i) for i in range(f.signals_in_file)])

# 2. 提取特征
result = eeg_analyze(eeg_data, fs=256, segment_len=30,
                     feat_include=['band_power', 'band_power_rel', 'apen'])

# 3. 构建训练集
X = result['segments']['feature_matrix']
y = sleep_stages       # 每段对应的睡眠分期标签
y_disease = patient_diagnosis  # 或疾病分类标签

# 4. 训练
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

clf = RandomForestClassifier()
scores = cross_val_score(clf, X, y, cv=5)
print(f'CV accuracy: {np.mean(scores):.3f}')

# 5. 特征重要性
clf.fit(X, y)
importances = list(zip(result['segments']['feat_names'],
                       clf.feature_importances_))
importances.sort(key=lambda x: -x[1])
for name, imp in importances[:10]:
    print(f'  {name}: {imp:.4f}')
```

## 运行测试

```bash
# EEG 测试（需要 PhysioNet EDF 文件，编辑 test_eeg.py 中的 EDF_PATH）
python tests/test_eeg.py

# HRV 测试（合成数据，无需外部文件）
python tests/test_hrv.py

# OSEA 测试（合成 ECG）
python osea/test_osea.py
```

## 依赖项

**Python 环境**：推荐 conda 环境 `ecg_ai`

| 包            | 用途            | 必需 |
|---------------|----------------|------|
| numpy         | 数组运算         | 是   |
| scipy         | 信号滤波, FFT    | 是   |
| hrv-analysis  | HRV 特征计算引擎  | 是   |
| pyedflib      | EDF 文件读取     | 测试  |
| pandas        | hrv-analysis 依赖 | 间接 |
| nolds         | 样本熵 (hrv-analysis) | 间接 |
| C++ 编译器    | 编译 OSEA 共享库 | 可选 (OSEA 功能需要) |

OSEA 模块为 **C++ 源码形式随仓库分发**，Python 端通过 ctypes 调用编译后的共享库（Windows `.dll` / Linux `.so` / macOS `.dylib`）。`pip install` 会自动检测平台和编译器进行编译；若编译器缺失，脚本会打印对应平台的安装指引。

## 注意事项

1. **Conda 环境**：使用 `ecg_ai` 环境，`conda activate ecg_ai`。
2. **安装方式**：`pip install -e .` 一键安装，自动编译 OSEA 并注册 `neurocardiokit` 包名。
3. **EEG 输入格式**：`(n_channels, n_samples)` ndarray，单位 µV。
4. **ECG 输入格式**（OSEA）：200 Hz 采样率，200 ADC units/mV，基线 ~0。
5. **频带标准**：delta (0.5-4Hz), theta (4-8Hz), alpha (8-13Hz), beta (13-30Hz), gamma (30-45Hz)。
6. **OSEA 跨平台编译**：`pip install` 自动检测平台编译（Windows 优先 MinGW，Linux 用 g++，macOS 用 clang++）。编译器缺失时打印安装指引（如 `sudo apt install build-essential`）。也可手动 `python osea/build_osea.py --compiler gcc`。
