# SAR-AudioData：无人机搜救场景呼救声/敲击声音频数据集

*语言：[English](../README.md) · **中文***

面向无人机搜救（Search and Rescue, SAR）场景的声音事件检测数据集。数据集聚焦两类关键的求救声音信号——**人声呼救**与**敲击声**（如被困人员敲击管道、墙体、金属物件发出的声响），并针对无人机在真实救援中不可避免的**旋翼噪声**进行了大量加噪增强，用于训练和评估在强噪声环境下依然鲁棒的声音事件检测（SED）模型。

本仓库开源其中已完成划分、可直接用于建模的四个子集：

| 子集 | 用途 | 音频数 | 有标注音频数 |
|------|------|-------:|-------------:|
| `呼救声DEV`  | 呼救声检测 · 训练/开发集 | 19184 | 17184 |
| `呼救声TEST` | 呼救声检测 · 测试集       | 715   | 715   |
| `敲击声DEV`  | 敲击声检测 · 训练/开发集 | 9811  | 7796  |
| `敲击声TEST` | 敲击声检测 · 测试集       | 1365  | 580   |

> 无标注（标注文件内容为 `[]`）的音频代表该段音频中不含目标事件，作为负样本参与训练与评估。

---

## 数据集用途

- **声音事件检测（SED）**：在连续音频中定位呼救声 / 敲击声出现的起止时间区间。
- **噪声鲁棒性研究**：数据集叠加了多种无人机旋翼噪声（悬停、急停、上下/左右直飞、降落、UAV 等工况）以及机械噪声、风噪、白噪声，可用于研究模型在真实无人机作业噪声下的检测性能。
- **无人机搜救应用**：为无人机搭载的机载声学检测系统提供训练与评测数据。

---

## 目录结构

四个子集结构一致，均由成对的音频与标注组成：

```
呼救声DEV/
└── data/
    ├── audio/          # WAV 音频文件
    └── annotations/    # 同名 JSON 标注文件

呼救声TEST/  敲击声DEV/  敲击声TEST/   # 结构同上
```

音频文件与标注文件**同名**（仅扩展名不同），一一对应：

```
data/audio/1-dev-XXXX.wav
data/annotations/1-dev-XXXX.json
```

---

## 标注格式

每个标注文件为 JSON 数组，数组中每个元素表示音频中一个目标事件的时间区间：

```json
[
  {
    "start_time": 0.5792577249575555,
    "end_time": 2.900395925297113,
    "label": "呼救声",
    "duration": 2.3211382003395578
  },
  {
    "start_time": 3.465149745331069,
    "end_time": 6.76895959252971,
    "label": "呼救声",
    "duration": 3.3038098471986412
  }
]
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `start_time` | float | 事件开始时间（秒） |
| `end_time`   | float | 事件结束时间（秒） |
| `label`      | string | 事件类别，取值为 `呼救声` 或 `敲击声` |
| `duration`   | float | 事件持续时间（秒），等于 `end_time - start_time` |

空数组 `[]` 表示该音频不含目标事件（负样本）。

---

## 各子集说明

- **呼救声DEV**：呼救声检测的训练/开发集，包含呼救声音频（含无人机噪声增强）与作为负样本补充的环境声。
- **呼救声TEST**：呼救声检测的测试集，均为叠加真实无人机工况噪声的呼救声音频数据。
- **敲击声DEV**：敲击声检测的训练/开发集，包含敲击声音频（多种信号增强）、呼救声音频作为负样本与作为负样本补充的环境声。
- **敲击声TEST**：敲击声检测的测试集，由叠加无人机噪声的敲击声音频与作为负样本补充的环境声组成。

---

## 音频规格说明

- **格式**：WAV，单声道（mono），16-bit PCM。
- **采样率**：因各子集来源不同，采样率**不统一**，混合有 16 kHz、44.1 kHz、48 kHz 等（少量其他采样率）。使用前建议按需统一重采样（如统一到 16 kHz）。
- **时长**：多数在 1–17 秒之间，均值约 4–5 秒。

> 由于数据来自采集语音、[ESC-50](https://github.com/karolpiczak/ESC-50)、[UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) 等多个来源，采样率与时长存在差异，请在数据加载阶段做统一预处理。

---

## 加噪工况与增强方式

数据集覆盖以下噪声与增强手段：

- **无人机旋翼噪声**：悬停、急停、上下直飞、左右直飞、降落、UAV 等真实工况录音。
- **信号处理增强**：时间拉伸（stretch ×0.9 / ×1.1）、白噪声（不同 SNR）、风噪、机械噪声。

---

## 数据预处理

仓库提供 `preprocess_data.py`，可将音频按滑动窗口切分并提取 Mel 频谱特征，保存为 `.npy` 文件，训练时直接加载，避免重复计算。

```bash
python preprocess_data.py \
    --audio_subdir XXXXX/audio \
    --annotation_subdir XXXXX/annotations \
    --output_dir data/preprocessed \
    --sample_rate 16000 --n_mels 40 \
    --window_size 3.0 --hop_size 1.0 \
    --event_type 敲击声
```

输出目录下会生成 `features.npy`（特征）、`labels.npy`（标签）与 `preprocess_config.json`（预处理参数及标签统计）。

---

## 下载申请

本数据集面向科研用途开放申请。如需获取完整数据，请填写申请表单，我们会尽快审核并回复：

**申请表单**：[表单](https://docs.google.com/forms/d/e/1FAIpQLSc5VrLp1BmhiSrvMD2KyTzKQ0xivBec1mPVJoik5immtQf7vg/viewform?usp=header)

在线示例与数据介绍页面见 `docs/`。

---

## 数据来源与引用

本数据集的环境声负样本部分取自以下公开数据集：

- **ESC-50**：[github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)
  
  > K. J. Piczak. *ESC: Dataset for Environmental Sound Classification.* Proceedings of the 23rd ACM International Conference on Multimedia, 2015.
- **UrbanSound8K**：[urbansounddataset.weebly.com/urbansound8k.html](https://urbansounddataset.weebly.com/urbansound8k.html)
  
  > J. Salamon, C. Jacoby and J. P. Bello. *A Dataset and Taxonomy for Urban Sound Research.* Proceedings of the 22nd ACM International Conference on Multimedia, 2014.

---

## 使用条款

- 本数据集仅供**学术研究与非商业用途**。
- 使用本数据集的研究成果请注明数据来源。
- 请勿将数据用于侵犯个人隐私或其他违法用途。
```
