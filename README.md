# SAR-AudioData: A Call-for-Help / Knocking Sound Dataset for UAV Search and Rescue

*Languages: **English** · [中文](docs/README_zh.md)*

A sound event detection dataset targeting UAV-based Search and Rescue (SAR) scenarios. The dataset focuses on two critical distress signals — **human calls for help** and **knocking sounds** (e.g., trapped people striking pipes, walls, or metal objects) — and provides extensive noise-augmented versions built around the **rotor noise** that is unavoidable in real UAV rescue operations. It is intended for training and evaluating sound event detection (SED) models that stay robust under strong-noise conditions.

This repository open-sources four ready-to-use subsets that have already been split for modeling:

| Subset | Purpose | # Audio | # Labeled |
|--------|---------|--------:|----------:|
| `呼救声DEV`  (help-DEV)   | Call-for-help detection · train/dev | 19184 | 17184 |
| `呼救声TEST` (help-TEST)  | Call-for-help detection · test      | 715   | 715   |
| `敲击声DEV`  (knock-DEV)  | Knocking-sound detection · train/dev | 9811  | 7796  |
| `敲击声TEST` (knock-TEST) | Knocking-sound detection · test     | 1365  | 580   |

> Audio with an empty annotation (annotation file content is `[]`) contains no target event and serves as a negative sample in training and evaluation.

---

## Intended Use

- **Sound Event Detection (SED)**: locate the onset/offset time intervals of call-for-help / knocking events within continuous audio.
- **Noise-robustness research**: the data is mixed with a variety of UAV rotor noises (hovering, emergency stop, ascending/descending, left-right flight, landing, UAV, and other conditions) as well as mechanical noise, wind noise, and white noise, enabling study of detection performance under realistic UAV operating noise.
- **UAV search-and-rescue applications**: provides training and evaluation data for airborne acoustic detection systems mounted on UAVs.

---

## Directory Structure

All four subsets share the same structure, consisting of paired audio and annotation files:

```
呼救声DEV/
└── data/
    ├── audio/          # WAV audio files
    └── annotations/    # JSON annotation files with matching names

呼救声TEST/  敲击声DEV/  敲击声TEST/   # same structure
```

Each audio file and its annotation file share the **same name** (differing only in extension), in one-to-one correspondence:

```
data/audio/1-dev-XXXX.wav
data/annotations/1-dev-XXXX.json
```

---

## Annotation Format

Each annotation file is a JSON array, where every element represents the time interval of one target event within the audio:

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

| Field | Type | Description |
|-------|------|-------------|
| `start_time` | float | Event start time (seconds) |
| `end_time`   | float | Event end time (seconds) |
| `label`      | string | Event class, either `呼救声` (call for help) or `敲击声` (knocking) |
| `duration`   | float | Event duration (seconds), equal to `end_time - start_time` |

An empty array `[]` indicates that the audio contains no target event (negative sample).

---

## Subset Descriptions

- **呼救声DEV (help-DEV)**: train/dev set for call-for-help detection, containing call-for-help audio (with UAV noise augmentation) plus environmental sounds added as negative samples.
- **呼救声TEST (help-TEST)**: test set for call-for-help detection, consisting entirely of call-for-help audio overlaid with realistic UAV operating noise.
- **敲击声DEV (knock-DEV)**: train/dev set for knocking-sound detection, containing knocking audio (with various signal augmentations), call-for-help audio used as negative samples, and environmental sounds added as negative samples.
- **敲击声TEST (knock-TEST)**: test set for knocking-sound detection, composed of knocking audio overlaid with UAV noise and environmental sounds added as negative samples.

---

## Audio Specifications

- **Format**: WAV, mono, 16-bit PCM.
- **Sample rate**: because the subsets come from different sources, the sample rate is **not uniform** — a mix of 16 kHz, 44.1 kHz, 48 kHz, etc. (with a small number of other rates). Resample to a common rate as needed before use (e.g., unify to 16 kHz).
- **Duration**: mostly between 1–17 seconds, averaging about 4–5 seconds.

> Because the data comes from multiple sources — collected speech, [ESC-50](https://github.com/karolpiczak/ESC-50), [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html), etc. — sample rate and duration vary. Please apply uniform preprocessing at the data-loading stage.

---

## Noise Conditions and Augmentation

The dataset covers the following noise and augmentation techniques:

- **UAV rotor noise**: real recordings under conditions such as hovering, emergency stop, ascending/descending flight, left-right flight, landing, and UAV.
- **Signal-processing augmentation**: time stretching (stretch ×0.9 / ×1.1), white noise (varying SNR), wind noise, and mechanical noise.

---

## Data Preprocessing

The repository provides `preprocess_data.py`, which splits audio into sliding windows, extracts Mel-spectrogram features, and saves them as `.npy` files so they can be loaded directly during training, avoiding repeated computation.

```bash
python preprocess_data.py \
    --audio_subdir XXXXX/audio \
    --annotation_subdir XXXXX/annotations \
    --output_dir data/preprocessed \
    --sample_rate 16000 --n_mels 40 \
    --window_size 3.0 --hop_size 1.0 \
    --event_type 敲击声
```

The output directory will contain `features.npy` (features), `labels.npy` (labels), and `preprocess_config.json` (preprocessing parameters and label statistics).

---

## Requesting Access

This dataset is open for research use by application. To obtain the full data, please fill out the request form; we will review and respond as soon as possible:

**Request form**: [Form](https://docs.google.com/forms/d/e/1FAIpQLSc5VrLp1BmhiSrvMD2KyTzKQ0xivBec1mPVJoik5immtQf7vg/viewform?usp=header)

An online demo and dataset introduction page is available under `docs/`.

---

## Data Sources and Citation

The environmental-sound negative samples in this dataset are drawn from the following public datasets:

- **ESC-50**: [github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)

  > K. J. Piczak. *ESC: Dataset for Environmental Sound Classification.* Proceedings of the 23rd ACM International Conference on Multimedia, 2015.
- **UrbanSound8K**: [urbansounddataset.weebly.com/urbansound8k.html](https://urbansounddataset.weebly.com/urbansound8k.html)

  > J. Salamon, C. Jacoby and J. P. Bello. *A Dataset and Taxonomy for Urban Sound Research.* Proceedings of the 22nd ACM International Conference on Multimedia, 2014.

When using this dataset, please also cite the original sources above.

---

## Terms of Use

- This dataset is for **academic research and non-commercial use only**.
- Please acknowledge the data source in any research results that use this dataset.
- Do not use the data to infringe on personal privacy or for any other unlawful purpose.
