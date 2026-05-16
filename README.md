# TrueSight
An open-source, fully offline desktop application for detecting deepfakes in images, videos, and audio files using a modular AI pipeline.

**License:** MIT  
**Platform:** Windows · macOS · Linux  
**Python:** 3.10+

---

## Features

- **Multi-modal detection** — images (JPG, PNG, BMP, TIFF, WEBP), videos (MP4, AVI, MOV, MKV), audio (MP3, WAV, FLAC, OGG)
- **Modular pipeline** — 7 independent detection modules, each targeting different manipulation cues
- **Runs fully offline** — no data leaves your machine, no accounts, no telemetry
- **Adjustable sensitivity** — confidence threshold slider to tune false-positive rate
- **PDF report export** — shareable analysis reports
- **CPU + optional GPU** — works on any mid-range machine; GPU (CUDA) speeds up CNN inference

---

## Detection Modules

| Module | Media | Method |
|---|---|---|
| Error Level Analysis | Image | Re-save comparison to detect local tampering |
| Frequency Analysis | Image / Video | DCT/FFT artifact detection (GAN fingerprints) |
| Face CNN (EfficientNet-B4) | Image / Video | Deep CNN trained on FaceForensics++ |
| Temporal Consistency | Video | Optical flow + frame-difference irregularity |
| Lip-Sync Analysis | Video | Audio-visual correlation via MediaPipe landmarks |
| Audio Spectrogram (LCNN) | Audio / Video | Mel-spectrogram CNN (ASVspoof 2019) |
| Noise Floor Consistency | Audio / Video | Background noise uniformity analysis |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/deepfake-detector.git
cd deepfake-detector
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration (requires CUDA 11.8+):
```bash
pip install onnxruntime-gpu
```

### 4. Download model weights (recommended)

```bash
python scripts/setup_models.py
```

> The app works without models using heuristic-only analysis, but CNN-based modules require the weights for meaningful results.

### 5. Run

```bash
python main.py
```

---

## Project Structure

```
deepfake-detector/
├── main.py                        # Entry point
├── requirements.txt
├── LICENSE
├── core/
│   ├── pipeline.py                # Orchestrates all detectors
│   └── aggregator.py              # Weighted score combination
├── detectors/
│   ├── base.py                    # Abstract base class
│   ├── image/
│   │   ├── ela.py                 # Error Level Analysis
│   │   ├── frequency.py           # DCT/FFT analysis
│   │   └── face_cnn.py            # EfficientNet-B4 face detector
│   ├── video/
│   │   ├── temporal.py            # Frame consistency
│   │   └── lipsync.py             # Audio-visual sync
│   └── audio/
│       ├── spectrogram.py         # LCNN mel-spectrogram classifier
│       └── noise_floor.py         # Noise consistency
├── ui/
│   ├── app.py                     # Main window (PySide6)
│   ├── style.qss                  # Stylesheet
│   └── components/
│       ├── drop_zone.py
│       ├── result_panel.py
│       └── verdict_widget.py
├── utils/
│   └── export.py                  # PDF report generation
├── models/                        # ONNX model weights (gitignored)
└── scripts/
    └── setup_models.py            # One-time model downloader
```

---

## Adding a New Detector

1. Create a file in `detectors/image/`, `detectors/video/`, or `detectors/audio/`
2. Subclass `BaseDetector` and implement `_detect()`
3. Register it in `core/pipeline.py` → `ALL_DETECTORS`
4. Optionally add its weight in `core/aggregator.py` → `WEIGHTS`

```python
from detectors.base import BaseDetector, DetectionResult, MediaType

class MyDetector(BaseDetector):
    name = "My Custom Detector"
    supported_types = [MediaType.IMAGE]

    def _detect(self, media_path: str, media_type: MediaType) -> DetectionResult:
        score = ...  # your logic
        return DetectionResult(
            module_name=self.name,
            score=score,
            confidence=0.75,
        )
```

---

## Third-Party Licenses

| Package | License |
|---|---|
| PySide6 | LGPL v3 |
| OpenCV | Apache 2.0 |
| MediaPipe | Apache 2.0 |
| ONNX Runtime | MIT |
| librosa | ISC |
| NumPy | BSD |
| Pillow | HPND (open) |
| ReportLab | BSD |
| HuggingFace Hub | Apache 2.0 |
| EfficientNet-B4 weights | Apache 2.0 (FF++ research) |
| LCNN weights | MIT (ASVspoof 2019 research) |

---

## Contributing

Pull requests are welcome. Please open an issue first for significant changes.

## Disclaimer

This tool is provided for research and educational purposes. Detection results are probabilistic and should not be used as sole evidence of manipulation. Always combine with expert human review for high-stakes decisions.
