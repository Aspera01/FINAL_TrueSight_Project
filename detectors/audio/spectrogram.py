"""
Audio Deepfake Detector
------------------------
Uses as1605/Deepfake-audio-detection-V2
  — Wav2Vec2-based model fine-tuned for audio deepfake detection
  — Apache 2.0 license
  — 99.7% accuracy on evaluation set
  — Labels: {0: "fake", 1: "real"}

Falls back to heuristic-only analysis if model not downloaded.
Requires: librosa
"""

import os
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from detectors.base import BaseDetector, DetectionResult, MediaType

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH  = MODELS_DIR / "audio_deepfake_wav2vec2.onnx"

SAMPLE_RATE   = 16000   # Wav2Vec2 expects 16 kHz mono
MAX_DURATION  = 30.0    # seconds

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}


class AudioSpectrogramDetector(BaseDetector):
    name            = "Audio Deepfake Detector (Wav2Vec2)"
    version         = "2.0.0"
    supported_types = [MediaType.AUDIO, MediaType.VIDEO]

    def __init__(self):
        self._session          = None
        self._librosa_available = False
        try:
            import librosa  # noqa
            self._librosa_available = True
        except ImportError:
            pass

    # ── model ─────────────────────────────────────────────

    def _load_model(self) -> bool:
        if self._session is not None:
            return True
        if not MODEL_PATH.exists():
            return False
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            providers = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(
                str(MODEL_PATH), sess_options=opts, providers=providers
            )
            return True
        except Exception:
            return False

    # ── audio loading ─────────────────────────────────────

    def _extract_audio_from_video(self, video_path: str) -> str | None:
        """Extract audio track from a video to a temp WAV. Returns path or None."""
        tmp = tempfile.NamedTemporaryFile(suffix="_wav2vec2.wav", delete=False).name
        try:
            # Try subprocess ffmpeg (works on any system with ffmpeg on PATH)
            result = subprocess.run(
                ["ffmpeg", "-i", video_path, "-vn",
                 "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
                 "-t", str(MAX_DURATION), "-y", tmp],
                capture_output=True, timeout=60,
            )
            if result.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                return tmp
        except Exception:
            pass

        # Fallback: ffmpeg-python wrapper
        try:
            import ffmpeg
            (
                ffmpeg.input(video_path)
                .output(tmp, acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE, t=MAX_DURATION)
                .overwrite_output()
                .run(quiet=True)
            )
            if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                return tmp
        except Exception:
            pass

        if os.path.exists(tmp):
            os.remove(tmp)
        return None

    def _load_audio(self, path: str) -> np.ndarray | None:
        if not self._librosa_available:
            return None

        import librosa

        is_video = Path(path).suffix.lower() in VIDEO_EXTS
        tmp = None

        try:
            if is_video:
                # librosa can't decode video containers directly — extract first
                tmp = self._extract_audio_from_video(path)
                if tmp is None:
                    return None
                load_path = tmp
            else:
                load_path = path

            y, _ = librosa.load(load_path, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION)
            return y.astype(np.float32)

        except Exception:
            return None
        finally:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)

    # ── model inference ───────────────────────────────────

    def _model_score(self, y: np.ndarray) -> float:
        """
        Run Wav2Vec2 ONNX model.
        Input shape: (1, sequence_length) — raw waveform float32 at 16 kHz
        Output: logits [fake_score, real_score]
        """
        # Pad or trim to exactly 5 seconds for consistent inference
        target_len = SAMPLE_RATE * 5
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # Normalise
        if y.std() > 0:
            y = (y - y.mean()) / y.std()

        inp = y[np.newaxis]   # (1, 80000)

        input_name = self._session.get_inputs()[0].name
        outputs    = self._session.run(None, {input_name: inp})
        logits     = outputs[0][0]

        exp   = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        # label 0 = fake → deepfake probability
        return float(probs[0])

    # ── heuristic fallback ────────────────────────────────

    def _heuristic_score(self, y: np.ndarray) -> tuple[float, dict]:
        import librosa

        # Pitch variance — synthetic voices tend to be flatter
        f0, voiced, _ = librosa.pyin(
            y, fmin=50, fmax=500, sr=SAMPLE_RATE, hop_length=160
        )
        voiced_f0   = f0[voiced] if voiced is not None else np.array([])
        pitch_var   = float(voiced_f0.std()) if len(voiced_f0) > 5 else 0.0
        pitch_score = max(0.0, 1.0 - pitch_var / 30.0)

        # Spectral flatness
        flatness      = librosa.feature.spectral_flatness(y=y, hop_length=160)[0]
        flat_score    = min(1.0, float(flatness.mean()) * 20.0)

        # Silence ratio
        rms           = librosa.feature.rms(y=y, hop_length=160)[0]
        silence_ratio = float((rms < 0.01).mean())
        sil_score     = min(1.0, abs(silence_ratio - 0.15) * 3.0)

        combined = 0.40 * pitch_score + 0.30 * flat_score + 0.30 * sil_score
        details  = {
            "pitch_variance":          round(pitch_var, 2),
            "spectral_flatness_mean":  round(float(flatness.mean()), 6),
            "silence_ratio":           round(silence_ratio, 4),
        }
        return combined, details

    # ── BaseDetector interface ────────────────────────────

    def _detect(self, media_path: str, media_type: MediaType) -> DetectionResult:
        if not self._librosa_available:
            return DetectionResult(
                module_name=self.name,
                score=0.0,
                confidence=0.0,
                error="librosa not installed. Run: pip install librosa",
            )

        y = self._load_audio(media_path)
        if y is None or len(y) < SAMPLE_RATE:
            return DetectionResult(
                module_name=self.name,
                score=0.0,
                confidence=0.1,
                details={"note": "Audio too short or unreadable."},
            )

        model_ready = self._load_model()
        heuristic, h_details = self._heuristic_score(y)

        if model_ready:
            try:
                model_sc    = self._model_score(y)
                final_score = 0.2 * heuristic + 0.8 * model_sc
                confidence  = 0.94
                details     = {
                    "model_score":     round(model_sc, 4),
                    "heuristic_score": round(heuristic, 4),
                    "model":           "Wav2Vec2 (Deepfake-audio-detection-V2)",
                    **h_details,
                }
            except Exception as e:
                # Model loaded but inference failed — fall back gracefully
                final_score = heuristic
                confidence  = 0.55
                details     = {
                    **h_details,
                    "note": f"Model inference failed ({e}), heuristic only.",
                }
        else:
            final_score = heuristic
            confidence  = 0.55
            details     = {
                **h_details,
                "note": "Model not downloaded. Run: python scripts/setup_models.py",
            }

        return DetectionResult(
            module_name=self.name,
            score=round(float(final_score), 4),
            confidence=confidence,
            details=details,
        )
