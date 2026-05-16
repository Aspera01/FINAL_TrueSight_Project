"""
Audio Spectrogram Detector
---------------------------
Analyses mel-spectrograms of audio to detect synthetic/cloned speech.
Trained voice cloning systems leave characteristic traces in the spectrogram
such as over-smooth formants, missing breath noise, and unnatural pitch contours.

Two-stage approach:
  1. Heuristic analysis (always runs): pitch variance, spectral flatness,
     silence ratio, formant stability.
  2. ONNX model inference (if model is downloaded): LCNN / RawNet2-style
     classifier trained on ASVspoof 2019 (open dataset).

Requires: librosa, soundfile
"""

import numpy as np
from pathlib import Path
from detectors.base import BaseDetector, DetectionResult, MediaType

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "audio_deepfake_lcnn.onnx"


class AudioSpectrogramDetector(BaseDetector):
    name = "Audio Spectrogram (LCNN)"
    version = "1.0.0"
    supported_types = [MediaType.AUDIO]#MediaType.VIDEO

    SAMPLE_RATE = 16000
    N_MELS = 80
    HOP_LENGTH = 160   # 10ms at 16kHz
    WIN_LENGTH = 400   # 25ms at 16kHz

    def __init__(self):
        self._session = None
        self._librosa_available = False
        try:
            import librosa  # noqa: F401
            self._librosa_available = True
        except ImportError:
            pass

    def _load_model(self) -> bool:
        if self._session is not None:
            return True
        if not MODEL_PATH.exists():
            return False
        try:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(str(MODEL_PATH), providers=providers)
            return True
        except Exception:
            return False

    def _load_audio(self, path: str) -> np.ndarray | None:
        """Load audio from file (audio or video) at target sample rate."""
        if not self._librosa_available:
            return None
        import librosa
        try:
            y, _ = librosa.load(path, sr=self.SAMPLE_RATE, mono=True, duration=30.0)
            return y
        except Exception:
            return None

    def _extract_mel(self, y: np.ndarray) -> np.ndarray:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.SAMPLE_RATE,
            n_mels=self.N_MELS,
            hop_length=self.HOP_LENGTH,
            win_length=self.WIN_LENGTH,
            fmax=8000,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel.astype(np.float32)

    def _heuristic_score(self, y: np.ndarray) -> tuple[float, dict]:
        """Compute heuristic deepfake indicators from raw waveform."""
        import librosa

        # 1. Pitch (F0) variance — cloned voices often have flatter pitch
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=500, sr=self.SAMPLE_RATE, hop_length=self.HOP_LENGTH
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
        pitch_var = float(voiced_f0.std()) if len(voiced_f0) > 5 else 0.0
        # Low pitch variance → more likely synthetic
        pitch_score = max(0.0, 1.0 - pitch_var / 30.0)

        # 2. Spectral flatness — synthetic speech often has flatter spectrum
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.HOP_LENGTH)[0]
        mean_flatness = float(flatness.mean())
        flatness_score = min(1.0, mean_flatness * 20.0)

        # 3. Zero-crossing rate consistency — natural speech is more variable
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.HOP_LENGTH)[0]
        zcr_cv = float(zcr.std() / (zcr.mean() + 1e-6))
        zcr_score = max(0.0, 1.0 - zcr_cv * 2.0)

        # 4. Silence ratio — TTS systems often have uniform silence patterns
        rms = librosa.feature.rms(y=y, hop_length=self.HOP_LENGTH)[0]
        silence_ratio = float((rms < 0.01).mean())
        silence_score = min(1.0, abs(silence_ratio - 0.15) * 3.0)

        combined = (
            0.35 * pitch_score
            + 0.25 * flatness_score
            + 0.20 * zcr_score
            + 0.20 * silence_score
        )
        return combined, {
            "pitch_variance": round(pitch_var, 2),
            "spectral_flatness_mean": round(mean_flatness, 6),
            "zcr_cv": round(zcr_cv, 4),
            "silence_ratio": round(silence_ratio, 4),
        }

    def _model_score(self, log_mel: np.ndarray) -> float:
        """Run ONNX LCNN model on mel spectrogram."""
        # Pad or crop to 300 frames (~3 seconds)
        target_frames = 300
        if log_mel.shape[1] < target_frames:
            pad = target_frames - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant", constant_values=-80)
        else:
            log_mel = log_mel[:, :target_frames]

        # Normalise
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        inp = log_mel[np.newaxis, np.newaxis]  # (1, 1, n_mels, frames)

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: inp})
        logits = outputs[0][0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        return float(probs[1])  # fake probability

    def _detect(self, media_path: str, media_type: MediaType) -> DetectionResult:
        if not self._librosa_available:
            return DetectionResult(
                module_name=self.name,
                score=0.0,
                confidence=0.0,
                error="librosa not installed. Run: pip install librosa",
            )

        y = self._load_audio(media_path)
        if y is None or len(y) < self.SAMPLE_RATE:
            return DetectionResult(
                module_name=self.name,
                score=0.0,
                confidence=0.1,
                details={"note": "Audio too short or unreadable."},
            )

        heuristic, details = self._heuristic_score(y)
        model_loaded = self._load_model()

        if model_loaded:
            log_mel = self._extract_mel(y)
            model_sc = self._model_score(log_mel)
            # Weighted blend: model is more reliable
            final_score = 0.3 * heuristic + 0.7 * model_sc
            confidence = 0.87
            details["model_score"] = round(model_sc, 4)
            details["heuristic_score"] = round(heuristic, 4)
            details["model"] = "lcnn_asvspoof2019"
        else:
            final_score = heuristic
            confidence = 0.55
            details["note"] = "Running heuristic only. Download model for better accuracy."

        return DetectionResult(
            module_name=self.name,
            score=round(float(final_score), 4),
            confidence=confidence,
            details=details,
        )
