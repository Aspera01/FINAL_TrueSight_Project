"""
Lip-Sync Mismatch Detector
----------------------------
Checks whether mouth movements correlate with the audio signal.
Deepfake videos often have mis-aligned lip movements and audio, especially
when only one modality has been manipulated.

Method:
  1. Extract audio energy envelope (RMS per frame window).
  2. Track mouth openness using facial landmarks (MediaPipe).
  3. Compute cross-correlation between audio energy and mouth aperture.
  4. Low correlation → suspicious.

Requires: mediapipe, librosa
"""

import numpy as np
import cv2
from pathlib import Path
from detectors.base import BaseDetector, DetectionResult, MediaType


class LipSyncDetector(BaseDetector):
    name = "Lip-Sync Analysis"
    version = "1.0.0"
    supported_types = [MediaType.VIDEO] #MediaType.VIDEO

    # MediaPipe face mesh indices for upper/lower lip landmarks
    UPPER_LIP_IDX = [13, 312, 311, 310, 415, 308]
    LOWER_LIP_IDX = [14, 317, 402, 318, 324, 78]

    def __init__(self):
        self._mp_available = False
        self._librosa_available = False
        self._ffmpeg_available = False
        self._check_deps()

    def _check_deps(self):
        try:
            import mediapipe  # noqa: F401
            self._mp_available = True
        except ImportError:
            pass
        try:
            import librosa  # noqa: F401
            self._librosa_available = True
        except ImportError:
            pass
        try:
            import ffmpeg  # noqa: F401
            self._ffmpeg_available = True
        except ImportError:
            pass

    def _extract_audio_rms(self, video_path: str, fps: float, n_frames: int) -> np.ndarray | None:
        """Extract per-video-frame RMS energy from audio track."""
        if not (self._librosa_available and self._ffmpeg_available):
            return None
        try:
            import ffmpeg
            import librosa

            tmp_audio = "/tmp/_deepfake_det_audio.wav"
            (
                ffmpeg.input(video_path)
                .output(tmp_audio, acodec="pcm_s16le", ac=1, ar=16000)
                .overwrite_output()
                .run(quiet=True)
            )
            y, sr = librosa.load(tmp_audio, sr=16000, mono=True)
            hop = int(sr / fps)
            rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
            # Align length to video frame count
            if len(rms) > n_frames:
                rms = rms[:n_frames]
            elif len(rms) < n_frames:
                rms = np.pad(rms, (0, n_frames - len(rms)))
            return rms.astype(np.float32)
        except Exception:
            return None

    def _extract_mouth_aperture(self, video_path: str, max_frames: int = 120) -> tuple[np.ndarray, float]:
        """Returns (mouth_aperture_per_frame, fps)."""
        if not self._mp_available:
            return np.array([]), 0.0

        import mediapipe as mp

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total // max_frames)

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

        apertures = []
        frame_idx = 0
        while cap.isOpened() and len(apertures) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    upper_y = np.mean([lm[i].y for i in self.UPPER_LIP_IDX])
                    lower_y = np.mean([lm[i].y for i in self.LOWER_LIP_IDX])
                    apertures.append(abs(lower_y - upper_y))
                else:
                    apertures.append(0.0)
            frame_idx += 1

        cap.release()
        face_mesh.close()
        return np.array(apertures, dtype=np.float32), fps

    def _detect(self, media_path: str, media_type: MediaType) -> DetectionResult:
        if not self._mp_available:
            return DetectionResult(
                module_name=self.name,
                score=0.0,
                confidence=0.0,
                error="mediapipe not installed. Run: pip install mediapipe",
            )

        apertures, fps = self._extract_mouth_aperture(media_path)

        if len(apertures) < 10:
            return DetectionResult(
                module_name=self.name,
                score=0.0,
                confidence=0.1,
                details={"note": "Not enough face frames for lip-sync analysis."},
            )

        rms = self._extract_audio_rms(media_path, fps, len(apertures))

        if rms is None or not self._librosa_available:
            # Fall back to mouth motion variability only.
            # Old formula: 0.5 - mouth_var*10 → nearly always returned ~0.5
            # which dragged the aggregate up on every video without audio.
            # New: return a genuinely neutral result with very low confidence
            # so the aggregator down-weights it appropriately.
            mouth_var = float(np.std(apertures))
            return DetectionResult(
                module_name=self.name,
                score=0.0,
                confidence=0.0,
                details={
                    "method": "mouth_variance_only",
                    "mouth_std": round(mouth_var, 6),
                    "note": "Audio unavailable; result excluded from aggregate.",
                },
            )

        # Cross-correlation between audio RMS and mouth aperture
        a = (apertures - apertures.mean()) / (apertures.std() + 1e-8)
        r = (rms - rms.mean()) / (rms.std() + 1e-8)
        corr = float(np.correlate(a, r, mode="full").max() / len(a))
        # corr in [-1, 1]; high positive = well synced
        # Map: 1 → 0 (authentic), -1 or 0 → 1 (suspicious)
        score = max(0.0, min(1.0, 0.5 - corr * 0.5))

        return DetectionResult(
            module_name=self.name,
            score=round(score, 4),
            confidence=0.75,
            details={
                "audio_mouth_correlation": round(corr, 4),
                "frames_analysed": len(apertures),
                "mouth_aperture_std": round(float(apertures.std()), 6),
            },
        )
