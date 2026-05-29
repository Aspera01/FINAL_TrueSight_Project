"""
Lip-Sync Mismatch Detector
----------------------------
Checks whether mouth movements correlate with the audio signal.
Deepfake videos often have mis-aligned lip movements and audio, especially
when only one modality has been manipulated.

Method:
  1. Extract audio energy envelope (RMS per frame window).
  2. Track mouth openness via MediaPipe face-mesh landmarks (preferred)
     OR frame-difference in the mouth region via cv2 (fallback).
  3. Compute cross-correlation between audio energy and mouth aperture.
  4. Low correlation → suspicious.

Requires: mediapipe (optional — cv2 fallback used if absent), librosa, ffmpeg
"""

import os
import tempfile
import numpy as np
import cv2
from pathlib import Path
from detectors.base import BaseDetector, DetectionResult, MediaType


class LipSyncDetector(BaseDetector):
    name = "Lip-Sync Analysis"
    version = "1.1.0"
    supported_types = [MediaType.VIDEO]

    # MediaPipe face mesh indices for upper/lower lip landmarks
    UPPER_LIP_IDX = [13, 312, 311, 310, 415, 308]
    LOWER_LIP_IDX = [14, 317, 402, 318, 324, 78]

    def __init__(self):
        self._mp_available       = False
        self._librosa_available  = False
        self._ffmpeg_available   = False
        self._check_deps()

    def _check_deps(self):
        try:
            import mediapipe  # noqa: F401
            self._mp_available = True
        except Exception:
            # Catches ImportError, OSError, protobuf errors, DLL failures, etc.
            pass
        try:
            import librosa  # noqa: F401
            self._librosa_available = True
        except Exception:
            pass
        try:
            import ffmpeg  # noqa: F401
            self._ffmpeg_available = True
        except Exception:
            pass

    # ── audio extraction ─────────────────────────────────────────────────

    def _extract_audio_rms(self, video_path: str, fps: float, n_frames: int) -> np.ndarray | None:
        """Extract per-video-frame RMS energy from the audio track."""
        if not self._librosa_available:
            return None

        import librosa

        tmp_audio = tempfile.NamedTemporaryFile(suffix="_lipsync_audio.wav", delete=False).name
        try:
            extracted = False

            # Try ffmpeg-python first
            if self._ffmpeg_available:
                try:
                    import ffmpeg
                    (
                        ffmpeg.input(video_path)
                        .output(tmp_audio, acodec="pcm_s16le", ac=1, ar=16000)
                        .overwrite_output()
                        .run(quiet=True)
                    )
                    extracted = True
                except Exception:
                    pass

            # Fallback: subprocess ffmpeg
            if not extracted:
                try:
                    import subprocess
                    result = subprocess.run(
                        ["ffmpeg", "-i", video_path, "-vn",
                         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                         "-y", tmp_audio],
                        capture_output=True, timeout=60,
                    )
                    extracted = result.returncode == 0
                except Exception:
                    pass

            if not extracted or not os.path.exists(tmp_audio):
                return None

            y, sr = librosa.load(tmp_audio, sr=16000, mono=True)
            hop = int(sr / fps)
            rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
            if len(rms) > n_frames:
                rms = rms[:n_frames]
            elif len(rms) < n_frames:
                rms = np.pad(rms, (0, n_frames - len(rms)))
            return rms.astype(np.float32)

        except Exception:
            return None
        finally:
            if os.path.exists(tmp_audio):
                os.remove(tmp_audio)

    # ── mouth aperture — MediaPipe (preferred) ───────────────────────────

    def _extract_mouth_aperture_mp(self, video_path: str, max_frames: int = 120) -> tuple[np.ndarray, float]:
        """Landmark-based mouth aperture using MediaPipe FaceMesh."""
        import mediapipe as mp

        cap  = cv2.VideoCapture(video_path)
        fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total // max_frames)

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

        apertures  = []
        frame_idx  = 0
        while cap.isOpened() and len(apertures) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    lm      = results.multi_face_landmarks[0].landmark
                    upper_y = np.mean([lm[i].y for i in self.UPPER_LIP_IDX])
                    lower_y = np.mean([lm[i].y for i in self.LOWER_LIP_IDX])
                    apertures.append(abs(lower_y - upper_y))
                else:
                    apertures.append(0.0)
            frame_idx += 1

        cap.release()
        face_mesh.close()
        return np.array(apertures, dtype=np.float32), fps

    # ── mouth aperture — cv2 fallback ────────────────────────────────────

    def _extract_mouth_aperture_cv2(self, video_path: str, max_frames: int = 120) -> tuple[np.ndarray, float]:
        """
        Fallback when MediaPipe is unavailable.
        Uses frame-difference in the mouth ROI (lower-face or fixed region)
        as a proxy for mouth movement.
        """
        cap      = cv2.VideoCapture(video_path)
        fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total // max_frames)

        # Load face cascade if available
        face_cascade = None
        try:
            cascade_dir = cv2.data.haarcascades
            face_xml    = cascade_dir + "haarcascade_frontalface_default.xml"
            if os.path.exists(face_xml):
                fc = cv2.CascadeClassifier(face_xml)
                if not fc.empty():
                    face_cascade = fc
        except Exception:
            pass

        apertures = []
        prev_roi  = None
        frame_idx = 0

        while cap.isOpened() and len(apertures) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w   = gray.shape
                roi    = None

                if face_cascade is not None:
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
                    )
                    if len(faces) > 0:
                        fx, fy, fw, fh = faces[0]
                        # Lower third of face ≈ mouth region
                        roi = gray[fy + fh * 2 // 3 : fy + fh, fx : fx + fw]

                if roi is None or roi.size == 0:
                    # Fixed fallback: centre-bottom of frame
                    roi = gray[h * 3 // 5 : h * 4 // 5, w // 4 : w * 3 // 4]

                if prev_roi is not None and roi.shape == prev_roi.shape:
                    diff = float(np.mean(np.abs(roi.astype(np.float32) - prev_roi.astype(np.float32))))
                    apertures.append(diff)
                else:
                    apertures.append(0.0)
                prev_roi = roi.copy()
            frame_idx += 1

        cap.release()
        return np.array(apertures, dtype=np.float32), fps

    # ── dispatcher ───────────────────────────────────────────────────────

    def _extract_mouth_aperture(self, video_path: str, max_frames: int = 120) -> tuple[np.ndarray, float, bool]:
        """Returns (apertures, fps, used_mediapipe)."""
        if self._mp_available:
            try:
                arr, fps = self._extract_mouth_aperture_mp(video_path, max_frames)
                return arr, fps, True
            except Exception:
                pass
        arr, fps = self._extract_mouth_aperture_cv2(video_path, max_frames)
        return arr, fps, False

    # ── main detection ───────────────────────────────────────────────────

    def _detect(self, media_path: str, media_type: MediaType) -> DetectionResult:  # noqa: ARG002
        apertures, fps, used_mp = self._extract_mouth_aperture(media_path)

        if len(apertures) < 10:
            return DetectionResult(
                module_name=self.name,
                score=0.0,
                confidence=0.1,
                details={"note": "Not enough face frames for lip-sync analysis."},
            )

        rms = self._extract_audio_rms(media_path, fps, len(apertures))

        if rms is None or not self._librosa_available:
            # Audio unavailable — return neutral, zero-confidence result
            # so the aggregator excludes it from the weighted average.
            return DetectionResult(
                module_name=self.name,
                score=0.0,
                confidence=0.0,
                details={
                    "method": "mouth_variance_only",
                    "mouth_std": round(float(np.std(apertures)), 6),
                    "note": "Audio unavailable; result excluded from aggregate.",
                },
            )

        # Cross-correlation between audio RMS and mouth aperture
        a    = (apertures - apertures.mean()) / (apertures.std() + 1e-8)
        r    = (rms - rms.mean()) / (rms.std() + 1e-8)
        corr = float(np.correlate(a, r, mode="full").max() / len(a))
        # corr ∈ [-1, 1]; high positive = well synced → authentic
        score = max(0.0, min(1.0, 0.5 - corr * 0.5))

        # Landmark tracking (MediaPipe) is more precise than frame-diff (cv2)
        confidence = 0.75 if used_mp else 0.45

        return DetectionResult(
            module_name=self.name,
            score=round(score, 4),
            confidence=confidence,
            details={
                "method":                "mediapipe_landmarks" if used_mp else "cv2_frame_diff",
                "audio_mouth_correlation": round(corr, 4),
                "frames_analysed":        len(apertures),
                "mouth_aperture_std":     round(float(apertures.std()), 6),
            },
        )
