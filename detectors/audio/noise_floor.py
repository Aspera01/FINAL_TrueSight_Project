"""
Noise Floor Consistency Detector
----------------------------------
Authentic recordings have consistent background noise throughout.
Synthetic or spliced audio often shows abrupt changes in the noise floor,
unnaturally clean segments, or mismatched background environments between
different parts of the recording.

No external model required.
"""

import numpy as np
from detectors.base import BaseDetector, DetectionResult, MediaType


class NoiseFloorDetector(BaseDetector):
    name = "Noise Floor Consistency"
    version = "1.0.0"
    supported_types = [MediaType.AUDIO]#MediaType.VIDEO

    SAMPLE_RATE = 16000
    SEGMENT_SEC = 0.5   # Analyse noise in 0.5s windows

    def _load_audio(self, path: str) -> np.ndarray | None:
        try:
            import librosa
            y, _ = librosa.load(path, sr=self.SAMPLE_RATE, mono=True, duration=60.0)
            return y
        except Exception:
            return None

    def _detect(self, media_path: str, media_type: MediaType) -> DetectionResult:
        try:
            import librosa
        except ImportError:
            return DetectionResult(
                module_name=self.name,
                score=0.0,
                confidence=0.0,
                error="librosa not installed.",
            )

        y = self._load_audio(media_path)
        if y is None or len(y) < self.SAMPLE_RATE:
            return DetectionResult(
                module_name=self.name,
                score=0.0,
                confidence=0.1,
                details={"note": "Audio too short."},
            )

        seg_len = int(self.SAMPLE_RATE * self.SEGMENT_SEC)
        segments = [y[i:i + seg_len] for i in range(0, len(y) - seg_len, seg_len)]

        # Measure noise floor per segment using silent frames only
        noise_floors = []
        spectral_centroids = []
        for seg in segments:
            rms_seg = np.sqrt(np.mean(seg ** 2))
            # Estimate noise as energy of the quietest 10% of frames
            frame_size = 160
            frames = [seg[j:j + frame_size] for j in range(0, len(seg) - frame_size, frame_size)]
            frame_rms = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
            threshold = np.percentile(frame_rms, 10)
            noise_frames = frame_rms[frame_rms <= threshold * 1.5]
            noise_floors.append(float(noise_frames.mean()) if len(noise_frames) > 0 else 0.0)

            # Spectral centroid of the segment
            sc = librosa.feature.spectral_centroid(y=seg, sr=self.SAMPLE_RATE)[0]
            spectral_centroids.append(float(sc.mean()))

        noise_arr = np.array(noise_floors)
        sc_arr = np.array(spectral_centroids)

        # High variance in noise floor = suspicious
        noise_cv = float(noise_arr.std() / (noise_arr.mean() + 1e-9))

        # Abrupt changes (sudden drops to near-zero = synthetic insertion)
        noise_diffs = np.abs(np.diff(noise_arr))
        spike_ratio = float((noise_diffs > noise_arr.mean() * 2).mean())

        # Centroid consistency (synthetic voices have very uniform centroid)
        centroid_cv = float(sc_arr.std() / (sc_arr.mean() + 1e-6))
        centroid_score = max(0.0, 0.5 - centroid_cv * 2.0)

        # Final score
        score = min(1.0,
            0.45 * min(1.0, noise_cv * 2.0)
            + 0.35 * min(1.0, spike_ratio * 3.0)
            + 0.20 * centroid_score
        )
        confidence = min(0.85, 0.4 + len(segments) / 60.0)

        return DetectionResult(
            module_name=self.name,
            score=round(float(score), 4),
            confidence=round(confidence, 4),
            details={
                "segments_analysed": len(segments),
                "noise_floor_cv": round(noise_cv, 4),
                "noise_spike_ratio": round(spike_ratio, 4),
                "centroid_cv": round(centroid_cv, 4),
                "mean_noise_floor": round(float(noise_arr.mean()), 6),
            },
        )
