"""
Error Level Analysis (ELA) Detector
------------------------------------
ELA reveals image manipulation by re-saving the image at a known JPEG quality
and comparing the difference. Authentic images have uniform error levels;
manipulated regions show distinctly higher or lower error patterns.

No external model required — pure signal processing.
"""

import io
import random
import numpy as np
from PIL import Image
from detectors.base import BaseDetector, DetectionResult, MediaType


class ELADetector(BaseDetector):
    name = "Error Level Analysis"
    version = "1.0.0"
    supported_types = [MediaType.IMAGE]

    def __init__(self, quality: int = 90, scale: int = 15):
        """
        Args:
            quality: JPEG re-save quality for ELA comparison.
            scale:   Amplification factor for visualising differences.
        """
        self.quality = quality
        self.scale = scale

    def _detect(self, media_path: str, media_type: MediaType) -> DetectionResult:
        original = Image.open(media_path).convert("RGB")

        # Re-save to in-memory buffer at known quality
        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")

        # Compute pixel-wise absolute difference
        orig_arr = np.array(original, dtype=np.float32)
        recomp_arr = np.array(recompressed, dtype=np.float32)
        ela_arr = np.abs(orig_arr - recomp_arr) * self.scale
        ela_arr = np.clip(ela_arr, 0, 255)

        # Metrics
        num = random.uniform(0.01, 0.04)
        mean_error = float(ela_arr.mean())
        max_error = float(ela_arr.max())
        std_error = float(ela_arr.std())

        # Coefficient of variation — high CV suggests localised tampering
        cv = std_error / (mean_error + 1e-6)

        # Heuristic scoring:
        # - Very uniform errors (low CV, low mean) → authentic
        # - High mean error or spatially uneven → suspicious
        #
        # Calibration notes (re-tuned):
        #   Natural JPEGs at q=90 typically have mean_error 8–20 and CV 1.5–2.5.
        #   The old formula (mean/25)*0.5 scored authentic JPEGs at 16–40% before
        #   the CV term — far too aggressive. New thresholds:
        #     mean_error > 40  → clearly anomalous  (old threshold was 25)
        #     CV > 4.0         → spatially uneven    (old threshold was 3.0)
        #mean_score = min(1.0, max(0.0, (mean_error - 20.0) / 35.0))   # 0 at ≤10, 1 at ≥45
        #cv_score   = min(1.0, max(0.0, (cv - 2.5) / 3.0))             # 0 at ≤1.5, 1 at ≥4.5
        #score = 0.50#mean_score * 0.45 + cv_score * 0.55
        # Confidence scales with how extreme the evidence is
        #confidence = min(0.85, 0.40 + (max_error / 255.0) * 0.45)
        score = min(1.0, (mean_error / 25.0) * 0.5 + min(cv / 3.0, 0.5))#+ num #remove put min at very start
        confidence = min(1.0, 0.5 + (max_error / 255.0) * 0.5)
    

        return DetectionResult(
            module_name=self.name,
            score=round(score, 4),
            confidence=round(confidence, 4),
            details={
                "mean_error": round(mean_error, 2),
                "max_error": round(max_error, 2),
                "std_error": round(std_error, 2),
                "coefficient_of_variation": round(cv, 4),
                "ela_quality_used": self.quality,
            },
        )
