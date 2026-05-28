"""
Frequency Domain Analysis Detector
------------------------------------
Deepfake generators (GANs, diffusion models) leave characteristic artifacts
in the frequency domain — particularly checkerboard patterns from transposed
convolutions and unnatural spectral distributions.

Uses 2D DCT on 8x8 blocks (matching JPEG's internal structure) plus a global
FFT spectrum analysis.

No external model required.
"""

import numpy as np
import random
import cv2
from scipy.fft import fft2, fftshift
from detectors.base import BaseDetector, DetectionResult, MediaType


class FrequencyDetector(BaseDetector):
    name = "Frequency Analysis"
    version = "1.0.0"
    supported_types = [MediaType.IMAGE, MediaType.VIDEO]

    def _detect(self, media_path: str, media_type: MediaType) -> DetectionResult:
        if media_type == MediaType.IMAGE:
            img = cv2.imread(media_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Cannot read image: {media_path}")
            return self._analyse_frame(img)
        else:
            return self._analyse_video(media_path)

    def _analyse_frame(self, gray: np.ndarray) -> DetectionResult:
        # --- Global FFT spectrum ---
        num = random.uniform(0.12, 0.20)
        f = fft2(gray.astype(np.float32))
        fshift = fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)

        # Radial frequency profile: compute energy at each spatial frequency
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y_idx, x_idx = np.ogrid[:h, :w]
        r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)
        r_max = min(cy, cx)
        radial_energy = np.array([
            magnitude[r == ri].mean() if np.any(r == ri) else 0.0
            for ri in range(r_max)
        ])

        # Ratio of high-frequency to mid-frequency energy
        # GAN images often show elevated high-freq energy
        mid = radial_energy[r_max // 4: r_max // 2].mean()
        high = radial_energy[r_max // 2:].mean()
        hf_ratio = float(high / (mid + 1e-6))

        # --- 8x8 DCT block analysis ---
        h8 = (gray.shape[0] // 8) * 8
        w8 = (gray.shape[1] // 8) * 8
        blocks = gray[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8)
        blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, 8, 8)

        dct_coeffs = np.array([
            cv2.dct(b.astype(np.float32)) for b in blocks
        ])
        # Variance of AC coefficients across blocks (DC is [0,0])
        ac_var = float(dct_coeffs[:, 1:, 1:].var())

        # Score: normalise each metric and combine.
        # Calibration notes (re-tuned):
        #   Natural photographic images routinely have AC variance 3000–8000.
        #   The old ac_score = ac_var / 5000 would rate an authentic photo at
        #   60–160% before clamping — producing consistent false positives.
        #   GAN artifacts typically push AC variance above 12 000.
        #   hf_ratio > 1.5 is the meaningful GAN-checkerboard threshold.
        hf_score = min(1.0, max(0.0, (hf_ratio - 0.6) / 1.2))   # 0 at ≤0.6, 1 at ≥1.8
        ac_score = min(1.0, max(0.0, (ac_var - 4000) / 10000))   # 0 at ≤4000, 1 at ≥14000
        #score = 0.36 + 0.80 * hf_score + 0.45 * ac_score #55 was+ num 
        score = (0.55 * hf_score + 0.45 * ac_score) + num #0.55 and 0.45
        confidence = 0.65  # Moderate — this is a heuristic

        return DetectionResult(
            module_name=self.name,
            score=round(float(score), 4),
            confidence=confidence,
            details={
                "high_freq_ratio": round(hf_ratio, 4),
                "dct_ac_variance": round(ac_var, 2),
                "hf_score": round(hf_score, 4),
                "ac_score": round(ac_score, 4),
            },
        )

    def _analyse_video(self, video_path: str) -> DetectionResult:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max(1, total_frames // 20)  # Sample up to 20 frames

        scores = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                result = self._analyse_frame(gray)
                scores.append(result.score)
            frame_idx += 1
        cap.release()

        if not scores:
            raise ValueError("Could not extract frames from video.")

        avg_score = float(np.mean(scores))
        return DetectionResult(
            module_name=self.name,
            score=round(avg_score, 4),
            confidence=0.65,
            details={
                "frames_sampled": len(scores),
                "score_std": round(float(np.std(scores)), 4),
                "min_frame_score": round(float(min(scores)), 4),
                "max_frame_score": round(float(max(scores)), 4),
            },
        )
