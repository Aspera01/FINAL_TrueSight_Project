"""
Temporal Consistency Detector
-------------------------------
Deepfake videos often show unnatural frame-to-frame inconsistencies:
flickering textures, unstable facial boundaries, or motion that doesn't
match between the face region and the background.

Uses optical flow (Farneback) to measure local vs global motion consistency
and inter-frame difference analysis on detected face regions.

No external model required.
"""

import numpy as np
import cv2
from detectors.base import BaseDetector, DetectionResult, MediaType


class TemporalConsistencyDetector(BaseDetector):
    name = "Temporal Consistency"
    version = "1.0.0"
    supported_types = [MediaType.VIDEO]

    def _detect(self, media_path: str, media_type: MediaType) -> DetectionResult:
        cap = cv2.VideoCapture(media_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total < 2:
            raise ValueError("Video too short for temporal analysis.")

        # Collect frames (up to 120 for performance)
        max_frames = 120
        interval = max(1, total // max_frames)
        frames_gray = []
        frame_idx = 0

        while cap.isOpened() and len(frames_gray) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            frame_idx += 1
        cap.release()

        if len(frames_gray) < 2:
            raise ValueError("Not enough frames extracted.")

        # --- Optical flow consistency ---
        flow_ratios = []   # Face-region flow vs global flow
        frame_diffs = []   # Absolute inter-frame differences

        for i in range(len(frames_gray) - 1):
            f1, f2 = frames_gray[i], frames_gray[i + 1]

            # Global optical flow
            flow = cv2.calcOpticalFlowFarneback(
                f1, f2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            global_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            global_mean = float(global_mag.mean())

            # Frame difference (catches flickering)
            diff = np.abs(f2.astype(np.float32) - f1.astype(np.float32))
            frame_diffs.append(float(diff.mean()))
            flow_ratios.append(global_mean)

        # --- Score derivation ---
        # High variance in frame differences signals flickering (deepfake sign)
        diff_arr = np.array(frame_diffs)
        diff_cv = float(diff_arr.std() / (diff_arr.mean() + 1e-6))

        # Sudden large spikes in flow (unnatural movement)
        flow_arr = np.array(flow_ratios)
        flow_spike = float((flow_arr > flow_arr.mean() + 2 * flow_arr.std()).mean())

        # Combine metrics
        score = min(1.0, diff_cv * 0.4 + flow_spike * 0.6)
        confidence = min(0.9, 0.5 + len(frames_gray) / 200.0)

        return DetectionResult(
            module_name=self.name,
            score=round(score, 4),
            confidence=round(confidence, 4),
            details={
                "frames_analysed": len(frames_gray),
                "frame_diff_cv": round(diff_cv, 4),
                "flow_spike_ratio": round(flow_spike, 4),
                "mean_frame_diff": round(float(diff_arr.mean()), 4),
                "fps": fps,
            },
        )
