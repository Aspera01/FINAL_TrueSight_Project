"""
Face / Image Deepfake Detector
--------------------------------
Uses onnx-community/Deep-Fake-Detector-v2-Model-ONNX
  — Vision Transformer (ViT-base-patch16-224) fine-tuned on real vs deepfake images
  — Apache 2.0 license
  — 92.12% accuracy, labels: {0: "Deepfake", 1: "Realism"}

Input:  224x224 RGB image, normalised with ImageNet mean/std
Output: logits [deepfake_score, real_score]
"""

import numpy as np
import cv2
from pathlib import Path
from detectors.base import BaseDetector, DetectionResult, MediaType

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH  = MODELS_DIR / "face_deepfake_vit.onnx"

# ViT ImageNet normalisation
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)


class FaceCNNDetector(BaseDetector):
    name            = "Face / Image Deepfake CNN"
    version         = "2.0.0"
    supported_types = [MediaType.IMAGE, MediaType.VIDEO]

    def __init__(self):
        self._session       = None
        self._mp_face       = None
        self._haar_cascade  = None
        self._init_face_detector()

    # ── face detector init ────────────────────────────────

    def _init_face_detector(self):
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                self._mp_face = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                )
                return
        except Exception:
            pass
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar_cascade = cv2.CascadeClassifier(cascade)

    # ── model loading ─────────────────────────────────────

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

    # ── preprocessing ─────────────────────────────────────

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """Resize → RGB → normalise → NCHW float32."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224))
        x   = rgb.astype(np.float32) / 255.0
        x   = (x - MEAN) / STD
        return x.transpose(2, 0, 1)[np.newaxis]          # (1, 3, 224, 224)

    def _infer(self, tensor: np.ndarray) -> float:
        """Returns deepfake probability [0, 1]."""
        name    = self._session.get_inputs()[0].name
        logits  = self._session.run(None, {name: tensor})[0][0]
        exp     = np.exp(logits - logits.max())
        probs   = exp / exp.sum()
        # label 0 = Deepfake, label 1 = Realism
        return float(probs[0])

    # ── face detection ────────────────────────────────────

    def _detect_faces(self, bgr: np.ndarray) -> list[tuple]:
        if self._mp_face:
            rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results = self._mp_face.process(rgb)
            boxes   = []
            if results.detections:
                h, w = bgr.shape[:2]
                for d in results.detections:
                    bb = d.location_data.relative_bounding_box
                    x  = int(bb.xmin * w);  y  = int(bb.ymin * h)
                    bw = int(bb.width * w); bh = int(bb.height * h)
                    boxes.append((max(0, x), max(0, y), bw, bh))
            return boxes
        else:
            gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            faces = self._haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            return [(x, y, w, h) for (x, y, w, h) in faces] if len(faces) else []

    def _crop_face(self, bgr: np.ndarray, box: tuple) -> np.ndarray:
        x, y, w, h = box
        pad = int(max(w, h) * 0.2)
        x1  = max(0, x - pad);      y1 = max(0, y - pad)
        x2  = min(bgr.shape[1], x + w + pad)
        y2  = min(bgr.shape[0], y + h + pad)
        return bgr[y1:y2, x1:x2]

    # ── analysis ──────────────────────────────────────────

    def _analyse_frame(self, bgr: np.ndarray) -> tuple[float, int]:
        """Returns (mean_deepfake_score, n_faces_detected)."""
        if not self._load_model():
            return 0.0, 0

        faces = self._detect_faces(bgr)

        if faces:
            # Score each face crop
            scores = []
            for box in faces:
                crop   = self._crop_face(bgr, box)
                tensor = self._preprocess(crop)
                scores.append(self._infer(tensor))
            return float(np.mean(scores)), len(faces)
        else:
            # No face found — run on the full image anyway
            tensor = self._preprocess(bgr)
            score  = self._infer(tensor)
            return score, 0

    # ── BaseDetector interface ────────────────────────────

    def _detect(self, media_path: str, media_type: MediaType) -> DetectionResult:
        model_ready = self._load_model()

        if not model_ready:
            return DetectionResult(
                module_name=self.name,
                score=0.0,
                confidence=0.0,
                error="Model not downloaded. Run: python scripts/setup_models.py",
                details={"model_loaded": False},
            )

        if media_type == MediaType.IMAGE:
            bgr = cv2.imread(media_path)
            if bgr is None:
                raise ValueError(f"Cannot read image: {media_path}")
            score, n_faces = self._analyse_frame(bgr)
            return DetectionResult(
                module_name=self.name,
                score=round(score, 4),
                confidence=0.92,
                details={
                    "faces_detected": n_faces,
                    "model": "ViT-base-patch16-224 (Deep-Fake-Detector-v2)",
                    "full_image_scored": n_faces == 0,
                },
            )

        else:  # VIDEO
            cap    = cv2.VideoCapture(media_path)
            total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stride = max(1, total // 30)
            scores = []
            idx    = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % stride == 0:
                    s, _ = self._analyse_frame(frame)
                    scores.append(s)
                idx += 1
            cap.release()

            if not scores:
                return DetectionResult(
                    module_name=self.name,
                    score=0.0,
                    confidence=0.1,
                    details={"note": "No frames could be extracted."},
                )

            return DetectionResult(
                module_name=self.name,
                score=round(float(np.mean(scores)), 4),
                confidence=0.90,
                details={
                    "frames_scored":  len(scores),
                    "score_std":      round(float(np.std(scores)), 4),
                    "max_frame_score": round(float(max(scores)), 4),
                    "model":          "ViT-base-patch16-224 (Deep-Fake-Detector-v2)",
                },
            )
