"""
Media Preview Widget
---------------------
Displays a visual preview of the loaded file, set immediately in __init__:
  Image  → scaled thumbnail
  Video  → single frame extracted near the start (cv2)
  Audio  → peak-per-pixel waveform drawn with QPainter
"""

import numpy as np
from pathlib import Path

from PySide6.QtWidgets import QLabel, QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

_RENDER_W = 600  # fixed render width; label centers it horizontally


class MediaPreviewWidget(QLabel):
    """
    QLabel subclass that immediately renders a media preview in its constructor.
    For audio, resizeEvent redraws the waveform at the actual widget width.
    """

    def __init__(self, path: str, height: int = 130, parent=None):
        super().__init__(parent)
        self._ext = Path(path).suffix.lower()
        self._height = height
        self._samples: np.ndarray | None = None  # kept for audio resizeEvent

        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAlignment(Qt.AlignCenter)
        self.setObjectName("previewFrame")

        self.setPixmap(self._build(path, _RENDER_W, height))

    # ── build pixmap once at construction ────────────────────────────────

    def _build(self, path: str, w: int, h: int) -> QPixmap:
        ext = self._ext
        try:
            if ext in IMAGE_EXTS:
                px = QPixmap(path)
                if not px.isNull():
                    return px.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            elif ext in VIDEO_EXTS:
                import cv2
                cap = cv2.VideoCapture(path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total * 0.1)))
                ok, frame = cap.read()
                cap.release()
                if ok:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    fh, fw, ch = rgb.shape
                    img = QImage(rgb.tobytes(), fw, fh, fw * ch, QImage.Format_RGB888)
                    return QPixmap.fromImage(img).scaled(
                        w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )

            elif ext in AUDIO_EXTS:
                import librosa
                y, _ = librosa.load(path, sr=4000, mono=True, duration=60)
                self._samples = y
                return self._draw_waveform(y, w, h)

        except Exception:
            pass

        return self._placeholder(w, h)

    # ── redraw waveform on resize (audio only) ────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        if self._ext in AUDIO_EXTS and self._samples is not None and w > 0:
            self.setPixmap(self._draw_waveform(self._samples, w, self._height))

    # ── helpers ──────────────────────────────────────────────────────────

    def _draw_waveform(self, samples: np.ndarray, w: int, h: int) -> QPixmap:
        px = QPixmap(w, h)
        px.fill(QColor("#0a1520"))
        painter = QPainter(px)
        mid_y = h // 2
        norm = float(np.abs(samples).max()) or 1.0

        painter.setPen(QPen(QColor("#1e3050"), 1))
        painter.drawLine(0, mid_y, w, mid_y)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#4d9de0"))

        n_bins = min(w, len(samples))
        for x, chunk in enumerate(np.array_split(samples, n_bins)):
            if len(chunk) == 0:
                continue
            amp = max(int(float(np.abs(chunk).max()) / norm * (mid_y - 4)), 1)
            painter.drawRect(x, mid_y - amp, max(w // n_bins, 1), amp * 2)

        painter.end()
        return px

    def _placeholder(self, w: int, h: int) -> QPixmap:
        px = QPixmap(w, h)
        px.fill(QColor("#0a1520"))
        return px
