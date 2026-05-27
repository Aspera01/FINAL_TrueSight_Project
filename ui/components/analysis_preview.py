"""
Analysis Preview Widget
------------------------
Animated media display shown while detection is running.

  Image  → still image with a bouncing horizontal scan line
  Video  → fast-forward frame playback (cv2 grab loop, ~3-4x speed)
  Audio  → waveform with a looping vertical scan line that lights up bars
"""

import numpy as np
from pathlib import Path

from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPixmap, QImage, QColor, QPen, QFont

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


class AnalysisPreviewWidget(QWidget):
    """
    Full-area animated preview. Call start() to begin animation, stop() to end.
    Cleans up the cv2 VideoCapture on stop().
    """

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self._path = file_path
        self._ext = Path(file_path).suffix.lower()

        # Shared animation state
        self._scan = 0.0        # 0.0–1.0
        self._scan_dir = 1      # ±1 (image bounce)

        # Image
        self._img_px: QPixmap | None = None

        # Video
        self._cap = None
        self._frame_step = 8
        self._frame_px: QPixmap | None = None

        # Audio
        self._bar_heights: np.ndarray | None = None  # pre-normalised, shape (N,)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setObjectName("analysisPreview")

        self._timer = QTimer(self)
        self._load()

    # ── source loading ───────────────────────────────────────────────────

    def _load(self):
        ext = self._ext
        try:
            if ext in IMAGE_EXTS:
                px = QPixmap(self._path)
                if not px.isNull():
                    self._img_px = px
                self._timer.setInterval(16)          # ~60 fps
                self._timer.timeout.connect(self._tick_image)

            elif ext in VIDEO_EXTS:
                import cv2
                self._cap = cv2.VideoCapture(self._path)
                fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
                # Show ~12 unique frames/s; advance enough frames for ~3-4x speed
                self._frame_step = max(1, int(fps / 12))
                self._timer.setInterval(83)          # ~12 fps display
                self._timer.timeout.connect(self._tick_video)
                self._tick_video()                   # first frame immediately

            elif ext in AUDIO_EXTS:
                import librosa
                y, _ = librosa.load(self._path, sr=4000, mono=True, duration=60)
                norm = float(np.abs(y).max()) or 1.0
                self._samples_norm = y / norm        # store normalised samples
                self._timer.setInterval(16)          # ~60 fps
                self._timer.timeout.connect(self._tick_audio)

        except Exception:
            pass

    # ── animation control ────────────────────────────────────────────────

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── timer callbacks ──────────────────────────────────────────────────

    def _tick_image(self):
        self._scan += self._scan_dir * 0.005
        if self._scan >= 1.0:
            self._scan, self._scan_dir = 1.0, -1
        elif self._scan <= 0.0:
            self._scan, self._scan_dir = 0.0, 1
        self.update()

    def _tick_video(self):
        if self._cap is None:
            return
        import cv2
        for _ in range(self._frame_step - 1):
            if not self._cap.grab():
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fh, fw, ch = rgb.shape
        img = QImage(rgb.tobytes(), fw, fh, fw * ch, QImage.Format_RGB888)
        self._frame_px = QPixmap.fromImage(img)
        self.update()

    def _tick_audio(self):
        # Full left-to-right sweep in ~10 s at 60 fps
        self._scan += 16 / 10_000
        if self._scan > 1.0:
            self._scan = 0.0
        self.update()

    # ── painting ─────────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor("#0a1520"))

        ext = self._ext
        if ext in IMAGE_EXTS:
            self._paint_image(painter, w, h)
        elif ext in VIDEO_EXTS:
            self._paint_video(painter, w, h)
        elif ext in AUDIO_EXTS:
            self._paint_audio(painter, w, h)

        painter.end()

    def _paint_image(self, p: QPainter, w: int, h: int):
        if self._img_px is None:
            return
        scaled = self._img_px.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        ox = (w - scaled.width()) // 2
        oy = (h - scaled.height()) // 2
        p.drawPixmap(ox, oy, scaled)

        # Horizontal scan line bouncing over the image
        sy = oy + int(self._scan * scaled.height())
        for off, alpha in ((0, 220), (1, 110), (2, 55), (3, 20),
                           (-1, 110), (-2, 55), (-3, 20)):
            p.setPen(QPen(QColor(77, 157, 224, alpha), 1))
            p.drawLine(ox, sy + off, ox + scaled.width(), sy + off)

    def _paint_video(self, p: QPainter, w: int, h: int):
        if self._frame_px is None:
            return
        scaled = self._frame_px.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        ox = (w - scaled.width()) // 2
        oy = (h - scaled.height()) // 2
        p.drawPixmap(ox, oy, scaled)

        # Fast-forward badge
        p.setPen(QColor("#88ccff"))
        f = QFont("Segoe UI", 10, QFont.Bold)
        p.setFont(f)
        p.drawText(ox + 10, oy + 22, "⏩  FAST FORWARD")

    def _paint_audio(self, p: QPainter, w: int, h: int):
        if not hasattr(self, "_samples_norm") or self._samples_norm is None:
            return
        samples = self._samples_norm
        mid_y = h // 2
        scan_x = int(self._scan * w)
        n_bins = min(w, len(samples))
        bar_w = max(w // n_bins, 1)

        # Centre line
        p.setPen(QPen(QColor("#1e3050"), 1))
        p.drawLine(0, mid_y, w, mid_y)

        # Waveform bars — lit (scanned) vs dim (upcoming)
        p.setPen(Qt.NoPen)
        for i, chunk in enumerate(np.array_split(samples, n_bins)):
            if len(chunk) == 0:
                continue
            amp = max(int(float(np.abs(chunk).max()) * (mid_y - 8)), 1)
            x = int(i * w / n_bins)
            p.setBrush(QColor("#4d9de0") if x <= scan_x else QColor("#1a3a5a"))
            p.drawRect(x, mid_y - amp, bar_w, amp * 2)

        # Vertical scan line with glow
        for off, alpha in ((0, 240), (1, 120), (2, 60), (3, 25),
                           (-1, 120), (-2, 60), (-3, 25)):
            p.setPen(QPen(QColor(136, 204, 255, alpha), 1))
            p.drawLine(scan_x + off, 0, scan_x + off, h)
