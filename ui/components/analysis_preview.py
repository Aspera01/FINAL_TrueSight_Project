"""
Analysis Preview Widget
------------------------
Animated media display shown while detection is running.

  Image  → scan line + bounding boxes revealing as the line passes them
  Video  → fast-forward frame playback + pulsing discrepancy boxes
  Audio  → waveform with coloured anomaly zones (yellow = maybe, red = likely)

All detection markers are visual-only placeholders generated from the file path
so they look consistent on repeated runs of the same file.
"""

import math
import random
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPixmap, QImage, QColor, QPen, QFont

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

_RED    = (231, 76,  60)   # "likely" – red
_YELLOW = (243, 156, 18)   # "maybe"  – yellow


# ── helper: fake detection box / zone generation ─────────────────────────────

def _gen_image_boxes(rng: random.Random) -> list[tuple]:
    """Return 2-4 (rx, ry, rw, rh, severity, label) boxes in [0-1] image space."""
    labels = [
        "Texture anomaly", "Edge inconsistency", "GAN fingerprint",
        "Compression artifact", "Boundary artifact", "Frequency artifact",
    ]
    boxes = []
    for _ in range(rng.randint(2, 4)):
        rx  = rng.uniform(0.04, 0.60)
        ry  = rng.uniform(0.04, 0.62)
        rw  = rng.uniform(0.14, 0.30)
        rh  = rng.uniform(0.10, 0.24)
        sev = rng.choice(["likely", "likely", "maybe"])
        lbl = rng.choice(labels)
        boxes.append((rx, ry, rw, rh, sev, lbl))
    return boxes


def _gen_video_boxes(rng: random.Random) -> list[tuple]:
    """Boxes positioned to look like face-region detections."""
    regions = [
        (rng.uniform(0.20, 0.38), rng.uniform(0.05, 0.18),
         rng.uniform(0.26, 0.42), rng.uniform(0.08, 0.14),
         "likely", "Face boundary"),
        (rng.uniform(0.28, 0.44), rng.uniform(0.22, 0.36),
         rng.uniform(0.14, 0.22), rng.uniform(0.06, 0.11),
         "maybe",  "Eye region"),
        (rng.uniform(0.22, 0.40), rng.uniform(0.56, 0.70),
         rng.uniform(0.20, 0.34), rng.uniform(0.07, 0.13),
         rng.choice(["likely", "maybe"]), "Jawline artifact"),
    ]
    return regions


def _gen_audio_zones(rng: random.Random) -> list[tuple]:
    """Return 2-3 (start, end, severity) zones in [0-1] waveform space."""
    zones = []
    pos   = 0.05
    while len(zones) < 3 and pos < 0.88:
        start = pos + rng.uniform(0.04, 0.18)
        end   = min(start + rng.uniform(0.06, 0.16), 0.94)
        sev   = rng.choice(["likely", "maybe", "maybe"])
        zones.append((start, end, sev))
        pos   = end + rng.uniform(0.06, 0.18)
        if len(zones) == 2:
            break
    return zones


# ── main widget ──────────────────────────────────────────────────────────────

class AnalysisPreviewWidget(QWidget):
    """Full-area animated preview. Call start() / stop() to control animation."""

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self._path = file_path
        self._ext  = Path(file_path).suffix.lower()

        # Seeded RNG so boxes are consistent for the same file
        rng = random.Random(hash(file_path) & 0xFFFF_FFFF)

        # Shared scan state
        self._scan     = 0.0
        self._scan_dir = 1
        self._first_pass_done    = False  # image: boxes reveal after first downward sweep
        self._audio_fully_scanned = False  # audio: zones stay revealed after first pass

        # Pre-generated markers
        self._img_boxes   = _gen_image_boxes(rng)
        self._vid_box_sets = [_gen_video_boxes(rng) for _ in range(5)]
        self._audio_zones = _gen_audio_zones(rng)

        # Video-specific
        self._cap          = None
        self._frame_step   = 8
        self._frame_px: QPixmap | None = None
        self._vid_set_idx  = 0    # which box set is currently shown
        self._vid_box_t    = 0    # ticks into current cycle (0 → _VID_CYCLE_LEN)
        self._VID_CYCLE_LEN = 30  # frames per set at ~12 fps ≈ 2.5 s

        # Audio-specific
        self._samples_norm: np.ndarray | None = None

        # Image-specific
        self._img_px: QPixmap | None = None

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setObjectName("analysisPreview")

        self._timer = QTimer(self)
        self._load()

    # ── source loading ───────────────────────────────────────────────────────

    def _load(self):
        ext = self._ext
        try:
            if ext in IMAGE_EXTS:
                px = QPixmap(self._path)
                if not px.isNull():
                    self._img_px = px
                self._timer.setInterval(16)
                self._timer.timeout.connect(self._tick_image)

            elif ext in VIDEO_EXTS:
                import cv2
                self._cap = cv2.VideoCapture(self._path)
                fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
                self._frame_step = max(1, int(fps / 12))
                self._timer.setInterval(83)
                self._timer.timeout.connect(self._tick_video)
                self._tick_video()

            elif ext in AUDIO_EXTS:
                import librosa
                y, _ = librosa.load(self._path, sr=4000, mono=True, duration=60)
                norm = float(np.abs(y).max()) or 1.0
                self._samples_norm = y / norm
                self._timer.setInterval(16)
                self._timer.timeout.connect(self._tick_audio)

        except Exception:
            pass

    # ── animation control ────────────────────────────────────────────────────

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── tick callbacks ───────────────────────────────────────────────────────

    def _tick_image(self):
        # 0.016 per tick at 60 fps ≈ 1 second per pass → ~2-3 passes in a typical image analysis
        self._scan += self._scan_dir * 0.016
        if self._scan >= 1.0:
            self._scan, self._scan_dir = 1.0, -1
            self._first_pass_done = True
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
        self._frame_px  = QPixmap.fromImage(img)
        self._vid_box_t += 1
        if self._vid_box_t >= self._VID_CYCLE_LEN:
            self._vid_box_t = 0
            self._vid_set_idx = (self._vid_set_idx + 1) % len(self._vid_box_sets)
        self.update()

    def _tick_audio(self):
        self._scan += 16 / 7_000   # ~7 s per full sweep
        if self._scan > 1.0:
            self._scan = 0.0
            self._audio_fully_scanned = True  # zones stay coloured on subsequent loops
        self.update()

    # ── painting ─────────────────────────────────────────────────────────────

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

    # ── image painting ───────────────────────────────────────────────────────

    def _paint_image(self, p: QPainter, w: int, h: int):
        if self._img_px is None:
            return

        scaled = self._img_px.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        ox = (w - scaled.width()) // 2
        oy = (h - scaled.height()) // 2
        iw, ih = scaled.width(), scaled.height()
        p.drawPixmap(ox, oy, scaled)

        # Discrepancy boxes
        f = QFont("Segoe UI", 7, QFont.Bold)
        p.setFont(f)
        for (rx, ry, rw, rh, sev, lbl) in self._img_boxes:
            box_cy = ry + rh / 2

            # Reveal logic: only show once scan has passed box center (downward)
            # or after the first full pass is done
            if not self._first_pass_done:
                if not (self._scan_dir > 0 and self._scan >= box_cy):
                    continue

            dist = abs(self._scan - box_cy)
            near = dist < 0.13

            col = _RED if sev == "likely" else _YELLOW
            a_border = 240 if near else 160
            a_fill   = 55  if near else 22

            bx = ox + int(rx * iw)
            by = oy + int(ry * ih)
            bw = int(rw * iw)
            bh = int(rh * ih)

            # Filled tint
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(*col, a_fill))
            p.drawRect(bx, by, bw, bh)

            # Border
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(*col, a_border), 2 if near else 1))
            p.drawRect(bx, by, bw, bh)

            # Corner label
            prefix = "!!" if sev == "likely" else "!"
            p.setPen(QColor(*col, a_border))
            p.drawText(bx + 3, by + 10, f"{prefix} {lbl}")

        # Horizontal scan line
        sy = oy + int(self._scan * ih)
        for off, alpha in ((0, 220), (1, 110), (2, 55), (3, 20),
                           (-1, 110), (-2, 55), (-3, 20)):
            p.setPen(QPen(QColor(77, 157, 224, alpha), 1))
            p.drawLine(ox, sy + off, ox + iw, sy + off)

    # ── video painting ───────────────────────────────────────────────────────

    def _paint_video(self, p: QPainter, w: int, h: int):
        if self._frame_px is None:
            return

        scaled = self._frame_px.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        ox = (w - scaled.width()) // 2
        oy = (h - scaled.height()) // 2
        iw, ih = scaled.width(), scaled.height()
        p.drawPixmap(ox, oy, scaled)

        # Fast-forward badge
        p.setPen(QColor("#88ccff"))
        f = QFont("Segoe UI", 10, QFont.Bold)
        p.setFont(f)
        p.drawText(ox + 10, oy + 22, "⏩  FAST FORWARD")

        # Cycling discrepancy boxes — fade in, hold, fade out, then swap set
        t = self._vid_box_t / self._VID_CYCLE_LEN   # 0.0 – 1.0 through cycle
        if t < 0.15:
            alpha_factor = t / 0.15          # fade in
        elif t > 0.78:
            alpha_factor = (1.0 - t) / 0.22  # fade out
        else:
            alpha_factor = 1.0               # hold

        lf = QFont("Segoe UI", 7, QFont.Bold)
        p.setFont(lf)

        for (rx, ry, rw, rh, sev, lbl) in self._vid_box_sets[self._vid_set_idx]:
            col      = _RED if sev == "likely" else _YELLOW
            a_border = int(220 * alpha_factor)
            a_fill   = int(50  * alpha_factor)
            if a_border < 5:
                continue

            bx = ox + int(rx * iw)
            by = oy + int(ry * ih)
            bw = int(rw * iw)
            bh = int(rh * ih)

            p.setPen(Qt.NoPen)
            p.setBrush(QColor(*col, a_fill))
            p.drawRect(bx, by, bw, bh)

            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(*col, a_border), 2))
            p.drawRect(bx, by, bw, bh)

            prefix = "!!" if sev == "likely" else "!"
            p.setPen(QColor(*col, a_border))
            p.drawText(bx + 3, by + 10, f"{prefix} {lbl}")

    # ── audio painting ───────────────────────────────────────────────────────

    def _paint_audio(self, p: QPainter, w: int, h: int):
        if self._samples_norm is None:
            return

        samples  = self._samples_norm
        mid_y    = h // 2
        scan_x   = int(self._scan * w)
        n_bins   = min(w, len(samples))
        bar_w    = max(w // n_bins, 1)

        # Build a zone lookup: for each bin index, which severity (or None)
        zone_map: dict[int, str] = {}
        for (start, end, sev) in self._audio_zones:
            for i in range(int(start * n_bins), min(int(end * n_bins) + 1, n_bins)):
                zone_map[i] = sev

        # Centre line
        p.setPen(QPen(QColor("#1e3050"), 1))
        p.drawLine(0, mid_y, w, mid_y)

        # Waveform bars — zones only colour once the scan line has passed them
        p.setPen(Qt.NoPen)
        for i, chunk in enumerate(np.array_split(samples, n_bins)):
            if len(chunk) == 0:
                continue
            amp     = max(int(float(np.abs(chunk).max()) * (mid_y - 8)), 1)
            x       = int(i * w / n_bins)
            scanned = x <= scan_x
            sev     = zone_map.get(i)

            # Reveal zone colour only behind the scan line (or after first full pass)
            zone_revealed = scanned or self._audio_fully_scanned

            if sev == "likely" and zone_revealed:
                col = QColor(*_RED,    230 if scanned else 160)
            elif sev == "maybe" and zone_revealed:
                col = QColor(*_YELLOW, 220 if scanned else 140)
            else:
                col = QColor(77, 157, 224, 255) if scanned else QColor(26, 58, 90, 255)

            p.setBrush(col)
            p.drawRect(x, mid_y - amp, bar_w, amp * 2)

        # Zone underline markers — reveal progressively as scan passes through
        lf = QFont("Segoe UI", 7, QFont.Bold)
        p.setFont(lf)
        for (start, end, sev) in self._audio_zones:
            col = _RED if sev == "likely" else _YELLOW
            zx  = int(start * w)
            ze  = int(end * w)
            # Only draw the portion the scan has covered (or full strip after first pass)
            revealed_end = ze if self._audio_fully_scanned else min(ze, scan_x)
            if revealed_end <= zx:
                continue
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(*col, 180))
            p.drawRect(zx, h - 8, max(revealed_end - zx, 2), 4)

            # Label appears once the scan has cleared the whole zone
            if scan_x >= ze or self._audio_fully_scanned:
                p.setPen(QColor(*col, 210))
                p.drawText(zx + 2, h - 11, "LIKELY" if sev == "likely" else "MAYBE")

        # Vertical scan line with glow
        for off, alpha in ((0, 240), (1, 120), (2, 60), (3, 25),
                           (-1, 120), (-2, 60), (-3, 25)):
            p.setPen(QPen(QColor(136, 204, 255, alpha), 1))
            p.drawLine(scan_x + off, 0, scan_x + off, h - 10)
