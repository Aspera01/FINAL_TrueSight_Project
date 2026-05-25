"""
Drop zone widget — file selection with media preview.
Shows upload prompt when empty; thumbnail/frame/waveform when a file is loaded.

Architecture note: the outer QVBoxLayout on DropZoneWidget is created ONCE and
never replaced. State changes swap the inner content QWidget instead, which avoids
Qt's silent refusal to replace a layout via QVBoxLayout(parent) while the old
layout is still pending deletion.
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from ui.components.media_preview import MediaPreviewWidget

SUPPORTED = (
    "Media Files (*.jpg *.jpeg *.png *.bmp *.tiff *.webp "
    "*.mp4 *.avi *.mov *.mkv *.wmv *.webm "
    "*.mp3 *.wav *.flac *.ogg *.m4a *.aac);;"
    "All Files (*.*)"
)


class DropZoneWidget(QWidget):
    file_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("dropZone")
        self.setMinimumHeight(180)

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(0)

        self._content: QWidget | None = None
        self._swap(self._empty_widget())

    # ── content swap ─────────────────────────────────

    def _swap(self, new: QWidget):
        """Replace the inner content widget without touching the outer layout."""
        if self._content is not None:
            self._outer.removeWidget(self._content)
            self._content.deleteLater()
        self._content = new
        self._outer.addWidget(new)

    # ── empty state ──────────────────────────────────

    def _empty_widget(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 28, 20, 28)

        icon = QLabel("⬆")
        icon.setObjectName("dropIcon")
        icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon)

        main_lbl = QLabel("Click to upload or drag and drop")
        main_lbl.setObjectName("dropMainLabel")
        main_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(main_lbl)

        sub_lbl = QLabel("Supported: MP4, AVI, MOV, MP3, WAV, JPG, PNG and more")
        sub_lbl.setObjectName("dropSubLabel")
        sub_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(sub_lbl)

        browse_btn = QPushButton("Browse file")
        browse_btn.setObjectName("browseBtn")
        browse_btn.setFixedWidth(130)
        browse_btn.clicked.connect(self._browse)
        layout.addWidget(browse_btn, alignment=Qt.AlignCenter)

        return w

    # ── loaded state ─────────────────────────────────

    def _loaded_widget(self, path: str) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(6)
        layout.setContentsMargins(12, 10, 12, 10)

        preview = MediaPreviewWidget(path, height=130)
        layout.addWidget(preview)

        name_lbl = QLabel(Path(path).name)
        name_lbl.setObjectName("previewFileName")
        name_lbl.setAlignment(Qt.AlignCenter)
        name_lbl.setWordWrap(True)
        layout.addWidget(name_lbl)

        ext = Path(path).suffix.upper().lstrip(".")
        size = Path(path).stat().st_size
        size_str = f"{size/1024/1024:.1f} MB" if size > 1024*1024 else f"{size/1024:.1f} KB"
        meta_lbl = QLabel(f"{ext}  ·  {size_str}  ·  Ready for analysis")
        meta_lbl.setObjectName("previewMeta")
        meta_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(meta_lbl)

        change_btn = QPushButton("Change file")
        change_btn.setObjectName("browseBtn")
        change_btn.setFixedWidth(130)
        change_btn.clicked.connect(self._browse)
        layout.addWidget(change_btn, alignment=Qt.AlignCenter)

        return w

    # ── public API ───────────────────────────────────

    def set_file(self, path: str):
        self._swap(self._loaded_widget(path))

    def reset(self):
        self._swap(self._empty_widget())

    # ── internals ────────────────────────────────────

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File", str(Path.home()), SUPPORTED
        )
        if path:
            self.file_selected.emit(path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            self.file_selected.emit(urls[0].toLocalFile())
