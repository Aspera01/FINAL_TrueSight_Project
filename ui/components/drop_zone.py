"""
Drop zone widget — simple file selection, no media preview.
Shows upload prompt when empty, filename + type when a file is loaded.
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent

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
        self._build_empty()

    # ── states ───────────────────────────────────────

    def _build_empty(self):
        self._clear()
        layout = QVBoxLayout(self)
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

    def _build_loaded(self, path: str):
        self._clear()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 28, 20, 28)

        # Checkmark icon
        icon = QLabel("✔")
        icon.setObjectName("dropIconLoaded")
        icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon)

        # File name
        name_lbl = QLabel(Path(path).name)
        name_lbl.setObjectName("dropFileLoaded")
        name_lbl.setAlignment(Qt.AlignCenter)
        name_lbl.setWordWrap(True)
        layout.addWidget(name_lbl)

        # File type + size
        ext = Path(path).suffix.upper().lstrip(".")
        size = Path(path).stat().st_size
        size_str = f"{size/1024/1024:.1f} MB" if size > 1024*1024 else f"{size/1024:.1f} KB"
        meta_lbl = QLabel(f"{ext}  ·  {size_str}  ·  Ready for analysis")
        meta_lbl.setObjectName("dropSubLabel")
        meta_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(meta_lbl)

        # Change file button
        change_btn = QPushButton("Change file")
        change_btn.setObjectName("browseBtn")
        change_btn.setFixedWidth(130)
        change_btn.clicked.connect(self._browse)
        layout.addWidget(change_btn, alignment=Qt.AlignCenter)

    # ── public API ───────────────────────────────────

    def set_file(self, path: str):
        self._build_loaded(path)

    def reset(self):
        self._build_empty()

    # ── internals ────────────────────────────────────

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File", str(Path.home()), SUPPORTED
        )
        if path:
            self.file_selected.emit(path)

    def _clear(self):
        old = self.layout()
        if old:
            while old.count():
                item = old.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            old.deleteLater()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            self.file_selected.emit(urls[0].toLocalFile())
