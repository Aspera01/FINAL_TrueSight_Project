"""
History Panel
--------------
Shows previous analysis sessions stored in the local JSON history file.
Each row is clickable and emits `history_item_selected(report_id)` so the
main window can navigate back into a stored result.
"""

from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor


class HistoryRowWidget(QFrame):
    """Single clickable row representing one past analysis."""

    clicked = Signal(int)   # emits report_id

    def __init__(self, report: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("historyRow")
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self._report_id = report["Report_ID"]
        self._build(report)

    def _build(self, r: dict):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        # Risk colour dot
        colour = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}.get(
            r.get("Risk_Level", "low"), "#4d7a9a"
        )
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {colour}; font-size: 10px;")
        dot.setFixedWidth(14)
        layout.addWidget(dot)

        # File name + verdict
        info_col = QVBoxLayout()
        info_col.setSpacing(2)

        name_lbl = QLabel(r.get("File_Name", "Unknown file"))
        name_lbl.setObjectName("historyFileName")
        name_lbl.setWordWrap(False)

        verdict_lbl = QLabel(f"{r.get('Verdict', '—')}  ·  {r.get('File_Type', '').capitalize()}")
        verdict_lbl.setObjectName("historyMeta")

        info_col.addWidget(name_lbl)
        info_col.addWidget(verdict_lbl)
        layout.addLayout(info_col, 1)

        # Score badge
        score_pct = round((r.get("Overall_Score") or 0) * 100)
        score_lbl = QLabel(f"{score_pct}%")
        score_lbl.setObjectName(
            "historyScoreHigh"   if r.get("Risk_Level") == "high"   else
            "historyScoreMedium" if r.get("Risk_Level") == "medium" else
            "historyScoreLow"
        )
        score_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        score_lbl.setFixedWidth(44)
        layout.addWidget(score_lbl)

        # Date
        raw_date = r.get("Generated_Date", "")
        try:
            dt = datetime.fromisoformat(raw_date)
            date_str = dt.strftime("%b %d  %H:%M")
        except Exception:
            date_str = raw_date[:16]
        date_lbl = QLabel(date_str)
        date_lbl.setObjectName("historyDate")
        date_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        date_lbl.setFixedWidth(90)
        layout.addWidget(date_lbl)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._report_id)
        super().mousePressEvent(event)


class HistoryPanel(QWidget):
    """
    Scrollable list of past analyses.
    Emits `history_item_selected(report_id)` when a row is clicked.
    """

    history_item_selected = Signal(int)

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self._db = db
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header row
        header = QWidget()
        header.setObjectName("historyHeader")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(4, 8, 4, 8)

        title = QLabel("RECENT ANALYSES")
        title.setObjectName("historySectionLabel")
        h_layout.addWidget(title)
        h_layout.addStretch()

        self._refresh_btn = QPushButton("↻  Refresh")
        self._refresh_btn.setObjectName("historyRefreshBtn")
        self._refresh_btn.setFixedWidth(90)
        self._refresh_btn.clicked.connect(self.refresh)
        h_layout.addWidget(self._refresh_btn)

        outer.addWidget(header)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._list_container = QWidget()
        self._list_layout = QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(4)
        self._list_layout.addStretch()

        scroll.setWidget(self._list_container)
        outer.addWidget(scroll, 1)

        self.refresh()

    def refresh(self):
        """Reload history from the database and redraw rows."""
        # Clear existing rows (leave the stretch at the end)
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        try:
            records = self._db.get_history(limit=50)
        except Exception:
            records = []

        if not records:
            empty = QLabel("No analyses yet. Run a detection to see history here.")
            empty.setObjectName("historyEmpty")
            empty.setAlignment(Qt.AlignCenter)
            empty.setWordWrap(True)
            self._list_layout.insertWidget(0, empty)
            return

        for record in records:
            row = HistoryRowWidget(record)
            row.clicked.connect(self.history_item_selected)
            self._list_layout.insertWidget(self._list_layout.count() - 1, row)
