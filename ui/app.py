"""
TrueSight — Main Application Window
Dark theme, no sidebar, media preview on upload.

Changes in this revision
-------------------------
- Home view: drop-zone + history panel below it (scrollable).
- Results view: replaces the home view; shows verdict + module cards.
  A "← New Analysis" back button returns to the home view.
- Detection Sensitivity is only shown on the home/upload view; it is
  hidden when results are displayed so it doesn't clutter the results.
- Fixed QSS dynamic objectName bug (polishStyle helper forces re-evaluation).
- Fixed _clear_result_cards() accumulation bug.
- History rows are clickable and restore a previous result in the results view.
"""

import os
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QSlider,
    QScrollArea, QFrame, QStackedWidget,
    QProgressBar, QMessageBox, QSplitter,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from core.pipeline import run_pipeline, detect_media_type
from core.aggregator import aggregate, AggregatedResult
from ui.components.result_panel import ResultPanel
from ui.components.drop_zone import DropZoneWidget
from ui.components.verdict_widget import VerdictWidget
from ui.components.history_panel import HistoryPanel
from utils.database import Database


# ──────────────────────────────────────────────── worker thread ──

class AnalysisWorker(QThread):
    progress = Signal(str, int, int)
    finished = Signal(object, object)
    error = Signal(str)

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path

    def run(self):
        try:
            media_type, results = run_pipeline(
                self.file_path,
                progress_callback=lambda name, cur, tot: self.progress.emit(name, cur, tot),
                parallel=True,
            )
            self.finished.emit(media_type, results)
        except Exception as e:
            self.error.emit(str(e))


# ──────────────────────────────────────────────── main window ──

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TrueSight — AI-Powered Deepfake Detection")
        self.setMinimumSize(960, 640)
        self.resize(1140, 780)

        self._current_file: str | None = None
        self._worker: AnalysisWorker | None = None
        self._threshold: float = 0.5
        self._last_result: AggregatedResult | None = None
        self._db = Database()

        self._build_ui()
        self.setAcceptDrops(True)

    # ─────────────────────────────────────────── UI BUILD ──

    def _build_ui(self):
        root_widget = QWidget()
        self.setCentralWidget(root_widget)
        root_layout = QVBoxLayout(root_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_header())

        body = QWidget()
        body.setObjectName("contentArea")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        body_layout.addWidget(splitter)
        root_layout.addWidget(body, 1)

    # ── Header ──────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("headerBar")
        header.setFixedHeight(72)
        layout = QVBoxLayout(header)
        layout.setContentsMargins(32, 0, 32, 0)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(2)

        title = QLabel("TrueSight")
        title.setObjectName("appTitle")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("AI-Powered Deepfake Detection")
        subtitle.setObjectName("appSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        return header

    # ── Left panel — stacked: home view / results view ──────

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 12, 24)
        layout.setSpacing(0)

        self._left_stack = QStackedWidget()
        self._left_stack.addWidget(self._build_home_view())     # index 0
        self._left_stack.addWidget(self._build_results_view())  # index 1
        layout.addWidget(self._left_stack)

        return panel

    # ── Home view (drop zone + history) ─────────────────────

    def _build_home_view(self) -> QWidget:
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        # Drop zone
        self.drop_zone = DropZoneWidget()
        self.drop_zone.file_selected.connect(self._on_file_selected)
        self.drop_zone.setMinimumHeight(200)
        self.drop_zone.setMaximumHeight(320)
        layout.addWidget(self.drop_zone)

        # Progress bar (visible during analysis, lives in home view)
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(5)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setObjectName("progressLabel")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)

        # History section header
        hist_header = QLabel("")
        hist_header.setObjectName("sectionLabel")
        layout.addWidget(hist_header)

        # History panel
        self.history_panel = HistoryPanel(self._db)
        self.history_panel.history_item_selected.connect(self._on_history_selected)
        layout.addWidget(self.history_panel, 1)

        return view

    # ── Results view (back button + module cards) ────────────

    def _build_results_view(self) -> QWidget:
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Back button row
        back_row = QHBoxLayout()
        self.back_btn = QPushButton("← New Analysis")
        self.back_btn.setObjectName("backBtn")
        self.back_btn.setFixedWidth(150)
        self.back_btn.clicked.connect(self._go_home)
        back_row.addWidget(self.back_btn)
        back_row.addStretch()

        self._results_file_label = QLabel("")
        self._results_file_label.setObjectName("historyMeta")
        self._results_file_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        back_row.addWidget(self._results_file_label)

        layout.addLayout(back_row)

        # Module result cards (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(8)
        self.results_layout.addStretch()
        scroll.setWidget(self.results_container)
        layout.addWidget(scroll, 1)

        return view

    # ── Right panel — verdict + settings + actions ───────────

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("rightPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 24, 24, 24)
        layout.setSpacing(14)

        # Verdict widget (always visible)
        self.verdict_widget = VerdictWidget()
        layout.addWidget(self.verdict_widget, 1) #remove 1 if doesnt work

        # Settings card — hidden while results are shown
        self._settings_card = QFrame()
        self._settings_card.setObjectName("settingsCard")
        settings_layout = QVBoxLayout(self._settings_card)
        settings_layout.setContentsMargins(16, 14, 16, 14)
        settings_layout.setSpacing(10)

        sens_label = QLabel("DETECTION SENSITIVITY")
        sens_label.setObjectName("sectionLabel")
        settings_layout.addWidget(sens_label)

        self.threshold_value_label = QLabel("50%")
        self.threshold_value_label.setObjectName("thresholdValueLabel")
        self.threshold_value_label.setAlignment(Qt.AlignCenter)
        settings_layout.addWidget(self.threshold_value_label)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(10, 90)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        settings_layout.addWidget(self.threshold_slider)

        slider_row = QHBoxLayout()
        low_lbl = QLabel("Low")
        low_lbl.setObjectName("sliderLabelLow")
        high_lbl = QLabel("High")
        high_lbl.setObjectName("sliderLabelHigh")
        slider_row.addWidget(low_lbl)
        slider_row.addStretch()
        slider_row.addWidget(high_lbl)
        settings_layout.addLayout(slider_row)

        note = QLabel(
            "ℹ  Sensitivity is intended for advanced or cybersecurity users. "
            "If you are a casual user, the default setting (50%) is recommended."
        )
        note.setObjectName("sensitivityNote")
        note.setWordWrap(True)
        settings_layout.addWidget(note)

        layout.addWidget(self._settings_card)
        layout.addStretch()

        # Action buttons
        self.start_btn = QPushButton("▶   Start Analysis")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_analyse)
        layout.addWidget(self.start_btn)

        self.export_btn = QPushButton("↓   Export Report (PDF)")
        self.export_btn.setObjectName("exportBtn")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export)
        layout.addWidget(self.export_btn)

        return panel

    # ─────────────────────────────── VIEW NAVIGATION ──

    def _go_home(self):
        """Switch left panel back to the home/upload view."""
        self._current_file = None
        self.drop_zone.reset()          # rebuild the empty upload state
        self.start_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self._left_stack.setCurrentIndex(0)
        self._settings_card.setVisible(True)
        self.history_panel.refresh()

    def _show_results(self):
        """Switch left panel to the results view and hide sensitivity."""
        self._left_stack.setCurrentIndex(1)
        self._settings_card.setVisible(False)

    # ─────────────────────────────────────────── SLOTS ──

    def _on_file_selected(self, path: str):
        try:
            detect_media_type(path)
        except ValueError:
            QMessageBox.warning(
                self, "Unsupported File",
                "This file type is not supported.\n\n"
                "Supported: JPG, PNG, MP4, AVI, MOV, MP3, WAV, FLAC and more."
            )
            return

        self._current_file = path
        self.start_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.verdict_widget.reset()
        self.drop_zone.set_file(path)
        # Return to home view if user picks a new file while on results view
        if self._left_stack.currentIndex() == 1:
            self._go_home()

    def _on_threshold_changed(self, value: int):
        self._threshold = value / 100.0
        self.threshold_value_label.setText(f"{value}%")

    def _on_analyse(self):
        if not self._current_file:
            return
        self._clear_result_cards()
        self.start_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Preparing analysis…")
        # Stay on home view so progress is visible during analysis
        self._left_stack.setCurrentIndex(0)
        self._settings_card.setVisible(True)

        self._worker = AnalysisWorker(self._current_file)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.start()

    def _on_progress(self, module_name: str, current: int, total: int):
        pct = int((current / max(total, 1)) * 100)
        self.progress_bar.setValue(pct)
        self.progress_label.setText(f"Running: {module_name}  ({current}/{total})")

    def _on_analysis_done(self, media_type, results):
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.start_btn.setEnabled(True)

        aggregated = aggregate(results, media_type, self._threshold)
        self._last_result = aggregated

        # Save to local database
        try:
            file_name = Path(self._current_file).name
            file_type = media_type.value.capitalize()
            file_id = self._db.save_media_file(file_name, file_type, self._current_file)
            self._db.save_analysis(file_id, aggregated)
        except Exception:
            pass

        self._populate_results(aggregated, results, Path(self._current_file).name)
        self._show_results()
        self.export_btn.setEnabled(True)

    def _populate_results(self, aggregated: AggregatedResult, results, filename: str):
        """Fill the results view with verdict and module cards."""
        self.verdict_widget.update_result(aggregated)
        self._results_file_label.setText(filename)
        self._clear_result_cards()
        for result in results:
            card = ResultPanel(result)
            self.results_layout.insertWidget(self.results_layout.count() - 1, card)

    def _on_analysis_error(self, error_msg: str):
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.start_btn.setEnabled(True)
        QMessageBox.critical(
            self, "Analysis Error",
            f"An error occurred during analysis:\n\n{error_msg}"
        )

    def _on_history_selected(self, report_id: int):
        """User clicked a history row — restore that stored result."""
        try:
            detail = self._db.get_report_detail(report_id)
        except Exception as e:
            QMessageBox.warning(self, "History Error", f"Could not load history entry:\n{e}")
            return
        if not detail:
            return

        import json
        from detectors.base import DetectionResult, MediaType
        from core.aggregator import AggregatedResult

        report = detail["report"]

        # Reconstruct DetectionResult objects from stored rows
        mod_results = []
        for row in detail["module_results"]:
            not_applicable = row["Discrepancy_Detected"] in ("N/A", None, "")
            dr = DetectionResult(
                module_name=row["Module_Name"],
                score=float(row["Probability_Score"]),
                confidence=float(row["Confidence"]),
                supported=not not_applicable,
                details=json.loads(row["Details_JSON"]) if row.get("Details_JSON") else {},
                error=None,
            )
            mod_results.append(dr)

        try:
            media_type = MediaType(report.get("File_Type", "image").lower())
        except ValueError:
            media_type = MediaType.IMAGE

        key_findings = json.loads(report.get("Key_Findings") or "[]")

        agg = AggregatedResult(
            overall_score=float(report["Overall_Score"]),
            overall_confidence=float(report.get("Confidence") or 0.0),
            verdict=report["Verdict"],
            risk_level=report["Risk_Level"],
            flagged=bool(report["Flagged"]),
            module_results=mod_results,
            media_type=media_type,
            threshold=float(report.get("Threshold_Used") or 0.5),
            key_findings=key_findings,
        )

        self._last_result = agg
        self._current_file = report.get("File_Path", "")
        self._populate_results(agg, mod_results, report.get("File_Name", "Historical result"))
        self._show_results()
        self.export_btn.setEnabled(True)

    def _on_export(self):
        if not self._last_result:
            return
        from utils.export import export_pdf_report
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report",
            str(Path.home() / "truesight_report.pdf"),
            "PDF Files (*.pdf)"
        )
        if save_path:
            source = self._current_file or ""
            try:
                export_pdf_report(self._last_result, source, save_path)
                QMessageBox.information(
                    self, "Export Complete",
                    f"Report saved to:\n{save_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))

    # ─────────────────────────────────────────── HELPERS ──

    def _clear_result_cards(self):
        """Remove all module cards, keeping only the trailing stretch."""
        while self.results_layout.count() > 1:
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    # ─────────────────────────── DRAG & DROP (whole window) ──

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            self._on_file_selected(urls[0].toLocalFile())
