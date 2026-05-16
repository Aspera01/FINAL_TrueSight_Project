"""
Module result card widget with expandable detailed analysis.
The expanded view shows the raw detection metrics as evidence
for cybersecurity investigation documentation.
"""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QWidget,
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve
from detectors.base import DetectionResult


# Human-readable labels for the raw detail keys returned by each module
DETAIL_LABELS: dict[str, str] = {
    # ELA
    "mean_error":                   "Mean pixel error",
    "max_error":                    "Max pixel error",
    "std_error":                    "Error std deviation",
    "coefficient_of_variation":     "Error coefficient of variation",
    "ela_quality_used":             "ELA re-save quality",
    # Frequency
    "high_freq_ratio":              "High-frequency energy ratio",
    "dct_ac_variance":              "DCT AC coefficient variance",
    "hf_score":                     "High-freq anomaly score",
    "ac_score":                     "DCT artifact score",
    "frames_sampled":               "Frames sampled",
    "score_std":                    "Score std deviation",
    "min_frame_score":              "Min frame score",
    "max_frame_score":              "Max frame score",
    # Temporal
    "frames_analysed":              "Frames analysed",
    "frame_diff_cv":                "Frame difference CV",
    "flow_spike_ratio":             "Optical flow spike ratio",
    "mean_frame_diff":              "Mean frame difference",
    "fps":                          "Video FPS",
    # Lip sync
    "audio_mouth_correlation":      "Audio-mouth correlation",
    "frames_with_faces":            "Frames with faces detected",
    "mouth_aperture_std":           "Mouth aperture std deviation",
    "method":                       "Analysis method",
    "mouth_std":                    "Mouth movement std deviation",
    # Face CNN
    "faces_detected":               "Faces detected",
    "model":                        "Model used",
    "model_loaded":                 "Model loaded",
    # Audio spectrogram
    "pitch_variance":               "Pitch (F0) variance",
    "spectral_flatness_mean":       "Mean spectral flatness",
    "zcr_cv":                       "Zero-crossing rate CV",
    "silence_ratio":                "Silence ratio",
    "model_score":                  "CNN model score",
    "heuristic_score":              "Heuristic score",
    # Noise floor
    "segments_analysed":            "Audio segments analysed",
    "noise_floor_cv":               "Noise floor CV",
    "noise_spike_ratio":            "Noise spike ratio",
    "centroid_cv":                  "Spectral centroid CV",
    "mean_noise_floor":             "Mean noise floor level",
    # General
    "note":                         "Note",
}


class ResultPanel(QFrame):
    def __init__(self, result: DetectionResult, parent=None):
        super().__init__(parent)
        self.setObjectName("resultCard")
        self.setFrameShape(QFrame.StyledPanel)
        self._result = result
        self._expanded = False
        self._build(result)

    def _build(self, result: DetectionResult):
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(14, 10, 14, 10)
        self._outer.setSpacing(6)

        # --- Header row ---
        header = QHBoxLayout()
        name_label = QLabel(result.module_name)
        name_label.setObjectName("moduleNameLabel")

        score_label = QLabel()
        if not result.supported:
            score_label.setText("N/A")
            score_label.setObjectName("scoreLabelNA")
        elif result.error:
            score_label.setText("Error")
            score_label.setObjectName("scoreLabelError")
        else:
            pct = round(result.score * 100)
            score_label.setText(f"{pct}%")
            obj = "scoreLabelHigh" if result.risk_level == "high" else \
                  "scoreLabelMedium" if result.risk_level == "medium" else "scoreLabelLow"
            score_label.setObjectName(obj)

        header.addWidget(name_label)
        header.addStretch()
        header.addWidget(score_label)
        self._outer.addLayout(header)

        # --- Progress bar ---
        if result.supported and not result.error:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(round(result.score * 100))
            bar.setTextVisible(False)
            bar.setFixedHeight(4)
            bar.setObjectName(
                "barHigh" if result.risk_level == "high" else
                "barMedium" if result.risk_level == "medium" else "barLow"
            )
            self._outer.addWidget(bar)

        # --- Status row with expand button ---
        status_row = QHBoxLayout()

        status = QLabel()
        status.setObjectName("moduleStatusLabel")
        if not result.supported:
            status.setText("Not applicable for this media type")
        elif result.error:
            status.setText(f"⚠  {result.error}")
            status.setObjectName("moduleStatusError")
        else:
            conf_pct = round(result.confidence * 100)
            status.setText(f"{result.label}  ·  {conf_pct}% confidence")

        status_row.addWidget(status)
        status_row.addStretch()

        # Only show expand button if there are details to show
        has_details = (
            result.supported
            and not result.error
            and result.details
            and any(
                k != "note" or v
                for k, v in result.details.items()
            )
        )
        if has_details:
            self._toggle_btn = QPushButton("View Details ▾")
            self._toggle_btn.setObjectName("detailToggleBtn")
            self._toggle_btn.setFixedWidth(100)
            self._toggle_btn.clicked.connect(self._toggle_details)
            status_row.addWidget(self._toggle_btn)

        self._outer.addLayout(status_row)

        # --- Expandable detail section ---
        self._detail_widget = QWidget()
        self._detail_widget.setObjectName("detailWidget")
        detail_layout = QVBoxLayout(self._detail_widget)
        detail_layout.setContentsMargins(0, 6, 0, 0)
        detail_layout.setSpacing(4)

        if has_details:
            # Separator line label
            sep = QLabel("─" * 40)
            sep.setObjectName("detailSeparator")
            detail_layout.addWidget(sep)

            header_lbl = QLabel("Detection Metrics  (Evidence Reference)")
            header_lbl.setObjectName("detailHeader")
            detail_layout.addWidget(header_lbl)

            for key, value in result.details.items():
                label_text = DETAIL_LABELS.get(key, key.replace("_", " ").title())
                row = QHBoxLayout()

                key_lbl = QLabel(f"{label_text}:")
                key_lbl.setObjectName("detailKey")
                key_lbl.setFixedWidth(200)

                # Format value nicely
                if isinstance(value, float):
                    val_text = f"{value:.4f}"
                elif isinstance(value, bool):
                    val_text = "Yes" if value else "No"
                else:
                    val_text = str(value)

                val_lbl = QLabel(val_text)
                val_lbl.setObjectName("detailValue")
                val_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

                row.addWidget(key_lbl)
                row.addWidget(val_lbl)
                row.addStretch()
                detail_layout.addLayout(row)

            # Interpretive note
            interp = self._get_interpretation(result)
            if interp:
                note_lbl = QLabel(f"ℹ  {interp}")
                note_lbl.setObjectName("detailNote")
                note_lbl.setWordWrap(True)
                detail_layout.addWidget(note_lbl)

        self._detail_widget.setVisible(False)
        self._outer.addWidget(self._detail_widget)

    def _toggle_details(self):
        self._expanded = not self._expanded
        self._detail_widget.setVisible(self._expanded)
        self._toggle_btn.setText("Hide Details ▴" if self._expanded else "View Details ▾")
        # Nudge layout to resize the card
        self.adjustSize()
        if self.parent():
            self.parent().adjustSize()

    def _get_interpretation(self, result: DetectionResult) -> str:
        """Return a short plain-language interpretation of the raw metrics."""
        interps = {
            "Error Level Analysis": (
                "Higher mean error and coefficient of variation suggest localised pixel "
                "tampering. Values above 15 for mean error are considered anomalous."
            ),
            "Frequency Analysis": (
                "A high-frequency ratio above 1.0 or DCT AC variance above 3000 "
                "is associated with GAN-generated or synthetically altered imagery."
            ),
            "Face CNN (EfficientNet-B4)": (
                "Score represents the model's confidence that the face region has been "
                "synthetically generated or manipulated. Trained on FaceForensics++."
            ),
            "Temporal Consistency": (
                "Frame difference CV above 0.5 or flow spike ratio above 0.1 indicates "
                "unnatural flickering across frames, a hallmark of deepfake video."
            ),
            "Lip-Sync Analysis": (
                "Audio-mouth correlation near 1.0 indicates good synchronization. "
                "Values below 0.3 suggest audio and video are misaligned."
            ),
            "Audio Spectrogram (LCNN)": (
                "Pitch variance below 10 Hz and spectral flatness above 0.05 are "
                "indicators of synthetic or cloned speech generation."
            ),
            "Noise Floor Consistency": (
                "A noise floor CV above 0.5 or spike ratio above 0.2 suggests "
                "the audio was edited or assembled from multiple sources."
            ),
        }
        return interps.get(result.module_name, "")
