"""Overall verdict display panel with Key Findings for cybersecurity reporting."""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QScrollArea, QWidget
)
from PySide6.QtCore import Qt
from core.aggregator import AggregatedResult


class VerdictWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("verdictCard")
        self.setFrameShape(QFrame.StyledPanel)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 16, 14, 16)
        layout.setSpacing(6)

        self.panel_label = QLabel("OVERALL VERDICT")
        self.panel_label.setObjectName("panelLabel")
        self.panel_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.panel_label)

        # "DEEPFAKE PROBABILITY" label sits above the number
        self.prob_label = QLabel("DEEPFAKE PROBABILITY")
        self.prob_label.setObjectName("verdictProbLabel")
        self.prob_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prob_label)

        self.score_label = QLabel("—")
        self.score_label.setObjectName("verdictScore")
        self.score_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.score_label)

        # Risk description — describes the level, not the number
        self.verdict_label = QLabel("Awaiting analysis")
        self.verdict_label.setObjectName("verdictText")
        self.verdict_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.verdict_label)

        self.confidence_label = QLabel("")
        self.confidence_label.setObjectName("verdictConfidence")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.confidence_label)

        self.flagged_label = QLabel("")
        self.flagged_label.setObjectName("flaggedLabel")
        self.flagged_label.setAlignment(Qt.AlignCenter)
        self.flagged_label.setWordWrap(True)
        layout.addWidget(self.flagged_label)

        # Key Findings
        self.findings_header = QLabel("KEY FINDINGS")
        self.findings_header.setObjectName("panelLabel")
        self.findings_header.setVisible(False)
        layout.addWidget(self.findings_header)

        self.findings_scroll = QScrollArea()
        self.findings_scroll.setWidgetResizable(True)
        self.findings_scroll.setFrameShape(QFrame.NoFrame)
        self.findings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.findings_scroll.setVisible(False)
        self.findings_scroll.setMaximumHeight(500) #300
        self.findings_scroll.setMinimumHeight(200) #remove it dont work

        self.findings_container = QWidget()
        self.findings_layout = QVBoxLayout(self.findings_container)
        self.findings_layout.setContentsMargins(0, 0, 0, 0)
        self.findings_layout.setSpacing(6)
        self.findings_scroll.setWidget(self.findings_container)
        layout.addWidget(self.findings_scroll)

    @staticmethod
    def _repolish(widget):
        """Force Qt to re-evaluate QSS rules after a dynamic objectName change."""
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def update_result(self, result: AggregatedResult):
        self.score_label.setText(f"{result.score_pct}%")
        self.score_label.setObjectName(
            "verdictScoreHigh"   if result.risk_level == "high"   else
            "verdictScoreMedium" if result.risk_level == "medium" else
            "verdictScoreLow"
        )
        self._repolish(self.score_label)

        # Clear risk descriptions — describes what the score means, not restating it
        risk_descriptions = {
            "low":    "Low risk of manipulation",
            "medium": "Moderate manipulation risk",
            "high":   "High risk — likely manipulated",
        }
        self.verdict_label.setText(risk_descriptions.get(result.risk_level, result.verdict))
        self.confidence_label.setText(f"Detection confidence: {result.confidence_pct}%")

        if result.flagged:
            self.flagged_label.setText("⚑  Flagged — exceeds sensitivity threshold")
            self.flagged_label.setObjectName("flaggedLabelActive")
        else:
            self.flagged_label.setText("✓  Within acceptable threshold")
            self.flagged_label.setObjectName("flaggedLabelOk")
        self._repolish(self.flagged_label)

        self._clear_findings()
        if result.key_findings:
            self.findings_header.setVisible(True)
            self.findings_scroll.setVisible(True)
            for i, finding in enumerate(result.key_findings):
                card = self._make_finding_card(finding, i, result.risk_level)
                self.findings_layout.addWidget(card)
            self.findings_layout.addStretch()

    def _make_finding_card(self, text: str, index: int, risk_level: str) -> QFrame:
        card = QFrame()
        card.setObjectName("findingCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)

        if index == 0:
            obj = "findingSummaryHigh" if risk_level == "high" else \
                  "findingSummaryMedium" if risk_level == "medium" else "findingSummaryLow"
        else:
            obj = "findingText"

        label = QLabel(text)
        label.setObjectName(obj)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(label)
        return card

    def _clear_findings(self):
        while self.findings_layout.count() > 0:
            item = self.findings_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.findings_header.setVisible(False)
        self.findings_scroll.setVisible(False)

    def reset(self):
        self.score_label.setText("—")
        self.score_label.setObjectName("verdictScore")
        self.verdict_label.setText("Awaiting analysis")
        self.confidence_label.setText("")
        self.flagged_label.setText("")
        self._clear_findings()
