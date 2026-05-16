"""
Key Findings Widget
--------------------
Displays plain-language investigation findings below the module result cards.
Separated from VerdictWidget so it has full width and proper space.
"""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QLabel,
)
from PySide6.QtCore import Qt
from core.aggregator import AggregatedResult


class KeyFindingsWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("keyFindingsCard")
        self.setFrameShape(QFrame.StyledPanel)
        self.setVisible(False)
        self._build()

    def _build(self):
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 14, 16, 14)
        self._layout.setSpacing(8)

        header = QLabel("KEY FINDINGS  —  Evidence Summary")
        header.setObjectName("keyFindingsHeader")
        self._layout.addWidget(header)

        self._findings_layout = QVBoxLayout()
        self._findings_layout.setSpacing(6)
        self._layout.addLayout(self._findings_layout)

    def update_result(self, result: AggregatedResult):
        self._clear()
        if not result.key_findings:
            self.setVisible(False)
            return

        for i, finding in enumerate(result.key_findings):
            card = self._make_card(finding, i, result.risk_level)
            self._findings_layout.addWidget(card)

        self.setVisible(True)

    def _make_card(self, text: str, index: int, risk_level: str) -> QFrame:
        card = QFrame()
        card.setObjectName("findingCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)

        if index == 0:
            obj = (
                "findingSummaryHigh"   if risk_level == "high"   else
                "findingSummaryMedium" if risk_level == "medium" else
                "findingSummaryLow"
            )
        else:
            obj = "findingText"

        lbl = QLabel(text)
        lbl.setObjectName(obj)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(lbl)
        return card

    def _clear(self):
        while self._findings_layout.count() > 0:
            item = self._findings_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def reset(self):
        self._clear()
        self.setVisible(False)
