"""
Base detector interface. All detection modules inherit from this.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class MediaType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


@dataclass
class DetectionResult:
    """Standardised result returned by every detector module."""
    module_name: str
    score: float                        # 0.0 = authentic, 1.0 = deepfake
    confidence: float                   # How confident the module is (0-1)
    supported: bool = True              # False if module can't process this media type
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def label(self) -> str:
        if not self.supported:
            return "N/A"
        if self.score < 0.35:
            return "Likely Authentic"
        elif self.score < 0.65:
            return "Suspicious"
        else:
            return "Likely Fake"

    @property
    def risk_level(self) -> str:
        if not self.supported:
            return "na"
        if self.score < 0.35:
            return "low"
        elif self.score < 0.65:
            return "medium"
        else:
            return "high"


class BaseDetector(ABC):
    """Abstract base class for all detection modules."""

    name: str = "Base Detector"
    version: str = "1.0.0"
    supported_types: list[MediaType] = []

    def supports(self, media_type: MediaType) -> bool:
        return media_type in self.supported_types

    def run(self, media_path: str, media_type: MediaType) -> DetectionResult:
        """Entry point. Handles errors gracefully."""
        if not self.supports(media_type):
            return DetectionResult(
                module_name=self.name,
                score=0.0,
                confidence=0.0,
                supported=False,
            )
        try:
            return self._detect(media_path, media_type)
        except Exception as e:
            return DetectionResult(
                module_name=self.name,
                score=0.0,
                confidence=0.0,
                error=str(e),
            )

    @abstractmethod
    def _detect(self, media_path: str, media_type: MediaType) -> DetectionResult:
        """Implement detection logic in subclasses."""
        ...
