"""
Score Aggregator
-----------------
Combines individual module scores into a single deepfake probability,
applying confidence-weighted averaging and sensitivity threshold logic.
Also generates plain-language Key Findings for cybersecurity investigation use.
"""

from dataclasses import dataclass, field
from detectors.base import DetectionResult, MediaType


# Module weights per media type (must sum to 1.0 for active modules)
WEIGHTS: dict[str, dict[str, float]] = {
    MediaType.IMAGE: {
        "Error Level Analysis":              0.20,
        "Frequency Analysis":                0.15,
        "Face / Image Deepfake CNN":         0.65,
    },
    MediaType.VIDEO: {
        "Frequency Analysis":                0.08,
        "Face / Image Deepfake CNN":         0.42,
        "Temporal Consistency":              0.25,
        "Lip-Sync Analysis":                 0.15,
        "Audio Deepfake Detector (Wav2Vec2)": 0.05,
        "Noise Floor Consistency":           0.05,
    },
    MediaType.AUDIO: {
        "Audio Deepfake Detector (Wav2Vec2)": 0.70,
        "Noise Floor Consistency":            0.30,
    },
}

# Plain-language finding templates per module, keyed by risk level
FINDING_TEMPLATES: dict[str, dict[str, str]] = {
    "Error Level Analysis": {
        "high":   "Error level analysis detected significant pixel-level inconsistencies, "
                  "suggesting localized image tampering or compositing.",
        "medium": "Error level analysis found moderate compression anomalies across regions "
                  "of the image, which may indicate partial manipulation.",
        "low":    "Error level analysis found uniform compression patterns consistent "
                  "with an unmodified image.",
    },
    "Frequency Analysis": {
        "high":   "Frequency domain analysis revealed strong GAN-associated artifacts "
                  "in the DCT spectrum, a common fingerprint of AI-generated imagery.",
        "medium": "Frequency domain analysis detected mild spectral irregularities "
                  "that may indicate synthetic generation or post-processing.",
        "low":    "Frequency domain analysis found a natural spectral distribution "
                  "with no significant synthetic artifacts.",
    },
    "Face / Image Deepfake CNN": {
        "high":   "The ViT deepfake classifier flagged high-confidence manipulation "
                  "indicators, including texture inconsistencies and unnatural facial "
                  "boundaries consistent with AI-generated imagery.",
        "medium": "The ViT deepfake classifier detected moderate anomalies partially "
                  "consistent with deepfake generation.",
        "low":    "The ViT deepfake classifier found no significant manipulation "
                  "artifacts in the analysed image or face region.",
    },
    "Temporal Consistency": {
        "high":   "Frame-to-frame analysis detected significant flickering and irregular "
                  "optical flow patterns, indicating inconsistent facial rendering across "
                  "video frames — a key indicator of deepfake video generation.",
        "medium": "Temporal analysis detected moderate inter-frame inconsistencies that "
                  "suggest possible manipulation in isolated segments of the video.",
        "low":    "Temporal analysis found consistent frame-to-frame motion patterns "
                  "with no significant irregularities.",
    },
    "Lip-Sync Analysis": {
        "high":   "Lip-sync analysis detected significant desynchronization between mouth "
                  "movements and the audio track, strongly suggesting that the audio or "
                  "video has been separately manipulated.",
        "medium": "Lip-sync analysis found moderate audio-visual misalignment that may "
                  "indicate partial manipulation of either the video or audio channel.",
        "low":    "Lip-sync analysis found the mouth movements and audio track to be "
                  "well-synchronized, consistent with authentic recording.",
    },
    "Audio Deepfake Detector (Wav2Vec2)": {
        "high":   "The Wav2Vec2 audio classifier identified strong synthetic speech "
                  "indicators, including unnatural pitch contours and spectral patterns "
                  "typical of AI voice cloning systems.",
        "medium": "The audio classifier detected moderate anomalies partially consistent "
                  "with AI-synthesised or cloned speech.",
        "low":    "The audio classifier found natural speech characteristics consistent "
                  "with authentic human speech.",
    },
    "Noise Floor Consistency": {
        "high":   "Background noise analysis detected abrupt changes in the noise floor "
                  "across audio segments, indicating possible audio splicing or "
                  "insertion of synthetic speech segments.",
        "medium": "Background noise analysis found moderate inconsistencies in the "
                  "ambient noise profile that may suggest audio editing.",
        "low":    "Background noise analysis found a consistent noise floor throughout "
                  "the recording, consistent with a single unedited audio source.",
    },
}


def _generate_key_findings(
    results: list[DetectionResult],
    overall_score: float,
    verdict: str,
    risk_level: str,
    flagged: bool,
) -> list[str]:
    """
    Generate a list of plain-language key findings for cybersecurity reporting.
    Findings are sorted by severity (highest score first) and only include
    modules that produced meaningful results.
    """
    findings = []

    # Overall summary finding
    if risk_level == "high":
        findings.append(
            f"Overall analysis indicates a HIGH probability ({round(overall_score * 100)}%) "
            f"of deepfake manipulation. The content is flagged as '{verdict}' and should "
            f"be treated as potentially fraudulent media."
        )
    elif risk_level == "medium":
        findings.append(
            f"Overall analysis returned a MODERATE manipulation probability "
            f"({round(overall_score * 100)}%). The content is classified as '{verdict}' "
            f"and warrants further investigation before use as evidence."
        )
    else:
        findings.append(
            f"Overall analysis returned a LOW manipulation probability "
            f"({round(overall_score * 100)}%). The content is classified as '{verdict}', "
            f"though results should always be corroborated with additional review."
        )

    # Sort module results by score descending — highest anomalies first
    active = [r for r in results if r.supported and not r.error and r.score > 0.0]
    active.sort(key=lambda r: r.score, reverse=True)

    for result in active:
        templates = FINDING_TEMPLATES.get(result.module_name)
        if not templates:
            continue
        level = result.risk_level
        if level == "na":
            continue
        text = templates.get(level, "")
        if text:
            findings.append(f"[{result.module_name}] {text}")

    # Flagged notice
    if flagged:
        findings.append(
            "⚑  This file exceeds the configured detection sensitivity threshold and "
            "is formally flagged for review. All findings above are documented as "
            "anomaly evidence for investigative purposes."
        )

    return findings


@dataclass
class AggregatedResult:
    overall_score: float        # 0.0 - 1.0
    overall_confidence: float   # 0.0 - 1.0
    verdict: str                # "Likely Authentic" | "Suspicious" | "Likely Fake"
    risk_level: str             # "low" | "medium" | "high"
    flagged: bool               # True if score exceeds user threshold
    module_results: list[DetectionResult]
    media_type: MediaType
    threshold: float
    key_findings: list[str] = field(default_factory=list)

    @property
    def score_pct(self) -> int:
        return round(self.overall_score * 100)

    @property
    def confidence_pct(self) -> int:
        return round(self.overall_confidence * 100)


def aggregate(
    results: list[DetectionResult],
    media_type: MediaType,
    threshold: float = 0.5,
) -> AggregatedResult:
    """
    Compute weighted average score from module results.

    Args:
        results:    List of DetectionResult from the pipeline.
        media_type: IMAGE | VIDEO | AUDIO.
        threshold:  User-set sensitivity threshold (0.0 - 1.0).
                    Content is flagged as deepfake if score >= threshold.

    Returns:
        AggregatedResult with final verdict.
    """
    weights = WEIGHTS.get(media_type, {})

    weighted_score = 0.0
    weighted_confidence = 0.0
    total_weight = 0.0

    for result in results:
        if not result.supported or result.error:
            continue
        w = weights.get(result.module_name, 0.0)
        if w == 0.0:
            # Give equal weight to unregistered modules
            w = 1.0 / max(len(results), 1)
        # Further weight by module's own confidence
        effective_w = w * result.confidence
        weighted_score += result.score * effective_w
        weighted_confidence += result.confidence * w
        total_weight += effective_w

    if total_weight == 0.0:
        overall_score = 0.0
        overall_confidence = 0.0
    else:
        overall_score = weighted_score / total_weight
        # Use the same confidence-weighted denominator so result stays in [0,1].
        # The previous expression used a plain-weighpython main.pyt sum which didn't match
        # total_weight, causing occasional > 1.0 outputs.
        overall_confidence = weighted_confidence / total_weight

    overall_score = max(0.0, min(1.0, overall_score))
    overall_confidence = max(0.0, min(1.0, overall_confidence))

    # Verdict thresholds (independent of user sensitivity)
    if overall_score < 0.35:
        verdict = "Likely Authentic"
        risk_level = "low"
    elif overall_score < 0.65:
        verdict = "Suspicious"
        risk_level = "medium"
    else:
        verdict = "Likely Fake"
        risk_level = "high"

    flagged = overall_score >= threshold

    key_findings = _generate_key_findings(
        results, overall_score, verdict, risk_level, flagged
    )

    return AggregatedResult(
        overall_score=round(overall_score, 4),
        overall_confidence=round(overall_confidence, 4),
        verdict=verdict,
        risk_level=risk_level,
        flagged=flagged,
        module_results=results,
        media_type=media_type,
        threshold=threshold,
        key_findings=key_findings,
    )
