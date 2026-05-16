"""
Detection Pipeline
-------------------
Loads the correct detector modules for a given media type,
runs them in sequence (or parallel), and returns all results.
"""

import os
import mimetypes
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from detectors.base import BaseDetector, DetectionResult, MediaType
from detectors.image.ela import ELADetector
from detectors.image.frequency import FrequencyDetector
from detectors.image.face_cnn import FaceCNNDetector
from detectors.video.temporal import TemporalConsistencyDetector
from detectors.video.lipsync import LipSyncDetector
from detectors.audio.spectrogram import AudioSpectrogramDetector
from detectors.audio.noise_floor import NoiseFloorDetector


# All registered detectors
ALL_DETECTORS: list[BaseDetector] = [
    ELADetector(),
    FrequencyDetector(),
    FaceCNNDetector(),
    TemporalConsistencyDetector(),
    LipSyncDetector(),
    AudioSpectrogramDetector(),
    NoiseFloorDetector(),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".jfif"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


def detect_media_type(file_path: str) -> MediaType:
    ext = Path(file_path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return MediaType.IMAGE
    elif ext in VIDEO_EXTENSIONS:
        return MediaType.VIDEO
    elif ext in AUDIO_EXTENSIONS:
        return MediaType.AUDIO
    # Fallback: check MIME type
    mime, _ = mimetypes.guess_type(file_path)
    if mime:
        if mime.startswith("image/"):
            return MediaType.IMAGE
        elif mime.startswith("video/"):
            return MediaType.VIDEO
        elif mime.startswith("audio/"):
            return MediaType.AUDIO
    raise ValueError(f"Unsupported file type: {ext}")


def run_pipeline(
    file_path: str,
    progress_callback: Callable[[str, int, int], None] | None = None,
    parallel: bool = True,
) -> tuple[MediaType, list[DetectionResult]]:
    """
    Run all applicable detectors on the given file.

    Args:
        file_path:         Path to the media file.
        progress_callback: Optional fn(module_name, current, total) for UI updates.
        parallel:          Run detectors concurrently (faster on multi-core).

    Returns:
        (media_type, list of DetectionResult)
    """
    media_type = detect_media_type(file_path)
    applicable = [d for d in ALL_DETECTORS if d.supports(media_type)]
    total = len(applicable)
    results = []

    if parallel:
        futures = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            for detector in applicable:
                future = executor.submit(detector.run, file_path, media_type)
                futures[future] = detector.name

            completed = 0
            for future in as_completed(futures):
                name = futures[future]
                result = future.result()
                results.append(result)
                completed += 1
                if progress_callback:
                    progress_callback(name, completed, total)
    else:
        for i, detector in enumerate(applicable):
            if progress_callback:
                progress_callback(detector.name, i, total)
            result = detector.run(file_path, media_type)
            results.append(result)
            if progress_callback:
                progress_callback(detector.name, i + 1, total)

    # Sort by detector registration order for consistent display
    name_order = [d.name for d in applicable]
    results.sort(key=lambda r: name_order.index(r.module_name) if r.module_name in name_order else 99)

    return media_type, results
