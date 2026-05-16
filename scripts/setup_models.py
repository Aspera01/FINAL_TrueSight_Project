"""
Model Setup Script
-------------------
Downloads open-source pretrained model weights from HuggingFace Hub.
Run this once before first use:

    python scripts/setup_models.py

All models are published under open licenses (Apache 2.0 / MIT / CC-BY).
No account or API key required for public models.
"""

import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# HuggingFace model registry
# Each entry: (repo_id, filename_in_repo, local_filename, description)
MODELS = [
    (
        "deepfake-detection/efficientnet-b4-faceforensics",
        "face_deepfake_efficientnet_b4.onnx",
        "face_deepfake_efficientnet_b4.onnx",
        "Face deepfake detector (EfficientNet-B4, trained on FaceForensics++, Apache 2.0)",
    ),
    (
        "deepfake-detection/lcnn-asvspoof2019",
        "audio_deepfake_lcnn.onnx",
        "audio_deepfake_lcnn.onnx",
        "Audio deepfake detector (LCNN, trained on ASVspoof 2019, MIT)",
    ),
]


def download_models(force: bool = False):
    try:
        from huggingface_hub import hf_hub_download
        from tqdm import tqdm
    except ImportError:
        print("ERROR: huggingface-hub and tqdm are required.")
        print("Run: pip install huggingface-hub tqdm")
        sys.exit(1)

    print(f"\nDownloading models to: {MODELS_DIR}\n")

    for repo_id, filename, local_name, description in MODELS:
        local_path = MODELS_DIR / local_name
        if local_path.exists() and not force:
            print(f"  ✓  {local_name} — already present, skipping.")
            continue

        print(f"  ↓  {local_name}")
        print(f"     {description}")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False,
            )
            # Rename if needed
            downloaded = MODELS_DIR / filename
            if downloaded.exists() and downloaded.name != local_name:
                downloaded.rename(local_path)
            print(f"     Saved to: {local_path}\n")
        except Exception as e:
            print(f"     WARNING: Could not download {local_name}: {e}")
            print(f"     The app will run in heuristic-only mode for this module.\n")

    print("Setup complete.\n")
    print("Note: The app works without models using heuristic analysis.")
    print("Download models for significantly better accuracy.\n")


if __name__ == "__main__":
    force = "--force" in sys.argv
    download_models(force=force)
