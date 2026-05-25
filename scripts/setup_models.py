"""
TrueSight — Model Setup Script
--------------------------------
Downloads and sets up pretrained model weights.

Image model:  onnx-community/Deep-Fake-Detector-v2-Model-ONNX  (Apache 2.0)
Audio model:  MelodyMachine/Deepfake-audio-detection-V2         (Apache 2.0)
              → downloaded as Safetensors, converted to ONNX locally

Run once:
    python scripts/setup_models.py

Add --force to re-download even if files already exist.
"""

import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FACE_MODEL  = MODELS_DIR / "face_deepfake_vit.onnx"
AUDIO_MODEL = MODELS_DIR / "audio_deepfake_wav2vec2.onnx"


def check_deps():
    missing = []
    for pkg in ["huggingface_hub", "onnxruntime", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)


def download_face_model(force: bool):
    if FACE_MODEL.exists() and not force:
        print(f"  ✓  face_deepfake_vit.onnx  already present, skipping.")
        return True

    print("  ↓  face_deepfake_vit.onnx  (~330 MB)")
    print("     ViT deepfake image detector (Apache 2.0)")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="onnx-community/Deep-Fake-Detector-v2-Model-ONNX",
            filename="onnx/model.onnx",
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        src = Path(downloaded)
        if src != FACE_MODEL:
            src.rename(FACE_MODEL)
        print(f"     ✓  Saved.\n")
        return True
    except Exception as e:
        print(f"     ✗  Failed: {e}\n")
        return False


def download_audio_model(force: bool):
    if AUDIO_MODEL.exists() and not force:
        print(f"  ✓  audio_deepfake_wav2vec2.onnx  already present, skipping.")
        return True

    print("  ↓  audio_deepfake_wav2vec2.onnx  (~360 MB)")
    print("     Wav2Vec2 audio deepfake detector (Apache 2.0)")
    print("     Step 1: Downloading Safetensors weights...")

    try:
        from huggingface_hub import snapshot_download
        import subprocess

        # Download the full model repo (Safetensors)
        local_dir = MODELS_DIR / "audio_tmp"
        snapshot_download(
            repo_id="MelodyMachine/Deepfake-audio-detection-V2",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
        )
        print("     Step 2: Converting to ONNX (this may take 1-2 minutes)...")

        # Use optimum-cli to export to ONNX
        result = subprocess.run(
            [
                sys.executable, "-m", "optimum.exporters.onnx",
                "--model", str(local_dir),
                "--task", "audio-classification",
                str(MODELS_DIR / "audio_onnx_export"),
            ],
            capture_output=True, text=True
        )

        exported = MODELS_DIR / "audio_onnx_export" / "model.onnx"
        if not exported.exists():
            # Try alternative optimum CLI path
            result2 = subprocess.run(
                [
                    sys.executable, "-c",
                    f"""
from optimum.onnxruntime import ORTModelForAudioClassification
model = ORTModelForAudioClassification.from_pretrained('{local_dir}', export=True)
model.save_pretrained('{MODELS_DIR / "audio_onnx_export"}')
"""
                ],
                capture_output=True, text=True
            )
            exported = MODELS_DIR / "audio_onnx_export" / "model.onnx"

        if exported.exists():
            exported.rename(AUDIO_MODEL)
            # Cleanup tmp dirs
            import shutil
            shutil.rmtree(local_dir, ignore_errors=True)
            shutil.rmtree(MODELS_DIR / "audio_onnx_export", ignore_errors=True)
            print(f"     ✓  Saved.\n")
            return True
        else:
            print(f"     ✗  ONNX export failed.")
            print(f"        stdout: {result.stdout[-300:] if result.stdout else ''}")
            print(f"        stderr: {result.stderr[-300:] if result.stderr else ''}")
            print(f"     App will use heuristic-only audio detection.\n")
            return False

    except Exception as e:
        print(f"     ✗  Failed: {e}")
        print(f"     Tip: Run: pip install optimum[onnxruntime]")
        print(f"     App will use heuristic-only audio detection.\n")
        return False


def main():
    force = "--force" in sys.argv
    check_deps()

    print(f"\nTrueSight — Model Setup")
    print(f"Models directory: {MODELS_DIR}\n")

    ok_face  = download_face_model(force)
    ok_audio = download_audio_model(force)

    total = sum([ok_face, ok_audio])
    print("─" * 50)
    print(f"Setup complete: {total}/2 models ready.")
    if not ok_audio:
        print("\nNote: Audio detection will use heuristic analysis only.")
        print("To enable the CNN audio model, install optimum:")
        print("  pip install optimum[onnxruntime]")
        print("Then re-run: python scripts/setup_models.py")
    print()


if __name__ == "__main__":
    main()
