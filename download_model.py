"""
download_model.py
─────────────────
Downloads model weights from HuggingFace Hub at Docker build time.
Run: python download_model.py
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

HF_REPO  = os.getenv("HF_MODEL_REPO",
                      "YOUR_HF_USERNAME/terrain-intelligence")
HF_TOKEN = os.getenv("HF_TOKEN", "")

DEST_DIR  = Path("./runs/deployed")
DEST_DIR.mkdir(parents=True, exist_ok=True)
DEST_PATH = DEST_DIR / "best.pth"

if DEST_PATH.exists():
    print(f"✅ Model already exists: {DEST_PATH}")
else:
    print(f"Downloading model from {HF_REPO}...")
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename="best.pth",
        token=HF_TOKEN or None,
        local_dir=str(DEST_DIR),
    )
    print(f"✅ Model downloaded to {path}")

# Update .env to point to downloaded model
env_path = Path(".env")
env_text = env_path.read_text() if env_path.exists() else ""

if "MODEL_CHECKPOINT" not in env_text:
    with open(env_path, "a") as f:
        f.write(f"\nMODEL_CHECKPOINT={DEST_PATH}\n")
    print(f"✅ MODEL_CHECKPOINT set to {DEST_PATH}")
else:
    lines = env_text.splitlines()
    lines = [
        f"MODEL_CHECKPOINT={DEST_PATH}"
        if l.startswith("MODEL_CHECKPOINT") else l
        for l in lines
    ]
    env_path.write_text("\n".join(lines))
    print(f"✅ MODEL_CHECKPOINT updated to {DEST_PATH}")