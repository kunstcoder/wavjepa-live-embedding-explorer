from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "labhamlet/wavjepa-base"
LOCAL_DIR = Path(__file__).resolve().parents[1] / "models" / "wavjepa-base"
HF_CACHE_DIR = Path(__file__).resolve().parents[1] / ".hf_cache"

HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        revision="main",
        local_dir=LOCAL_DIR,
        cache_dir=str(HF_CACHE_DIR / "hub"),
        allow_patterns=[
            "*.json",
            "*.md",
            "*.py",
            "*.safetensors",
        ],
    )

    print(f"Downloaded {REPO_ID} into {LOCAL_DIR}")


if __name__ == "__main__":
    main()
