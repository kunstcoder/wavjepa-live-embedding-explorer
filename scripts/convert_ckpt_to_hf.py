from __future__ import annotations

import argparse
from pathlib import Path

from app.services.model_artifacts import convert_checkpoint_to_hf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a WavJEPA PyTorch/Lightning checkpoint into an HF-compatible safetensors directory.",
    )
    parser.add_argument("checkpoint", help="Path to the .ckpt/.pt/.pth checkpoint file.")
    parser.add_argument(
        "--output-dir",
        help="Destination directory for the converted HF artifact. Defaults to .hf_cache/converted_models/<fingerprint>.",
    )
    parser.add_argument(
        "--template-dir",
        help="Template HF model directory used to copy Python modules and feature extractor config. Defaults to models/wavjepa-base.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing converted directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = convert_checkpoint_to_hf(
        args.checkpoint,
        output_dir=args.output_dir,
        template_dir=args.template_dir,
        force=args.force,
    )
    print(f"Converted checkpoint into {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
