from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from app.services.knn_eval import (
    DatasetKNNReport,
    discover_dataset_roots,
    evaluate_dataset_knn,
    load_dataset_examples,
    summarize_reports,
)
from app.services.wavjepa import WavJEPAService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate WavJEPA mean-pooled embeddings with X-ARES-style weighted kNN "
            "on datasets that follow dataset_root/{train,test}/*.wav + *.json."
        )
    )
    parser.add_argument(
        "dataset_paths",
        nargs="+",
        help=(
            "One or more dataset directories, or parent directories whose immediate children "
            "contain train/ and test/."
        ),
    )
    parser.add_argument(
        "--label-key",
        default=None,
        help=(
            "Dot-separated JSON key for the label, for example 'label' or 'metadata.class_name'. "
            "If omitted, common top-level keys are inferred automatically."
        ),
    )
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature used for similarity-weighted voting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of test embeddings to score per kNN batch.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Optional limit for train/test files per dataset, useful for debugging.",
    )
    parser.add_argument(
        "--model-source",
        default=None,
        help="Optional HF model directory or checkpoint path. Overrides WAVJEPA_MODEL_SOURCE.",
    )
    parser.add_argument(
        "--knn-device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cpu",
        help="Device for kNN similarity search. Encoder inference still follows WavJEPAService device detection.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the full evaluation result as JSON.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.model_source is not None:
        os.environ["WAVJEPA_MODEL_SOURCE"] = str(Path(args.model_source).expanduser().resolve())

    service = WavJEPAService()
    dataset_roots = discover_dataset_roots(args.dataset_paths)
    knn_device = resolve_knn_device(args.knn_device)

    print(f"Encoder device: {service.device}")
    print(f"Evaluating {len(dataset_roots)} dataset(s) with k={args.k}, temperature={args.temperature}")

    reports: list[DatasetKNNReport] = []

    for index, dataset_root in enumerate(dataset_roots, start=1):
        dataset = load_dataset_examples(
            dataset_root,
            label_key=args.label_key,
            limit_per_split=args.limit_per_split,
        )
        print(
            f"[{index}/{len(dataset_roots)}] {dataset.name}: "
            f"train={len(dataset.train)} test={len(dataset.test)}"
        )

        report = evaluate_dataset_knn(
            dataset,
            service=service,
            label_key=args.label_key,
            k=args.k,
            temperature=args.temperature,
            batch_size=args.batch_size,
            device=knn_device,
        )
        reports.append(report)

        print(
            f"  accuracy={report.accuracy:.4f} "
            f"macro_f1={report.macro_f1:.4f} "
            f"weighted_f1={report.weighted_f1:.4f}"
        )

    summary = summarize_reports(reports)
    print(
        "Summary: "
        f"weighted_accuracy={summary['weightedAccuracy']:.4f} "
        f"weighted_macro_f1={summary['weightedMacroF1']:.4f} "
        f"weighted_weighted_f1={summary['weightedWeightedF1']:.4f}"
    )

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": service.describe_artifact(),
            "config": {
                "k": args.k,
                "temperature": args.temperature,
                "batchSize": args.batch_size,
                "labelKey": args.label_key,
                "limitPerSplit": args.limit_per_split,
                "knnDevice": knn_device,
            },
            "datasets": [report.to_dict() for report in reports],
            "summary": summary,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {output_path}")


def resolve_knn_device(raw_device: str) -> str:
    if raw_device != "auto":
        return raw_device

    if torch.cuda.is_available():
        return "cuda"

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


if __name__ == "__main__":
    main()
