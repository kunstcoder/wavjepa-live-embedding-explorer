from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn.functional as F

from .audio import load_audio_from_bytes
from .wavjepa import WavJEPAService


COMMON_LABEL_KEYS = (
    "label",
    "labels",
    "class",
    "category",
    "target",
    "genre",
    "instrument",
    "emotion",
    "command",
    "intent",
    "speaker",
    "speaker_id",
    "language",
)


@dataclass(frozen=True)
class AudioExample:
    split: str
    sample_id: str
    audio_path: Path
    metadata_path: Path
    label: str


@dataclass(frozen=True)
class DatasetExamples:
    name: str
    root: Path
    train: list[AudioExample]
    test: list[AudioExample]


@dataclass(frozen=True)
class DatasetKNNReport:
    name: str
    root: str
    train_size: int
    test_size: int
    num_classes: int
    embedding_dim: int
    k: int
    temperature: float
    pooling: str
    label_key: str | None
    accuracy: float
    macro_f1: float
    weighted_f1: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_dataset_roots(paths: Sequence[str | Path]) -> list[Path]:
    dataset_roots: list[Path] = []
    seen: set[Path] = set()

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

        candidates: list[Path]

        if has_train_test_split(path):
            candidates = [path]
        else:
            candidates = [
                child
                for child in sorted(path.iterdir())
                if child.is_dir() and has_train_test_split(child)
            ]

        if not candidates:
            raise FileNotFoundError(
                f"Could not find a dataset root under {path}. Expected train/ and test/ directories."
            )

        for candidate in candidates:
            if candidate not in seen:
                dataset_roots.append(candidate)
                seen.add(candidate)

    return dataset_roots


def has_train_test_split(path: Path) -> bool:
    return (path / "train").is_dir() and (path / "test").is_dir()


def load_dataset_examples(
    dataset_root: str | Path,
    *,
    label_key: str | None = None,
    limit_per_split: int | None = None,
) -> DatasetExamples:
    root = Path(dataset_root).expanduser().resolve()

    if not has_train_test_split(root):
        raise FileNotFoundError(f"Expected train/ and test/ under dataset root: {root}")

    train_examples = load_split_examples(
        root / "train",
        split="train",
        label_key=label_key,
        limit=limit_per_split,
    )
    test_examples = load_split_examples(
        root / "test",
        split="test",
        label_key=label_key,
        limit=limit_per_split,
    )

    train_labels = {example.label for example in train_examples}
    missing_labels = sorted({example.label for example in test_examples} - train_labels)

    if missing_labels:
        missing = ", ".join(missing_labels[:10])
        raise ValueError(
            f"Test split contains labels unseen in train split for {root.name}: {missing}"
        )

    return DatasetExamples(
        name=root.name,
        root=root,
        train=train_examples,
        test=test_examples,
    )


def load_split_examples(
    split_dir: Path,
    *,
    split: str,
    label_key: str | None,
    limit: int | None,
) -> list[AudioExample]:
    wav_files = sorted(
        path for path in split_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"
    )

    if not wav_files:
        raise FileNotFoundError(f"No .wav files were found under {split_dir}")

    examples: list[AudioExample] = []

    for wav_path in wav_files:
        metadata_path = wav_path.with_suffix(".json")

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing matching JSON metadata for {wav_path}. Expected {metadata_path.name}"
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        label = extract_label(metadata, label_key=label_key)
        sample_id = str(wav_path.relative_to(split_dir).with_suffix(""))

        examples.append(
            AudioExample(
                split=split,
                sample_id=sample_id,
                audio_path=wav_path,
                metadata_path=metadata_path,
                label=label,
            )
        )

        if limit is not None and len(examples) >= limit:
            break

    return examples


def extract_label(metadata: Any, *, label_key: str | None) -> str:
    if label_key is not None:
        return normalize_label_value(resolve_label_key(metadata, label_key))

    if isinstance(metadata, dict):
        for key in COMMON_LABEL_KEYS:
            if key in metadata:
                return normalize_label_value(metadata[key])

        scalar_candidates = [
            value
            for value in metadata.values()
            if is_supported_label_value(value)
        ]

        if len(scalar_candidates) == 1:
            return normalize_label_value(scalar_candidates[0])

        keys = ", ".join(sorted(metadata.keys()))
        raise KeyError(
            "Could not infer a label field from JSON metadata. "
            f"Available top-level keys: {keys}. Use --label-key."
        )

    if is_supported_label_value(metadata):
        return normalize_label_value(metadata)

    raise TypeError(
        "Unsupported JSON metadata format for label extraction. "
        "Use --label-key to point to a scalar field."
    )


def resolve_label_key(metadata: Any, label_key: str) -> Any:
    current = metadata

    for token in label_key.split("."):
        if isinstance(current, dict):
            if token not in current:
                raise KeyError(f"Label key '{label_key}' was not found in metadata.")
            current = current[token]
            continue

        if isinstance(current, list):
            try:
                index = int(token)
            except ValueError as exc:
                raise KeyError(
                    f"Label key '{label_key}' requires a numeric index at '{token}'."
                ) from exc

            if index < 0 or index >= len(current):
                raise IndexError(f"Label key '{label_key}' index {index} is out of range.")

            current = current[index]
            continue

        raise KeyError(f"Label key '{label_key}' could not be resolved at '{token}'.")

    return current


def is_supported_label_value(value: Any) -> bool:
    if isinstance(value, (str, int, float, bool)):
        return True

    if isinstance(value, (list, tuple)) and len(value) == 1:
        return is_supported_label_value(value[0])

    return False


def normalize_label_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Multi-label JSON metadata is not supported by this evaluator.")
        return normalize_label_value(value[0])

    if isinstance(value, bool):
        return str(value).lower()

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    if isinstance(value, (str, int, float)):
        return str(value)

    raise TypeError(f"Unsupported label value type: {type(value).__name__}")


def encode_examples(
    examples: Sequence[AudioExample],
    *,
    service: WavJEPAService,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []

    for example in examples:
        audio_sample = load_audio_from_bytes(example.audio_path.read_bytes())
        embedding_summary = service.embed_waveform(audio_sample.waveform)
        embeddings.append(embedding_summary.pooled_embedding)

    return np.stack(embeddings, axis=0).astype(np.float32, copy=False)


class WeightedKNNClassifier:
    def __init__(
        self,
        train_embeddings: np.ndarray,
        train_labels: Sequence[int],
        *,
        num_classes: int,
        k: int = 10,
        temperature: float = 0.07,
        device: str | torch.device = "cpu",
    ) -> None:
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        if temperature <= 0:
            raise ValueError("temperature must be greater than 0.")

        self.device = torch.device(device)
        self.k = min(k, int(train_embeddings.shape[0]))
        self.temperature = temperature
        self.num_classes = num_classes
        self.train_embeddings = F.normalize(
            torch.from_numpy(train_embeddings).to(self.device),
            dim=1,
            p=2,
        )
        self.train_labels = torch.as_tensor(train_labels, device=self.device, dtype=torch.long)

    @torch.inference_mode()
    def predict(
        self,
        test_embeddings: np.ndarray,
        *,
        batch_size: int = 256,
    ) -> np.ndarray:
        predictions: list[torch.Tensor] = []

        for start in range(0, len(test_embeddings), batch_size):
            batch = torch.from_numpy(test_embeddings[start : start + batch_size]).to(self.device)
            batch = F.normalize(batch, dim=1, p=2)

            similarity = batch @ self.train_embeddings.transpose(0, 1)
            topk_similarity, topk_indices = similarity.topk(
                self.k,
                dim=1,
                largest=True,
                sorted=True,
            )
            neighbor_labels = self.train_labels[topk_indices]
            neighbor_weights = torch.softmax(topk_similarity / self.temperature, dim=1)

            class_scores = (
                F.one_hot(neighbor_labels, num_classes=self.num_classes).float()
                * neighbor_weights.unsqueeze(-1)
            ).sum(dim=1)
            predictions.append(class_scores.argmax(dim=1).cpu())

        return torch.cat(predictions, dim=0).numpy()


def evaluate_dataset_knn(
    dataset: DatasetExamples,
    *,
    service: WavJEPAService,
    label_key: str | None,
    k: int = 10,
    temperature: float = 0.07,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> DatasetKNNReport:
    label_to_index = build_label_index(example.label for example in dataset.train)
    y_train = np.asarray([label_to_index[example.label] for example in dataset.train], dtype=np.int64)
    y_test = np.asarray([label_to_index[example.label] for example in dataset.test], dtype=np.int64)

    train_embeddings = encode_examples(dataset.train, service=service)
    test_embeddings = encode_examples(dataset.test, service=service)

    classifier = WeightedKNNClassifier(
        train_embeddings,
        y_train,
        num_classes=len(label_to_index),
        k=k,
        temperature=temperature,
        device=device,
    )
    predictions = classifier.predict(test_embeddings, batch_size=batch_size)

    return DatasetKNNReport(
        name=dataset.name,
        root=str(dataset.root),
        train_size=len(dataset.train),
        test_size=len(dataset.test),
        num_classes=len(label_to_index),
        embedding_dim=int(train_embeddings.shape[1]),
        k=classifier.k,
        temperature=temperature,
        pooling="mean",
        label_key=label_key,
        accuracy=float(accuracy_score(y_test, predictions)),
        macro_f1=float(f1_score(y_test, predictions, average="macro")),
        weighted_f1=float(f1_score(y_test, predictions, average="weighted")),
    )


def build_label_index(labels: Iterable[str]) -> dict[str, int]:
    label_to_index: dict[str, int] = {}

    for label in labels:
        if label not in label_to_index:
            label_to_index[label] = len(label_to_index)

    return label_to_index


def summarize_reports(reports: Sequence[DatasetKNNReport]) -> dict[str, Any]:
    if not reports:
        return {
            "datasetCount": 0,
            "weightedAccuracy": 0.0,
            "weightedMacroF1": 0.0,
            "weightedWeightedF1": 0.0,
        }

    total_test = sum(report.test_size for report in reports)

    return {
        "datasetCount": len(reports),
        "weightedAccuracy": sum(report.accuracy * report.test_size for report in reports) / total_test,
        "weightedMacroF1": sum(report.macro_f1 * report.test_size for report in reports) / total_test,
        "weightedWeightedF1": sum(report.weighted_f1 * report.test_size for report in reports) / total_test,
    }
