from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from threading import RLock
from typing import Any

from huggingface_hub import snapshot_download
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch


HF_CACHE_DIR = Path(__file__).resolve().parents[2] / ".hf_cache"
HF_HUB_CACHE_DIR = HF_CACHE_DIR / "hub"

HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HUB_CACHE_DIR))


MODEL_REPO_ID = "labhamlet/wavjepa-base"
MODEL_REVISION = "main"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "wavjepa-base"
MODEL_PACKAGE_NAME = "local_wavjepa_base"


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


@dataclass
class EmbeddingSummary:
    pooled_embedding: np.ndarray
    temporal_steps: int
    embedding_dim: int
    pooled_norm: float
    rms_energy: float


class WavJEPAService:
    def __init__(self) -> None:
        self.device = detect_device()
        self._feature_extractor = None
        self._model = None
        self._lock = RLock()

    def is_snapshot_available(self) -> bool:
        return (MODEL_DIR / "model.safetensors").exists()

    def ensure_snapshot(self) -> Path:
        if self.is_snapshot_available():
            return MODEL_DIR

        with self._lock:
            if self.is_snapshot_available():
                return MODEL_DIR

            MODEL_DIR.mkdir(parents=True, exist_ok=True)

            snapshot_download(
                repo_id=MODEL_REPO_ID,
                revision=MODEL_REVISION,
                local_dir=MODEL_DIR,
                cache_dir=str(HF_HUB_CACHE_DIR),
                allow_patterns=[
                    "*.json",
                    "*.md",
                    "*.py",
                    "*.safetensors",
                ],
            )

        return MODEL_DIR

    def load(self) -> None:
        if self._model is not None and self._feature_extractor is not None:
            return

        with self._lock:
            if self._model is not None and self._feature_extractor is not None:
                return

            model_path = str(self.ensure_snapshot())
            feature_extractor_class, config_class, model_class = load_local_model_classes()
            config = config_class.from_pretrained(model_path)
            self._feature_extractor = feature_extractor_class.from_pretrained(model_path)
            self._model = model_class.from_pretrained(
                model_path,
                config=config,
                local_files_only=True,
            )
            self._model.to(self.device)
            self._model.eval()

    @torch.inference_mode()
    def embed_waveform(self, waveform: np.ndarray) -> EmbeddingSummary:
        self.load()

        inputs = self._feature_extractor(
            [waveform],
            sampling_rate=16_000,
            return_tensors="pt",
        )

        input_values = inputs["input_values"].to(self.device)
        outputs = self._model(input_values)

        frame_embeddings = outputs[0].detach().cpu().numpy().squeeze(0)
        pooled_embedding = frame_embeddings.mean(axis=0)

        return EmbeddingSummary(
            pooled_embedding=np.asarray(pooled_embedding, dtype=np.float32),
            temporal_steps=int(frame_embeddings.shape[0]),
            embedding_dim=int(frame_embeddings.shape[1]),
            pooled_norm=float(np.linalg.norm(pooled_embedding)),
            rms_energy=float(np.sqrt(np.mean(np.square(waveform)))),
        )


def project_embeddings(
    embeddings: np.ndarray,
    method: str,
    dimensions: int,
) -> tuple[np.ndarray, str]:
    if dimensions not in {2, 3}:
        raise ValueError("Projection dimensions must be 2 or 3.")

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    sample_count, feature_count = embeddings.shape
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)

    if sample_count == 1:
        return np.zeros((1, dimensions), dtype=np.float32), "identity"

    effective_method = method

    if method == "tsne":
        perplexity = max(1.0, min(30.0, float(sample_count - 1)))

        if sample_count <= dimensions:
            effective_method = "pca"
        else:
            projection = TSNE(
                n_components=dimensions,
                init="pca",
                learning_rate="auto",
                perplexity=perplexity,
                random_state=42,
            ).fit_transform(centered)
            return np.asarray(projection, dtype=np.float32), effective_method

    component_count = min(dimensions, sample_count, feature_count)
    projection = PCA(n_components=component_count, random_state=42).fit_transform(centered)

    if projection.shape[1] < dimensions:
        padding = np.zeros((sample_count, dimensions - projection.shape[1]), dtype=np.float32)
        projection = np.hstack([projection, padding])

    return np.asarray(projection, dtype=np.float32), effective_method


def serialize_vector(vector: np.ndarray) -> list[float]:
    return [float(value) for value in vector.tolist()]


def load_local_model_classes() -> tuple[type[Any], type[Any], type[Any]]:
    if MODEL_PACKAGE_NAME not in sys.modules:
        init_path = MODEL_DIR / "__init__.py"

        if not init_path.exists():
            init_path.write_text("", encoding="utf-8")

        spec = importlib.util.spec_from_file_location(
            MODEL_PACKAGE_NAME,
            init_path,
            submodule_search_locations=[str(MODEL_DIR)],
        )

        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create import spec for {MODEL_DIR}")

        package = importlib.util.module_from_spec(spec)
        sys.modules[MODEL_PACKAGE_NAME] = package
        spec.loader.exec_module(package)

    feature_module = importlib.import_module(f"{MODEL_PACKAGE_NAME}.feature_extraction_wavjepa")
    config_module = importlib.import_module(f"{MODEL_PACKAGE_NAME}.configuration_wavjepa")
    model_module = importlib.import_module(f"{MODEL_PACKAGE_NAME}.modeling_wavjepa")

    return (
        feature_module.WavJEPAFeatureExtractor,
        config_module.WavJEPAConfig,
        model_module.WavJEPAModel,
    )


def build_projection_response(
    filenames: list[str],
    summaries: list[EmbeddingSummary],
    coordinates: np.ndarray,
    requested_method: str,
    effective_method: str,
    dimensions: int,
    durations: list[float],
    sample_rates: list[int],
    extra_fields: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    points = []

    for index, (filename, summary, coordinate, duration, sample_rate) in enumerate(
        zip(filenames, summaries, coordinates, durations, sample_rates)
    ):
        points.append(
            {
                "id": index,
                "label": filename,
                "coordinates": serialize_vector(coordinate),
                "durationSeconds": round(duration, 3),
                "sampleRate": sample_rate,
                "temporalSteps": summary.temporal_steps,
                "embeddingDim": summary.embedding_dim,
                "pooledNorm": round(summary.pooled_norm, 4),
                "rmsEnergy": round(summary.rms_energy, 6),
            }
        )

        if extra_fields is not None:
            points[-1].update(extra_fields[index])

    return {
        "requestedMethod": requested_method,
        "effectiveMethod": effective_method,
        "dimensions": dimensions,
        "pointCount": len(points),
        "points": points,
        "model": {
            "repoId": MODEL_REPO_ID,
            "localPath": str(MODEL_DIR),
        },
    }
