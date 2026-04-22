from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

from .audio import compute_rms_energy
from .model_artifacts import (
    BASE_MODEL_REPO_ID,
    ResolvedModelArtifact,
    load_local_model_classes,
    resolved_model_artifact,
)


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
        self._resolved_artifact: ResolvedModelArtifact | None = None
        self._lock = RLock()

    def is_snapshot_available(self) -> bool:
        try:
            artifact = resolved_model_artifact()
        except FileNotFoundError:
            return False

        return (artifact.model_dir / "model.safetensors").exists() or (
            artifact.model_dir / "pytorch_model.bin"
        ).exists()

    def ensure_model_artifact(self) -> ResolvedModelArtifact:
        with self._lock:
            if self._resolved_artifact is not None:
                return self._resolved_artifact

            self._resolved_artifact = resolved_model_artifact()
            return self._resolved_artifact

    def load(self) -> None:
        if self._model is not None and self._feature_extractor is not None:
            return

        with self._lock:
            if self._model is not None and self._feature_extractor is not None:
                return

            artifact = self.ensure_model_artifact()
            model_path = str(artifact.model_dir)
            feature_extractor_class, config_class, model_class = load_local_model_classes(artifact.model_dir)
            config = config_class.from_pretrained(model_path)
            self._feature_extractor = feature_extractor_class.from_pretrained(model_path)
            self._model = model_class.from_pretrained(
                model_path,
                config=config,
                local_files_only=True,
            )
            self._model.to(self.device)
            self._model.eval()

    def describe_artifact(self) -> dict[str, Any]:
        artifact = self.ensure_model_artifact()
        return {
            "repoId": BASE_MODEL_REPO_ID,
            "sourcePath": str(artifact.source_path),
            "localPath": str(artifact.model_dir),
            "format": artifact.format,
            "converted": artifact.converted,
        }

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
            rms_energy=compute_rms_energy(waveform),
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
    model_metadata: dict[str, Any] | None = None,
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
        "model": model_metadata if model_metadata is not None else service_model_metadata(),
    }


def service_model_metadata() -> dict[str, Any]:
    artifact = resolved_model_artifact()
    return {
        "repoId": BASE_MODEL_REPO_ID,
        "sourcePath": str(artifact.source_path),
        "localPath": str(artifact.model_dir),
        "format": artifact.format,
        "converted": artifact.converted,
    }
