from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
import time
from typing import Any
from uuid import uuid4

import numpy as np

from .wavjepa import EmbeddingSummary, build_projection_response, project_embeddings


@dataclass
class LiveSessionPoint:
    label: str
    summary: EmbeddingSummary
    duration_seconds: float
    sample_rate: int
    elapsed_seconds: float
    chunk_index: int
    captured_at: float = field(default_factory=time.time)


@dataclass
class LiveProjectionSession:
    session_id: str
    points: list[LiveSessionPoint] = field(default_factory=list)
    compare_points: dict[str, list[LiveSessionPoint]] = field(
        default_factory=lambda: {"wavjepa": [], "audioJepa": []}
    )


class LiveSessionStore:
    def __init__(self, max_points: int = 90) -> None:
        self.max_points = max_points
        self._sessions: dict[str, LiveProjectionSession] = {}
        self._lock = Lock()

    def create_session(self) -> str:
        session_id = uuid4().hex

        with self._lock:
            self._sessions[session_id] = LiveProjectionSession(session_id=session_id)

        return session_id

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)

            del self._sessions[session_id]

    def point_count(self, session_id: str) -> int:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)

            session = self._sessions[session_id]
            compare_count = max((len(points) for points in session.compare_points.values()), default=0)
            return max(len(session.points), compare_count)

    def append_chunk(
        self,
        session_id: str,
        label: str,
        summary: EmbeddingSummary,
        duration_seconds: float,
        sample_rate: int,
        dimensions: int,
        elapsed_seconds: float,
        chunk_index: int,
        model_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)

            session = self._sessions[session_id]
            session.points.append(
                LiveSessionPoint(
                    label=label,
                    summary=summary,
                    duration_seconds=duration_seconds,
                    sample_rate=sample_rate,
                    elapsed_seconds=elapsed_seconds,
                    chunk_index=chunk_index,
                )
            )

            if len(session.points) > self.max_points:
                session.points = session.points[-self.max_points :]

            matrix = np.vstack([point.summary.pooled_embedding for point in session.points])
            coordinates, effective_method = project_embeddings(
                matrix,
                method="pca",
                dimensions=dimensions,
            )

            response = build_projection_response(
                filenames=[point.label for point in session.points],
                summaries=[point.summary for point in session.points],
                coordinates=coordinates,
                requested_method="pca",
                effective_method=effective_method,
                dimensions=dimensions,
                durations=[point.duration_seconds for point in session.points],
                sample_rates=[point.sample_rate for point in session.points],
                extra_fields=[
                    {
                        "elapsedSeconds": round(point.elapsed_seconds, 3),
                        "chunkIndex": point.chunk_index,
                        "mode": "live",
                    }
                    for point in session.points
                ],
                model_metadata=model_metadata,
            )

        response["live"] = True
        response["sessionId"] = session_id
        return response

    def append_compare_chunk(
        self,
        session_id: str,
        label: str,
        wavjepa_summary: EmbeddingSummary,
        audio_jepa_summary: EmbeddingSummary,
        wavjepa_duration_seconds: float,
        audio_jepa_duration_seconds: float,
        wavjepa_sample_rate: int,
        audio_jepa_sample_rate: int,
        dimensions: int,
        elapsed_seconds: float,
        chunk_index: int,
        wavjepa_model_metadata: dict[str, Any] | None = None,
        audio_jepa_model_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)

            session = self._sessions[session_id]
            session.compare_points["wavjepa"].append(
                LiveSessionPoint(
                    label=label,
                    summary=wavjepa_summary,
                    duration_seconds=wavjepa_duration_seconds,
                    sample_rate=wavjepa_sample_rate,
                    elapsed_seconds=elapsed_seconds,
                    chunk_index=chunk_index,
                )
            )
            session.compare_points["audioJepa"].append(
                LiveSessionPoint(
                    label=label,
                    summary=audio_jepa_summary,
                    duration_seconds=audio_jepa_duration_seconds,
                    sample_rate=audio_jepa_sample_rate,
                    elapsed_seconds=elapsed_seconds,
                    chunk_index=chunk_index,
                )
            )

            for model_key, points in session.compare_points.items():
                if len(points) > self.max_points:
                    session.compare_points[model_key] = points[-self.max_points :]

            wavjepa_response, wavjepa_effective_method = self._build_live_model_response(
                points=session.compare_points["wavjepa"],
                dimensions=dimensions,
                model_metadata=wavjepa_model_metadata,
            )
            audio_jepa_response, audio_jepa_effective_method = self._build_live_model_response(
                points=session.compare_points["audioJepa"],
                dimensions=dimensions,
                model_metadata=audio_jepa_model_metadata,
            )

        effective_method = (
            wavjepa_effective_method
            if wavjepa_effective_method == audio_jepa_effective_method
            else f"{wavjepa_effective_method}/{audio_jepa_effective_method}"
        )

        return {
            "compare": True,
            "live": True,
            "sharedProjection": False,
            "projectionMode": "modelLocalNormalized",
            "requestedMethod": "pca",
            "effectiveMethod": effective_method,
            "dimensions": dimensions,
            "pointCount": max(
                wavjepa_response["pointCount"],
                audio_jepa_response["pointCount"],
            ),
            "sessionId": session_id,
            "models": {
                "wavjepa": wavjepa_response,
                "audioJepa": audio_jepa_response,
            },
        }

    def _build_live_model_response(
        self,
        points: list[LiveSessionPoint],
        dimensions: int,
        model_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str]:
        matrix = np.vstack([point.summary.pooled_embedding for point in points])
        coordinates, effective_method = project_embeddings(
            matrix,
            method="pca",
            dimensions=dimensions,
        )
        coordinates = normalize_live_compare_coordinates(coordinates)

        response = build_projection_response(
            filenames=[point.label for point in points],
            summaries=[point.summary for point in points],
            coordinates=coordinates,
            requested_method="pca",
            effective_method=effective_method,
            dimensions=dimensions,
            durations=[point.duration_seconds for point in points],
            sample_rates=[point.sample_rate for point in points],
            extra_fields=[
                {
                    "elapsedSeconds": round(point.elapsed_seconds, 3),
                    "chunkIndex": point.chunk_index,
                    "mode": "live",
                }
                for point in points
            ],
            model_metadata=model_metadata,
        )
        return response, effective_method


def normalize_live_compare_coordinates(coordinates: np.ndarray) -> np.ndarray:
    centered = coordinates.astype(np.float32, copy=False)

    if centered.shape[0] <= 1:
        return centered

    centered = centered - centered.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=1).max()

    if not np.isfinite(radius) or radius <= 1e-8:
        return centered

    return centered / radius
