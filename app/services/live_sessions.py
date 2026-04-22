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
