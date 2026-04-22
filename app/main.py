from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

from .services.audio import compute_rms_energy, load_audio_from_bytes
from .services.live_sessions import LiveSessionStore
from .services.wavjepa import (
    WavJEPAService,
    build_projection_response,
    project_embeddings,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DEFAULT_LIVE_MIN_RMS_DBFS = float(os.environ.get("WAVJEPA_LIVE_MIN_RMS_DBFS", "-45.0"))
DEFAULT_LIVE_MIN_RMS_ENERGY = 10 ** (DEFAULT_LIVE_MIN_RMS_DBFS / 20.0)

app = FastAPI(
    title="WavJEPA Embedding Explorer",
    description="Extract WavJEPA embeddings and visualize them in 2D or 3D.",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

service = WavJEPAService()
live_sessions = LiveSessionStore()


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, object]:
    model_info: dict[str, object] | None = None

    try:
        model_info = service.describe_artifact()
    except FileNotFoundError:
        model_info = None

    return {
        "status": "ok",
        "device": service.device,
        "modelCached": service.is_snapshot_available(),
        "model": model_info,
    }


@app.post("/api/embeddings")
async def create_embeddings(
    files: list[UploadFile] = File(...),
    method: str = Form("pca"),
    dimensions: int = Form(2),
) -> dict[str, object]:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one audio file.")

    if method not in {"pca", "tsne"}:
        raise HTTPException(status_code=400, detail="method must be one of: pca, tsne")

    if dimensions not in {2, 3}:
        raise HTTPException(status_code=400, detail="dimensions must be 2 or 3")

    filenames: list[str] = []
    sample_rates: list[int] = []
    durations: list[float] = []
    summaries = []

    for uploaded_file in files:
        payload = await uploaded_file.read()

        if not payload:
            raise HTTPException(status_code=400, detail=f"{uploaded_file.filename or 'audio'} is empty.")

        try:
            sample = load_audio_from_bytes(payload)
            summary = service.embed_waveform(sample.waveform)
        except Exception as exc:  # pragma: no cover - runtime failures depend on local env.
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process {uploaded_file.filename or 'audio'}: {exc}",
            ) from exc

        filenames.append(uploaded_file.filename or f"audio-{len(filenames) + 1}")
        sample_rates.append(sample.target_sample_rate)
        durations.append(sample.duration_seconds)
        summaries.append(summary)

    matrix = np.vstack([summary.pooled_embedding for summary in summaries])
    coordinates, effective_method = project_embeddings(matrix, method=method, dimensions=dimensions)

    return build_projection_response(
        filenames=filenames,
        summaries=summaries,
        coordinates=coordinates,
        requested_method=method,
        effective_method=effective_method,
        dimensions=dimensions,
        durations=durations,
        sample_rates=sample_rates,
        model_metadata=service.describe_artifact(),
    )


@app.post("/api/live-sessions")
async def create_live_session() -> dict[str, str]:
    return {"sessionId": live_sessions.create_session()}


@app.delete("/api/live-sessions/{session_id}")
async def clear_live_session(session_id: str) -> dict[str, str]:
    try:
        live_sessions.clear_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Live session not found.") from exc

    return {
        "status": "cleared",
        "sessionId": session_id,
    }


@app.post("/api/live-sessions/{session_id}/chunks")
async def push_live_chunk(
    session_id: str,
    file: UploadFile = File(...),
    dimensions: int = Form(2),
    chunk_index: int = Form(0),
    elapsed_seconds: float = Form(0.0),
    min_rms_energy: float = Form(DEFAULT_LIVE_MIN_RMS_ENERGY),
) -> dict[str, object]:
    if dimensions not in {2, 3}:
        raise HTTPException(status_code=400, detail="dimensions must be 2 or 3")

    if min_rms_energy < 0:
        raise HTTPException(status_code=400, detail="min_rms_energy must be non-negative")

    payload = await file.read()

    if not payload:
        raise HTTPException(status_code=400, detail="Live audio chunk is empty.")

    try:
        sample = load_audio_from_bytes(payload)
        rms_energy = compute_rms_energy(sample.waveform)

        if rms_energy < min_rms_energy:
            return {
                "live": True,
                "sessionId": session_id,
                "accepted": False,
                "skipped": True,
                "skipReason": "low_rms",
                "pointCount": live_sessions.point_count(session_id),
                "chunkIndex": chunk_index,
                "elapsedSeconds": round(elapsed_seconds, 3),
                "rmsEnergy": round(rms_energy, 6),
                "minRmsEnergy": round(min_rms_energy, 6),
            }

        summary = service.embed_waveform(sample.waveform)
        response = live_sessions.append_chunk(
            session_id=session_id,
            label=f"t+{elapsed_seconds:05.1f}s",
            summary=summary,
            duration_seconds=sample.duration_seconds,
            sample_rate=sample.target_sample_rate,
            dimensions=dimensions,
            elapsed_seconds=elapsed_seconds,
            chunk_index=chunk_index,
            model_metadata=service.describe_artifact(),
        )
        response["accepted"] = True
        response["skipped"] = False
        response["minRmsEnergy"] = round(min_rms_energy, 6)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Live session not found.") from exc
    except Exception as exc:  # pragma: no cover - runtime failures depend on local env.
        raise HTTPException(status_code=500, detail=f"Failed to process live chunk: {exc}") from exc

    return response


def parse_env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def run() -> None:
    import uvicorn

    host = os.environ.get("WAVJEPA_HOST", "0.0.0.0")
    port = int(os.environ.get("WAVJEPA_PORT", "8000"))
    reload_enabled = parse_env_flag("WAVJEPA_RELOAD", default=False)
    ssl_certfile = os.environ.get("WAVJEPA_SSL_CERTFILE")
    ssl_keyfile = os.environ.get("WAVJEPA_SSL_KEYFILE")

    uvicorn.run(
        "app.main:app" if reload_enabled else app,
        host=host,
        port=port,
        reload=reload_enabled,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )


if __name__ == "__main__":
    run()
