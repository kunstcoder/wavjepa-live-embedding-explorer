from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16_000


@dataclass
class AudioSample:
    waveform: np.ndarray
    original_sample_rate: int
    target_sample_rate: int
    duration_seconds: float


def compute_rms_energy(waveform: np.ndarray) -> float:
    if waveform.size == 0:
        return 0.0

    return float(np.sqrt(np.mean(np.square(waveform))))


def load_audio_from_bytes(raw_bytes: bytes, target_sample_rate: int = TARGET_SAMPLE_RATE) -> AudioSample:
    if not raw_bytes:
        raise ValueError("Empty audio payload.")

    waveform, original_sample_rate = sf.read(
        BytesIO(raw_bytes),
        dtype="float32",
        always_2d=True,
    )

    if waveform.size == 0:
        raise ValueError("Audio file does not contain samples.")

    mono_waveform = waveform.mean(axis=1)

    if original_sample_rate != target_sample_rate:
        gcd = math.gcd(original_sample_rate, target_sample_rate)
        mono_waveform = resample_poly(
            mono_waveform,
            target_sample_rate // gcd,
            original_sample_rate // gcd,
        ).astype(np.float32)

    duration_seconds = float(len(mono_waveform) / target_sample_rate)

    return AudioSample(
        waveform=np.asarray(mono_waveform, dtype=np.float32),
        original_sample_rate=int(original_sample_rate),
        target_sample_rate=target_sample_rate,
        duration_seconds=duration_seconds,
    )
