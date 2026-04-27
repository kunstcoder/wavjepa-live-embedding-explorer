from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from huggingface_hub import hf_hub_download
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .audio import compute_rms_energy
from .model_artifacts import HF_HUB_CACHE_DIR, MODELS_DIR
from .wavjepa import EmbeddingSummary, detect_device


AUDIO_JEPA_MODEL_REPO_ID = "ltuncay/Audio-JEPA"
AUDIO_JEPA_CKPT_FILENAME = "JEPA.ckpt"
AUDIO_JEPA_MODEL_SOURCE_ENV_VAR = "AUDIO_JEPA_MODEL_SOURCE"
AUDIO_JEPA_REPO_ID_ENV_VAR = "AUDIO_JEPA_MODEL_REPO_ID"
AUDIO_JEPA_CKPT_FILENAME_ENV_VAR = "AUDIO_JEPA_CKPT_FILENAME"
AUDIO_JEPA_MODEL_DIR = MODELS_DIR / "audio-jepa"

AUDIO_JEPA_SAMPLE_RATE = 32_000
AUDIO_JEPA_CLIP_SECONDS = 10
AUDIO_JEPA_TARGET_TIME_BINS = 256
AUDIO_JEPA_MELS = 128
AUDIO_JEPA_PATCH_SIZE = (16, 16)
AUDIO_JEPA_EMBED_DIM = 768


def _optional_kaldi_fbank():
    try:
        from torchaudio.compliance.kaldi import fbank
    except Exception:
        return None

    return fbank


def normalize_audio(waveform: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    waveform = waveform - waveform.mean()
    peak = waveform.abs().max()
    return waveform / (peak + eps)


def hop_and_frame_for_target(
    *,
    num_samples: int,
    sample_rate: int,
    target_frames: int,
    ratio: float = 2.5,
) -> tuple[float, float]:
    hop_samples = max(1, int(round(num_samples / (target_frames - 1 + ratio))))
    frame_samples = int(round(ratio * hop_samples))
    return 1000.0 * frame_samples / sample_rate, 1000.0 * hop_samples / sample_rate


def hz_to_mel(frequency: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def mel_to_hz(mels: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mels / 2595.0) - 1.0)


def build_mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> torch.Tensor:
    mel_edges = np.linspace(float(hz_to_mel(f_min)), float(hz_to_mel(f_max)), n_mels + 2)
    hz_edges = mel_to_hz(mel_edges)
    fft_frequencies = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = hz_edges[index : index + 3]

        if center == left or right == center:
            continue

        up_slope = (fft_frequencies - left) / (center - left)
        down_slope = (right - fft_frequencies) / (right - center)
        filters[index] = np.maximum(0.0, np.minimum(up_slope, down_slope))

    return torch.from_numpy(filters)


class AudioJEPAMelSpecTransform(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int = AUDIO_JEPA_SAMPLE_RATE,
        n_mels: int = AUDIO_JEPA_MELS,
        clip_seconds: int = AUDIO_JEPA_CLIP_SECONDS,
        target_time_bins: int = AUDIO_JEPA_TARGET_TIME_BINS,
        f_min: int = 20,
        f_max: int | None = None,
        log: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.clip_seconds = int(clip_seconds)
        self.target_time_bins = int(target_time_bins)
        self.f_min = int(f_min)
        self.f_max = int(f_max if f_max is not None else sample_rate // 2)
        self.log = bool(log)
        self._kaldi_fbank = _optional_kaldi_fbank()

        frame_ms, hop_ms = hop_and_frame_for_target(
            num_samples=self.sample_rate * self.clip_seconds,
            sample_rate=self.sample_rate,
            target_frames=self.target_time_bins,
        )
        self.frame_length_ms = frame_ms
        self.hop_length_ms = hop_ms
        self.frame_length_samples = max(1, int(round(self.frame_length_ms * self.sample_rate / 1000.0)))
        self.hop_length_samples = max(1, int(round(self.hop_length_ms * self.sample_rate / 1000.0)))
        self.register_buffer(
            "mel_filterbank",
            build_mel_filterbank(
                sample_rate=self.sample_rate,
                n_fft=self.frame_length_samples,
                n_mels=self.n_mels,
                f_min=float(self.f_min),
                f_max=float(self.f_max),
            ),
            persistent=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.float().cpu()
        waveform = waveform - waveform.mean()

        if self._kaldi_fbank is not None:
            spec = self._kaldi_fbank(
                waveform,
                sample_frequency=self.sample_rate,
                frame_length=self.frame_length_ms,
                frame_shift=self.hop_length_ms,
                num_mel_bins=self.n_mels,
                high_freq=self.f_max,
                low_freq=self.f_min,
                use_log_fbank=self.log,
                window_type="hanning",
                snip_edges=True,
            )
        else:
            spec = self._fallback_fbank(waveform.squeeze(0))

        spec = self._fit_time_bins(spec)
        return spec.unsqueeze(0)

    def _fallback_fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.numel() < self.frame_length_samples:
            waveform = F.pad(waveform, (0, self.frame_length_samples - waveform.numel()))

        frames = waveform.unfold(0, self.frame_length_samples, self.hop_length_samples)
        if frames.numel() == 0:
            frames = waveform[: self.frame_length_samples].unsqueeze(0)

        window = torch.hann_window(
            self.frame_length_samples,
            periodic=False,
            dtype=waveform.dtype,
            device=waveform.device,
        )
        spectrum = torch.fft.rfft(frames * window, n=self.frame_length_samples, dim=-1)
        power = spectrum.real.square() + spectrum.imag.square()
        mel_energies = power @ self.mel_filterbank.to(power.device, dtype=power.dtype).transpose(0, 1)
        mel_energies = torch.clamp(mel_energies, min=1e-10)

        if self.log:
            return torch.log(mel_energies)

        return mel_energies

    def _fit_time_bins(self, spec: torch.Tensor) -> torch.Tensor:
        frame_count = spec.shape[0]

        if frame_count < self.target_time_bins:
            padding = self.target_time_bins - frame_count
            return F.pad(spec, (0, 0, 0, padding))

        if frame_count > self.target_time_bins:
            return spec[: self.target_time_bins]

        return spec


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class AudioJEPAMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class AudioJEPAMHA(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_proj_bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=qkv_proj_bias)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.dropout(attention)
        x = attention @ v
        x = x.transpose(1, 2).reshape(batch_size, token_count, embed_dim)
        return self.proj(x)


class AudioJEPABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AudioJEPAMHA(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            qkv_proj_bias=qkv_bias,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = AudioJEPAMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AudioJEPAPatchEmbed(nn.Module):
    def __init__(
        self,
        *,
        input_size: tuple[int, int] = (AUDIO_JEPA_TARGET_TIME_BINS, AUDIO_JEPA_MELS),
        patch_size: tuple[int, int] = AUDIO_JEPA_PATCH_SIZE,
        in_chans: int = 1,
        embed_dim: int = AUDIO_JEPA_EMBED_DIM,
    ) -> None:
        super().__init__()
        input_height, input_width = input_size
        patch_height, patch_width = patch_size
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches_h = input_height // patch_height
        self.num_patches_w = input_width // patch_width
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)

    def patch_grid(self, x: torch.Tensor) -> tuple[int, int]:
        return x.shape[-2] // self.patch_size[0], x.shape[-1] // self.patch_size[1]


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even.")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    output = np.einsum("m,d->md", positions.reshape(-1), omega)
    return np.concatenate([np.sin(output), np.cos(output)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even.")

    height_positions = np.arange(grid_h, dtype=np.float64)
    width_positions = np.arange(grid_w, dtype=np.float64)
    grid = np.meshgrid(width_positions, height_positions)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_h, grid_w)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class AudioJEPAVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_size: tuple[int, int] = (AUDIO_JEPA_TARGET_TIME_BINS, AUDIO_JEPA_MELS),
        patch_size: tuple[int, int] = AUDIO_JEPA_PATCH_SIZE,
        in_chans: int = 1,
        embed_dim: int = AUDIO_JEPA_EMBED_DIM,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.num_features = self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.patch_size = patch_size
        self.patch_embed = AudioJEPAPatchEmbed(
            input_size=input_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim),
            requires_grad=False,
        )
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim,
            self.patch_embed.num_patches_h,
            self.patch_embed.num_patches_w,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                AudioJEPABlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[index],
                    norm_layer=norm_layer,
                )
                for index in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_grid = self.patch_embed.patch_grid(x)
        x = self.patch_embed(x)
        x = x + self.interpolate_pos_encoding(patch_grid)

        for block in self.blocks:
            x = block(x)

        return self.norm(x)

    def interpolate_pos_encoding(self, patch_grid: tuple[int, int]) -> torch.Tensor:
        patch_h, patch_w = patch_grid
        if patch_h == self.patch_embed.num_patches_h and patch_w == self.patch_embed.num_patches_w:
            return self.pos_embed

        pos_embed = self.pos_embed.reshape(
            1,
            self.patch_embed.num_patches_h,
            self.patch_embed.num_patches_w,
            self.embed_dim,
        ).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(patch_h, patch_w), mode="bicubic", align_corners=False)
        return pos_embed.permute(0, 2, 3, 1).reshape(1, patch_h * patch_w, self.embed_dim)


@dataclass(frozen=True)
class ResolvedAudioJEPAArtifact:
    repo_id: str
    checkpoint_path: Path
    source_path: str
    cached: bool


class AudioJEPAService:
    def __init__(self) -> None:
        self.device = detect_device()
        self._model: AudioJEPAVisionTransformer | None = None
        self._mel_transform = AudioJEPAMelSpecTransform()
        self._resolved_artifact: ResolvedAudioJEPAArtifact | None = None
        self._lock = RLock()

    def is_checkpoint_available(self) -> bool:
        source = os.environ.get(AUDIO_JEPA_MODEL_SOURCE_ENV_VAR)
        if source:
            return Path(source).expanduser().is_file()

        return self.default_checkpoint_path().is_file()

    def ensure_model_artifact(self, *, download: bool = True) -> ResolvedAudioJEPAArtifact:
        with self._lock:
            if self._resolved_artifact is not None:
                if self._resolved_artifact.cached or not download:
                    return self._resolved_artifact

                if self._resolved_artifact.checkpoint_path.is_file():
                    self._resolved_artifact = ResolvedAudioJEPAArtifact(
                        repo_id=self._resolved_artifact.repo_id,
                        checkpoint_path=self._resolved_artifact.checkpoint_path,
                        source_path=self._resolved_artifact.source_path,
                        cached=True,
                    )
                    return self._resolved_artifact

            source = os.environ.get(AUDIO_JEPA_MODEL_SOURCE_ENV_VAR)
            if source:
                checkpoint_path = Path(source).expanduser().resolve()
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(f"Audio-JEPA checkpoint does not exist: {checkpoint_path}")

                self._resolved_artifact = ResolvedAudioJEPAArtifact(
                    repo_id=self.repo_id(),
                    checkpoint_path=checkpoint_path,
                    source_path=str(checkpoint_path),
                    cached=True,
                )
                return self._resolved_artifact

            checkpoint_path = self.default_checkpoint_path()
            if checkpoint_path.is_file():
                self._resolved_artifact = ResolvedAudioJEPAArtifact(
                    repo_id=self.repo_id(),
                    checkpoint_path=checkpoint_path,
                    source_path=self.remote_checkpoint_url(),
                    cached=True,
                )
                return self._resolved_artifact

            if not download:
                self._resolved_artifact = ResolvedAudioJEPAArtifact(
                    repo_id=self.repo_id(),
                    checkpoint_path=checkpoint_path,
                    source_path=self.remote_checkpoint_url(),
                    cached=False,
                )
                return self._resolved_artifact

            AUDIO_JEPA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id(),
                filename=self.checkpoint_filename(),
                revision="main",
                cache_dir=str(HF_HUB_CACHE_DIR),
                local_dir=str(AUDIO_JEPA_MODEL_DIR),
            )
            self._resolved_artifact = ResolvedAudioJEPAArtifact(
                repo_id=self.repo_id(),
                checkpoint_path=Path(downloaded_path).resolve(),
                source_path=self.remote_checkpoint_url(),
                cached=True,
            )
            return self._resolved_artifact

    def load(self) -> None:
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            artifact = self.ensure_model_artifact(download=True)
            model = AudioJEPAVisionTransformer()
            state = self.load_encoder_state_dict(artifact.checkpoint_path, model.state_dict().keys())
            missing, unexpected = model.load_state_dict(state, strict=False)
            self.validate_loaded_state_dict(state, missing, unexpected)
            model.to(self.device)
            model.eval()
            self._model = model

    @torch.inference_mode()
    def embed_waveform(self, waveform: np.ndarray) -> EmbeddingSummary:
        self.load()
        frame_embeddings = self.encode_waveform(waveform)
        pooled_embedding = frame_embeddings.mean(axis=0)

        return EmbeddingSummary(
            pooled_embedding=np.asarray(pooled_embedding, dtype=np.float32),
            temporal_steps=int(frame_embeddings.shape[0]),
            embedding_dim=int(frame_embeddings.shape[1]),
            pooled_norm=float(np.linalg.norm(pooled_embedding)),
            rms_energy=compute_rms_energy(waveform),
        )

    @torch.inference_mode()
    def encode_waveform(self, waveform: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Audio-JEPA model is not loaded.")

        wav = torch.as_tensor(waveform, dtype=torch.float32).flatten()
        if wav.numel() == 0:
            raise ValueError("Audio waveform is empty.")

        clip_samples = AUDIO_JEPA_SAMPLE_RATE * AUDIO_JEPA_CLIP_SECONDS
        n_chunks = max(1, math.ceil(wav.numel() / clip_samples))
        chunk_embeddings: list[torch.Tensor] = []

        for chunk_index in range(n_chunks):
            start = chunk_index * clip_samples
            end = min(start + clip_samples, wav.numel())
            chunk = wav[start:end]
            valid_samples = int(chunk.numel())

            if valid_samples < clip_samples:
                chunk = F.pad(chunk, (0, clip_samples - valid_samples))

            spec = self.waveform_to_spectrogram(chunk)
            patch_embeddings = self.spectrogram_to_patch_embeddings(spec)
            valid_t = max(
                1,
                min(
                    AUDIO_JEPA_TARGET_TIME_BINS // AUDIO_JEPA_PATCH_SIZE[0],
                    int(math.ceil((valid_samples / clip_samples) * (AUDIO_JEPA_TARGET_TIME_BINS // AUDIO_JEPA_PATCH_SIZE[0]))),
                ),
            )
            chunk_embeddings.append(patch_embeddings[:valid_t])

        output = torch.cat(chunk_embeddings, dim=0)
        return output.detach().cpu().numpy().astype(np.float32, copy=False)

    def waveform_to_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = normalize_audio(waveform)
        spec = self._mel_transform(waveform)
        return spec.to(self.device)

    def spectrogram_to_patch_embeddings(self, spec: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("Audio-JEPA model is not loaded.")

        x = spec.unsqueeze(0)
        feats = self._model(x)
        time_patches = AUDIO_JEPA_TARGET_TIME_BINS // AUDIO_JEPA_PATCH_SIZE[0]
        freq_patches = AUDIO_JEPA_MELS // AUDIO_JEPA_PATCH_SIZE[1]
        feats = feats.reshape(1, time_patches, freq_patches, -1)
        return feats.mean(dim=2).squeeze(0)

    def describe_artifact(self) -> dict[str, Any]:
        artifact = self.ensure_model_artifact(download=False)
        return {
            "repoId": artifact.repo_id,
            "sourcePath": artifact.source_path,
            "localPath": str(artifact.checkpoint_path),
            "format": "ckpt",
            "converted": False,
            "cached": artifact.cached,
        }

    @staticmethod
    def repo_id() -> str:
        return os.environ.get(AUDIO_JEPA_REPO_ID_ENV_VAR, AUDIO_JEPA_MODEL_REPO_ID)

    @staticmethod
    def checkpoint_filename() -> str:
        return os.environ.get(AUDIO_JEPA_CKPT_FILENAME_ENV_VAR, AUDIO_JEPA_CKPT_FILENAME)

    @classmethod
    def default_checkpoint_path(cls) -> Path:
        return AUDIO_JEPA_MODEL_DIR / cls.checkpoint_filename()

    @classmethod
    def remote_checkpoint_url(cls) -> str:
        return f"https://huggingface.co/{cls.repo_id()}/resolve/main/{cls.checkpoint_filename()}"

    @staticmethod
    def load_encoder_state_dict(
        checkpoint_path: Path,
        expected_keys: Any,
    ) -> dict[str, torch.Tensor]:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = extract_state_dict_from_checkpoint(payload)
        expected_key_set = set(expected_keys)

        for prefix in (
            "target_encoder.",
            "encoder.",
            "model.target_encoder.",
            "model.encoder.",
            "module.target_encoder.",
            "module.encoder.",
        ):
            stripped = {
                normalize_checkpoint_key(key[len(prefix) :]): value
                for key, value in state.items()
                if key.startswith(prefix) and isinstance(value, torch.Tensor)
            }
            filtered = {key: value for key, value in stripped.items() if key in expected_key_set}
            if filtered:
                return filtered

        direct = {
            normalize_checkpoint_key(key): value
            for key, value in state.items()
            if isinstance(value, torch.Tensor) and normalize_checkpoint_key(key) in expected_key_set
        }
        if direct:
            return direct

        raise ValueError("Audio-JEPA checkpoint did not contain target_encoder or encoder weights.")

    @staticmethod
    def validate_loaded_state_dict(
        state: Mapping[str, torch.Tensor],
        missing: list[str],
        unexpected: list[str],
    ) -> None:
        if unexpected:
            raise ValueError("Unexpected Audio-JEPA checkpoint keys: " + ", ".join(sorted(unexpected[:12])))

        if not any(key.startswith("blocks.") for key in state):
            raise ValueError("Audio-JEPA checkpoint did not load transformer block weights.")

        critical_missing = [
            key
            for key in missing
            if key.startswith(("patch_embed.proj.", "blocks.0.", "norm."))
        ]
        if critical_missing:
            raise ValueError(
                "Audio-JEPA checkpoint is missing critical inference weights: "
                + ", ".join(sorted(critical_missing[:12]))
            )


def extract_state_dict_from_checkpoint(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                return {str(k): v for k, v in candidate.items() if isinstance(v, torch.Tensor)}

        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return {str(k): v for k, v in payload.items() if isinstance(v, torch.Tensor)}

    raise ValueError("Unsupported Audio-JEPA checkpoint format.")


def normalize_checkpoint_key(key: str) -> str:
    normalized = key.replace("._orig_mod.", ".").replace("_orig_mod.", "")
    normalized = normalized.replace(".attn.Wqkv.", ".attn.qkv.")
    normalized = normalized.replace(".attn.out_proj.", ".attn.proj.")
    for prefix in ("module.", "model."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized
