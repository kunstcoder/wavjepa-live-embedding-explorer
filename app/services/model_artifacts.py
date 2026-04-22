from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import importlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from huggingface_hub import snapshot_download
from transformers import PreTrainedModel, PretrainedConfig
import torch


BASE_MODEL_REPO_ID = "labhamlet/wavjepa-base"
BASE_MODEL_REVISION = "main"

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
HF_CACHE_DIR = WORKSPACE_ROOT / ".hf_cache"
HF_HUB_CACHE_DIR = HF_CACHE_DIR / "hub"
CONVERTED_MODELS_DIR = HF_CACHE_DIR / "converted_models"
MODELS_DIR = WORKSPACE_ROOT / "models"
BASE_MODEL_DIR = MODELS_DIR / "wavjepa-base"
MODEL_SOURCE_ENV_VAR = "WAVJEPA_MODEL_SOURCE"
MODEL_TEMPLATE_ENV_VAR = "WAVJEPA_TEMPLATE_DIR"
PYTHON_MODULE_FILENAMES = (
    "__init__.py",
    "audio_extractor.py",
    "configuration_wavjepa.py",
    "feature_extraction_wavjepa.py",
    "model.py",
    "modeling_wavjepa.py",
    "pos_embed.py",
    "types.py",
    "utils.py",
)

HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CONVERTED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HUB_CACHE_DIR))


@dataclass(frozen=True)
class ResolvedModelArtifact:
    source_path: Path
    model_dir: Path
    format: str
    converted: bool


def ensure_base_model_snapshot() -> Path:
    if (BASE_MODEL_DIR / "model.safetensors").exists():
        return BASE_MODEL_DIR

    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=BASE_MODEL_REPO_ID,
        revision=BASE_MODEL_REVISION,
        local_dir=BASE_MODEL_DIR,
        cache_dir=str(HF_HUB_CACHE_DIR),
        allow_patterns=[
            "*.json",
            "*.md",
            "*.py",
            "*.safetensors",
        ],
    )

    return BASE_MODEL_DIR


def resolve_model_source_path(source: str | Path | None = None) -> Path:
    raw_source = source if source is not None else os.environ.get(MODEL_SOURCE_ENV_VAR, str(BASE_MODEL_DIR))
    return Path(raw_source).expanduser().resolve()


def is_checkpoint_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".ckpt", ".pt", ".pth", ".bin"}


def is_hf_model_directory(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").exists()
        and (path / "preprocessor_config.json").exists()
        and ((path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists())
    )


def resolved_model_artifact(source: str | Path | None = None) -> ResolvedModelArtifact:
    source_path = resolve_model_source_path(source)

    if is_hf_model_directory(source_path):
        return ResolvedModelArtifact(
            source_path=source_path,
            model_dir=source_path,
            format="hf",
            converted=False,
        )

    if is_checkpoint_file(source_path):
        model_dir = convert_checkpoint_to_hf(source_path)
        return ResolvedModelArtifact(
            source_path=source_path,
            model_dir=model_dir,
            format="ckpt",
            converted=True,
        )

    if source_path == BASE_MODEL_DIR or str(source_path) == str(BASE_MODEL_DIR):
        base_model_dir = ensure_base_model_snapshot()
        return ResolvedModelArtifact(
            source_path=base_model_dir,
            model_dir=base_model_dir,
            format="hf",
            converted=False,
        )

    raise FileNotFoundError(
        f"Unsupported model source: {source_path}. Expected an HF model directory or a checkpoint file."
    )


def load_local_model_classes(model_dir: Path) -> tuple[type[Any], type[Any], type[Any]]:
    package_name = build_local_package_name(model_dir)

    if package_name not in sys.modules:
        spec = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        package = importlib.util.module_from_spec(spec)
        package.__path__ = [str(model_dir)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package

    feature_module = importlib.import_module(f"{package_name}.feature_extraction_wavjepa")
    config_module = importlib.import_module(f"{package_name}.configuration_wavjepa")
    model_module = importlib.import_module(f"{package_name}.modeling_wavjepa")

    return (
        feature_module.WavJEPAFeatureExtractor,
        config_module.WavJEPAConfig,
        model_module.WavJEPAModel,
    )


def convert_checkpoint_to_hf(
    checkpoint_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    template_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    template_dir = Path(template_dir).expanduser().resolve() if template_dir is not None else default_template_dir()
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else CONVERTED_MODELS_DIR / build_checkpoint_fingerprint(checkpoint_path)
    )

    if is_hf_model_directory(output_dir) and not force:
        return output_dir

    if not is_hf_model_directory(template_dir):
        raise FileNotFoundError(
            f"Template model directory is invalid: {template_dir}. "
            "Expected config.json, preprocessor_config.json, and model weights."
        )

    if output_dir == template_dir:
        raise ValueError("Conversion output_dir must be different from template_dir.")

    if force and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_python_modules(template_dir, output_dir)

    checkpoint_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    raw_state_dict = extract_state_dict_from_checkpoint(checkpoint_payload)

    feature_extractor_class, config, model = instantiate_from_template(template_dir, raw_state_dict)
    normalized_state_dict = normalize_checkpoint_state_dict(raw_state_dict, model.state_dict().keys())
    missing_keys, unexpected_keys = model.load_state_dict(normalized_state_dict, strict=False)
    validate_loaded_state_dict(normalized_state_dict, missing_keys, unexpected_keys)

    feature_extractor = feature_extractor_class.from_pretrained(str(template_dir))
    feature_extractor.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)

    metadata = {
        "checkpointPath": str(checkpoint_path),
        "templateDir": str(template_dir),
        "loadedKeyCount": len(normalized_state_dict),
        "missingKeys": missing_keys,
        "unexpectedKeys": unexpected_keys,
    }
    (output_dir / "conversion_metadata.json").write_text(
        json_dump(metadata),
        encoding="utf-8",
    )

    return output_dir


def instantiate_from_template(
    template_dir: Path,
    raw_state_dict: Mapping[str, Any],
) -> tuple[type[Any], PretrainedConfig, PreTrainedModel]:
    feature_extractor_class, config_class, model_class = load_local_model_classes(template_dir)
    config_overrides = infer_config_overrides(raw_state_dict)
    config = config_class.from_pretrained(str(template_dir), **config_overrides)
    model = model_class(config)
    return feature_extractor_class, config, model


def infer_config_overrides(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    canonical_keys = canonicalize_keys(state_dict)

    encoder_layer_count = infer_layer_count(canonical_keys, "encoder")
    decoder_layer_count = infer_layer_count(canonical_keys, "decoder")
    encoder_d_model = infer_tensor_dim(
        canonical_keys,
        preferred_keys=("pos_encoding_encoder", "encoder.layers.0.self_attn.out_proj.weight"),
    )
    decoder_d_model = infer_tensor_dim(
        canonical_keys,
        preferred_keys=("pos_encoding_decoder", "mask_token", "decoder.layers.0.self_attn.out_proj.weight"),
    )

    overrides: dict[str, Any] = {}

    if encoder_layer_count is not None:
        overrides["encoder_num_layers"] = encoder_layer_count
    if decoder_layer_count is not None:
        overrides["decoder_num_layers"] = decoder_layer_count
    if encoder_d_model is not None:
        overrides["encoder_d_model"] = encoder_d_model
        overrides["encoder_nhead"] = infer_attention_heads(encoder_d_model)
    if decoder_d_model is not None:
        overrides["decoder_d_model"] = decoder_d_model
        overrides["decoder_nhead"] = infer_attention_heads(decoder_d_model)

    if encoder_d_model and encoder_d_model >= 1024:
        overrides["model_size"] = "large"

    return overrides


def infer_attention_heads(embedding_dim: int) -> int:
    for candidate in (16, 12, 8, 4, 2, 1):
        if embedding_dim % candidate == 0:
            return candidate

    return 1


def infer_layer_count(state_dict: Mapping[str, Any], component: str) -> int | None:
    layer_indices = []
    markers = (f"{component}.layers.", f"model.{component}.layers.")

    for key in state_dict:
        marker = next((candidate for candidate in markers if key.startswith(candidate)), None)

        if marker is None:
            continue

        suffix = key[len(marker) :]
        index_token = suffix.split(".", 1)[0]

        if index_token.isdigit():
            layer_indices.append(int(index_token))

    return max(layer_indices) + 1 if layer_indices else None


def infer_tensor_dim(state_dict: Mapping[str, Any], preferred_keys: Iterable[str]) -> int | None:
    for key in preferred_keys:
        for candidate in (key, f"model.{key}", key.removeprefix("model.")):
            tensor = state_dict.get(candidate)

            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue

            if tensor.ndim == 1:
                return int(tensor.shape[0])
            if tensor.ndim >= 2:
                return int(tensor.shape[-1])

    return None


def extract_state_dict_from_checkpoint(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for candidate_key in ("state_dict", "model_state_dict"):
            candidate = payload.get(candidate_key)
            if isinstance(candidate, Mapping):
                return coerce_state_dict(candidate)

        if is_tensor_mapping(payload):
            return coerce_state_dict(payload)

    raise ValueError("Unsupported checkpoint format: could not locate a state_dict mapping.")


def normalize_checkpoint_state_dict(
    raw_state_dict: Mapping[str, torch.Tensor],
    expected_keys: Iterable[str],
) -> dict[str, torch.Tensor]:
    expected_key_set = set(expected_keys)
    normalized: dict[str, torch.Tensor] = {}

    for raw_key, value in raw_state_dict.items():
        for candidate in candidate_keys(raw_key):
            if candidate in expected_key_set:
                normalized[candidate] = value
                break

    if not any(key.startswith("model.encoder.") for key in normalized):
        teacher_keys = {
            key.replace("model.teacher_encoder.", "model.encoder.", 1): value
            for key, value in normalized.items()
            if key.startswith("model.teacher_encoder.")
        }
        normalized.update({key: value for key, value in teacher_keys.items() if key in expected_key_set})

    if not any(key.startswith("model.teacher_encoder.") for key in normalized):
        teacher_keys = {
            key.replace("model.encoder.", "model.teacher_encoder.", 1): value
            for key, value in normalized.items()
            if key.startswith("model.encoder.")
        }
        normalized.update({key: value for key, value in teacher_keys.items() if key in expected_key_set})

    return normalized


def validate_loaded_state_dict(
    normalized_state_dict: Mapping[str, torch.Tensor],
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> None:
    if not normalized_state_dict:
        raise ValueError("No checkpoint tensors matched the expected WavJEPA model keys.")

    encoder_keys = [key for key in normalized_state_dict if key.startswith("model.encoder.")]
    extractor_keys = [key for key in normalized_state_dict if key.startswith("model.extract_audio.")]

    if not encoder_keys or not extractor_keys:
        raise ValueError(
            "Checkpoint conversion did not recover the expected encoder/extractor weights. "
            "Verify that the checkpoint comes from WavJEPA training."
        )

    critical_missing = [
        key
        for key in missing_keys
        if key.startswith(("model.encoder.", "model.extract_audio.", "model.feature_norms.", "model.pos_encoding_encoder"))
    ]

    if critical_missing:
        raise ValueError(
            "Converted checkpoint is missing critical inference weights: "
            + ", ".join(sorted(critical_missing[:12]))
        )

    if unexpected_keys:
        # The conversion intentionally ignores optimizer state and other training-only keys.
        return


def candidate_keys(raw_key: str) -> list[str]:
    canonical = canonicalize_key(raw_key)
    ordered_candidates: list[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate not in ordered_candidates:
            ordered_candidates.append(candidate)

    add(canonical)
    add(f"model.{canonical}")

    if canonical.startswith("model."):
        add(canonical[len("model.") :])

    if canonical.startswith("teacher_encoder."):
        add(f"model.{canonical}")
        add(f"model.encoder.{canonical.split('teacher_encoder.', 1)[1]}")

    if canonical.startswith("encoder."):
        add(f"model.teacher_encoder.{canonical.split('encoder.', 1)[1]}")

    return ordered_candidates


def canonicalize_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {canonicalize_key(key): value for key, value in state_dict.items()}


def canonicalize_key(raw_key: str) -> str:
    key = raw_key.replace("._orig_mod.", ".").replace("_orig_mod.", "")

    changed = True
    while changed:
        changed = False
        for prefix in ("state_dict.", "module.", "network.", "net."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True

    if key.startswith("model.model."):
        key = key[len("model.") :]

    return key


def build_local_package_name(model_dir: Path) -> str:
    digest = hashlib.sha1(str(model_dir).encode("utf-8")).hexdigest()[:12]
    return f"local_wavjepa_{digest}"


def build_checkpoint_fingerprint(checkpoint_path: Path) -> str:
    stat = checkpoint_path.stat()
    signature = f"{checkpoint_path.resolve()}:{stat.st_size}:{int(stat.st_mtime)}"
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return f"{checkpoint_path.stem}-{digest}"


def default_template_dir() -> Path:
    configured = os.environ.get(MODEL_TEMPLATE_ENV_VAR)
    return Path(configured).expanduser().resolve() if configured else ensure_base_model_snapshot()


def copy_python_modules(source_dir: Path, target_dir: Path) -> None:
    for filename in PYTHON_MODULE_FILENAMES:
        shutil.copy2(source_dir / filename, target_dir / filename)


def coerce_state_dict(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {str(key): value for key, value in payload.items() if isinstance(value, torch.Tensor)}


def is_tensor_mapping(payload: Mapping[str, Any]) -> bool:
    return bool(payload) and all(isinstance(value, torch.Tensor) for value in payload.values())


def json_dump(data: Mapping[str, Any]) -> str:
    import json

    return json.dumps(data, indent=2, sort_keys=True)
