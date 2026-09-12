"""Portable, CPU-first Checkpoint save/load utilities."""

from __future__ import annotations

import hashlib
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import torch
from torch import Tensor, nn

CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    payload: Mapping[str, Any]
    sha256: str

    @property
    def state_dict(self) -> Mapping[str, Tensor]:
        state = self.payload.get("model_state_dict", self.payload.get("state_dict"))
        if not isinstance(state, Mapping):
            raise KeyError("Checkpoint contains neither model_state_dict nor state_dict")
        return state

    @property
    def step(self) -> int:
        return int(self.payload.get("step", self.payload.get("steps", 0)))


def _to_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value


def environment_metadata() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    step: int,
    model_config: Mapping[str, Any],
    vocabulary: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Atomically save a backend-independent Checkpoint and return its SHA-256."""

    if step < 0:
        raise ValueError("step must be non-negative")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state_dict = _to_cpu(model.state_dict())
    payload: MutableMapping[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "step": step,
        "model_state_dict": state_dict,
        # Retain the historical key for controlled compatibility with yc_model.py.
        "state_dict": state_dict,
        "model_config": dict(model_config),
        "vocabulary": dict(vocabulary),
        "environment": environment_metadata(),
        "metadata": dict(metadata or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = _to_cpu(optimizer.state_dict())
    if scheduler is not None:
        payload["scheduler_state_dict"] = _to_cpu(scheduler.state_dict())

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    return sha256_file(destination)


def load_checkpoint(path: str | Path) -> LoadedCheckpoint:
    """Load a trusted Checkpoint onto CPU regardless of its training backend."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        # PyTorch <2.0 does not accept weights_only.
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint payload must be a mapping")
    return LoadedCheckpoint(source, payload, sha256_file(source))


def load_model_state(
    model: nn.Module,
    checkpoint: LoadedCheckpoint | str | Path,
    *,
    strict: bool = True,
) -> LoadedCheckpoint:
    loaded = checkpoint if isinstance(checkpoint, LoadedCheckpoint) else load_checkpoint(checkpoint)
    model.load_state_dict(loaded.state_dict, strict=strict)
    return loaded


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move a restored optimizer state only after CPU-first deserialization."""

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)
