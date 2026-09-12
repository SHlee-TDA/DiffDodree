"""Device selection shared by local notebooks, Colab, and Linux servers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

DevicePreference = Literal["auto", "cpu", "mps", "cuda"]


@dataclass(frozen=True)
class RuntimeDevice:
    """Resolved execution backend and its safe mixed-precision policy."""

    device: torch.device
    backend: Literal["cpu", "mps", "cuda"]
    amp_enabled: bool
    dtype: torch.dtype

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "device": str(self.device),
            "backend": self.backend,
            "amp_enabled": self.amp_enabled,
            "dtype": str(self.dtype).removeprefix("torch."),
        }


def _mps_is_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_built() and backend.is_available())


def resolve_device(preference: DevicePreference | str = "auto") -> RuntimeDevice:
    """Resolve CUDA, Apple MPS, or CPU without invoking backend-specific code.

    ``auto`` prefers CUDA, then MPS, then CPU. Automatic mixed precision is
    enabled only for CUDA because the project must also run reliably on Apple
    Silicon, where operator and dtype support varies by PyTorch release.
    """

    requested = preference.lower().strip()
    allowed = {"auto", "cpu", "mps", "cuda"}
    if requested not in allowed:
        raise ValueError(f"Unknown device preference {preference!r}; expected one of {sorted(allowed)}")

    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif _mps_is_available():
            requested = "mps"
        else:
            requested = "cpu"
    elif requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    elif requested == "mps" and not _mps_is_available():
        raise RuntimeError("MPS was requested, but the installed PyTorch build cannot use MPS")

    backend = requested
    if backend == "cuda":
        return RuntimeDevice(torch.device("cuda"), "cuda", True, torch.float16)
    if backend == "mps":
        return RuntimeDevice(torch.device("mps"), "mps", False, torch.float32)
    return RuntimeDevice(torch.device("cpu"), "cpu", False, torch.float32)
