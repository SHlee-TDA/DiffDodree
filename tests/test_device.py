from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from gugakgen.device import RuntimeDevice, resolve_device


def test_explicit_cpu_runtime() -> None:
    runtime = resolve_device("cpu")
    assert isinstance(runtime, RuntimeDevice)
    assert runtime.device == torch.device("cpu")
    assert runtime.backend == "cpu"
    assert runtime.amp_enabled is False
    assert runtime.dtype is torch.float32
    assert runtime.as_dict()["device"] == "cpu"


def test_auto_falls_back_to_cpu() -> None:
    with (
        patch("gugakgen.device.torch.cuda.is_available", return_value=False),
        patch("gugakgen.device._mps_is_available", return_value=False),
    ):
        runtime = resolve_device("auto")
    assert runtime.backend == "cpu"


def test_auto_prefers_cuda() -> None:
    with patch("gugakgen.device.torch.cuda.is_available", return_value=True):
        runtime = resolve_device("auto")
    assert runtime.device == torch.device("cuda")
    assert runtime.backend == "cuda"
    assert runtime.amp_enabled is True
    assert runtime.dtype is torch.float16


def test_auto_uses_mps_when_cuda_is_unavailable() -> None:
    with (
        patch("gugakgen.device.torch.cuda.is_available", return_value=False),
        patch("gugakgen.device._mps_is_available", return_value=True),
    ):
        runtime = resolve_device("auto")
    assert runtime.device == torch.device("mps")
    assert runtime.backend == "mps"
    assert runtime.amp_enabled is False
    assert runtime.dtype is torch.float32


@pytest.mark.skipif(
    not (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ),
    reason="Apple MPS is not available in this runtime",
)
def test_explicit_mps_on_available_hardware() -> None:
    runtime = resolve_device("mps")
    assert runtime.backend == "mps"
    assert runtime.device.type == "mps"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available in this runtime")
def test_explicit_cuda_on_available_hardware() -> None:
    runtime = resolve_device("cuda")
    assert runtime.backend == "cuda"
    assert runtime.device.type == "cuda"
    assert runtime.amp_enabled is True


def test_invalid_preference_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown device preference"):
        resolve_device("tpu")


def test_unavailable_explicit_backend_is_rejected() -> None:
    with (
        patch("gugakgen.device.torch.cuda.is_available", return_value=False),
        pytest.raises(RuntimeError, match="CUDA was requested"),
    ):
        resolve_device("cuda")
