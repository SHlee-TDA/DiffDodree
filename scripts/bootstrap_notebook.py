#!/usr/bin/env python3
"""Validate and optionally install dependencies for tutorial notebooks.

This module deliberately imports only the Python standard library.  In Colab it
never asks pip to install or upgrade PyTorch, preserving the CUDA-enabled build
provided by the runtime.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

REQUIREMENT_IMPORT_NAMES = {
    "ipywidgets": "ipywidgets",
    "matplotlib": "matplotlib",
    "music21": "music21",
    "networkx": "networkx",
    "numpy": "numpy",
    "pandas": "pandas",
    "pyyaml": "yaml",
    "torch": "torch",
    "tqdm": "tqdm",
}

_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


@dataclass(frozen=True)
class Requirement:
    """One installable requirement and the module used to verify it."""

    specifier: str
    distribution: str
    import_name: str


def _distribution_name(specifier: str) -> str:
    match = _NAME_PATTERN.match(specifier)
    if match is None:
        raise ValueError(f"지원하지 않는 requirement 형식입니다: {specifier!r}")
    return match.group(1)


def _normalise_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _iter_requirement_lines(path: Path, seen: set[Path] | None = None) -> Iterable[str]:
    """Yield concrete requirement lines, recursively expanding ``-r`` files."""

    seen = set() if seen is None else seen
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"requirements file 순환 참조가 발견되었습니다: {resolved}")
    seen.add(resolved)

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-r ", "--requirement ")):
            include = line.split(maxsplit=1)[1]
            yield from _iter_requirement_lines(path.parent / include, seen)
            continue
        if line.startswith("-"):
            raise ValueError(f"지원하지 않는 pip option입니다: {line}")
        yield line

    seen.remove(resolved)


def read_requirements(path: Path) -> list[Requirement]:
    requirements: list[Requirement] = []
    for specifier in _iter_requirement_lines(path):
        distribution = _distribution_name(specifier)
        import_name = REQUIREMENT_IMPORT_NAMES.get(
            _normalise_name(distribution), distribution.replace("-", "_")
        )
        requirements.append(Requirement(specifier, distribution, import_name))
    return requirements


def module_is_available(requirement: Requirement) -> bool:
    """Return whether a requirement's import target can be discovered."""

    return importlib.util.find_spec(requirement.import_name) is not None


def install_requirements(requirements: Sequence[Requirement]) -> None:
    if not requirements:
        return
    specs = [requirement.specifier for requirement in requirements]
    print("누락된 dependency를 설치합니다:", ", ".join(specs), flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *specs],
        check=True,
    )
    importlib.invalidate_caches()


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--profile", choices=("local", "colab"), default="local")
    parser.add_argument("--install-missing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    requirements_name = "colab.txt" if args.profile == "colab" else "base.txt"
    requirements_path = project_root / "requirements" / requirements_name
    src_root = project_root / "src"

    if not requirements_path.is_file():
        raise FileNotFoundError(f"requirements file을 찾지 못했습니다: {requirements_path}")
    if not src_root.is_dir():
        raise FileNotFoundError(f"src folder를 찾지 못했습니다: {src_root}")

    requirements = read_requirements(requirements_path)
    if args.profile == "colab" and any(
        _normalise_name(item.distribution) == "torch" for item in requirements
    ):
        raise RuntimeError(
            "requirements/colab.txt에는 torch를 포함할 수 없습니다. "
            "Colab의 CUDA PyTorch build를 유지해야 합니다."
        )

    missing = [item for item in requirements if not module_is_available(item)]
    if missing and not args.install_missing:
        names = ", ".join(item.distribution for item in missing)
        raise RuntimeError(
            f"누락된 dependency가 있습니다: {names}. --install-missing을 사용하세요."
        )
    install_requirements(missing)

    # PyTorch is an environment dependency. Colab must already provide it.
    torch_requirement = Requirement("torch", "torch", "torch")
    if not module_is_available(torch_requirement):
        if args.profile == "colab":
            raise RuntimeError(
                "현재 Colab runtime에서 PyTorch를 찾지 못했습니다. "
                "GPU runtime을 다시 시작한 뒤 Cell 1을 재실행하세요."
            )
        if args.install_missing:
            install_requirements([Requirement("torch>=2.1", "torch", "torch")])
        else:
            raise RuntimeError("PyTorch가 설치되어 있지 않습니다.")

    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    importlib.import_module("gugakgen")

    checked = [*requirements]
    if not any(_normalise_name(item.distribution) == "torch" for item in checked):
        checked.append(torch_requirement)

    print("dependency 확인 완료")
    for item in checked:
        print(f"- {item.distribution}: {package_version(item.distribution)}")
    print(f"- gugakgen source: {src_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
