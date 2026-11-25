"""Run generator-v1 experiments on the local CPU (e.g. overnight), one config after another.

    python scripts/run_local.py configs/chart_diffusion_v1_lambda_sweep_haegeum.yaml \
        configs/chart_diffusion_v1_haegeum.yaml --data-from "/path/to/original/data"
    python scripts/run_local.py CONFIG --mode smoke                 # code path only (20 step, 1 Seed)
    python scripts/run_local.py CONFIG --run-tag 20261126_0140_cpu_lambda_sweep_haegeum   # resume

This is the same code path as experiments/chart_diffusion_colab.ipynb (``run_experiment``), minus the
CUDA requirement. Each finished unit is one JSON, so an interrupted run resumes with the printed RUN_TAG,
and a RUN_TAG started with other code or another config stops instead of being overwritten.

- Scores that are not in the repository (haegeum) are copied from ``--data-from`` (the original
  project's ``data/``); the measures file is then built (or checked) with scripts/build_measures.py.
- ``--results-root``: keep the run folders somewhere durable (e.g. the original project folder) by
  making this checkout's run folder a symlink to ``<results-root>/<run_dir>``.
- After a run: ``config_used.yaml`` and three MusicXML pieces per arm (first Seed) in ``musicxml/``.

Start it detached so it survives closing the terminal and the Mac does not idle-sleep::

    nohup caffeinate -i nice -n 10 python scripts/run_local.py ... > run_local.log 2>&1 &
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from gugakgen.config import load_yaml  # noqa: E402
from gugakgen.device import resolve_device  # noqa: E402
from gugakgen.workflows import chart_diffusion_experiment as v1  # noqa: E402

KST = timezone(timedelta(hours=9))


def log(message: str) -> None:
    print(f"[{datetime.now(KST):%m-%d %H:%M:%S}] {message}", flush=True)


def prepare_scores(piece_config: Path, data_from: Path | None) -> None:
    """Copy missing source scores from ``data_from`` and build (or check) the measures file."""
    for pattern in load_yaml(piece_config)["data"]["source_globs"]:
        folder, name = pattern.rsplit("/", 1)
        if list((ROOT / folder).glob(name)):
            continue
        found = list((data_from / Path(folder).name).glob(name)) if data_from else []
        if len(found) != 1:
            raise FileNotFoundError(f"악보가 없습니다: {pattern} (--data-from으로 원래 폴더의 data/를 주세요)")
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
        shutil.copy2(found[0], ROOT / folder / found[0].name)
        log(f"복사: {found[0].name}")
    subprocess.run([sys.executable, str(ROOT / "scripts/build_measures.py"), str(piece_config)], check=True)


def link_results(run_base: Path, results_root: Path) -> None:
    """Make ``run_base`` (inside this checkout) a symlink to the same relative folder under ``results_root``."""
    target = (results_root / run_base.relative_to(ROOT)).resolve()
    target.mkdir(parents=True, exist_ok=True)
    if run_base.is_symlink():
        if run_base.resolve() != target:
            raise RuntimeError(f"{run_base}는 이미 다른 곳({run_base.resolve()})을 가리킵니다.")
        return
    if run_base.exists():
        if any(run_base.iterdir()):
            raise RuntimeError(f"{run_base}에 이미 결과가 있어 링크로 바꾸지 않습니다.")
        run_base.rmdir()
    run_base.parent.mkdir(parents=True, exist_ok=True)
    run_base.symlink_to(target, target_is_directory=True)
    log(f"결과 폴더: {run_base.relative_to(ROOT)} → {target}")


def default_tag(config: dict) -> str:
    tag = datetime.now(KST).strftime("%Y%m%d_%H%M") + "_cpu"
    if config["training"].get("fixed_steps"):
        tag += "_lambda_sweep"
    if "haegeum" in config["piece_config"]:
        tag += "_haegeum"
    return tag


def export_listening(report: dict, config_path: Path, run_dir: Path) -> None:
    setup = v1.load_setup(config_path, ROOT)
    part = load_yaml(ROOT / setup["config"]["piece_config"])["piece"]["part_name_in_files"]
    first_seed = report["full"][0]["seed"]
    for unit in report["full"]:
        if unit["seed"] != first_seed:
            continue
        if list((run_dir / "musicxml").glob(f"{unit['arm']}_s{first_seed}_*.musicxml")):
            continue  # 이어서 실행한 경우: 이미 저장한 악보는 그대로 둡니다
        for path in v1.export_musicxml(unit, setup, run_dir / "musicxml"):
            if part != "Piri":  # export_musicxml은 파트 이름을 Piri로 적습니다
                path.write_text(path.read_text(encoding="utf-8").replace(
                    "<part-name>Piri</part-name>", f"<part-name>{part}</part-name>"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("configs", nargs="+", type=Path, help="실행할 설정 파일(적은 순서대로 하나씩)")
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument("--run-tag", nargs="+", help="이어서 실행할 RUN_TAG(설정 수만큼). 없으면 시작 시각으로 새로 만듭니다")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU 스레드 수 (기본 4)")
    parser.add_argument("--data-from", type=Path, help="저장소에 없는 악보(해금)를 복사해 올 원래 폴더의 data/")
    parser.add_argument("--results-root", type=Path, help="결과를 실제로 쓸 폴더(예: 원래 project 폴더). 이 checkout의 결과 폴더가 그곳을 가리키게 합니다")
    args = parser.parse_args()
    if args.run_tag and len(args.run_tag) != len(args.configs):
        parser.error("--run-tag는 설정 파일 수만큼 주어야 합니다.")

    torch.set_num_threads(args.threads)
    runtime = resolve_device("cpu")
    log(f"torch {torch.__version__}, CPU 스레드 {torch.get_num_threads()}개, 설정 {len(args.configs)}개")
    for i, config_path in enumerate(args.configs):
        config_path = (ROOT / config_path).resolve() if not config_path.is_absolute() else config_path
        config = load_yaml(config_path)
        prepare_scores(ROOT / config["piece_config"], args.data_from)
        run_base = ROOT / config["artifacts"]["run_dir"]
        if args.results_root:
            link_results(run_base, args.results_root)
        tag = args.run_tag[i] if args.run_tag else default_tag(config)
        run_dir = run_base / (tag + ("_smoke" if args.mode == "smoke" else ""))
        log(f"시작: {config_path.relative_to(ROOT)} · RUN_TAG = {tag}  ← 다시 실행할 때 --run-tag로 이 값을 주세요")
        started = time.time()
        report = v1.run_experiment(config_path, ROOT, tag, runtime, run_mode=args.mode, log=log)
        used = run_dir / "config_used.yaml"
        if not used.exists():
            shutil.copy2(config_path, used)
        export_listening(report, config_path, run_dir)
        log(f"끝: {run_dir} ({(time.time() - started) / 3600:.2f}시간)")


if __name__ == "__main__":
    main()
