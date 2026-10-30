"""Запуск обучения на Kaggle: wheel в приватный dataset, kernel-script с зеркальными флагами."""

from __future__ import annotations

import ast
import enum
import glob
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro
import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from tyro.conf import OmitArgPrefixes

from compare_attention.cli.train import _derived_flags, load_experiment
from compare_attention.kaggle_kernel import (  # noqa: F401
    _KERNEL_TEMPLATE,
    SECRETS,
    secrets_block,
)
from compare_attention.models.config import GPT2Config
from compare_attention.train.config import DatasetSpec, Optim, Schedule

Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]
Sleep = Callable[[float], None]


def _cli_run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """CLI-запуск с захватом stdout/stderr для разбора статусов."""
    subprocess_out = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return subprocess_out


COMPLETE = "complete"
ERROR = "error"
RUNNING = "running"
_CANCELLED = "cancelled"

_WHEEL_MOUNT = "/kaggle/input"
KAGGLE_WORK = "/kaggle/working"


class KaggleArgs(BaseModel):
    """Зеркало флагов train плюс организация Kaggle."""

    model_config = ConfigDict(frozen=True)

    schedule: OmitArgPrefixes[Schedule] = Field(
        default_factory=lambda: Schedule(train_flops=10_000_000_000)
    )
    optim: OmitArgPrefixes[Optim] = Field(default_factory=Optim)
    model: OmitArgPrefixes[GPT2Config] = Field(default_factory=GPT2Config)
    kv: OmitArgPrefixes[DatasetSpec] = Field(default_factory=DatasetSpec)
    config: Path | None = Field(
        default=None,
        description="YAML-файл эксперимента: уезжает в датасет, ядро запускает train --config.",
    )
    owner: str | None = Field(
        default=None, description="Kaggle-логин; по умолчанию KAGGLE_USERNAME"
    )
    slug: str = Field(default="compare-attention-train", description="Имя kernel-запуска")
    wheel_dataset: str = Field(
        default="compare-attention-wheel", description="Приватный dataset с wheel проекта"
    )
    accelerator: str = Field(
        default="NvidiaTeslaT4", description="GPU Kaggle: NvidiaTeslaT4/NvidiaL4/NvidiaTeslaA100"
    )
    tracker: Literal["clearml", "disabled"] = Field(
        default="clearml",
        description="Куда отражать метрики и логи запуска; смоук-прогонам удобен disabled",
    )
    timeout: int = Field(default=9 * 3600, description="Лимит запуска в секундах")
    poll_seconds: int = Field(default=60, description="Пауза между проверками статуса")


@dataclass(frozen=True)
class Kernel:
    """Сгенерированный kernel-файл с метаданными."""

    code: str
    metadata: dict[str, object]


def render_kernel(args: KaggleArgs, owner: str) -> Kernel:
    """Собрать kernel.py и kernel-metadata.json для push."""
    wheel_glob = f"{_WHEEL_MOUNT}/**/*.whl"
    result_dir = f"{KAGGLE_WORK}/{args.slug}"
    dataset_dir = f"{KAGGLE_WORK}/kv_retrieval"
    experiment = load_experiment(args.config) if args.config is not None else None
    spec = experiment.kv if experiment is not None else args.kv
    vocab = (experiment.model if experiment is not None else args.model).vocab_size
    prepare = [
        f"--vocab-size={vocab}",
        f"--input-seq-len={spec.input_seq_len}",
        f"--num-kv-pairs={spec.num_kv_pairs}",
        f"--power-a={spec.power_a}",
        f"--n-train={spec.n_train}",
        f"--n-val={spec.n_val}",
        f"--n-test={spec.n_test}",
        f"--seed={spec.dataset_seed}",
        f"--output={dataset_dir}",
    ]
    if experiment is not None:
        train = [
            f"--config={_WHEEL_MOUNT}/datasets/{owner}/{args.wheel_dataset}/{args.config.name}",
            f"--result-dir={result_dir}",
        ]
    else:
        train = [f"--data={dataset_dir}", f"--result-dir={result_dir}"]
        train += _derived_flags(args.schedule)
        train += _derived_flags(args.optim)
        train += _derived_flags(args.model)
        train += [f"--tracker={args.tracker}"]
    code = _KERNEL_TEMPLATE.format(
        wheel_glob=wheel_glob, prepare=_dump(prepare), train=_dump(train), secrets=secrets_block()
    )
    metadata: dict[str, object] = {
        "id": f"{owner}/{args.slug}",
        "title": args.slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "dataset_sources": [f"{owner}/{args.wheel_dataset}"],
        "enable_internet": True,
    }
    return Kernel(code=code, metadata=metadata)


def _dump(arguments: list[str]) -> str:
    """Список аргументов внутрь kernel-файла без проблем с кавычками."""
    return json.dumps(arguments)


def _flags(spec: BaseModel) -> list[str]:
    """Pydantic-спецификация в плоский список --flag=value (bool --no-flag/--flag)."""
    flags: list[str] = []
    for name, _field in spec.model_fields.items():
        value = getattr(spec, name)
        rendered = name.replace("_", "-")
        if isinstance(value, enum.Enum):
            flags.append(f"--{rendered}={value.name}")
            continue
        if isinstance(value, bool):
            flags.append(f"--{rendered}" if value else f"--no-{rendered}")
            continue
        flags += [f"--{rendered}={value}"]
    return flags


def stage_wheel_dataset(staging: Path, wheel: Path, owner: str, slug: str) -> None:
    """Каталог-заготовка для Kaggle dataset: wheel и метаданные."""
    staging.mkdir(parents=True, exist_ok=True)
    target = staging / wheel.name
    target.write_bytes(wheel.read_bytes())
    metadata = {
        "id": f"{owner}/{slug}",
        "title": "compare attention wheel",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def upload_wheel_dataset(staging: Path, owner: str, slug: str, run: Runner | None = None) -> None:
    """Обновить существующий dataset или создать новый."""
    del owner, slug  # иденты уже вписаны в dataset-metadata.json
    kaggle = run or _cli_run
    attempts = [
        ["kaggle", "datasets", "version", "-p", str(staging), "-m", "refresh wheel"],
        ["kaggle", "datasets", "create", "-p", str(staging)],
    ]
    for attempt in attempts:
        completed = kaggle(attempt)
        if completed.returncode == 0:
            return
    raise RuntimeError("не удалось создать или обновить dataset с wheel")


def verify_wheel(wheel: Path) -> None:
    """Каждый .py внутри колеса обязан разбираться питоном 3.13 (Kaggle)."""
    with zipfile.ZipFile(wheel) as zf:
        for name in zf.namelist():
            if not name.endswith(".py"):
                continue
            source = zf.read(name).decode("utf-8")
            try:
                ast.parse(source, feature_version=(3, 13))
            except SyntaxError as exc:
                raise RuntimeError(
                    f"{name} не является 3.13-совместимым: {exc.msg} (строка {exc.lineno})"
                ) from exc


_DEBUG_GLOBS = ("dataset-verify*", "ds-*", "output-log*")


def cleanup_debug_dirs(staging: Path) -> None:
    """Убрать временные каталоги диагностики; результат скачивания (output) не трогаем."""
    for pattern in _DEBUG_GLOBS:
        for ghost in staging.glob(pattern):
            if ghost.is_dir():
                shutil.rmtree(ghost, ignore_errors=True)


_REQUIREMENT_EXCLUSIONS = (
    ("--no-emit-package", "torch"),
    ("--no-emit-package", "triton"),
    ("--no-emit-package", "triton-windows"),
)


def stage_requirements(staging: Path, run: Runner | None = None) -> None:
    """Pinned-версии из uv.lock -> requirements.txt; Kaggle-стек torch не трогаем."""
    if run is None:
        run = _cli_run  # тест/основной поток вызывают с cwd корня проекта
    command = [
        "uv",
        "export",
        "--format",
        "requirements-txt",
        "--no-dev",
        "--no-hashes",
        "--no-annotate",
        "--no-emit-project",
        *(flag for pair in _REQUIREMENT_EXCLUSIONS for flag in pair),
        "-o",
        str(staging / "requirements.txt"),
    ]
    root = Path.cwd().resolve()
    while root != root.parent and not (root / "pyproject.toml").is_file():
        root = root.parent
    if run is _cli_run:
        subprocess.run(command, check=True, cwd=str(root), capture_output=True, text=True)
    else:
        run(command)


def stage_experiment(staging: Path, args: KaggleArgs) -> None:
    """Копия YAML эксперимента в датасет; data переписывается на рабочую папку ядра."""
    experiment = load_experiment(args.config)
    payload = experiment.model_dump(mode="json")
    payload["data"] = f"{KAGGLE_WORK}/kv_retrieval"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / args.config.name).write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def cleanup_wheel_artifacts(staging: Path) -> None:
    """Убрать локальные копии wheel: собранные и уже уехавшие в датасет."""
    for folder, suffixes in (("wheel", {".whl"}), ("dataset", {".whl", ".env"})):
        root = staging / folder
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if path.is_file() and path.suffix in suffixes:
                path.unlink()


def build_wheel(staging: Path, run: Runner | None = None) -> Path:
    """Собрать wheel проекта в каталог staging/wheel."""
    kaggle = run or _cli_run
    dist = staging / "wheel"
    if dist.is_dir():
        shutil.rmtree(dist)
    kaggle(["uv", "build", "--wheel", "-o", str(dist)])
    wheels = glob.glob(str(dist / "*.whl"))
    if not wheels:
        raise RuntimeError("uv build не положил wheel в каталог")
    wheel = Path(wheels[0])
    verify_wheel(wheel)
    return wheel


def push_kernel(
    staging: Path,
    accelerator: str,
    timeout: int,
    run: Runner | None = None,
    immediately: bool = True,
) -> None:
    """Загрузить kernel и запустить его на выбранном ускорителе."""
    kaggle = run or _cli_run
    command = [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(staging),
        "--accelerator",
        accelerator,
        "--timeout",
        str(timeout),
    ]
    if not immediately:
        command.append("--no-run")
    completed = kaggle(command)
    if completed.returncode != 0:
        raise RuntimeError("kaggle kernels push не прошёл")


STATUS_TOKENS: dict[str, str] = {
    COMPLETE: COMPLETE,
    ERROR: ERROR,
    RUNNING: RUNNING,
    "queued": RUNNING,
    "cancelacknowledgement": _CANCELLED,
    "kernelworkerstatus.running": RUNNING,
    "kernelworkerstatus.complete": COMPLETE,
    "kernelworkerstatus.error": ERROR,
    "kernelworkerstatus.queued": RUNNING,
}

TERMINAL = {COMPLETE, ERROR, _CANCELLED}


def _status_of(text: str) -> str:
    """Разобрать вывод `kaggle kernels status`."""
    lowered = text.lower()
    for token, canon in STATUS_TOKENS.items():
        if token in lowered:
            return canon
    raise RuntimeError(f"непонятный статус запуска: {text.strip()!r}")


def poll_status(
    owner_slug: str,
    poll_seconds: int = 60,
    poll: Runner | None = None,
    sleep: Sleep | None = None,
    timeout: float | None = None,
) -> str:
    """Опрашивать статус до терминального состояния ('complete'/'error'/'cancelled')."""
    kaggle = poll or _cli_run
    nap = sleep or time.sleep
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        completed = kaggle(["kaggle", "kernels", "status", owner_slug])
        status = _status_of(completed.stdout or completed.stderr)
        if status in TERMINAL:
            return status
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"запуск {owner_slug} не завершился за отведённое время")
        nap(poll_seconds)


_KERNEL_KEYS = SECRETS


def prepare_env_file(source: Path, target: Path) -> Path:
    """Минимум ключей в env-файле датасета: KAGGLE_* и незаполненные не попадают."""
    if not source.is_file():
        return target
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() in _KERNEL_KEYS and value.strip():
            rows.append(f"{key.strip()}={value.strip()}")
    target.write_text(chr(10).join(rows) + chr(10), encoding="utf-8")
    return target


def poll_dataset_ready(
    owner_slug: str,
    poll: Runner | None = None,
    sleep: Sleep | None = None,
    timeout: float = 900.0,
) -> None:
    """Ждать, пока последняя версия dataset будет готова к монтированию."""
    kaggle = poll or _cli_run
    nap = sleep or time.sleep
    deadline = time.monotonic() + timeout
    while True:
        completed = kaggle(["kaggle", "datasets", "status", owner_slug])
        status = (completed.stdout or "").lower()
        if "ready" in status:
            return
        if "error" in status:
            raise RuntimeError(f"dataset {owner_slug} сломан на стороне Kaggle")
        if time.monotonic() > deadline:
            raise TimeoutError(f"dataset {owner_slug} так и не стал ready")
        nap(15)


def download_output(path: Path, owner_slug: str, run: Runner | None = None) -> None:
    """Скачать артефакты последнего запуска."""
    kaggle = run or _cli_run
    command = ["kaggle", "kernels", "output", owner_slug, "-p", str(path)]
    completed = kaggle(command)
    if completed.returncode != 0:
        raise RuntimeError("kaggle kernels output не прошёл")


def _normalize_kaggle_credentials() -> None:
    """Kaggle CLI 2.2.5 хочет KAGGLE_API_TOKEN; учитывает и устаревший KAGGLE_KEY."""
    if not os.getenv("KAGGLE_API_TOKEN") and os.getenv("KAGGLE_KEY"):
        os.environ["KAGGLE_API_TOKEN"] = os.environ["KAGGLE_KEY"]


def _prepare_kaggle_cli() -> None:
    _normalize_kaggle_credentials()
    # Windows-баг CLI 2.2.5: рабочий каталог загрузок не создаётся сам
    Path(tempfile.gettempdir()).joinpath(".kaggle-uploads").mkdir(exist_ok=True)


def main(argv: list[str] | None = None, run: Runner | None = None) -> int:
    load_dotenv(find_dotenv(usecwd=True))
    _prepare_kaggle_cli()
    args = tyro.cli(KaggleArgs, args=argv)
    owner = args.owner or os.getenv("KAGGLE_USERNAME")
    if not owner:
        raise SystemExit("укажите --owner или переменную KAGGLE_USERNAME")
    staging = Path("kaggle") / args.slug
    staging.mkdir(parents=True, exist_ok=True)
    wheel = build_wheel(staging, run=run)
    stage_wheel_dataset(staging / "dataset", wheel, owner=owner, slug=args.wheel_dataset)
    prepare_env_file(Path(".env"), staging / "dataset" / "kaggle-env.env")
    stage_requirements(staging / "dataset", run=run)
    if args.config is not None:
        stage_experiment(staging / "dataset", args)
    upload_wheel_dataset(staging / "dataset", owner=owner, slug=args.wheel_dataset, run=run)
    cleanup_wheel_artifacts(staging)
    poll_dataset_ready(f"{owner}/{args.wheel_dataset}", poll=run)
    kernel = render_kernel(args, owner=owner)
    (staging / "kernel.py").write_text(kernel.code, encoding="utf-8")
    (staging / "kernel-metadata.json").write_text(json.dumps(kernel.metadata), encoding="utf-8")
    push_kernel(staging, accelerator=args.accelerator, timeout=args.timeout, run=run)
    owner_slug = f"{owner}/{args.slug}"
    status = poll_status(
        owner_slug,
        poll_seconds=args.poll_seconds,
        poll=run,
        timeout=float(args.timeout),
    )
    if status != COMPLETE:
        return 5
    download_output(staging / "output", owner_slug, run=run)
    cleanup_debug_dirs(staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
