"""Команда обучения до бюджета FLOPs."""

from __future__ import annotations

import enum
import logging
import os
import sys
import time
from pathlib import Path
from typing import Literal

import tyro
import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from compare_attention.data.config import META_NAME, KVRetrievalConfig
from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.models.config import GPT2Config
from compare_attention.train.backup import BackupTracker
from compare_attention.train.config import DatasetSpec, Optim, Schedule, TrainConfig
from compare_attention.train.hf import upload_results
from compare_attention.train.loop import train
from compare_attention.train.tracker import GroupTracker, select

_FROZEN = ConfigDict(frozen=True)

logger = logging.getLogger(__name__)


class TrainArgs(BaseModel):
    """Учить GPT-2, пока не кончится бюджет FLOPs."""

    model_config = _FROZEN

    config: Path | None = Field(
        default=None, description="YAML-файл эксперимента; флаги CLI переопределяют его значения."
    )
    data: Path = Field(description="Каталог kv_retrieval с meta.json.")
    schedule: tyro.conf.OmitArgPrefixes[Schedule] = Field(description="Бюджет, батч и тип.")
    optim: tyro.conf.OmitArgPrefixes[Optim] = Field(default_factory=Optim)
    model: tyro.conf.OmitArgPrefixes[GPT2Config] = Field(default_factory=GPT2Config)
    kv: tyro.conf.OmitArgPrefixes[DatasetSpec] = Field(
        default_factory=DatasetSpec,
        description="Параметры генерации kv_retrieval: датасет готовится сам, если meta.json нет.",
    )
    result_dir: Path | None = Field(
        default=None,
        description="Каталог результата: train.log, model.pt и metrics.jsonl. "
        "По умолчанию models/<штамп времени>, с --config — models/<имя эксперимента>/<штамп>.",
    )
    tracker: Literal["clearml", "disabled"] = Field(
        default="clearml", description="Куда отражать метрики и логи запуска."
    )
    hf_repo: str | None = Field(
        default=None,
        description="Приватный репозиторий Hugging Face (например, user/model), "
        "куда после обучения сливается каталог результата.",
    )


def load_experiment(path: Path) -> TrainArgs:
    """YAML-файл эксперимента в список аргументов обучения."""
    return TrainArgs(**yaml.safe_load(path.read_text(encoding="utf-8")))


def _config_from_argv(argv: list[str] | None) -> Path | None:
    """Достать --config до разбора tyro: YAML должен стать дефолтом заранее."""
    candidates = list(argv or [])
    for index, item in enumerate(candidates):
        if item == "--config" and index + 1 < len(candidates):
            return Path(candidates[index + 1])
        if item.startswith("--config="):
            return Path(item[len("--config=") :])
    return None


def _derived_flags(spec: BaseModel) -> list[str]:
    """Pydantic-спецификация в плоский список --flag=value (bool --flag/--no-flag)."""
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


def _flag_of(token: str) -> str:
    flag = token[2:].split("=")[0]
    if flag.startswith("no-"):
        flag = flag[3:]
    return flag


def _merge_yaml_into_argv(argv: list[str] | None, defaults: TrainArgs) -> list[str]:
    """YAML-значения разворачиваются в флаги и подклеиваются к не заданным явно."""
    tokens = [f"--data={defaults.data}"] if defaults.data else []
    tokens += _derived_flags(defaults.schedule)
    tokens += _derived_flags(defaults.optim)
    tokens += _derived_flags(defaults.model)
    if defaults.kv is not None:
        tokens += _derived_flags(defaults.kv)
    if defaults.tracker:
        tokens += [f"--tracker={defaults.tracker}"]
    if defaults.hf_repo:
        tokens += [f"--hf-repo={defaults.hf_repo}"]
    if defaults.result_dir:
        tokens += [f"--result-dir={defaults.result_dir}"]
    present = {_flag_of(item) for item in (argv or []) if item.startswith("--")}
    return list(argv or []) + [token for token in tokens if _flag_of(token) not in present]


def parse_train_args(argv: list[str] | None = None) -> TrainArgs:
    """Парсим CLI; секции из --config разворачиваются в флаги, явные флаги их перекрывают."""
    if argv is None:
        argv = list(sys.argv[1:])
    config = _config_from_argv(argv)
    if config is not None:
        defaults = load_experiment(config)
        argv = _merge_yaml_into_argv(argv, defaults)
    return tyro.cli(TrainArgs, args=argv)


def _ensure_dataset(args: TrainArgs) -> None:
    """Подготовить датасет из kv-секции, если данных ещё нет."""
    if (args.data / META_NAME).is_file():
        return
    spec = KVRetrievalConfig(
        vocab_size=args.model.vocab_size,
        input_seq_len=args.kv.input_seq_len,
        num_kv_pairs=args.kv.num_kv_pairs,
        power_a=args.kv.power_a,
        n_train=args.kv.n_train,
        n_val=args.kv.n_val,
        n_test=args.kv.n_test,
        seed=args.kv.dataset_seed,
    )
    prepare_kv_retrieval(spec, args.data)
    logger.info("датасет подготовлен: %s", args.data)


def _configure_logging(log_path: Path) -> None:
    """И писать в терминал, и параллельно складывать ход обучения в train.log."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_train_args(argv)
    load_dotenv(find_dotenv(usecwd=True))
    payload = args.model_dump()
    config_path = payload.pop("config")
    payload.pop("kv")
    experiment = Path(config_path).stem if config_path else None
    if experiment:
        result_dir = payload.pop("result_dir") or Path("models") / experiment / time.strftime(
            "%Y%m%d-%H%M%S"
        )
    else:
        result_dir = payload.pop("result_dir") or Path("models") / time.strftime("%Y%m%d-%H%M%S")
    target = Path(result_dir)
    kind = payload.pop("tracker")
    hf_repo = payload.pop("hf_repo") or os.getenv("HF_REPO")
    target.mkdir(parents=True, exist_ok=True)
    _configure_logging(target / "train.log")
    _ensure_dataset(args)
    config = TrainConfig.model_validate(payload)
    trackers = GroupTracker(select(kind), BackupTracker(target))
    train(config, tracker=trackers)
    if hf_repo:
        return _upload(target, hf_repo)
    return 0


def _upload(target: Path, hf_repo: str) -> int:
    """Слить каталог результата в приватный репозиторий Hugging Face."""
    try:
        url = upload_results(target, hf_repo)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("не удалось выгрузить результат в Hugging Face")
        return 4
    logger.info("артефакты запуска в Hugging Face: %s", url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
