"""Команды подготовки датасета kv_retrieval."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import tyro
from pydantic import BaseModel, ConfigDict, Field

from compare_attention.data.config import KVRetrievalConfig
from compare_attention.data.store import prepare_kv_retrieval

_DEFAULT_OUTPUT = Path("data/kv_retrieval")
_FROZEN = ConfigDict(frozen=True)


class PrepareArgs(BaseModel):
    """Подготовить фиксированные train/val/test выборки kv_retrieval (MQAR)."""

    model_config = _FROZEN

    output: Path = Field(
        default=_DEFAULT_OUTPUT,
        description="Каталог с meta.json и массивами .npy.",
    )
    force: bool = Field(
        default=False,
        description="Перезаписать датасет, даже если параметры в каталоге другие.",
    )
    config: tyro.conf.OmitArgPrefixes[KVRetrievalConfig] = Field(default_factory=KVRetrievalConfig)


class CleanArgs(BaseModel):
    """Удалить подготовленный каталог kv_retrieval."""

    model_config = _FROZEN

    output: Path = Field(
        default=_DEFAULT_OUTPUT,
        description="Каталог с meta.json и массивами .npy.",
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = tyro.cli(PrepareArgs, args=argv)
    prepare_kv_retrieval(args.config, args.output, force=args.force)
    return 0


def clean_main(argv: list[str] | None = None) -> int:
    """Удалить каталог подготовленного датасета."""
    args = tyro.cli(CleanArgs, args=argv)
    shutil.rmtree(args.output, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
