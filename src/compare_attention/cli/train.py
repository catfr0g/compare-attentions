"""Команда обучения до бюджета FLOPs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import tyro
from pydantic import BaseModel, ConfigDict, Field

from compare_attention.models.config import GPT2Config
from compare_attention.train.config import Optim, Schedule, TrainConfig
from compare_attention.train.loop import train

_FROZEN = ConfigDict(frozen=True)


class TrainArgs(BaseModel):
    """Учить GPT-2, пока не кончится бюджет FLOPs."""

    model_config = _FROZEN

    data: Path = Field(description="Каталог kv_retrieval с meta.json.")
    schedule: tyro.conf.OmitArgPrefixes[Schedule] = Field(description="Бюджет, батч и тип.")
    optim: tyro.conf.OmitArgPrefixes[Optim] = Field(default_factory=Optim)
    model: tyro.conf.OmitArgPrefixes[GPT2Config] = Field(default_factory=GPT2Config)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = tyro.cli(TrainArgs, args=argv)
    train(TrainConfig.model_validate(args.model_dump()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
