"""Флаги CLI обучения для тестов."""

from pathlib import Path


def train_flags(
    data: Path,
    *,
    tracker: str = "clearml",
    result_dir: Path | None = None,
    hf_repo: str | None = None,
) -> list[str]:
    """Флаги запуска обучения на подготовленных данных."""
    flags = [
        "--data",
        str(data),
        "--train-flops",
        "1",
        "--batch-size",
        "2",
        "--learning-rate",
        "0.01",
        "--weight-decay",
        "0",
        "--n-layer",
        "1",
        "--n-head",
        "2",
        "--n-embd",
        "8",
        "--vocab-size",
        "32",
        "--block-size",
        "16",
        "--dropout",
        "0",
        "--seed",
        "0",
        "--no-compile",
        "--tracker",
        tracker,
    ]
    if result_dir is not None:
        flags += ["--result-dir", str(result_dir)]
    if hf_repo is not None:
        flags += ["--hf-repo", hf_repo]
    return flags
