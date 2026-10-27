"""Команда подготовки kv_retrieval."""

from pathlib import Path

import tyro

from compare_attention.cli.prepare_kv_retrieval import PrepareArgs, clean_main, main
from compare_attention.data.config import KVRetrievalConfig
from compare_attention.data.store import load_config

ROOT = Path(__file__).resolve().parents[3]


def test_cli_writes_the_requested_config(tmp_path: Path) -> None:
    output = tmp_path / "cli"
    expected = KVRetrievalConfig(
        vocab_size=32,
        input_seq_len=16,
        num_kv_pairs=2,
        power_a=0.5,
        n_train=2,
        n_val=1,
        n_test=2,
        seed=9,
    )

    main(
        [
            "--output",
            str(output),
            "--vocab-size",
            "32",
            "--input-seq-len",
            "16",
            "--num-kv-pairs",
            "2",
            "--power-a",
            "0.5",
            "--n-train",
            "2",
            "--n-val",
            "1",
            "--n-test",
            "2",
            "--seed",
            "9",
        ]
    )

    assert load_config(output) == expected


def test_prepare_command_is_registered() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'prepare-kv-retrieval = "compare_attention.cli.prepare_kv_retrieval:main"' in project


def test_prepare_script_uses_dataset_defaults() -> None:
    args = tyro.cli(PrepareArgs, args=[])

    assert args == PrepareArgs()


def test_clean_removes_the_dataset_directory(tmp_path: Path) -> None:
    output = tmp_path / "kv_retrieval"
    marker = output / "meta.json"
    marker.parent.mkdir()
    marker.write_text("{}", encoding="utf-8")

    clean_main(["--output", str(output)])

    assert not output.exists()
