# pylint: disable=redefined-outer-name
"""Эксперименты через YAML: полный прогон CLI."""

import json
import textwrap
from pathlib import Path

import pytest

from compare_attention.cli.prepare_kv_retrieval import main as prepare_main
from compare_attention.cli.train import main as train_main
from compare_attention.cli.train import parse_train_args


@pytest.fixture(autouse=True)
def clean_hf_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    "Ограждает прогон от репозиториев, выбранных в окружении."
    monkeypatch.delenv("HF_REPO", raising=False)


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    prepare_main(
        [
            "--output",
            str(tmp_path / "kv_retrieval"),
            "--vocab-size",
            "64",
            "--input-seq-len",
            "16",
            "--num-kv-pairs",
            "2",
            "--n-train",
            "4",
            "--n-val",
            "1",
            "--n-test",
            "1",
            "--seed",
            "5",
        ]
    )
    return tmp_path / "kv_retrieval"


def _experiment_path(tmp_path: Path, extra_lines: str, name: str = "smoke") -> Path:
    experiment = textwrap.dedent(extra_lines)
    path = tmp_path / f"{name}.yaml"
    path.write_text(experiment, encoding="utf-8")
    return path


def test_the_yaml_experiment_runs(prepared: Path, tmp_path: Path) -> None:
    experiment = _experiment_path(
        tmp_path,
        f"""
        data: {prepared}
        schedule:
          train_flops: 1
          batch_size: 2
          compile: false
        model:
          vocab_size: 64
          block_size: 16
          n_layer: 1
          n_head: 2
          n_embd: 8
          dropout: 0.0
        optim:
          learning_rate: 0.01
          weight_decay: 0.0
        tracker: disabled
        """,
    )
    args = parse_train_args(["--config", str(experiment)])

    assert args.kv.input_seq_len == 512
    assert args.model.vocab_size == 64

    code = train_main(["--config", str(experiment), "--result-dir", str(tmp_path / "res")])

    assert code == 0
    assert (tmp_path / "res" / "train.log").exists()
    rows = [
        json.loads(line)
        for line in (tmp_path / "res" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["type"] == "step"


def test_the_yaml_experiment_prepares_the_missing_dataset(tmp_path: Path) -> None:
    data = tmp_path / "kv_retrieval"
    experiment = _experiment_path(
        tmp_path,
        f"""
        data: {data}
        schedule:
          train_flops: 1
          batch_size: 2
          compile: false
        model:
          vocab_size: 64
          block_size: 16
          n_layer: 1
          n_head: 2
          n_embd: 8
          dropout: 0.0
        kv:
          input_seq_len: 16
          num_kv_pairs: 2
          power_a: 0.01
          n_train: 2
          n_val: 1
          n_test: 1
          dataset_seed: 3
        optim:
          learning_rate: 0.01
          weight_decay: 0.0
        tracker: disabled
        """,
        name="auto-prepare",
    )

    code = train_main(["--config", str(experiment), "--result-dir", str(tmp_path / "res")])

    assert code == 0
    assert (data / "meta.json").is_file()
    assert (tmp_path / "res" / "train.log").exists()


def test_the_default_result_dir_groups_by_the_experiment_name(
    prepared: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    experiment = _experiment_path(
        tmp_path,
        f"""
        data: {prepared}
        schedule:
          train_flops: 1
          batch_size: 2
          compile: false
        model:
          vocab_size: 64
          block_size: 16
          n_layer: 1
          n_head: 2
          n_embd: 8
          dropout: 0.0
        tracker: disabled
        """,
    )

    code = train_main(["--config", str(experiment)])

    assert code == 0
    runs = list((tmp_path / "models" / "smoke").iterdir())
    assert len(runs) == 1
    assert (runs[0] / "train.log").exists()
