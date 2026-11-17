"""Неизменяемые входы серии: без сети и чтения credentials."""

import json
from pathlib import Path

import pytest
import yaml

from compare_attention import kaggle_artifacts
from compare_attention.data.config import KVRetrievalConfig
from compare_attention.data.store import prepare_kv_retrieval


def make_inputs(root: Path) -> tuple[list[Path], Path]:
    """Два семейства с одинаковым именем seed-файла и маленькие реальные данные."""
    (root / "src/compare_attention").mkdir(parents=True)
    (root / "src/compare_attention/model.py").write_text("VALUE = 1\n", encoding="utf-8")
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        (root / name).write_text("fixture\n", encoding="utf-8")
    configs = []
    spec = {"input_seq_len": 8, "num_kv_pairs": 1, "n_train": 8, "n_val": 2, "n_test": 2}
    for family in ("classic", "deltanet"):
        config = root / "experiments" / family / "seed_42.yaml"
        config.parent.mkdir(parents=True)
        config.write_text(
            yaml.safe_dump(
                dict(
                    {
                        "data": "unused",
                        "schedule": {"train_flops": 12},
                        "kv": spec,
                        "model": {"vocab_size": 32},
                    }
                )
            ),
            encoding="utf-8",
        )
        configs.append(config)
    data = root / "data"
    prepare_kv_retrieval(KVRetrievalConfig(vocab_size=32, **spec), data)
    return configs, data


def test_snapshot_preserves_paths_and_excludes_secrets(tmp_path: Path) -> None:
    configs, data = make_inputs(tmp_path)
    (tmp_path / ".env").write_text("HF_TOKEN=DO_NOT_READ\n", encoding="utf-8")
    snapshot = kaggle_artifacts.snapshot(tmp_path, configs, data)
    assert set(snapshot["configs"]) == {
        "experiments/classic/seed_42.yaml",
        "experiments/deltanet/seed_42.yaml",
    }
    assert "DO_NOT_READ" not in json.dumps(snapshot)
    assert ".env" not in snapshot["source"]
    before = snapshot["fingerprint"]
    (tmp_path / "src/compare_attention/model.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert kaggle_artifacts.snapshot(tmp_path, configs, data)["fingerprint"] != before


def test_data_manifest_detects_corruption(tmp_path: Path) -> None:
    _, data = make_inputs(tmp_path)
    manifest = kaggle_artifacts.data_manifest(data)
    kaggle_artifacts.verify_files(data, manifest["files"])
    with (data / "train_inputs.npy").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        kaggle_artifacts.verify_files(data, manifest["files"])
