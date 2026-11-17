"""Public opt-in flag and real module entry point under two-process torchrun."""

import json

import yaml

from compare_attention.cli.train import parse_train_args
from tests.compare_attention.train.test_distributed import fixture, launch


def test_distributed_flag_is_operational_and_legacy_serialization_stays_unchanged(tmp_path):
    config = fixture(tmp_path)
    payload = config.model_dump(mode="json", exclude={"dataset"})
    payload.update(tracker="disabled", result_dir=str(tmp_path / "result"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    default = parse_train_args(["--config", str(path)])
    assert default.distributed is False
    assert "distributed" not in default.model_dump()
    payload["distributed"] = True
    path.write_text(yaml.safe_dump(payload))
    assert parse_train_args(["--config", str(path)]).distributed is True
    assert parse_train_args(["--config", str(path), "--no-distributed"]).distributed is False


def test_actual_module_entry_under_torchrun_writes_exactly_one_result(tmp_path):
    config = fixture(tmp_path)
    payload = config.model_dump(mode="json", exclude={"dataset"})
    payload.update(tracker="disabled", result_dir=str(tmp_path / "result"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    launch(tmp_path, "--config", str(path), "--distributed", module="compare_attention.cli.train")
    rows = [
        json.loads(line) for line in (tmp_path / "result/metrics.jsonl").read_text().splitlines()
    ]
    assert sum(row["type"] == "step" for row in rows) == 4
    assert sum(row["type"] == "result" for row in rows) == 1
    assert rows[-1]["provenance"]["schema_version"] == 3
