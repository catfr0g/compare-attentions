"""Reuse the verified immutable KV input across code releases without another upload."""

import json
from pathlib import Path

from compare_attention import kaggle_series
from tests.compare_attention.test_kaggle_artifacts import make_inputs
from tests.compare_attention.test_kaggle_series import Runner


def test_reuse_checks_remote_manifest_and_avoids_data_create(tmp_path: Path, monkeypatch) -> None:
    configs, data = make_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = Runner()
    old = kaggle_series.stage_series(tmp_path, configs, data, tmp_path / "cache", "user", runner)
    kaggle_series.publish_series(old, runner)
    old_state = json.loads(old.read_text())
    (tmp_path / "uv.lock").write_text("updated source release", encoding="utf-8")
    new = kaggle_series.stage_series(tmp_path, configs, data, tmp_path / "cache", "user", runner)

    def remote(command):
        if "dataset_download_file" in " ".join(command):
            target = Path(command[-1])
            target.mkdir(parents=True, exist_ok=True)
            (target / "data-manifest.json").write_text(json.dumps(old_state["identity"]["data"]))
        return runner(command)

    before = len([cmd for cmd in runner.commands if "dataset_create_new" in " ".join(cmd)])
    kaggle_series.reuse_private_data(new, old, run=remote)
    kaggle_series.publish_series(new, remote)
    state = json.loads(new.read_text())
    after = len([cmd for cmd in runner.commands if "dataset_create_new" in " ".join(cmd)])
    assert after - before == 1  # Only the changed source release is published.
    assert state["published"] == {"data": "ready", "release": "ready"}
    assert state["data_id"] == old_state["data_id"]
    assert state["reused_data"]["source_series"] == str(old.resolve())
