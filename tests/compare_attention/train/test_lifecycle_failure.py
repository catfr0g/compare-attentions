"""Secret-safe failure evidence from the real public normal-training lifecycle."""

import json

import pytest

from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.train.backup import BackupTracker
from compare_attention.train.loop import train
from compare_attention.train.tracker import DisabledTracker, GroupTracker
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.train.factories import train_config


class FailingFinish(DisabledTracker):
    def finish(self, _result, _model):
        raise LookupError("PRIVATE_MESSAGE_Bearer_secret_C:/private/model")


def test_public_normal_finish_persists_original_safe_failure(tmp_path):
    data = tmp_path / "data"
    prepare_kv_retrieval(data_config(), data)
    out = tmp_path / "out"
    tracker = GroupTracker(BackupTracker(out), FailingFinish())
    with pytest.raises(LookupError, match="PRIVATE_MESSAGE"):
        train(train_config(data, seed=303), tracker)
    path = out / "lifecycle-failure.json"
    assert path.is_file(), "finish escaped without durable original failure evidence"
    report = json.loads(path.read_bytes())
    assert report["phase"] == "finish"
    assert report["exception_class"] == "LookupError"
    assert report["frames"][-1]["basename"] == "test_lifecycle_failure.py"
    assert report["frames"][-1]["function"] == "finish"
    assert report["frames"][-1]["line"] > 0
    assert "PRIVATE_MESSAGE" not in path.read_text()
    assert "secret" not in path.read_text()
    assert (out / "checkpoints").exists() is False
    assert (out / "model.pt").is_file()


def test_gloo_original_finish_facts_exist_before_coordinated_wrapper(tmp_path):
    from tests.compare_attention.train.test_distributed import fixture, launch

    fixture(tmp_path)
    launch(tmp_path, str(tmp_path), module="tests.compare_attention.train.lifecycle_worker")
    rows = [json.loads((tmp_path / f"rank-{rank}-observed.json").read_bytes()) for rank in (0, 1)]
    assert rows[0] == rows[1]
    assert rows[0]["exception_class"] == "LookupError"
    assert rows[0]["frames"][-1]["basename"] == "lifecycle_worker.py"


def test_gloo_console_retains_primary_through_service_sanitizer(tmp_path):
    from tests.compare_attention.train.test_distributed import fixture, launch

    fixture(tmp_path)
    (tmp_path / "sanitized-service").touch()
    launch(tmp_path, str(tmp_path), module="tests.compare_attention.train.lifecycle_worker")
    console = (tmp_path / "torchrun.log").read_text(encoding="utf-8", errors="replace")
    assert "LookupError" in console
    assert "clearml_flush" in console
    assert "PRIVATE_MESSAGE" not in console
    assert "SAFE_SERVICE_WRAPPER" not in console


def test_secondary_failed_identity_write_cannot_mask_primary(tmp_path):
    data = tmp_path / "data"
    prepare_kv_retrieval(data_config(), data)
    out = tmp_path / "out"

    class BrokenIdentity(BackupTracker):
        def log_loss(self, _loss, _flops):
            raise LookupError("PRIVATE_PRIMARY")

        def record_identity(self, identity):
            if identity["status"] == "failed":
                raise OSError("PRIVATE_SECONDARY")
            super().record_identity(identity)

    with pytest.raises(LookupError, match="PRIVATE_PRIMARY"):
        train(train_config(data, seed=404), BrokenIdentity(out))
    report = json.loads((out / "lifecycle-failure.json").read_bytes())
    assert report["exception_class"] == "LookupError"
    assert "PRIVATE" not in json.dumps(report)


def test_initial_identity_failure_has_identity_phase(tmp_path):
    data = tmp_path / "data"
    prepare_kv_retrieval(data_config(), data)
    out = tmp_path / "out"

    class BrokenInitialIdentity(BackupTracker):
        def record_identity(self, _identity):
            raise LookupError("PRIVATE_IDENTITY")

    with pytest.raises(LookupError):
        train(train_config(data, seed=404), BrokenInitialIdentity(out))
    report = json.loads((out / "lifecycle-failure.json").read_bytes())
    assert report["phase"] == "identity_write"
    assert report["exception_class"] == "LookupError"


def test_unavailable_diagnostic_sink_does_not_replace_original(tmp_path):
    data = tmp_path / "data"
    prepare_kv_retrieval(data_config(), data)

    class BrokenDiagnostic(FailingFinish):
        def record_failure(self, _report):
            raise OSError("diagnostic-unavailable")

    with pytest.raises(LookupError, match="PRIVATE_MESSAGE"):
        train(train_config(data, seed=404), BrokenDiagnostic())
