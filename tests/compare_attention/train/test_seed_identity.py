"""Real offline ClearML initialization must not replace the training seed."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from clearml import Task
from clearml.backend_api.session.session import Session
from clearml.backend_interface.base import InterfaceBase

from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.models.gpt2 import GPT2
from compare_attention.train import loop
from compare_attention.train.tracker import ClearMLTracker
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.models.factories import gpt2_config
from tests.compare_attention.train.factories import train_config, train_result


def _offline_task(_config):
    return Task.init(
        project_name="compare-attention-seed-test",
        task_name="seed initialization regression",
        reuse_last_task_id=False,
        auto_connect_frameworks=False,
        auto_connect_arg_parser=False,
        auto_resource_monitoring=False,
        auto_connect_streams=False,
    )


@pytest.mark.integration
@pytest.mark.parametrize("attention", ["classic", "deltanet"])
def test_real_clearml_preserves_requested_seed_and_independent_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attention: str
) -> None:
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    monkeypatch.setenv("CLEARML_CACHE_DIR", str(tmp_path / "clearml"))
    monkeypatch.setattr(Session, "_offline_mode", True)
    monkeypatch.setattr(InterfaceBase, "_offline_mode", True)
    prepare_kv_retrieval(data_config(), tmp_path / "data")
    initialized = []

    def capture_model(config):
        effective_seed = torch.initial_seed()
        model = GPT2(config)
        values = b"".join(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            for tensor in model.parameters()
        )
        initialized.append((effective_seed, hashlib.sha256(values).hexdigest()))
        return model

    # This test targets initialization, not Triton forward support on CPU.
    monkeypatch.setattr(loop, "GPT2", capture_model)
    monkeypatch.setattr(loop, "_run", lambda config, *_args: train_result(config))
    for seed in (42, 6767, 42):
        config = train_config(
            tmp_path / "data",
            seed=seed,
            model=gpt2_config(n_layer=1, n_embd=32, attention=attention),
        )
        loop.train(config, ClearMLTracker(task_factory=_offline_task))

    assert [seed for seed, _digest in initialized] == [42, 6767, 42]
    assert initialized[0][1] == initialized[2][1]
    assert initialized[0][1] != initialized[1][1]


def test_identity_artifact_records_real_weights_and_runtime(tmp_path: Path) -> None:
    from compare_attention.train.backup import BackupTracker  # pylint: disable=import-outside-toplevel

    prepare_kv_retrieval(data_config(), tmp_path / "data")
    config = train_config(tmp_path / "data", seed=6767)
    loop.train(config, BackupTracker(tmp_path / "result"))
    identity = json.loads((tmp_path / "result" / "run_identity.json").read_text())
    assert identity["requested_seed"] == identity["effective_seed"] == 6767
    assert identity["attention"] == "classic"
    assert identity["status"] == "completed"
    assert identity["config"]["schedule"]["seed"] == 6767
    assert identity["runtime"]["torch"] == torch.__version__
    assert identity["runtime"]["fla"]
    assert "triton" in identity["runtime"]
    assert "gpu" in identity["runtime"]
    assert len(identity["initial_parameters_sha256"]) == 64
    assert len(identity["final_parameters_sha256"]) == 64
    assert identity["initial_parameters_sha256"] != identity["final_parameters_sha256"]
    # Reconstruct the exact initialized tensors, not a config/file digest.
    from compare_attention.train.identity import parameter_fingerprint  # pylint: disable=import-outside-toplevel

    torch.manual_seed(6767)
    rebuilt = GPT2(config.model)
    assert identity["initial_parameters_sha256"] == parameter_fingerprint(rebuilt)
    rebuilt.load_state_dict(torch.load(tmp_path / "result" / "model.pt", weights_only=True))
    assert identity["final_parameters_sha256"] == parameter_fingerprint(rebuilt)
    assert identity["schema_version"] == 2
    assert identity["accounting"] == identity["recorded_flops"]
    assert identity["accounting"]["policy"] == "algorithmic_matmul_conv_v2"
    assert identity["compiler"]["backend_invocations"] == 0


def test_compile_request_does_not_claim_unexecuted_backend() -> None:
    model = GPT2(gpt2_config(n_layer=1))
    evidence = {}
    loop._compile(model, True, evidence)  # pylint: disable=protected-access
    assert evidence["requested"] is True
    assert evidence["backend_invocations"] == 0
    assert evidence["backend_invocation_verified"] is False
