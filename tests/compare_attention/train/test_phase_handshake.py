"""Successful hot-phase coordination must not serialize Python objects."""

import pytest
import torch

from compare_attention.train.distributed import DistributedRun


@pytest.mark.parametrize(
    "phase,phase_id",
    [("forward", 1), ("scaled_optimizer_step", 2), ("optimizer_step", 3), ("log_loss", 4)],
)
def test_successful_hot_phase_uses_one_fixed_tensor_collective(monkeypatch, phase, phase_id):
    run = DistributedRun(0, 2, 0, torch.device("cpu"))
    headers = []

    def gather(outputs, value):
        headers.append(value.clone())
        for output in outputs:
            output.copy_(value)

    def objects(*_args):
        raise AssertionError("successful forward serialized an object")

    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    monkeypatch.setattr(torch.distributed, "all_gather_object", objects)
    sentinel = object()
    assert run.call(lambda: sentinel, phase=phase) is sentinel
    assert len(headers) == 1
    assert headers[0].dtype == torch.int64
    assert headers[0].tolist() == [phase_id, 0]


def test_real_gloo_warm_protocol_failure_and_accepted_wheel_resume(tmp_path):
    import json

    from compare_attention.train.flops import estimate_step
    from tests.compare_attention.train.test_distributed import fixture, launch

    config = fixture(tmp_path, dropout=0.2)
    config = config.model_copy(
        update={
            "schedule": config.schedule.model_copy(
                update={
                    "dtype": "float16",
                    "checkpoint_every_steps": 2,
                    "checkpoint_keep_last": 2,
                    "train_flops": estimate_step(config).total * 7,
                }
            )
        }
    )
    (tmp_path / "config.json").write_text(config.model_dump_json())
    launch(tmp_path, str(tmp_path), module="tests.compare_attention.train.phase_handshake_worker")
    for rank in range(2):
        proof = json.loads((tmp_path / f"phase-proof-{rank}.json").read_text(encoding="utf-8"))
        assert proof["group_destroyed"] is True
        assert proof["protocol"]["warm_hot_object_calls"] == 0
        assert proof["protocol"]["diagnostic_cases"] == 7
        assert proof["trajectory"]["updates"] == 7
        assert proof["trajectory"]["gradients_bitwise"] is True
        assert proof["trajectory"]["next_rng_bitwise"] is True
        assert [row["mode"] for row in proof["failures"]] == [
            "forward",
            "optimizer",
            "scaled_optimizer",
            "writer",
            "callback",
        ]


def test_single_device_control_is_bitwise_and_never_enters_handshake(tmp_path):
    import json

    from compare_attention.train.flops import estimate_step
    from tests.compare_attention.train.phase_handshake_worker import reference, single_control
    from tests.compare_attention.train.test_distributed import fixture

    config = fixture(tmp_path, dropout=0.2)
    config = config.model_copy(
        update={
            "schedule": config.schedule.model_copy(
                update={
                    "dtype": "float16",
                    "checkpoint_every_steps": 2,
                    "checkpoint_keep_last": 2,
                    "train_flops": estimate_step(config).total * 7,
                }
            )
        }
    )
    proof = single_control(tmp_path, config, reference(tmp_path))
    assert proof["distributed_calls"] == 0
    assert proof["next_rng_bitwise"] is True
    (tmp_path / "single-proof.json").write_text(json.dumps(proof, indent=2))
