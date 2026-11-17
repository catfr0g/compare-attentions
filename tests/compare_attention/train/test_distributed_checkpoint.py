"""Semantic DDP-envelope checks on genuinely produced two-process snapshots."""

import copy

import pytest
import torch

from compare_attention.exploration_gate import verify_exploration_run
from compare_attention.flop_gate import _verify_identity
from compare_attention.seed_gate import _read_run
from compare_attention.train.checkpoint import _digest, load_checkpoint
from compare_attention.train.distributed_checkpoint import validate
from tests.compare_attention.train.test_distributed import fixture, launch


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("ddp-envelope")
    config = fixture(root)
    config = config.model_copy(
        update={"schedule": config.schedule.model_copy(update={"checkpoint_every_steps": 1})}
    )
    (root / "config.json").write_text(config.model_dump_json())
    launch(root)
    return root, load_checkpoint(root / "distributed/checkpoints/latest.json")


def test_per_rank_compiler_identity_cannot_be_rebound_with_a_valid_checksum(artifact):
    _, payload = artifact
    broken = copy.deepcopy(payload)
    broken["distributed"]["ranks"][1]["identity"]["compiler"]["backend"] = "foreign"
    broken["integrity_sha256"] = _digest(broken)
    with pytest.raises(ValueError, match="compiler"):
        validate(broken)


@pytest.mark.parametrize("gate", ["flops", "exploration", "seed"])
def test_legacy_single_device_gates_reject_real_ddp_identity_schema(artifact, gate):
    root, payload = artifact
    directory = root / "distributed"
    with pytest.raises(ValueError, match="schema"):
        if gate == "flops":
            _verify_identity(
                directory, attention="classic", expected=payload["identity"]["accounting"]
            )
        elif gate == "exploration":
            verify_exploration_run(directory)
        else:
            _read_run(directory, seed=0, attention="classic")


def test_cpu_validator_accepts_structural_selected_cuda_rng_without_cuda_access(
    artifact, monkeypatch
):
    _, payload = artifact
    remote = copy.deepcopy(payload)
    remote["distributed"]["topology"]["device_type"] = "cuda"
    for rank, row in enumerate(remote["distributed"]["ranks"]):
        row["rng"]["cuda"] = [torch.ones(5056, dtype=torch.uint8)]
        row["rng"]["cuda_devices"] = [["Tesla T4", [7, 5]]]
        row["identity"]["runtime"]["gpu"] = {
            "name": "Tesla T4",
            "index": rank,
            "compute_capability": [7, 5],
            "total_memory_bytes": 16 * 1024**3,
        }
    remote["rng"] = remote["distributed"]["ranks"][0]["rng"]
    remote["identity"] = remote["distributed"]["ranks"][0]["identity"]
    remote["integrity_sha256"] = _digest(remote)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Structural CPU verifier must not access CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    validate(remote)


def test_shared_rank_zero_identity_cannot_disagree_with_the_rank_record(artifact):
    _, payload = artifact
    broken = copy.deepcopy(payload)
    broken["identity"] = copy.deepcopy(broken["identity"])
    broken["identity"]["effective_seed"] += 1
    broken["integrity_sha256"] = _digest(broken)
    with pytest.raises(ValueError, match="canonical"):
        validate(broken)


def test_ddp_version_is_an_exact_integer_not_a_numeric_lookalike(artifact):
    _, payload = artifact
    broken = copy.deepcopy(payload)
    broken["schema_version"] = 2.0
    broken["integrity_sha256"] = _digest(broken)
    with pytest.raises(ValueError, match="schema"):
        validate(broken)


def test_rank_initialization_identity_must_agree_before_resume_trackers_start(artifact):
    _, payload = artifact
    broken = copy.deepcopy(payload)
    broken["distributed"]["ranks"][1]["identity"]["initial_parameters_sha256"] = "0" * 64
    broken["integrity_sha256"] = _digest(broken)
    with pytest.raises(ValueError, match="initialization"):
        validate(broken)
