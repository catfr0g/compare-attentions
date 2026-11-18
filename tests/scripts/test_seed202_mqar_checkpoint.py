"""Genuine FULL metadata validation; constructors and restoration are prohibited."""

import copy
import os
import sys
from pathlib import Path

import pytest

WORK = Path("C:/Users/Michael/AppData/Local/hermes/cache/scratch/mqar202-repair-work-v1")
WHEEL = Path(os.environ.get("SEED202_EXTRACTED_WHEEL", WORK / "wheel"))
DOWNLOAD = Path(
    "E:/HermesArtifacts/compare-attention/sorting-mqar-seed202-pilot-v1/"
    "remaining-three-after-tail-v1/qualified-launch-bridge-v1/"
    "execution-classic-20261118-v1/failed-download-v1"
)


def native_case(payload):
    return {
        "id": "mqar-k64-classic-seed202",
        "task": "mqar-k64",
        "family": "classic",
        "seed": 202,
        "config": payload["config"],
        "updates": 10780,
        "step_flops": 463856467968,
        "global_batch": 64,
        "local_batch": 32,
        "world_size": 2,
        "validation_steps": list(range(980, 10781, 980)),
        "accounting": payload["identity"]["accounting"],
        "requested_flops": 5_000_000_000_000_000,
        "test_split_permitted": False,
    }


@pytest.fixture(scope="module")
def full():
    sys.path.insert(0, str(WHEEL))
    import torch

    import compare_attention

    assert Path(compare_attention.__file__).resolve().is_relative_to(WHEEL.resolve())

    (path,) = DOWNLOAD.rglob("step_00000980.pt")
    yield path, torch.load(path, weights_only=True, map_location="cpu")
    sys.path.remove(str(WHEEL))


def test_genuine_full_structure_without_model_optimizer_or_rng_restore(full, monkeypatch):
    import torch
    from seed202_mqar_checkpoint import validate_original

    from compare_attention.train.distributed_checkpoint import validate

    assert callable(validate)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("HOST_SCIENTIFIC_EXECUTION_PROHIBITED")

    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(torch.random, "fork_rng", forbidden)
    monkeypatch.setattr(torch, "set_rng_state", forbidden)
    proof = validate_original(full[0])
    assert proof["step"] == 980
    assert proof["remaining_updates"] == 9800
    assert proof["next_step"] == 981
    assert proof["scope"] == "AUTHENTIC_FULL_METADATA_NO_TARGET_RESTORE"
    assert proof["optimizer_states"] == 52


@pytest.mark.parametrize("field", ["shape", "optimizer_order", "scaler", "runtime", "cursor"])
def test_rehashed_structure_corruption_rejected(full, field):
    from seed202_mqar_checkpoint import descriptor, validate_structure

    from compare_attention.train.checkpoint import _digest

    payload = copy.deepcopy(full[1])
    expected = descriptor(full[1])
    if field == "shape":
        payload["model"]["wpe.weight"] = payload["model"]["wpe.weight"][:1].clone()
    elif field == "optimizer_order":
        payload["optimizer_parameters"][0].reverse()
    elif field == "scaler":
        payload["precision"]["scaler"]["scale"] = 64.0
    elif field == "runtime":
        payload["distributed"]["ranks"][1]["identity"]["runtime"]["python"] = "3.13.15"
    else:
        payload["cursor"] += 1
    payload["integrity_sha256"] = _digest(payload)
    with pytest.raises(ValueError):
        validate_structure(payload, expected)
