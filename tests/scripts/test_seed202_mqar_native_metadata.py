"""Full adapted native predicate replay on genuine FULL without constructors."""

import copy
import importlib
import sys

import pytest
from test_seed202_mqar_builder import HISTORY, WORK
from test_seed202_mqar_checkpoint import native_case

pytest_plugins = ["test_seed202_mqar_checkpoint"]


def test_complete_native_metadata_validator_on_genuine_full(tmp_path, full, monkeypatch):
    from build_seed202_mqar_continuation import stage_science_helpers, stage_source

    source = stage_source(WORK / "parent", HISTORY, tmp_path / "source")
    helpers = stage_science_helpers(WORK / "parent", tmp_path / "science")
    old = {k: v for k, v in sys.modules.items() if k.startswith("seed202_pilot")}
    for name in old:
        del sys.modules[name]
    paths = [str(source), str(helpers / "operator")]
    sys.path[:0] = paths
    try:
        native = importlib.import_module("seed202_pilot.mqar_host_native")
        import torch

        def forbidden(*_args, **_kwargs):
            raise AssertionError("HOST_CONSTRUCTOR_OR_RNG_RESTORE_FORBIDDEN")

        monkeypatch.setattr(torch.optim, "AdamW", forbidden)
        monkeypatch.setattr(torch.random, "fork_rng", forbidden)
        payload = full[1]
        case = {
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
        proof = native.validate_full(payload, case, step=980)
        assert proof["updates"] == 980
        assert (
            proof["final_sha256"]
            == "2dcc90547529ac31e750c9e7476c8a5dd4eb8322025873996db436dca4a5079b"
        )
    finally:
        for path in paths:
            sys.path.remove(path)
        for name in list(sys.modules):
            if name.startswith("seed202_pilot"):
                del sys.modules[name]
        sys.modules.update(old)


@pytest.mark.parametrize("corruption", ["model_structure", "optimizer_betas"])
def test_rehashed_container_type_changes_do_not_weaken_native_predicates(
    tmp_path, full, corruption
):
    from build_seed202_mqar_continuation import stage_science_helpers, stage_source

    from compare_attention.train.checkpoint import _digest

    source = stage_source(WORK / "parent", HISTORY, tmp_path / "source")
    helpers = stage_science_helpers(WORK / "parent", tmp_path / "science")
    old = {k: v for k, v in sys.modules.items() if k.startswith("seed202_pilot")}
    for name in old:
        del sys.modules[name]
    paths = [str(source), str(helpers / "operator")]
    sys.path[:0] = paths
    try:
        native = importlib.import_module("seed202_pilot.mqar_host_native")
        payload = copy.deepcopy(full[1])
        if corruption == "model_structure":
            payload["model_structure"] = [list(row) for row in payload["model_structure"]]
        else:
            group = payload["optimizer"]["param_groups"][0]
            group["betas"] = list(group["betas"])
        payload["integrity_sha256"] = _digest(payload)
        with pytest.raises(ValueError):
            native.validate_full(payload, native_case(payload), step=980)
    finally:
        for path in paths:
            sys.path.remove(path)
        for name in list(sys.modules):
            if name.startswith("seed202_pilot"):
                del sys.modules[name]
        sys.modules.update(old)
