"""Genuine tensor structure only: forbid model, optimizer and RNG restoration."""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_genuine_full_native_structural_source_runtime_prefix_proof():
    assert importlib.util.find_spec("mqar_final_tail_structural") is not None, (
        "genuine FULL structural MQAR audit is missing"
    )
    from mqar_final_tail_native import audit
    from mqar_final_tail_test_layout import CANDIDATE, DATA, INPUT

    got = audit(
        INPUT / "original", None, DATA, CANDIDATE / "payload/configs/mqar-k64-classic-seed202.yaml"
    )
    assert got["updates"] == 10780 and got["optimizer_states"] == 52
    assert got["actual_flops"] == 5000372724695040
    assert got["optimizer_constructions"] == got["checkpoint_rng_restores"] == 0
    assert got["original_identity_file_status"] == "initialized"
    assert got["original_provider_errors"] == [145778849, 145850896]


def test_structural_corruptions_rejected_without_restore():
    import copy
    import json

    import pytest
    import torch
    from mqar_final_tail_full import inspect_full
    from mqar_final_tail_structural import derive_case
    from mqar_final_tail_test_layout import CANDIDATE, DATA, INPUT

    payload = torch.load(
        INPUT / "original/out/checkpoints/step_00010780.pt", map_location="cpu", weights_only=True
    )
    result = json.loads((INPUT / "original/out/metrics.jsonl").read_text().splitlines()[-1])
    case = derive_case(CANDIDATE / "payload/configs/mqar-k64-classic-seed202.yaml", DATA)
    faults = [
        lambda p: p["precision"].update(optimizer_steps=10779),
        lambda p: p["distributed"]["ranks"][0]["identity"]["runtime"].update(torch="2.11.0+cu128"),
        lambda p: p["distributed"]["ranks"][0].update(rank=True),
        lambda p: p["model"].update({"lm_head.weight": p["model"]["lm_head.weight"].clone() + 1}),
        lambda p: p["model"].update({"wte.weight": p["model"]["wte.weight"].half()}),
        lambda p: p["optimizer"]["state"][0].update(step=torch.tensor(10779.0)),
        lambda p: p["config"]["model"].update(n_layer=8),
    ]
    for fault in faults:
        changed = copy.deepcopy(payload)
        fault(changed)
        with pytest.raises((ValueError, TypeError)):
            inspect_full(changed, result, case)
