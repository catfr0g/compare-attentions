"""Genuine native CPU structural replay; no object state restoration or model construction."""

import json
from pathlib import Path

from mqar_final_tail_native import (
    CONFIG_SHA,
    FINAL,
    FULL_SHA,
    MODEL_SHA,
    REMOTE_DATA,
    digest,
    require,
)


def derive_case(config, data):
    from compare_attention.cli.train import parse_train_args
    from compare_attention.train.config import TrainConfig
    from compare_attention.train.flops import estimate_step

    require(digest(config) == CONFIG_SHA, "ORIGINAL_CONFIG_BYTES")
    args = parse_train_args(["--config", str(config), "--data", str(data)])
    parsed = TrainConfig(data=args.data, model=args.model, schedule=args.schedule, optim=args.optim)
    cost = estimate_step(parsed)
    cfg = parsed.model_dump(mode="json")
    cfg["data"] = REMOTE_DATA
    require(
        cfg["dataset"]["num_kv_pairs"] == 64 and cfg["model"]["attention"] == "classic",
        "MQAR_CLASSIC_FIXED_RECIPE",
    )
    budget = parsed.schedule.train_flops
    updates = (budget + cost.total - 1) // cost.total
    cadence = parsed.schedule.val_every_steps
    points = list(range(cadence, updates, cadence)) + [updates]
    require(updates == 10780 and cost.total == 463856467968, "ORIGINAL_SCIENTIFIC_BUDGET")
    return {
        "config": cfg,
        "step_flops": cost.total,
        "accounting": cost.to_dict(),
        "updates": updates,
        "requested_flops": budget,
        "actual_flops": updates * cost.total,
        "overshoot_flops": updates * cost.total - budget,
        "validation_steps": points,
        "checkpoint_steps": points,
    }


def audit(root, historical, data, config):
    """Bind old actual restore/capture receipts separately from newly replayed tensor checks."""
    del historical
    import torch
    from mqar_final_tail_full import inspect_full
    from mqar_final_tail_package import check
    from seed202_pilot.cloud_contract import verify_history

    from compare_attention.train.stopping import EarlyStopping

    root = Path(root)
    manifest = check(root.parent)
    before = {k: v for k, v in manifest["files"].items() if k != "native-audit.json"}
    require(digest(root / "out/model.pt") == MODEL_SHA, "GENUINE_WEIGHTS")
    require(digest(root / "out/checkpoints/step_00010780.pt") == FULL_SHA, "GENUINE_FULL")
    case = derive_case(config, data)
    events = [json.loads(line) for line in (root / "out/metrics.jsonl").read_text().splitlines()]
    verified = verify_history(events, case)
    old = [
        json.loads(line) for line in (root / "original980-metrics.jsonl").read_text().splitlines()
    ]
    require(
        events[: len(old)] == old and len([r for r in old if r["type"] == "step"]) == 980,
        "EXACT_ORIGINAL980_PREFIX",
    )
    payload = torch.load(
        root / "out/checkpoints/step_00010780.pt", map_location="cpu", weights_only=True
    )
    states = inspect_full(payload, verified["result"], case)
    require(
        payload["losses"] == verified["losses"]
        and payload["validation_history"] == verified["validations"],
        "FULL_HISTORIES",
    )
    schedule = case["config"]["schedule"]
    stopper = EarlyStopping(
        schedule["early_stopping_patience"],
        schedule["early_stopping_min_delta"],
        schedule["early_stopping_min_steps"],
    )
    for row in verified["validations"]:
        stopper.observe(row["loss"], row["flops"] // case["step_flops"])
    require(stopper.state_dict() == payload["early_stopping"], "STOPPING_REPLAY")
    weights = torch.load(root / "out/model.pt", map_location="cpu", weights_only=True)
    require(
        set(weights) == set(payload["model"])
        and all(torch.equal(weights[k], payload["model"][k]) for k in weights),
        "FULL_MODEL_EQUAL",
    )
    identity = json.loads((root / "out/run_identity.json").read_bytes())
    require(
        identity["status"] == "initialized"
        and identity["final_parameters_sha256"] is None
        and identity["config"] == case["config"],
        "ORIGINAL_IDENTITY_NOT_REPAIRED",
    )
    audit_pin = json.loads((root.parent / "historical-evidence-pins.json").read_bytes())
    proof = root / "diagnosis-audit.json"
    require(digest(proof) == audit_pin["diagnosis-audit.json"]["sha256"], "HISTORICAL_AUDIT_PIN")
    original_audit = json.loads(proof.read_bytes())
    require(
        original_audit["successful_total_updates"] == 10780
        and original_audit["prefix_history_exactly_equal"] is True
        and original_audit["restore_actual_cuda"] is True
        and original_audit["final_parameters_sha256"] == FINAL,
        "ACTUAL_HISTORY_PROOF",
    )
    receipts = []
    for rank in (0, 1):
        name = f"restore-equality-rank-{rank}.json"
        receipt = json.loads((root / name).read_bytes())
        expected = original_audit["restore_receipts"][rank]
        require(
            digest(root / name) == expected["sha256"]
            and receipt["rank"] == rank
            and receipt["scope"] == "ACTUAL_TARGET_NATIVE_RESTORE_EQUALITY_BEFORE_981"
            and receipt["step"] == 980
            and receipt["next_step"] == 981
            and receipt["native_restore_unchanged"] is True
            and receipt["rng_checked_after_history_replay"] is True
            and receipt["fields"] == expected["fields"],
            "ACTUAL_CUDA_RESTORE_RECEIPT",
        )
        receipts.append(expected)
    require(
        original_audit["shared_capture_pairs"]
        == [
            {"step": 980 * i, "shared_fields": 9, "ranks": 2, "status": "PASS"}
            for i in range(1, 12)
        ],
        "HISTORICAL_SHARED_CAPTURE_PROOF",
    )
    latest = json.loads((root / "out/checkpoints/latest.json").read_bytes())
    require(latest["path"] == "step_00010780.pt" and latest["sha256"] == FULL_SHA, "LATEST_FULL")
    require(
        {k: v for k, v in check(root.parent)["files"].items() if k != "native-audit.json"}
        == before,
        "AUDIT_CHANGED_INPUT",
    )
    return {
        "schema": "MQAR202_FINAL10780_QUALIFIED_NATIVE_V1",
        "updates": 10780,
        "actual_flops": case["actual_flops"],
        "requested_flops": case["requested_flops"],
        "final_parameters_sha256": FINAL,
        "optimizer_states": states,
        "optimizer_constructions": 0,
        "optimizer_updates": 0,
        "checkpoint_rng_restores": 0,
        "original_identity_file_status": "initialized",
        "fabricated_producer_envelopes": False,
        "original_provider_errors": [145778849, 145850896],
        "original_prefix_exact": True,
        "actual_cuda_restore_receipts": receipts,
        "shared_capture_proof_scope": (
            "AUTHENTICATED_EXISTING_FULL_STRUCTURAL_AUDIT_NOT_NEW_RESTORE"
        ),
        "shared_capture_points": original_audit["shared_capture_pairs"],
        "historical_audit_sha256": digest(proof),
        "raw_bindings": before,
        "missing_original_rank_source_service_envelopes": True,
        "missing_original_first_hf_backup": True,
        "acceptance": "QUALIFIED_POLICY_REQUIRED_NOT_ORDINARY_COMPLETE",
    }
