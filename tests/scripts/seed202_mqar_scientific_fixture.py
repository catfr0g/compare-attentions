"""ENGINEERING ONLY: synthetic future metadata, zero model/optimizer/RNG execution."""

import copy
import json
import shutil
from pathlib import Path

from build_seed202_mqar_continuation import original_files


def extended(original, step, case):
    from compare_attention.train.checkpoint import _digest
    from compare_attention.train.stopping import EarlyStopping

    saved = copy.deepcopy(original)
    saved.update(
        step=step,
        flops=step * case["step_flops"],
        cursor=step * 64 % 100000,
        losses=original["losses"] + [1.0] * (step - 980),
    )
    saved["validation_history"] = [
        dict(original["validation_history"][0], flops=point * case["step_flops"])
        for point in case["checkpoint_steps"]
        if point <= step
    ]
    schedule = saved["config"]["schedule"]
    stopper = EarlyStopping(
        schedule["early_stopping_patience"],
        schedule["early_stopping_min_delta"],
        schedule["early_stopping_min_steps"],
    )
    for event in saved["validation_history"]:
        stopper.observe(event["loss"], event["flops"] // case["step_flops"])
    saved["early_stopping"] = stopper.state_dict()
    for state in saved["optimizer"]["state"].values():
        state["step"].fill_(step)
    saved["precision"]["optimizer_steps"] = step
    saved["precision"]["scaler"]["_growth_tracker"] = step % 2000
    saved["integrity_sha256"] = _digest(saved)
    return saved


def produce(root, monkeypatch):
    """Real native saver, eleven SDK HTTP publications and unchanged native consumers."""
    import torch
    from quality_recovery import RecoveryPublisher
    from quality_services import HFTransport
    from quality_tracker import QualityTracker
    from seed202_mqar_http_fixture import HFHTTPFixture
    from seed202_pilot.mqar_host_native import SHARED, validate_full
    from seed202_pilot.storage import canonical_bytes, create_json, digest
    from test_seed202_mqar_checkpoint import native_case

    from compare_attention.train.checkpoint import _digest
    from compare_attention.train.config import (
        GPT2Config,
        KVRetrievalConfig,
        Optim,
        Schedule,
        TrainConfig,
    )
    from compare_attention.train.flops import estimate_step

    original_root, pins = original_files()
    full = torch.load(
        original_root / "out/checkpoints/step_00000980.pt", weights_only=True, map_location="cpu"
    )
    case = native_case(full)
    config = TrainConfig.model_construct(
        data=full["config"]["data"],
        dataset=KVRetrievalConfig.model_validate(full["config"]["dataset"]),
        model=GPT2Config.model_validate(full["config"]["model"]),
        schedule=Schedule.model_validate(full["config"]["schedule"]),
        optim=Optim.model_validate(full["config"]["optim"]),
    )
    rank_cost = estimate_step(
        config.model_copy(
            update={"schedule": config.schedule.model_copy(update={"batch_size": 32})}
        )
    )
    case.update(
        checkpoint_steps=case["validation_steps"],
        actual_flops=10780 * case["step_flops"],
        rank_step_flops=rank_cost.total,
        rank_accounting=rank_cost.to_dict(),
    )
    root = Path(root)
    root.mkdir(parents=True)
    for name in pins:
        if name.startswith("diagnostic/"):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original_root / name, target)
    fixture = HFHTTPFixture(monkeypatch)
    publisher = RecoveryPublisher(
        root / "recovery.jsonl",
        "engineering/mqar/recovery",
        HFTransport({"HF_TOKEN": "hf_ENGINEERING_ONLY", "HF_REPO": "engineering/mqar"}),
    )
    tracker = QualityTracker(root / "out", case, publisher, validate=validate_full)
    tracker.start(config)
    snapshot = root.parent / "ENGINEERING_ONLY_snapshot"
    snapshot.mkdir()
    periodic = {}
    summary = json.loads((original_root / "diagnostic/capture-0001-summary.json").read_bytes())
    final = None
    for index, step in enumerate(case["checkpoint_steps"], 1):
        saved = full if step == 980 else extended(full, step, case)
        if index > 1:
            row = copy.deepcopy(summary)
            row["capture_index"] = index
            capture = {key: saved[key] for key in SHARED}
            for rank in (0, 1):
                path = root / "diagnostic" / f"capture-{index:04d}-rank-{rank}.pt"
                torch.save(capture, path)
                row["ranks"][rank]["field_digests"] = {
                    key: _digest({key: saved[key]}) for key in SHARED
                }
            create_json(root / "diagnostic" / f"capture-{index:04d}-summary.json", row)
        tracker.save_checkpoint(saved, step, 2)
        publication = publisher.records()[-1]
        periodic[publication["revision"]] = publication
        for name in publication["files"]:
            target = snapshot / "periodic" / publication["revision"] / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(fixture.files[publication["prefix"] + "/" + name])
        final = saved
    proof = validate_full(final, case, step=10780)
    identity = json.loads((original_root / "out/run_identity.json").read_bytes())
    identity.pop("failure", None)
    precision = {
        "mode": "amp_float16",
        "parameter_dtype": "float32",
        "autocast_dtype": "float16",
        "grad_scaler_enabled": True,
        "optimizer_steps": 10780,
        "skipped_optimizer_steps": 0,
        "initial_scale": 128.0,
        "final_scale": 128.0,
        "max_scale": 128.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
    }
    identity.update(
        status="completed",
        stop_reason="budget_exhausted",
        final_parameters_sha256=proof["final_sha256"],
        precision=precision,
        recorded_flops=case["actual_flops"],
    )
    for rank, wire in enumerate(identity["distributed"]["ranks"]):
        wire.update(
            compiler=full["distributed"]["ranks"][rank]["identity"]["compiler"],
            precision=precision,
            final_parameters_sha256=proof["final_sha256"],
            status="completed",
            stop_reason="budget_exhausted",
        )
    # Preserve exact first980 producer events. Only future engineering metadata is synthetic.
    events = [
        json.loads(line) for line in (original_root / "out/metrics.jsonl").read_text().splitlines()
    ]
    for step in range(981, 10781):
        events.append({"type": "step", "loss": 1.0, "flops": step * case["step_flops"]})
        if step in case["validation_steps"]:
            events.append(dict(full["validation_history"][0], flops=step * case["step_flops"]))
    quality = {
        "train_loss": sum(final["losses"]) / 10780,
        "val_loss": final["validation_history"][-1]["loss"],
        "val_accuracy": final["validation_history"][-1]["accuracy"],
    }
    events.append(
        {
            "type": "result",
            "steps": 10780,
            "train_flops": case["actual_flops"],
            "config": case["config"],
            "stop_reason": "budget_exhausted",
            "quality": quality,
            "provenance": identity,
        }
    )
    (root / "out/metrics.jsonl").write_bytes(
        b"".join(canonical_bytes(event) + b"\n" for event in events)
    )
    create_json(root / "out/run_identity.json", identity)
    torch.save(final["model"], root / "out/model.pt")
    # Genuine measured execution metadata is immutable input to this engineering fixture.
    reference = Path(
        "E:/HermesArtifacts/compare-attention/ddp-equal-flops-quality-v1/"
        "quality-continuation-parent-v1/collector-v1/files-b843a6afff644b94a4bb5e2aa39420e2/download/"
        "continuation-e3e26f0c46672e3284288fa70d5c44a6/sorting-deltanet-global64"
    )
    for rank in (0, 1):
        record = json.loads((reference / f"rank-{rank}.json").read_bytes())
        local = copy.deepcopy(identity)
        local["runtime"] = full["distributed"]["ranks"][rank]["identity"]["runtime"]
        local["compiler"] = identity["distributed"]["ranks"][rank]["compiler"]
        record.update(identity=local, rank=rank, selected_device=rank)
        create_json(root / f"rank-{rank}.json", record)
    create_json(snapshot / "snapshot.json", {"periodic": periodic})
    rows = publisher.complete(case["checkpoint_steps"])
    create_json(root / "periodic-readback.json", {"publications": rows})
    assert fixture.commits == 11
    assert digest(root / "out/model.pt")
    return case, identity, quality, snapshot, rows, config, fixture
