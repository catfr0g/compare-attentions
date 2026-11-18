"""Exact original evidence and cloud-only native MQAR continuation boundary."""

from pathlib import Path

CHECKPOINT = "out/checkpoints/step_00000980.pt"
CASE = "mqar-k64-classic-seed202"
SOURCE_SHA = "19ff980d566691e4fde21130f3fff90400e8773011206b86a09cb21e5661ff6a"


def provenance():
    from seed202_pilot.mqar_resume_pins import FILES

    return {
        "original_kernel": "zimike/seed202-pilot-786acf2c5ec864ce/1",
        "original_numeric_id": 145778849,
        "original_status": "ERROR",
        "original_source_sha256": SOURCE_SHA,
        "original_hf_first_backup": "ABSENT",
        "resume_step": 980,
        "next_live_step": 981,
        "historical_replay_steps": 980,
        "historical_replay_validations": 1,
        "remaining_updates": 9800,
        "total_updates": 10780,
        "total_budget_unchanged": 5_000_000_000_000_000,
        "first_continuation_publication": "SALVAGED_BOUNDARY_NOT_ORIGINAL_BACKUP",
        "original_files": FILES,
    }


def load_resume(directory, case):
    """Authenticate every input byte before native parsing and target validation."""
    import torch
    from seed202_pilot.cloud_native import validate_full, verify_shared
    from seed202_pilot.mqar_checkpoint import validate_original
    from seed202_pilot.mqar_resume_pins import FILES
    from seed202_pilot.storage import digest

    from compare_attention.train.checkpoint import load_checkpoint

    root = Path(directory)
    if case["id"] != CASE or case["updates"] != 10780:
        raise ValueError("EXACT_MQAR_CONTINUATION_CASE")
    names = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if names != set(FILES):
        raise ValueError("EXACT_MQAR_ORIGINAL_INVENTORY")
    for name, pin in FILES.items():
        path = root / name
        if (
            path.is_symlink()
            or path.stat().st_size != pin["bytes"]
            or digest(path) != pin["sha256"]
        ):
            raise ValueError("EXACT_MQAR_ORIGINAL_BYTES")
    validate_original(root / CHECKPOINT)
    if not torch.cuda.is_available():
        raise ValueError("CLOUD_TARGET_VALIDATION_ONLY")
    payload = load_checkpoint(root / CHECKPOINT)
    validate_full(payload, case, step=980)
    verify_shared(root / "diagnostic", [980], case)
    if payload["flops"] != 454579338608640 or payload["flops"] >= case["requested_flops"]:
        raise ValueError("EXACT_MQAR_REMAINING_BUDGET")
    return payload


def continuation_tracker(directory, case, publisher, payload):
    """Publish preserved980 into a new namespace; native saver owns keep2 pruning."""
    from quality_tracker import QualityTracker
    from seed202_pilot.cloud_native import validate_full

    if case["id"] != CASE or payload["step"] != 980:
        raise ValueError("EXACT_MQAR_TRACKER_BOUNDARY")
    validate_full(payload, case, step=980)

    class ContinuationTracker(QualityTracker):
        def start(self, config):
            super().start(config)
            self.save_checkpoint(payload, 980, 2)

    return ContinuationTracker(directory, case, publisher, validate=validate_full)
