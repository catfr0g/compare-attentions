"""MQAR resume worker around the unchanged public trainer and native restore."""

import os

from seed202_pilot.cloud_case import finish as ordinary_finish
from seed202_pilot.cloud_case import normal_return_envelope

__all__ = ["worker", "finish"]


def finish(args, case):
    from seed202_pilot.mqar_authority import validate_args

    validate_args(args)
    return ordinary_finish(args, case)


def worker(args, case):
    from quality_inherit import leaf
    from quality_recovery import RecoveryPublisher, create_json, require
    from quality_services import HFTransport
    from seed202_pilot.cloud_completion import CompletionTracker, TrainingServiceTracker
    from seed202_pilot.cloud_native import qualify_execution
    from seed202_pilot.continuation import install_diagnostic
    from seed202_pilot.mqar_authority import validate_args
    from seed202_pilot.mqar_continuation import (
        CHECKPOINT,
        continuation_tracker,
        load_resume,
        provenance,
    )
    from seed202_pilot.mqar_restore_gate import observe_restore
    from seed202_pilot.mqar_resume_pins import FILES
    from seed202_pilot.mqar_worker_paths import bound_train, training_config

    from compare_attention.train.tracker import DisabledTracker

    rank = int(os.environ.get("RANK", "-1"))
    require(
        int(os.environ.get("WORLD_SIZE", "0")) == 2
        and int(os.environ.get("LOCAL_RANK", "-1")) == rank
        and rank in (0, 1),
        "EXACT_DDP_RANKS",
    )
    root = args.output
    authority = validate_args(args)
    execution_runtime = leaf("runtime").runtime_attestation()
    qualify_execution(execution_runtime)
    resume_payload = load_resume(args.resume_input, case)
    if rank == 0:
        require(not (root / "out").exists(), "FRESH_RESULT_REQUIRED")
        root.mkdir(parents=True, exist_ok=True)
        mounted = root / "mounted-authority"
        mounted.mkdir()
        for name in authority["files"]:
            with (mounted / name).open("xb") as stream:
                stream.write((args.authority_root / name).read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
        for name in FILES:
            if name.startswith("diagnostic/"):
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write((args.resume_input / name).read_bytes())
                    stream.flush()
                    os.fsync(stream.fileno())
        create_json(root / "resume-provenance.json", provenance())
    install_diagnostic(root / "diagnostic")
    if rank == 0:
        values = leaf("services").credentials(args.private, rank=0)
        publisher = RecoveryPublisher(
            root / "recovery.jsonl",
            "seed202-pilot-v1/" + args.nonce + "/" + case["id"] + "/recovery",
            HFTransport(values),
        )
        backup = continuation_tracker(root / "out", case, publisher, resume_payload)
        service = TrainingServiceTracker(
            values,
            "seed202-" + args.nonce + "-" + case["id"],
            updates=case["updates"],
            validations=len(case["checkpoint_steps"]),
        )
        tracker = CompletionTracker(backup, service)
    else:
        tracker = DisabledTracker()
    config = training_config(case)
    train = bound_train(case)
    with observe_restore(resume_payload, root, rank):
        result = train(
            config, tracker=tracker, distributed=True, resume_from=args.resume_input / CHECKPOINT
        )
    import torch

    imported = leaf("paired_worker").imported(args.wheel_destination, args.imported_source_pins)
    create_json(
        root / f"source-rank-{rank}.json",
        {
            "wheel_sha256": "512cb5b073e47deca36d2bfb41f2d183e0711b8817b9a0aa18dbd2e07591b2d3",
            "imported_modules": imported,
            "no_training_method_replacement": True,
        },
    )
    create_json(
        root / f"rank-{rank}.json",
        {
            "identity": result.provenance,
            "rank": rank,
            "selected_device": rank,
            "execution_runtime": execution_runtime,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(rank)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(rank)),
        },
    )
    if rank == 0:
        create_json(
            root / "training-task.json", normal_return_envelope(case["id"], result, service)
        )
        publisher.complete(case["checkpoint_steps"])
