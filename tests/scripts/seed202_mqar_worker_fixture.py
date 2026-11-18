"""Run the public resume worker with a no-optimizer native-train body fixture."""

import contextlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace


def worker_return(root, case, identity, quality, config, candidate, monkeypatch):
    import torch
    from build_seed202_mqar_continuation import original_files
    from quality_recovery import create_json
    from seed202_pilot import continuation, mqar_continuation, mqar_restore_gate, mqar_worker_paths
    from seed202_pilot.mqar_checkpoint import validate_original
    from seed202_pilot.mqar_resume_case import worker
    from seed202_pilot.storage import digest

    from compare_attention.train import loop
    from compare_attention.train.config import TrainConfig
    from compare_attention.train.distributed import local_call
    from compare_attention.train.distributed_checkpoint import single_view
    from compare_attention.train.result import Quality, Speed, TrainResult

    original, _pins = original_files()
    saved = validate_original(original / mqar_continuation.CHECKPOINT)
    del saved
    payload = torch.load(
        original / mqar_continuation.CHECKPOINT, weights_only=True, map_location="cpu"
    )
    rank_file = root / "rank-1.json"
    record = json.loads(rank_file.read_bytes())
    shutil.copyfile(rank_file, root.parent / "ENGINEERING_ONLY_rank1_before_worker.json")
    rank_file.unlink()
    request = next(
        row
        for row in json.loads((candidate / "PLAN-UNAPPROVED.json").read_bytes())["deployments"]
        if row["case_id"] == case["id"]
    )
    wheel = candidate / "payload/inherited/training-wheel.whl"
    with zipfile.ZipFile(wheel) as archive:
        pins = {
            name: __import__("hashlib").sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.startswith("compare_attention/")
        }
    args = SimpleNamespace(
        output=root,
        resume_input=original,
        case=case["id"],
        nonce=request["nonce"],
        private=root.parent / "UNREAD_SYNTHETIC_PRIVATE.env",
        science=Path("UNUSED"),
        mqar=Path("UNUSED"),
        authority_root=candidate / "payload/deploy" / case["id"] / "resume-input",
        evidence_sha256=request["evidence_sha256"],
        authority_sha256=request["authority_sha256"],
        wheel_destination=Path(sys.modules["compare_attention"].__file__).resolve().parent.parent,
        imported_source_pins=pins,
    )
    # The mounted view excludes the SDK's local dataset-metadata.json control file.
    mounted = root.parent / "ENGINEERING_ONLY_worker_mount"
    mounted.mkdir()
    for name in request["mounted_authority_files"]:
        shutil.copyfile(args.authority_root / name, mounted / name)
    args.authority_root = mounted
    seen = []

    class Control:
        def resume(self):
            seen.append("native-trace-return")

    def no_step():
        raise AssertionError("NO_OPTIMIZER_IN_ENGINEERING_WORKER")

    @contextlib.contextmanager
    def metadata_observer(saved_payload, directory, rank):
        expected = single_view(saved_payload, rank)
        with mqar_restore_gate.observe_return(
            expected,
            directory,
            rank,
            resume=Control.resume,
            restore=Control.resume,
            step=no_step,
            capture=lambda _control: expected,
            collective=local_call,
            compare=mqar_restore_gate.compare_snapshot,
            write=create_json,
        ):
            yield

    def body(actual, tracker, *, resume_from):
        assert actual is config and tracker.__class__.__name__ == "DisabledTracker"
        assert resume_from == original / mqar_continuation.CHECKPOINT
        Control().resume()
        return TrainResult(
            steps=10780,
            train_flops=case["actual_flops"],
            quality=Quality(**quality),
            speed=Speed(train_tokens_per_second=1, peak_gpu_bytes=1),
            config=config,
            provenance=record["identity"],
            stop_reason="budget_exhausted",
        )

    def dispatch(actual, tracker, resume_from, distributed, runner):
        assert distributed is True and runner is body
        seen.append("public-train")
        return runner(actual, tracker, resume_from=resume_from)

    with monkeypatch.context() as boundary:
        import quality_inherit

        original_leaf = quality_inherit.leaf
        runtime = original_leaf("runtime")
        boundary.setattr(
            quality_inherit,
            "leaf",
            lambda name: runtime if name == "runtime" else original_leaf(name),
        )
        for name, value in (("RANK", "1"), ("LOCAL_RANK", "1"), ("WORLD_SIZE", "2")):
            boundary.setenv(name, value)
        boundary.setattr(runtime, "runtime_attestation", lambda: record["execution_runtime"])
        boundary.setattr(mqar_continuation, "load_resume", lambda *_args: payload)
        boundary.setattr(mqar_restore_gate, "observe_restore", metadata_observer)
        boundary.setattr(continuation, "install_diagnostic", lambda path: seen.append(path.name))
        boundary.setattr(mqar_worker_paths, "training_config", lambda _case: config)
        boundary.setattr(mqar_worker_paths, "bound_train", lambda _case: loop.train)
        boundary.setattr(TrainConfig, "model_validate", lambda _raw: config)
        boundary.setattr(loop, "_train", body)
        boundary.setattr(loop, "run_training", dispatch)
        boundary.setattr(
            torch.cuda, "max_memory_allocated", lambda _rank: record["peak_allocated_bytes"]
        )
        boundary.setattr(
            torch.cuda, "max_memory_reserved", lambda _rank: record["peak_reserved_bytes"]
        )
        worker(args, case)
    assert seen == ["diagnostic", "public-train", "native-trace-return"]
    assert json.loads(rank_file.read_bytes()) == record
    source = json.loads((root / "source-rank-1.json").read_bytes())
    assert source["wheel_sha256"] == digest(wheel)
    assert (
        source["imported_modules"]["compare_attention.train.loop"]["member"]
        == "compare_attention/train/loop.py"
    )
    return args
