"""Exact read-only host collection/service/composite continuation after tail submission."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mqar_final_tail_host_contract import exact
from mqar_final_tail_host_live import predecessors
from mqar_final_tail_native import require
from mqar_final_tail_runner import create_json


def provider_readback(plan, provider):
    from seed202_pilot.provider import kernel_readback

    root = Path(plan["candidate_root"])
    return kernel_readback(
        provider,
        json.loads((root / "kernel/kernel-metadata.json").read_bytes()),
        (root / "kernel/kernel.py").read_bytes(),
    )


def service_observer(args, plan):
    from mqar_final_tail_package import check

    check(args.science_dir)
    check(args.input)
    require(Path(args.data).resolve() == Path(plan["data_directory"]).resolve(), "EXACT_DATA_ALIAS")
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        in {
            "SYSTEMROOT",
            "PATH",
            "HOME",
            "USERPROFILE",
            "USERNAME",
            "USER",
            "LOGNAME",
            "TMPDIR",
            "TMP",
            "TEMP",
            "UV_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "CUDA_VISIBLE_DEVICES",
        }
    }
    env.update(
        CUDA_VISIBLE_DEVICES="-1",  # Host service observer, never the GPU scoring worker.
        CLEARML_CONFIG_FILE=str(Path(os.environ["TMPDIR"]) / "tail-no-host-config.conf"),
        CLEARML_CACHE_DIR=str(Path(os.environ["TMPDIR"]) / "tail-service-cache"),
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="0",
        CLEARML_OFFLINE_MODE="0",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    command = [
        shutil.which("uv"),
        "run",
        "--offline",
        "--no-project",
        "--no-sync",
        "--python",
        sys.executable,
        "python",
        "-B",
        str(Path(plan["candidate_root"]) / "payload/src/mqar_final_tail_service_cli.py"),
    ]
    for name, value in {
        "release": Path(plan["candidate_root"]) / "payload",
        "science": args.science_dir,
        "input": args.input,
        "work": args.work,
        "private": args.private,
        "nonce": plan["nonce"],
        "output": args.output,
        "data": args.data,
    }.items():
        command += ["--" + name, str(value)]
    from mqar_final_tail_host_actions import recorded
    from mqar_final_tail_host_science import paths as authenticated_paths
    from seed202_pilot.prepared_files import lease

    receipt = recorded(plan, "submit")["source_authority"]["proof"]["preparation"]["receipt"]
    science_paths, _science_pins = authenticated_paths(
        args.science_dir, receipt["science_archive"], plan["science_sha256"]
    )
    input_paths, _input_pins = authenticated_paths(
        args.input,
        Path(plan["candidate_root"]) / "dataset/recovery-input.zip.bin",
        plan["dataset_files"]["recovery-input.zip.bin"]["sha256"],
    )
    with lease(science_paths + input_paths):
        result = subprocess.run(
            command,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=600,
            check=False,
        )
    require(result.returncode == 0, "TAIL_INDEPENDENT_OBSERVER_FAILED")
    return json.loads(Path(args.output).read_bytes())


def execute_observation(args, plan, factory):
    from mqar_final_tail_host_actions import acknowledged

    require(
        acknowledged(plan, "publish") and acknowledged(plan, "submit"), "TAIL_BOTH_ACK_REQUIRED"
    )
    provider = factory(args)
    predecessors(plan, provider)
    before = provider_readback(plan, provider)
    if args.command == "status":
        create_json(args.output, before)
        return {"status": "EXACT_PROVIDER_STATUS", "typed_status": before["typed_status"]}
    require(before["typed_status"] == "COMPLETE", "TAIL_COMPLETE_REQUIRED")
    if args.command == "collect":
        from seed202_pilot.collector import collect
        from seed202_pilot.provider import KernelOutputTransport

        transport = KernelOutputTransport(provider, plan["kernel_target"] + "/1")
        try:
            value = collect(
                transport.list_page,
                transport.download,
                args.destination,
                Path(plan["journal_directory"]) / "collect",
                authority={
                    "scope": "TAIL_EXACT_PROVIDER_COMPLETE",
                    "case_id": "mqar-k64-classic-seed202-final10780-tail",
                    "nonce": plan["nonce"],
                    "plan": plan,
                    "provider": before,
                },
                fresh_provider=lambda: provider_readback(plan, provider),
            )
        finally:
            transport.close()
        create_json(args.index, value)
    elif args.command == "services":
        service_observer(args, plan)
    else:
        from mqar_final_tail_composite import composite
        from mqar_final_tail_package import inventory

        index = json.loads(Path(args.index).read_bytes())
        require(
            exact(index["inventory"], inventory(args.destination))
            and exact(index["provider_before"], before)
            and exact(index["provider_after"], before),
            "TAIL_EXACT_COLLECTED_BYTES",
        )
        candidate = Path(plan["candidate_root"])
        frozen = json.loads((candidate / "plan.json").read_bytes())
        observed = {"ref" if k == "target" else k: v for k, v in before.items()}
        composite(
            candidate / "payload",
            args.result,
            json.loads(Path(args.native).read_bytes()),
            json.loads(Path(args.service_proof).read_bytes()),
            observed,
            json.loads(Path(args.policy_review).read_bytes()),
            frozen,
            args.data,
            args.output,
        )
    require(exact(provider_readback(plan, provider), before), "TAIL_OBSERVER_IDENTITY_DRIFT")
    return {"status": "TAIL_READ_ONLY_CONSUMER_FINISHED", "command": args.command}
