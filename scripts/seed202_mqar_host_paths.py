"""Derived source binding around the unchanged native host data-path mapper."""

import hashlib
import json
from pathlib import Path


def verify_expected_native(function, directory, local_case, domains):
    """Keep the inherited finite-domain gate; recognize the scoped history code."""
    from seed202_pilot.host_paths import verify_expected_native as inherited

    def scoped_validator(root, case):
        try:
            return function(root, case)
        except ValueError as error:
            if local_case["id"] == "mqar-k64-classic-seed202" and error.args == (
                "SEED202_HISTORY_MISMATCH",
            ):
                raise ValueError("QUALITY_HISTORY_MISMATCH") from None
            raise

    return inherited(scoped_validator, directory, local_case, domains)


def expected_data_paths(config_path, spec):
    """Reproduce the authenticated saved main before declaring finite data domains."""
    from seed202_pilot import render
    from seed202_pilot.cli import _candidate, _deployment
    from seed202_pilot.contract import require_authorized_case
    from seed202_pilot.host_resume import exact, require
    from seed202_pilot.storage import digest

    require(
        digest(Path(render.__file__))
        == "d076e5d5665922b1c644156c6be42d21e07e892296f1449a64e122fe56c9bffa",
        "AUTHENTIC_PARENT_RENDER_SOURCE",
    )

    approved = require_authorized_case(spec.case_id)
    require(spec == approved, "EXPECTED_PATH_TASK")
    config_path = Path(config_path).resolve()
    candidate = config_path.parents[2]
    require(
        config_path == candidate / "payload/configs" / (approved.case_id + ".yaml"),
        "EXPECTED_PATH_CONFIG_ROOT",
    )
    _descriptor, plan = _candidate(candidate)
    row = _deployment(plan, approved.case_id)
    require(
        plan == json.loads((candidate / "payload/PLAN-UNAPPROVED.json").read_bytes()),
        "EXPECTED_PATH_PLAN",
    )
    runtime = candidate / "payload/inherited/runtime-order.zip.bin"
    if row.get("mode") == "MQAR_CLASSIC_STEP980_V1":
        from seed202_pilot.mqar_host_resume import binding
        from seed202_pilot.mqar_render import render_kernel

        binding(candidate, row)
        source, metadata = render_kernel(
            approved.case_id,
            row["nonce"],
            row["worker_sha256"],
            runtime,
            resume_sha256=row["resume_input_sha256"],
            evidence_sha256=row["evidence_sha256"],
            authority_sha256=row["authority_sha256"],
        )
    else:
        from seed202_pilot.fresh_mode import binding
        from seed202_pilot.render import render_kernel

        binding(candidate, row)
        source, metadata = render_kernel(
            approved.case_id, row["nonce"], row["worker_sha256"], runtime
        )
    deploy = candidate / "payload/deploy" / approved.case_id
    require(
        (row["task"], row["family"], row["seed"]) == (approved.task, approved.family, approved.seed)
        and digest(config_path) == row["config_sha256"]
        and hashlib.sha256(source.encode()).hexdigest() == row["kernel_source_sha256"]
        and (deploy / "kernel/kernel.py").read_bytes() == source.encode()
        and exact(json.loads((deploy / "kernel/kernel-metadata.json").read_bytes()), metadata),
        "EXPECTED_PATH_DERIVED_EXECUTION_BINDING",
    )
    if approved.task == "sorting":
        return ("/kaggle/temp/sorting-science-fcff1b5daf574b70/data/sorting",)
    return (
        "/kaggle/input/datasets/zimike/ca-kv-1c660aa5f54d88ea9e0c1ce4",
        "/kaggle/input/ca-kv-1c660aa5f54d88ea9e0c1ce4",
    )
