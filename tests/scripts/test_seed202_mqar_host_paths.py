"""Whole derived saved-main binding before declaring the two MQAR data domains."""

import os
import uuid
from pathlib import Path


def test_second_declared_mount_domain_replays_scoped_history_error():
    from seed202_mqar_host_paths import verify_expected_native

    domains = (
        "/kaggle/input/datasets/zimike/ca-kv-1c660aa5f54d88ea9e0c1ce4",
        "/kaggle/input/ca-kv-1c660aa5f54d88ea9e0c1ce4",
    )
    seen = []

    def full_validator(_directory, case):
        seen.append(case["config"]["data"])
        if case["config"]["data"] != domains[1]:
            raise ValueError("SEED202_HISTORY_MISMATCH")
        return {"all_native_predicates_reached": True}

    proof, case = verify_expected_native(
        full_validator,
        Path("unused"),
        {"id": "mqar-k64-classic-seed202", "config": {"data": "ENGINEERING_HOST_ONLY"}},
        domains,
    )
    assert proof == {"all_native_predicates_reached": True}
    assert seen == list(domains) and case["config"]["data"] == domains[1]


def test_derived_mqar_saved_main_reaches_exact_host_data_domains():
    from seed202_pilot.contract import require_authorized_case
    from seed202_pilot.mqar_host_paths import expected_data_paths

    candidate = Path(os.environ["SEED202_FROZEN_CANDIDATE"])
    from seed202_mqar_binding_fixture import engineering_bind

    candidate = engineering_bind(
        candidate, candidate.parent / ("ENGINEERING_domains_" + uuid.uuid4().hex)
    )
    config = candidate / "payload/configs/mqar-k64-classic-seed202.yaml"
    assert expected_data_paths(config, require_authorized_case("mqar-k64-classic-seed202")) == (
        "/kaggle/input/datasets/zimike/ca-kv-1c660aa5f54d88ea9e0c1ce4",
        "/kaggle/input/ca-kv-1c660aa5f54d88ea9e0c1ce4",
    )
