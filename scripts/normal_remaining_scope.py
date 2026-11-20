"""Exact remaining Sorting/MQAR group; pure metadata shared by host and target."""

import hashlib
import json
from pathlib import Path

AUTHORIZATION_SHA256 = "f08d6713c0dd8cc6c213bb040ec67dca13bccfb5d8285cf5100c2d9def339250"
CASE_IDS = tuple(
    f"{task}-{family}-seed{seed}"
    for seed in (303, 404)
    for task in ("sorting", "mqar-k64")
    for family in ("classic", "deltanet")
)


def require(condition, code):
    if not condition:
        raise ValueError(code)


def case_rows():
    """No prospective cases, budget changes, test split or concurrent permission."""
    common = {
        "requested_flops": 5_000_000_000_000_000,
        "world_size": 2,
        "global_batch": 64,
        "local_batch": 32,
        "learning_rate": 0.001,
        "lr_schedule": "constant",
        "checkpoint_keep_last": 2,
        "periodic_events": 10,
        "total_full_events": 11,
        "test_split_permitted": False,
        "case_deadline_seconds": 28800,
    }
    return tuple(
        {
            **common,
            "case_id": case_id,
            "task": task,
            "family": family,
            "seed": seed,
            "primary_metric": (
                "all_1000_val_unrestricted_greedy_eos_last_weights"
                if task == "sorting"
                else "same_position_per_query_token_and_whole_row_exact"
            ),
        }
        for case_id, (seed, task, family) in zip(
            CASE_IDS,
            (
                (seed, task, family)
                for seed in (303, 404)
                for task in ("sorting", "mqar-k64")
                for family in ("classic", "deltanet")
            ),
            strict=True,
        )
    )


def case_specs(spec_type):
    """Instantiate the existing frozen CaseSpec, not a second training contract."""
    return tuple(spec_type(**row) for row in case_rows())


def permission(authorization):
    """Authenticate original human bytes, without manufacturing a new approval."""
    require(
        type(authorization) is dict
        and set(authorization) == {"path", "sha256"}
        and authorization["sha256"] == AUTHORIZATION_SHA256,
        "UNIFIED_GENERATION_HUMAN_PROVENANCE",
    )
    raw = Path(authorization["path"]).read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == AUTHORIZATION_SHA256,
        "UNIFIED_GENERATION_HUMAN_PROVENANCE",
    )
    value = json.loads(raw)
    require(
        value["kind"] == "DIRECT_HUMAN_REMAINING_SEEDS_AUTHORIZATION"
        and value["case_count"] == len(CASE_IDS)
        and tuple(row["case_id"] for row in value["cases"]) == CASE_IDS
        and value["execution_order"] == "sequential in listed order; never concurrent allocations"
        and value["historical_false_future_flags_remain_immutable"] is True,
        "REMAINING_EXACT_PERMISSION",
    )
    return value


def require_prefix(case_id, accepted_ids):
    """Empty new-series prefix is legitimate only for the first authorized case."""
    require(case_id in CASE_IDS, "UNIFIED_GENERATION_SCOPE")
    index = CASE_IDS.index(case_id)
    require(tuple(accepted_ids) == CASE_IDS[:index], "REMAINING_ACCEPTED_PREFIX")
    return index
