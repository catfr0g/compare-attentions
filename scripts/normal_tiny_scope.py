"""Finite independent Tiny Shakespeare series; never extends historical cases."""

import hashlib
import json
from pathlib import Path

CASE_IDS = tuple(
    f"Tiny{family.title()}{seed}" for seed in (202, 303, 404) for family in ("classic", "deltanet")
)
# Original user request receipt, not a technical review or provider approval.
AUTHORIZATION_SHA256 = "2f42a3423cc09100355a5af1c70ee516e80ee8cede4be24a8493d2c0c7e09bf8"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def case_rows():
    return tuple(
        {
            "case_id": case_id,
            "task": "tiny_shakespeare_char",
            "family": family,
            "seed": seed,
            "requested_flops": 5_000_000_000_000_000,
            "world_size": 2,
            "global_batch": 64,
            "local_batch": 32,
            "learning_rate": 0.001,
            "lr_schedule": "constant",
            "checkpoint_keep_last": 2,
            "periodic_events": 10,
            "total_full_events": 11,
            "primary_metric": "last_full_budget_validation_character_cross_entropy",
            "test_split_permitted": False,
            "case_deadline_seconds": 28800,
        }
        for case_id, (seed, family) in zip(
            CASE_IDS,
            ((seed, family) for seed in (202, 303, 404) for family in ("classic", "deltanet")),
            strict=True,
        )
    )


def case_specs(spec_type):
    return tuple(spec_type(**row) for row in case_rows())


def require_prefix(case_id, accepted_ids):
    require(case_id in CASE_IDS, "UNIFIED_GENERATION_SCOPE")
    index = CASE_IDS.index(case_id)
    require(tuple(accepted_ids) == CASE_IDS[:index], "TINY_ACCEPTED_PREFIX")
    return index


def permission(authorization):
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
        value["canonical_task"] == "tiny_shakespeare_char"
        and value["model_seeds"] == [202, 303, 404]
        and value["architectures"] == ["classic", "deltanet"]
        and value["world_size"] == 2
        and value["global_batch"] == 64
        and value["requested_training_flops_per_run"] == 5_000_000_000_000_000,
        "TINY_EXACT_PERMISSION",
    )
    return value
