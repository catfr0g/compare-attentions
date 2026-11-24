"""Finite independent RULER Variable Tracking authority, not historical cases."""

import hashlib
import json
from pathlib import Path

CASE_IDS = tuple(
    f"Ruler{family.title()}{seed}" for seed in (202, 303, 404) for family in ("classic", "deltanet")
)
AUTHORIZATION_SHA256 = "2930d72cbe32a38bd9a3fde622acfebccb8f0dc459d4004f1793951214b0e88d"

# Actual preparation02 fully derived configs, canonical JSON excluding only data mount.
CONFIG_PINS = {
    "RulerClassic202": "07f8b47ad42164d2d6c6878cf3d44b5d29d8f8ba488def5c83040e5e6aec7679",
    "RulerDeltanet202": "8a2222a8eba3510b781eee4c631faa887c071c253661b04919891bddefa3b31e",
    "RulerClassic303": "75433fdbd2ee5bc16249f112a47c605ad111bebe0a236d35644e60b7da08b140",
    "RulerDeltanet303": "50bbdd1e4f83b50aad81ebf2880f408736720cf53140cfc01f5afb69bdd6999e",
    "RulerClassic404": "0377bdb66f3248e926032ab7b9e7e90b23696b2a4cf70b84eb7142916805b862",
    "RulerDeltanet404": "1c1666762df399c565f99fe1e440f18df03867cd769bdf970c1e72ce3bb9a037",
}


def require_run(run):
    """Finite source authority cannot silently change any approved scientific field."""
    case_id = run["id"]
    require(case_id in CASE_IDS, "UNIFIED_GENERATION_SCOPE")
    config = {k: v for k, v in run["config"].items() if k != "data"}
    raw = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    require(
        hashlib.sha256(raw).hexdigest() == CONFIG_PINS[case_id], "RULER_FROZEN_SCIENTIFIC_CONFIG"
    )
    row = case_rows()[CASE_IDS.index(case_id)]
    require(
        (run["task"], run["family"], run["seed"]) == (row["task"], row["family"], row["seed"]),
        "RULER_CASE_IDENTITY",
    )


def require(condition, code):
    if not condition:
        raise ValueError(code)


def case_rows():
    return tuple(
        {
            "case_id": case_id,
            "task": "ruler_variable_tracking",
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
            "primary_metric": "generated_chain_exact_match",
            "test_split_permitted": False,
            "case_deadline_seconds": 28800,
        }
        for case_id, (seed, family) in zip(
            CASE_IDS,
            ((seed, family) for seed in (202, 303, 404) for family in ("classic", "deltanet")),
            strict=True,
        )
    )


def operational_case_rows():
    """Future Classic404 policy; accepted cases and Delta404 retain historical limits."""
    return tuple(
        {**row, "case_deadline_seconds": 7200} if row["case_id"] == "RulerClassic404" else row
        for row in case_rows()
    )


def operational_case_specs(spec_type):
    return tuple(spec_type(**row) for row in operational_case_rows())


def case_specs(spec_type):
    return tuple(spec_type(**row) for row in case_rows())


def require_prefix(case_id, accepted_ids):
    require(case_id in CASE_IDS, "UNIFIED_GENERATION_SCOPE")
    index = CASE_IDS.index(case_id)
    require(tuple(accepted_ids) == CASE_IDS[:index], "RULER_ACCEPTED_PREFIX")
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
        value["dataset_task"] == "ruler_variable_tracking"
        and value["scientific_cases"]
        == [{"family": row["family"], "model_seed": row["seed"]} for row in case_rows()]
        and value["protocol"]["world_size"] == 2
        and value["protocol"]["global_batch"] == 64
        and value["protocol"]["requested_algorithmic_training_flops_per_accepted_trajectory"]
        == 5_000_000_000_000_000
        and value["extra_replacement_or_retraining_attempts_authorized"] == 0,
        "RULER_EXACT_PERMISSION",
    )
    return value
