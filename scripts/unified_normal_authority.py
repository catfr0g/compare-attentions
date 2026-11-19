"""Explicit post-review NORMAL source authority; no review or decision is created here."""

import hashlib
import json
from pathlib import Path

SCHEMA = "UNIFIED_NORMAL_EXECUTION_AUTHORITY_V1"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def read(path):
    return json.loads(Path(path).read_bytes())


def pin(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def validate(plan_path, review_path, decision_path):
    plan, review, decision = map(read, (plan_path, review_path, decision_path))
    require(
        len(plan["cases"]) == 1 and plan["engineering_only"] is False,
        "UNIFIED_AUTHORITY_SCIENTIFIC_PLAN",
    )
    case_id = next(iter(plan["cases"]))
    bound = {
        "case_id": case_id,
        "source_identity_sha256": plan["source_identity_sha256"],
        "plan_sha256": pin(plan_path)["sha256"],
    }
    require(
        type(review) is dict and set(review) == set(bound) | {"status", "independent", "reviewer"},
        "UNIFIED_AUTHORITY_REVIEW_SCHEMA",
    )
    require(
        all(review[k] == v for k, v in bound.items())
        and review["status"] == "PASS"
        and review["independent"] is True
        and type(review["reviewer"]) is str
        and bool(review["reviewer"])
        and review["reviewer"] != "codex-single-implementation-owner",
        "UNIFIED_AUTHORITY_INDEPENDENT_REVIEW",
    )
    require(
        type(decision) is dict
        and set(decision) == set(bound) | {"action", "approved", "review_sha256"},
        "UNIFIED_AUTHORITY_DECISION_SCHEMA",
    )
    require(
        all(decision[k] == v for k, v in bound.items())
        and decision["action"] == "NORMAL_FRESH_TRAINING"
        and decision["approved"] is True
        and decision["review_sha256"] == pin(review_path)["sha256"],
        "UNIFIED_AUTHORITY_EXECUTION_DECISION",
    )
    return {
        "schema": SCHEMA,
        "status": "BOUND_EXTERNAL_REVIEW_AND_DECISION",
        "normal_plan_sha256": bound["plan_sha256"],
        "files": {
            "independent-review.json": pin(review_path),
            "execution-decision.json": pin(decision_path),
        },
    }


def binding(candidate):
    payload = Path(candidate) / "payload"
    path = payload / "unified/normal-authority.json"
    value = read(path)
    require(value["schema"] == SCHEMA, "UNIFIED_AUTHORITY_SCHEMA")
    if value["status"] == "BOUND_EXTERNAL_REVIEW_AND_DECISION":
        expected = validate(
            payload / "unified/normal-plan.json",
            payload / "unified/authority/independent-review.json",
            payload / "unified/authority/execution-decision.json",
        )
        require(value == expected, "UNIFIED_AUTHORITY_EXACT_BINDING")
    else:
        require(
            value
            == {"schema": SCHEMA, "status": "UNBOUND_REVIEW_AND_DECISION_REQUIRED", "files": {}},
            "UNIFIED_AUTHORITY_UNBOUND_SCHEMA",
        )
    return {
        "normal_authority_sha256": pin(path)["sha256"],
        "normal_authority_bound": value["status"] == "BOUND_EXTERNAL_REVIEW_AND_DECISION",
    }


def before_auth(args):
    """An unbound review candidate may be inspected, but never publish or allocate."""
    if args.command in {"publish", "submit"}:
        require(
            binding(args.candidate_dir)["normal_authority_bound"],
            "UNIFIED_NORMAL_EXECUTION_AUTHORITY_REQUIRED",
        )
