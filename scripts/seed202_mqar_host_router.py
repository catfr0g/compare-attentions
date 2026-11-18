"""Finite dispatch around unchanged legacy host guards; no runtime patching."""

import json
from pathlib import Path

from seed202_pilot.host_resume import ACTIONS, exact, pin, require, verify_extracted

__all__ = [
    "ACTIONS",
    "exact",
    "pin",
    "require",
    "verify_extracted",
    "binding",
    "original",
    "scoped_targets",
    "validate_proof",
    "validate_sources",
    "validate_absence",
]
MODE = "MQAR_CLASSIC_STEP980_V1"


def selected(deployment):
    if deployment.get("mode") == MODE:
        from seed202_pilot import mqar_host_resume

        mqar_host_resume.scoped_targets(deployment)
        return mqar_host_resume
    require(
        deployment.get("case_id") != "mqar-k64-classic-seed202"
        or "resume_input_target" not in deployment,
        "EXPLICIT_MQAR_CONTINUATION_MODE_REQUIRED",
    )
    from seed202_pilot import host_resume

    return host_resume


def binding(candidate, deployment):
    return selected(deployment).binding(candidate, deployment)


def original(candidate):
    plan = json.loads((Path(candidate) / "PLAN-UNAPPROVED.json").read_bytes())
    rows = [row for row in plan["deployments"] if row.get("mode") == MODE]
    require(len(rows) <= 1, "SINGLE_MQAR_CONTINUATION")
    if rows:
        return selected(rows[0]).original(candidate)
    from seed202_pilot import host_resume

    return host_resume.original(candidate)


def scoped_targets(deployment):
    return selected(deployment).scoped_targets(deployment)


def validate_proof(plan, proof, action="publish"):
    return selected(plan["deployment"]).validate_proof(plan, proof, action)


def validate_sources(plan, source, action):
    return selected(plan["deployment"]).validate_sources(plan, source, action)


def validate_absence(plan, proof, action):
    return selected(plan["deployment"]).validate_absence(plan, proof, action)
