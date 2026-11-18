"""Narrow external native-schema repair; scientific wheel and saved bytes stay unchanged."""

from __future__ import annotations

import hashlib

NATIVE_SHA256 = "bb7f9014967d48ebb4ef097061bf583e9e75266c20d7f5d7109610c7dfba31ac"


def task_matches(config, spec):
    """Resolve the declared task from the frozen policy, not an optional saved key."""
    from seed202_pilot.cloud_contract import canonical
    from seed202_pilot.contract import MQAR_DATA
    from seed202_pilot.matrix import config_payload

    declared = config_payload(spec)["dataset_task"]
    if spec.task == "sorting":
        # Preserve the Sorting predicate exactly; do not retrofit the KV schema.
        return declared == "sorting" and config["dataset"]["task"] == "sorting"
    if spec.task != "mqar-k64" or declared != "kv_retrieval":
        return False
    from compare_attention.data.config import KVRetrievalConfig

    saved = config["dataset"]
    if type(saved) is not dict or set(saved) != set(KVRetrievalConfig.model_fields):
        return False
    parsed = KVRetrievalConfig.model_validate(saved, strict=True).model_dump(mode="json")
    return canonical(saved) == canonical(parsed) == canonical(MQAR_DATA["spec"])


def repair_native(raw):
    """Accept only the authenticated V7 function source and replace its faulty predicate."""
    if hashlib.sha256(raw).hexdigest() != NATIVE_SHA256:
        raise ValueError("EXACT_V7_NATIVE_SOURCE_REQUIRED")
    before = (
        '    spec = require_authorized_case(case["id"])\n'
        "    _require(\n"
        '        (case["task"], case["family"], case["seed"]) '
        "== (spec.task, spec.family, spec.seed)\n"
        '        and config["dataset"]["task"] == '
        '("sorting" if spec.task == "sorting" else "kv_retrieval"),'
    )
    after = (
        "    from seed202_pilot.native_schema import task_matches\n\n"
        '    spec = require_authorized_case(case["id"])\n'
        "    _require(\n"
        '        (case["task"], case["family"], case["seed"]) '
        "== (spec.task, spec.family, spec.seed)\n"
        "        and task_matches(config, spec),"
    )
    text = raw.decode("utf-8")
    if text.count(before) != 1:
        raise ValueError("EXACT_V7_NATIVE_PREDICATE_REQUIRED")
    return text.replace(before, after).encode("utf-8")
