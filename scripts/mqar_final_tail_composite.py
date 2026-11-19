"""Whole-object MQAR qualified acceptance, preserving original ERROR + resume ERROR."""

import json
from pathlib import Path

from mqar_final_tail_native import FINAL, FULL_SHA, MODEL_SHA, canonical, digest, require
from mqar_final_tail_runner import create_json

POLICY = "RECOVERED_TAIL_WITH_ORIGINAL_TELEMETRY_GAP"


def require_review(review, candidate_sha256):
    require(
        type(review) is dict
        and set(review) == {"status", "independent", "scope", "candidate_sha256", "policy"}
        and review["status"] == "PASS"
        and review["independent"] is True
        and review["scope"] == "MQAR202_FINAL10780_TAIL_RECOVERY"
        and review["candidate_sha256"] == candidate_sha256
        and review["policy"] == POLICY,
        "INDEPENDENT_MQAR_QUALIFIED_REVIEW_REQUIRED",
    )


def persist(path, value):
    path = Path(path)
    if path.exists():
        require(
            canonical(json.loads(path.read_bytes())) == canonical(value), "COMPOSITE_REPLAY_DRIFT"
        )
    else:
        create_json(path, value)
    require(
        canonical(json.loads(path.read_bytes())) == canonical(value), "WHOLE_COMPOSITE_READBACK"
    )
    return value


def composite(release, result, native, services, provider, review, plan, data, output):
    from mqar_final_tail_historical import verify as historical
    from mqar_final_tail_package import check
    from mqar_final_tail_report import load_report, no_optimization, quality_metrics
    from mqar_final_tail_tracking import FILES, TRAINING_TASK, binding, verify_unsampled_events

    release, result = Path(release), Path(result)
    check(release)
    require_review(review, digest(release / "manifest.json"))
    expected_native = json.loads((release / "evidence/native-audit.json").read_bytes())
    require(canonical(native) == canonical(expected_native), "EXACT_NATIVE_WHOLE_OBJECT")
    require(
        native["updates"] == 10780
        and native["actual_flops"] == 5000372724695040
        and native["final_parameters_sha256"] == FINAL
        and native["original_identity_file_status"] == "initialized"
        and native["fabricated_producer_envelopes"] is False,
        "GENUINE_NATIVE_QUALIFICATION",
    )
    require(
        provider["typed_status"] == "COMPLETE"
        and type(provider["head"]) is int
        and type(provider["version"]) is int
        and provider["head"] == provider["version"] == 1
        and type(provider["numeric_id"]) is int
        and provider["numeric_id"] not in {145778849, 145850896}
        and provider["private"] is True
        and provider["ref"] == plan["target"]
        and provider["source_sha256"] == plan["kernel_sha256"],
        "DISTINCT_RECOVERY_COMPLETE_IDENTITY",
    )
    require(
        plan["original_errors"] == [145778849, 145850896]
        and plan["optimizer_updates_permitted"] == 0,
        "MQAR_RECOVERY_SCOPE",
    )
    require(check(release)["files"] == plan["files"], "ALL_REVIEWED_RELEASE_BYTES")
    require({p.name for p in result.iterdir()} == FILES, "WHOLE_RESULT_FILESET")
    report, metrics = load_report(result, data)
    proof = json.loads((result / "no-optimization.json").read_bytes())
    no_optimization(proof)
    hf11 = json.loads((result / "historical-hf-proof.json").read_bytes())
    old = historical(release, hf11)
    from mqar_final_tail_authority import verify as verify_authority

    authority = verify_authority(
        json.loads((result / "execution-authority.json").read_bytes()),
        release,
        plan["nonce"],
        plan["target"],
        result / "historical-hf-proof.json",
    )
    require(authority["independent_review"] == review, "ACTUAL_POLICY_REVIEW_BINDING")
    require(
        type(services) is dict
        and set(services) == {"scope", "services"}
        and services["scope"]
        == "FRESH_EXACT_PRIVATE_REVISION_AND_DISTINCT_INFERENCE_ARTIFACT_GETS",
        "FRESH_INDEPENDENT_CPU_SERVICE_OBSERVER_REQUIRED",
    )
    observed = services["services"]
    require(
        set(observed)
        == {"hf", "clearml", "validation", "native_audit_sha256", "qualified_composite_pending"},
        "EXACT_SERVICE_ENVELOPE",
    )
    tracking, hf = observed["clearml"], observed["hf"]
    journal = result.parent / (result.name + "-services") / "hf.jsonl"
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    require(
        bool(rows) and canonical(rows[-1]) == canonical(hf) and rows[-1]["state"] == "verified",
        "FINAL_ORIGINAL_JOURNAL_REVISION_BINDING",
    )
    require(
        observed["validation"]
        == {
            "correct": metrics["correct"],
            "rows": 1000,
            "query_accuracy": metrics["accuracy"],
            "supervised_count": metrics["supervised_count"],
            "whole_example_exact_match": metrics["whole_example_exact_match"],
        }
        and observed["qualified_composite_pending"] is True,
        "COMPLETE_SERVICE_VALIDATION_BINDING",
    )
    expected_metrics = quality_metrics(report)
    require(tracking["metrics"] == expected_metrics, "TASK_CORRECT_SCIENTIFIC_METRICS")
    events = tracking["unsampled"]
    normalized = verify_unsampled_events(
        {k: v for k, v in events.items() if k != "raw_pages"}, expected_metrics
    )
    require(all(events[k] == normalized[k] for k in normalized), "WHOLE_UNSAMPLED_EVENTS")
    require(
        type(events.get("raw_pages")) is list and bool(events["raw_pages"]), "RAW_PAGES_RETAINED"
    )
    raw = [row for page in events["raw_pages"] for row in page["events"]]
    require(
        all(row.get("task", tracking["task_id"]) == tracking["task_id"] for row in raw),
        "EXACT_INFERENCE_EVENT_TASK_IDENTITY",
    )
    require(
        all(
            type(p["total"]) is int and p["total"] == len(expected_metrics)
            for p in events["raw_pages"]
        )
        and verify_unsampled_events({"events": raw, "total": len(raw)}, expected_metrics)
        == normalized,
        "UNSAMPLED_RAW_PAGE_CLOSURE",
    )
    identifier = json.loads((result / "inference-task.json").read_bytes())["task_id"]
    require(
        tracking["status"] == "completed"
        and tracking["type"] == "inference"
        and tracking["binding"] == binding(plan["nonce"])
        and tracking["task_id"] == identifier != TRAINING_TASK,
        "DISTINCT_INFERENCE_TASK",
    )
    require(
        set(hf["files"]) == set(tracking["artifacts"]) == FILES
        and hf["state"] == "verified"
        and hf["prefix"] == "mqar202-final10780-tail-recovery/" + plan["nonce"] + "/FINAL",
        "EXACT_FINAL_NAMESPACE",
    )
    for name in FILES:
        require(
            hf["files"][name]
            == {"bytes": (result / name).stat().st_size, "sha256": digest(result / name)}
            and tracking["artifacts"][name] == digest(result / name),
            "ALL_DESTINATION_FILE_BYTES",
        )
    receipt = json.loads((result / "source-model-receipt.json").read_bytes())
    provenance = json.loads((result / "recovery-provenance.json").read_bytes())
    require(
        receipt["native_audit_sha256"]
        == observed["native_audit_sha256"]
        == provenance["native_audit_sha256"]
        and receipt["historical_source_and_service_evidence"] == old
        and receipt["model_sha256"] == MODEL_SHA
        and receipt["full_sha256"] == FULL_SHA
        and receipt["original_provider_errors"] == [145778849, 145850896]
        and provenance["binding"] == binding(plan["nonce"])
        and provenance["execution_authority_sha256"] == digest(result / "execution-authority.json")
        and provenance["release_manifest_sha256"] == digest(release / "manifest.json"),
        "FINAL_SOURCE_NATIVE_PROVENANCE",
    )
    return persist(
        output,
        {
            "schema": "MQAR202_FINAL10780_QUALIFIED_COMPOSITE_V1",
            "qualification": (
                "ORIGINAL_ERROR_RESUME_ERROR_INITIALIZED_IDENTITY_MISSING_ENVELOPES_PRESERVED"
            ),
            "original_provider_errors": old["original_error_jobs"],
            "original_telemetry": old,
            "native": native,
            "recovery_provider": provider,
            "services": observed,
            "weight_binding": {"parameters": FINAL, "model": MODEL_SHA, "full": FULL_SHA},
            "public_loader_provenance": None,
            "validation": metrics,
            "no_optimization": proof,
            "review": review,
            "execution_authority": authority,
            "plan": plan,
            "later_pilots_authorized": False,
        },
    )
