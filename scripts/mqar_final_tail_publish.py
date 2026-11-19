"""Service-only finish, separate from scoring and original training tasks."""

import json
from pathlib import Path

from mqar_final_tail_native import FULL_SHA, MODEL_SHA, canonical, digest, require
from mqar_final_tail_report import load_report, no_optimization
from mqar_final_tail_runner import create_json
from mqar_final_tail_tracking import FILES, publish_final


def finish(services, output, data, release, native, topology, *, revision=None, read_only=False):
    from quality_services import HFTransport

    output, release = Path(output), Path(release)
    report, metrics = load_report(output, data)
    proof = json.loads((output / "no-optimization.json").read_bytes())
    no_optimization(proof)
    source = json.loads((output / "scientific-source.json").read_bytes())
    require(
        set(source) == {"scope", "wheel_sha256", "operator_manifest_sha256", "imported_modules"}
        and source["scope"] == "ACTUAL_SELECTED_WORKER_AUTHENTICATED_IMPORT_ALIASES"
        and source["wheel_sha256"]
        == "512cb5b073e47deca36d2bfb41f2d183e0711b8817b9a0aa18dbd2e07591b2d3"
        and source["operator_manifest_sha256"] == digest(release / "manifest.json"),
        "SCIENTIFIC_SOURCE_BINDING",
    )
    import compare_attention

    wheel_root = Path(compare_attention.__file__).resolve().parent.parent
    require(
        {"compare_attention.cli.evaluate", "compare_attention.evaluation_checkpoint"}
        <= set(source["imported_modules"]),
        "PUBLIC_SCIENTIFIC_ALIAS_PROOF",
    )
    for module, expected in source["imported_modules"].items():
        require(
            module == "compare_attention" or module.startswith("compare_attention."),
            "SCIENTIFIC_ALIAS_NAME",
        )
        path = wheel_root.joinpath(*module.split("."))
        path = (
            path.with_suffix(".py") if path.with_suffix(".py").is_file() else path / "__init__.py"
        )
        require(digest(path) == expected, "SCIENTIFIC_ALIAS_BYTES")
    from mqar_final_tail_authority import verify as verify_authority
    from mqar_final_tail_historical import verify as verify_historical

    authority = json.loads((output / "execution-authority.json").read_bytes())
    verify_authority(
        authority,
        release,
        services.nonce,
        "zimike/mqar202-final10780-tail-" + services.nonce[:16],
        output / "historical-hf-proof.json",
    )
    provenance = {
        "binding": services.bound,
        "topology": topology,
        "release_manifest_sha256": digest(release / "manifest.json"),
        "native_audit_sha256": digest(native),
        "scientific_source_sha256": digest(output / "scientific-source.json"),
        "execution_authority_sha256": digest(output / "execution-authority.json"),
        "qualification": (
            "RECOVERED_MQAR_CLASSIC_FINAL10780_TAIL_ERRORS_AND_MISSING_PRODUCER_ENVELOPES_PRESERVED"
        ),
    }
    receipt = {
        "model_sha256": MODEL_SHA,
        "full_sha256": FULL_SHA,
        "native_audit_sha256": digest(native),
        "original_provider_errors": [145778849, 145850896],
        "historical_source_and_service_evidence": verify_historical(
            release, services.historical_proof
        ),
        "original_training_task": services.bound["source_training_task_id"],
        "original_telemetry_missing_train": 0,
        "original_telemetry_missing_val": 0,
        "original_final_producer_envelopes": "ABSENT_NOT_FABRICATED",
    }
    for name, value in (
        ("recovery-provenance.json", provenance),
        ("source-model-receipt.json", receipt),
    ):
        if (output / name).exists():
            require(
                canonical(json.loads((output / name).read_bytes())) == canonical(value),
                "SALVAGE_SOURCE_DRIFT",
            )
        else:
            require(not read_only, "READ_ONLY_REQUIRES_EXISTING_ARTIFACTS")
            create_json(output / name, value)
    require({path.name for path in output.iterdir()} == FILES, "EXACT_RESULT_FILES")
    folder = output.parent / (output.name + "-services")
    if read_only:
        require((folder / "hf.jsonl").is_file(), "GET_ONLY_FINAL_JOURNAL_REQUIRED")
        records = [json.loads(line) for line in (folder / "hf.jsonl").read_text().splitlines()]
        require(
            bool(records) and records[-1]["state"] == "verified", "GET_ONLY_VERIFIED_FINAL_REQUIRED"
        )
    from mqar_final_tail_service_errors import instrument

    transport = instrument(HFTransport(services.values), output.parent)
    hf = publish_final(
        folder,
        "mqar202-final10780-tail-recovery/" + services.nonce + "/FINAL",
        {name: output / name for name in sorted(FILES)},
        transport,
        revision=revision,
    )
    tracking = services.complete_tracking(output, report, read_only=read_only)
    result = {
        "hf": hf,
        "clearml": tracking,
        "validation": {
            "correct": metrics["correct"],
            "rows": 1000,
            "query_accuracy": metrics["accuracy"],
            "supervised_count": metrics["supervised_count"],
            "whole_example_exact_match": metrics["whole_example_exact_match"],
        },
        "native_audit_sha256": digest(native),
        "qualified_composite_pending": True,
    }
    path = folder / "service-readback.json"
    if path.exists():
        require(
            canonical(json.loads(path.read_bytes())) == canonical(result), "SERVICE_READBACK_DRIFT"
        )
    else:
        require(not read_only, "GET_ONLY_SERVICE_RECEIPT_REQUIRED")
        create_json(path, result)
    return result
