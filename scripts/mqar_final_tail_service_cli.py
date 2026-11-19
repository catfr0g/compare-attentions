"""Fresh independent destination GETs or explicitly authorized service-only salvage."""

import os
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

from mqar_final_tail_native import canonical, require  # noqa: E402


def reconcile_hf(services, args):
    """GET-only salvage of the SAME final operation, never scoring or ClearML completion."""
    folder = args.work / "result-services"
    journal = folder / "hf.jsonl"
    require(journal.is_file(), "EXISTING_FINAL_JOURNAL_REQUIRED")
    import re

    from mqar_final_tail_authority import verify as verify_authority
    from mqar_final_tail_report import load_report, no_optimization
    from mqar_final_tail_tracking import FILES, publish_final
    from quality_services import HFTransport

    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    require(
        bool(rows) and rows[-1]["state"] in {"pending", "unknown", "verified"},
        "EXISTING_FINAL_JOURNAL_REQUIRED",
    )
    require(
        type(args.hf_revision) is str and re.fullmatch("[0-9a-f]{40}", args.hf_revision),
        "EXACT_UNKNOWN_ACK_IMMUTABLE_REVISION_REQUIRED",
    )
    output = args.work / "result"
    require({p.name for p in output.iterdir()} == FILES, "EXACT_RESULT_FILES")
    load_report(output, args.data)
    no_optimization(json.loads((output / "no-optimization.json").read_bytes()))
    verify_authority(
        json.loads((output / "execution-authority.json").read_bytes()),
        args.release,
        args.nonce,
        "zimike/mqar202-final10780-tail-" + args.nonce[:16],
        output / "historical-hf-proof.json",
    )
    from mqar_final_tail_native import digest

    source_sha = digest(journal)
    reconciliation = args.output.parent / (args.output.stem + "-hf-get-only")
    reconciliation.mkdir(parents=True, exist_ok=False)
    with (reconciliation / "hf.jsonl").open("xb") as stream:
        stream.write(journal.read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    result = publish_final(
        reconciliation,
        "mqar202-final10780-tail-recovery/" + args.nonce + "/FINAL",
        {n: output / n for n in sorted(FILES)},
        HFTransport(services.values),
        revision=args.hf_revision,
    )
    require(digest(journal) == source_sha, "ORIGINAL_UNKNOWN_ACK_JOURNAL_PRESERVED")
    return {
        "schema": "MQAR202_FINAL_HF_GET_ONLY_RECONCILIATION_V1",
        "remote_mutations": 0,
        "original_journal_sha256": source_sha,
        "reconciliation_journal_sha256": digest(reconciliation / "hf.jsonl"),
        "hf": result,
        "scientific_acceptance": False,
        "clearml_completion": "NOT_ASSERTED",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("release", "science", "input", "work", "private", "output", "data"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--salvage-decision", type=Path)
    parser.add_argument("--hf-revision")
    parser.add_argument("--hf-reconcile-only", action="store_true")
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    require(
        os.environ.get("HF_HUB_OFFLINE") == "0" and os.environ.get("CLEARML_OFFLINE_MODE") == "0",
        "FRESH_ONLINE_MODE_REQUIRED_BEFORE_IMPORTS",
    )
    from mqar_final_tail_package import bind_wheel, check, imported_sources

    check(args.release)
    check(args.science)
    check(args.input)
    from mqar_final_tail_data import verify_files

    verify_files(args.data)
    if args.salvage_decision:
        from mqar_final_tail_cli import decision
        from mqar_final_tail_runtime import guard as runtime_guard

        runtime_guard()
        decision(args.salvage_decision, args.release, args.nonce)
    wheel = Path(os.environ["TMPDIR"]) / ("tail-service-wheel-" + uuid.uuid4().hex)
    pins = bind_wheel(args.science / "compare_attention-0.1.0-py3-none-any.whl", wheel)
    sys.path[1:1] = [
        str(args.release / "seed202"),
        str(args.science / "operator"),
        str(args.science / "src"),
        str(args.science / "tools"),
        str(args.science / "operator/inherited"),
    ]
    with (
        open(os.devnull, "w") as sink,
        contextlib.redirect_stdout(sink),
        contextlib.redirect_stderr(sink),
    ):
        from mqar_final_tail_publish import finish
        from mqar_final_tail_runner import create_json
        from mqar_final_tail_tracking import RecoveryServices

        services = RecoveryServices.from_private(
            args.private, args.nonce, args.input / "original", args.release
        )
        identifier = json.loads((args.work / "result/inference-task.json").read_bytes())["task_id"]
        services.attach(identifier)
        services.historical_proof = json.loads(
            (args.work / "result/historical-hf-proof.json").read_bytes()
        )
        if args.hf_reconcile_only:
            require(args.salvage_decision is None, "HF_RECONCILIATION_IS_GET_ONLY")
            result = reconcile_hf(services, args)
            imported_sources(wheel, pins)
            create_json(args.output, result)
            return 0
        recorded = json.loads((args.work / "result/recovery-provenance.json").read_bytes())
        result = finish(
            services,
            args.work / "result",
            args.data,
            args.release,
            args.work / "native-audit.json",
            recorded["topology"],
            revision=args.hf_revision,
            read_only=args.salvage_decision is None,
        )
        expected = json.loads((args.work / "result-services/service-readback.json").read_bytes())
        require(canonical(result) == canonical(expected), "FRESH_SERVICES_DIFFER")
        imported_sources(wheel, pins)
        create_json(
            args.output,
            {
                "scope": "FRESH_EXACT_PRIVATE_REVISION_AND_DISTINCT_INFERENCE_ARTIFACT_GETS",
                "services": result,
            },
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        name = type(error).__name__
        print(
            json.dumps(
                {
                    "status": "MQAR_SERVICE_STOPPED",
                    "exception_class": name
                    if name
                    in {
                        "ValueError",
                        "OSError",
                        "RuntimeError",
                        "FileNotFoundError",
                        "FileExistsError",
                        "PermissionError",
                        "KeyError",
                        "TypeError",
                    }
                    else "Exception",
                }
            )
        )
        raise SystemExit(1) from None
