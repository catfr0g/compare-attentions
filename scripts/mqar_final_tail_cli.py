"""Explicitly reviewed MQAR inference-only worker. No training route exists."""

import os
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

from mqar_final_tail_native import canonical, digest, require  # noqa: E402
from mqar_final_tail_package import check  # noqa: E402


def decision(path, release, nonce):
    value = json.loads(Path(path).read_bytes())
    review_path = Path(path).with_name("independent-review.json")
    review = json.loads(review_path.read_bytes())
    require(
        set(value)
        == {
            "status",
            "independent",
            "scope",
            "manifest_sha256",
            "execution_authorized",
            "telemetry_policy",
            "independent_review_sha256",
            "historical_hf_proof_sha256",
            "nonce",
            "kernel_target",
            "allocations_permitted",
        }
        and value["status"] == "PASS"
        and value["independent"] is True
        and value["scope"] == "MQAR202_FINAL10780_INFERENCE_PUBLICATION_ONLY"
        and value["manifest_sha256"] == digest(Path(release) / "manifest.json")
        and value["execution_authorized"] is True
        and value["nonce"] == nonce
        and value["kernel_target"] == "zimike/mqar202-final10780-tail-" + nonce[:16]
        and type(value["allocations_permitted"]) is int
        and value["allocations_permitted"] == 1
        and value["telemetry_policy"]
        == "EXPLICITLY_ACCEPT_ORIGINAL_GAP_WITH_QUALIFIED_DERIVED_NATIVE_EVIDENCE"
        and value["independent_review_sha256"] == digest(review_path)
        and value["historical_hf_proof_sha256"] == digest(Path(path).with_name("hf11-proof.json")),
        "NEW_INDEPENDENT_EXECUTION_DECISION_REQUIRED",
    )
    require(
        set(review) == {"status", "independent", "scope", "candidate_sha256", "policy"}
        and review["status"] == "PASS"
        and review["independent"] is True
        and review["scope"] == "MQAR202_FINAL10780_TAIL_RECOVERY"
        and review["candidate_sha256"] == value["manifest_sha256"]
        and review["policy"] == "RECOVERED_TAIL_WITH_ORIGINAL_TELEMETRY_GAP",
        "EXACT_INDEPENDENT_QUALIFIED_REVIEW",
    )
    return value


def run(args):
    from mqar_final_tail_runtime import guard

    args.failure_phase = "worker_runtime"
    topology = guard()  # Actual hardware and prepared library pins before credentials/model.
    args.failure_phase = "qualification"
    for root in (args.release, args.input, args.science):
        check(root)
    decision(args.decision, args.release, args.nonce)
    from mqar_final_tail_data import verify_files

    verify_files(args.data)
    require(
        os.environ.get("HF_HUB_OFFLINE") == "0" and os.environ.get("CLEARML_OFFLINE_MODE") == "0",
        "FRESH_ONLINE_REQUIRED",
    )
    from mqar_final_tail_package import bind_wheel, imported_sources

    wheel_root = args.work / "wheel"
    pins = bind_wheel(args.science / "compare_attention-0.1.0-py3-none-any.whl", wheel_root)
    sys.path[1:1] = [
        str(args.release / "seed202"),
        *[str(args.science / n) for n in ("operator", "src", "tools", "operator/inherited")],
    ]
    from mqar_final_tail_native import audit
    from mqar_final_tail_publish import finish
    from mqar_final_tail_runner import create_json, execute_once
    from mqar_final_tail_tracking import RecoveryServices

    native = audit(
        args.input / "original",
        None,
        args.data,
        args.release / "configs/mqar-k64-classic-seed202.yaml",
    )
    require(
        canonical(native) == canonical(json.loads((args.input / "native-audit.json").read_bytes())),
        "EXACT_REVIEWED_NATIVE_REPLAY",
    )
    create_json(args.work / "native-audit.json", native)
    config = json.loads((args.input / "original/out/metrics.jsonl").read_text().splitlines()[-1])[
        "config"
    ]["model"]
    output = args.work / "result"
    historical_proof = json.loads(args.decision.with_name("hf11-proof.json").read_bytes())
    from mqar_final_tail_historical import verify

    verify(args.release, historical_proof)
    services = RecoveryServices.from_private(
        args.private, args.nonce, args.input / "original", args.release
    )
    services.historical_proof = historical_proof
    from mqar_final_tail_service_errors import instrument

    instrument(services.service, args.work)
    args.failure_phase = "evaluation"
    execute_once(args.input / "original/out/model.pt", config, args.data, output, services.start)
    with (output / "historical-hf-proof.json").open("xb") as stream:
        stream.write(args.decision.with_name("hf11-proof.json").read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    from mqar_final_tail_authority import capture

    create_json(output / "execution-authority.json", capture(args.decision))
    imported = imported_sources(wheel_root, pins)
    create_json(
        output / "scientific-source.json",
        {
            "scope": "ACTUAL_SELECTED_WORKER_AUTHENTICATED_IMPORT_ALIASES",
            "wheel_sha256": digest(args.science / "compare_attention-0.1.0-py3-none-any.whl"),
            "operator_manifest_sha256": digest(args.release / "manifest.json"),
            "imported_modules": imported,
        },
    )
    args.failure_phase = "publication"
    result = finish(
        services, output, args.data, args.release, args.work / "native-audit.json", topology
    )
    imported_sources(wheel_root, pins)
    for root in (args.release, args.input, args.science):
        check(root)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("release", "science", "input", "work", "private", "decision", "data"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--nonce", required=True)
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    try:
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            result = run(args)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        from mqar_final_tail_errors import persist, public

        phase = getattr(args, "failure_phase", "entry")
        try:
            persist(args.work / "failure.json", error, phase)
        except Exception:
            print(json.dumps(public(error, phase), sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
