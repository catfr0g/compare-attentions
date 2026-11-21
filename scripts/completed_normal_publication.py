"""Bounded Delta303 completed-science publisher; no train, finish, model or scoring."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath

INDEX_SHA = "44322fee040c559ad6dfd207e9383a9fb717fefedefa600a89f09049047816e3"
SOURCE = "115df61b5390f324fd06fc81bb091d04a0bce74dfa8587be5aa560b01ed27e2e"
WHEEL = "a1e1db146ac0fef3583edbc6bee1d41ed1e83dffe85702c09667d34c96047e90"
NONCE = "007d90ea4622493a8fe41959713a816a"
SCOPE = "MQAR_DELTA303_COMPLETED_FINAL_PUBLICATION_V1"


def require(ok, code):
    if not ok:
        raise ValueError(code)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    """Durable create-only external metadata, outside immutable original evidence."""
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def pin(path):
    return {"bytes": Path(path).stat().st_size, "sha256": digest(path)}


def prepare(terminal, candidate, output):
    """Join previously qualified immutable bytes into one actual review subject."""
    terminal, candidate, output = map(Path, (terminal, candidate, output))
    require(digest(terminal / "INDEX.JSON") == INDEX_SHA, "COMPLETED_PUBLICATION_INDEX")
    index = read(terminal / "INDEX.JSON")["files"]
    selected = [name for name in index if name.startswith("rawoutputs/normal-" + NONCE + "/")]
    evidence_names = [
        "tensor-budget-diagnosis.json",
        "hf-full-and-quality-service-summary.json",
        "hf-periodic-readback.json",
        "download-index.json",
        "service-drift-diagnosis.json",
    ]
    for name in [*selected, *evidence_names]:
        require(pin(terminal / name) == index[name], "COMPLETED_PUBLICATION_EVIDENCE_BYTES")
    package = candidate / "payload/normal-package"
    require(
        digest(package / "normal-source-identity.json") == SOURCE, "COMPLETED_PUBLICATION_SOURCE"
    )
    require(
        digest(package / "compare_attention-0.1.0-py3-none-any.whl") == WHEEL,
        "COMPLETED_PUBLICATION_WHEEL",
    )
    root = terminal / ("rawoutputs/normal-" + NONCE)
    case = read(root / "training-task.json")["case"]
    audit = read(terminal / "tensor-budget-diagnosis.json")
    provider = read(terminal / "download-index.json")
    require(
        provider["provider_before"] == provider["provider_after"]
        and provider["provider_before"]["numeric_id"] == 146358896
        and provider["provider_before"]["typed_status"] == "ERROR",
        "COMPLETED_PUBLICATION_PROVIDER",
    )
    require(
        case["id"] == "mqar-k64-deltanet-seed303"
        and case["seed"] == 303
        and case["updates"] == audit["completed_updates"] == audit["planned_updates"] == 11817
        and case["actual_flops"] == audit["completed_flops"] == 5000421949046784
        and audit["requested_flops"] == 5_000_000_000_000_000
        and audit["cached_quality_label_and_final_weight_binding_verified"] is True,
        "COMPLETED_PUBLICATION_NATIVE",
    )
    native = read(root / "native-proof.json")
    require(native == audit["native_verification"], "COMPLETED_PUBLICATION_NATIVE_PROOF")
    require(
        all(
            row["precision"]["skipped_optimizer_steps"] == 0
            and row["meta_schema_full_finite_adam_rng_distributed_valid"] is True
            for row in audit["retained_full"]
        ),
        "COMPLETED_PUBLICATION_FULL",
    )
    require(not output.exists(), "COMPLETED_PUBLICATION_CREATE_ONLY")
    output.mkdir(parents=True)
    payload = output / "payload"
    shutil.copytree(root, payload)
    provenance = payload / "posttraining-provenance"
    provenance.mkdir()
    shutil.copyfile(terminal / "INDEX.JSON", provenance / "terminal-index.json")
    for name in evidence_names:
        shutil.copyfile(terminal / name, provenance / name)
    for name, source in {
        "normal-source-identity.json": package / "normal-source-identity.json",
        "original-normal-plan.json": candidate / "payload/unified/normal-plan.json",
        "scientific-wheel.whl": package / "compare_attention-0.1.0-py3-none-any.whl",
    }.items():
        shutil.copyfile(source, provenance / name)
    qualification = {
        "scope": SCOPE,
        "provider_status": "ERROR",
        "provider_job": 146358896,
        "provider_version": 1,
        "original_failure": "NORMAL_FRESH_UNSAMPLED_DRIFT",
        "historical_failed_fresh_payload_retained": False,
        "ordinary_completion_fabricated": False,
        "original_source_identity_sha256": SOURCE,
        "scientific_wheel_sha256": WHEEL,
        "native_host_scope": (
            "INHERITED_PINNED_CPU_TENSOR_AND_META_SCHEMA_AUDIT_NO_SCIENTIFIC_EXECUTION"
        ),
        "tracking_policy": "AUXILIARY_NONBLOCKING_V1",
        "training_resume_rescoring_executed": False,
        "final_publication": "UNATTEMPTED_REQUIRES_PARENT_TECHNICAL_INSPECTION",
    }
    save(payload / "posttraining-qualification.json", qualification)
    subject = {
        "schema": SCOPE,
        "case": case,
        "prefix": "normal-training/" + NONCE + "/final",
        "step": 11817,
        "payload": str(payload.resolve()),
        "files": {
            p.relative_to(payload).as_posix(): pin(p) for p in payload.rglob("*") if p.is_file()
        },
        "native": native,
        "quality": audit["cached_quality"],
        "qualification": qualification,
        "terminal_index_sha256": INDEX_SHA,
        "publisher_sha256": digest(__file__),
        "next_launch_authorized": False,
    }
    save(output / "publication-subject.json", subject)
    return subject


def validate_payload(subject):
    root = Path(subject["payload"])
    require(
        subject["schema"] == SCOPE
        and subject["prefix"] == "normal-training/" + NONCE + "/final"
        and subject["step"] == 11817
        and subject["publisher_sha256"] == digest(__file__),
        "COMPLETED_PUBLICATION_SUBJECT",
    )
    paths = list(root.rglob("*"))
    require(
        not root.is_symlink()
        and not root.is_junction()
        and all(not p.is_symlink() and not p.is_junction() for p in paths),
        "COMPLETED_PUBLICATION_PAYLOAD_LINK",
    )
    require(
        {p.relative_to(root).as_posix() for p in paths if p.is_file()} == set(subject["files"]),
        "COMPLETED_PUBLICATION_PAYLOAD_FILES",
    )
    for name, expected in subject["files"].items():
        part = PurePosixPath(name)
        require(
            not part.is_absolute()
            and ".." not in part.parts
            and "\\" not in name
            and ":" not in name,
            "COMPLETED_PUBLICATION_PAYLOAD_PATH",
        )
        require(pin(root / name) == expected, "COMPLETED_PUBLICATION_PAYLOAD_BYTES")
    provenance = root / "posttraining-provenance"
    require(digest(provenance / "terminal-index.json") == INDEX_SHA, "COMPLETED_PUBLICATION_INDEX")
    original = read(provenance / "terminal-index.json")["files"]
    prefix = "rawoutputs/normal-" + NONCE + "/"
    for name, expected in original.items():
        if name.startswith(prefix):
            require(
                subject["files"].get(name.removeprefix(prefix)) == expected,
                "COMPLETED_PUBLICATION_ORIGINAL_BYTES",
            )
    audit_name = "tensor-budget-diagnosis.json"
    require(
        pin(provenance / audit_name) == original[audit_name], "COMPLETED_PUBLICATION_NATIVE_PROOF"
    )
    audit = read(provenance / audit_name)
    require(
        subject["case"] == read(root / "training-task.json")["case"]
        and subject["native"] == audit["native_verification"] == read(root / "native-proof.json")
        and subject["quality"] == audit["cached_quality"]
        and subject["qualification"] == read(root / "posttraining-qualification.json"),
        "COMPLETED_PUBLICATION_NATIVE_PROOF",
    )
    return {name: root / name for name in subject["files"]}


def reviewed(subject, review):
    require(
        set(review) == {"scope", "status", "independent", "reviewer", "subject_sha256"}
        and review["scope"] == SCOPE
        and review["status"] == "PASS"
        and review["independent"] is True
        and type(review["reviewer"]) is str
        and bool(review["reviewer"])
        and review["subject_sha256"] == hashlib.sha256(canonical(subject).encode()).hexdigest(),
        "COMPLETED_PUBLICATION_TECHNICAL_REVIEW",
    )


def preflight(subject, transport):
    """GET-only private namespace absence; prove this is the original periodic repository."""

    def operation():
        from huggingface_hub import HfApi

        token, repo = transport.values["HF_TOKEN"], transport.values["HF_REPO"]
        api = HfApi(token=token)
        info = api.repo_info(repo, repo_type="model", token=token)
        require(info.private is True, "COMPLETED_PUBLICATION_PRIVATE")
        listing = api.list_repo_files(repo, repo_type="model", revision=info.sha, token=token)
        prefix = subject["prefix"] + "/"
        require(
            not any(name.startswith(prefix) for name in listing),
            "COMPLETED_PUBLICATION_ALREADY_PRESENT",
        )
        require(
            any(name.startswith("normal-training/" + NONCE + "/periodic/") for name in listing),
            "COMPLETED_PUBLICATION_ORIGINAL_REPOSITORY",
        )
        return {
            "private": True,
            "revision": info.sha,
            "prefix": subject["prefix"],
            "matching_final_files": 0,
        }

    return transport.invoke("completed_final_preflight_get", operation)


def execute(subject_path, review_path, private, *, revision=None):
    """One reviewed final write; unknown ACK permits only explicit immutable reconciliation."""
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "COMPLETED_PUBLICATION_HOST_MASK")
    subject_path = Path(subject_path)
    subject, review = read(subject_path), read(review_path)
    reviewed(subject, review)
    files = validate_payload(subject)
    from quality_inherit import leaf
    from quality_recovery import RecoveryPublisher
    from quality_services import HFTransport

    values = leaf("services").credentials(private, rank=0)
    transport = HFTransport(values)
    output = subject_path.parent
    journal = output / "hf-final.jsonl"
    publisher = RecoveryPublisher(journal, subject["prefix"], transport)
    require(
        not (output / "qualified-acceptance.json").exists(),
        "COMPLETED_PUBLICATION_ALREADY_ACCEPTED",
    )
    lock = output / "single-writer.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        records = publisher.records()
        if not records:
            require(revision is None, "COMPLETED_PUBLICATION_RECONCILE_WITHOUT_INTENT")
            absence = preflight(subject, transport)
            save(output / "final-absence-before-write.json", absence)
            save(
                output / "reviewed-write-intent.json",
                {
                    "subject_sha256": digest(subject_path),
                    "review_sha256": digest(review_path),
                    "prefix": subject["prefix"],
                    "files": subject["files"],
                },
            )
            final = publisher.publish(subject["step"], files)
        else:
            require(
                records[-1]["files"] == subject["files"]
                and records[-1]["prefix"] == subject["prefix"],
                "COMPLETED_PUBLICATION_JOURNAL_BINDING",
            )
            require(
                revision is not None, "COMPLETED_PUBLICATION_UNKNOWN_GET_ONLY_REVISION_REQUIRED"
            )
            final = publisher.reconcile(subject["step"], revision)
        # RecoveryPublisher verifies immutable private listing and every byte before this receipt.
        accepted = {
            "schema": "QUALIFIED_COMPLETED_NORMAL_ACCEPTANCE_V1",
            "status": "ACCEPTED_COMPLETED_NORMAL_WITH_FINISH_TAIL_QUALIFICATION",
            "case": subject["case"],
            "native": subject["native"],
            "task_val": subject["quality"],
            "qualification": subject["qualification"],
            "original_provider_status": "ERROR",
            "ordinary_completion": False,
            "final_publication": final,
            "subject_sha256": digest(subject_path),
            "technical_review_sha256": digest(review_path),
            "next_launch_authorized": False,
        }
        save(output / "qualified-acceptance.json", accepted)
        return accepted
    finally:
        lock.unlink()


def replay_completed(spec, recorded, inputs=None):
    """Bounded case3 whole-object replay, preserving ERROR rather than ordinary backfill."""
    require(set(spec) == {"subject", "review", "journal"}, "COMPLETED_PUBLICATION_REPLAY_SCHEMA")
    subject, review = read(spec["subject"]), read(spec["review"])
    reviewed(subject, review)
    validate_payload(subject)
    if inputs is not None:
        files = [Path(spec[key]).resolve() for key in spec]
        files.extend(p.resolve() for p in Path(subject["payload"]).rglob("*") if p.is_file())
        require(
            all(str(path) in inputs["files"] for path in files),
            "COMPLETED_PUBLICATION_REPLAY_CLOSURE",
        )
    records = [json.loads(line) for line in Path(spec["journal"]).read_text().splitlines()]
    require(
        bool(records) and records[-1]["state"] == "verified",
        "COMPLETED_PUBLICATION_REPLAY_VERIFIED",
    )
    final = records[-1]
    require(
        final["files"] == subject["files"]
        and final["prefix"] == subject["prefix"]
        and final["step"] == subject["step"],
        "COMPLETED_PUBLICATION_JOURNAL_BINDING",
    )
    expected = {
        "schema": "QUALIFIED_COMPLETED_NORMAL_ACCEPTANCE_V1",
        "status": "ACCEPTED_COMPLETED_NORMAL_WITH_FINISH_TAIL_QUALIFICATION",
        "case": subject["case"],
        "native": subject["native"],
        "task_val": subject["quality"],
        "qualification": subject["qualification"],
        "original_provider_status": "ERROR",
        "ordinary_completion": False,
        "final_publication": final,
        "subject_sha256": digest(spec["subject"]),
        "technical_review_sha256": digest(spec["review"]),
        "next_launch_authorized": False,
    }
    require(canonical(expected) == canonical(recorded), "COMPLETED_PUBLICATION_WHOLE_ACCEPTANCE")
    return expected


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--revision")
    args = parser.parse_args(argv)
    execute(args.subject, args.review, args.private, revision=args.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
