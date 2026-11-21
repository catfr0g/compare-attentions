"""One completed SortingClassic404 publication, preserving ERROR and absent cloud proof."""

import argparse
import hashlib
import os
import shutil
from pathlib import Path, PurePosixPath

from completed_normal_publication import canonical, digest, pin, read, require, save
from unified_normal_profiles import REMAINING_FOUR, authenticate_package, selected

INDEX_SHA = "87f36a08c90933f1015dabd9a64384861077f4c8471106d37d2e8d018b2259a0"
NONCE = "964cfc8c1f0241299a04b74c0785e4b7"
SCOPE = "SORTING_CLASSIC404_COMPLETED_FINAL_PUBLICATION_V1"
PREFIX = "normal-training/" + NONCE + "/final"
EVIDENCE = (
    "DIAGNOSIS.json",
    "METRICS-ifcached.json",
    "native-inspection.json",
    "failure-facts.json",
    "download-index.json",
    "hf-services-summary.json",
    "hf-current-listing.json",
    "collection-summary.json",
)


def qualification():
    return {
        "scope": SCOPE,
        "provider_status": "ERROR",
        "provider_job": 146454675,
        "provider_version": 1,
        "original_failure": "ModuleNotFoundError",
        "failure_phase": "finish",
        "failure_stage": "finish",
        "historical_missing_module_name": "NOT_RETAINED",
        "cloud_native_proof": "ABSENT_NOT_BACKFILLED",
        "ordinary_completion_fabricated": False,
        "original_source_identity_sha256": selected(REMAINING_FOUR)["source"],
        "native_host_scope": "INHERITED_PINNED_CPU_WEIGHTS_META_SCHEMA_AND_CACHED_LABEL_AUDIT_ONLY",
        "tracking_policy": "AUXILIARY_NONBLOCKING_V1",
        "training_resume_rescoring_executed": False,
        "next_launch_authorized": False,
    }


def facts(payload):
    """Facts come only from pinned original evidence, never caller subject headers."""
    root = Path(payload)
    audit = read(root / "posttraining-provenance/DIAGNOSIS.json")
    case = read(root / "training-task.json")["case"]
    require(
        case["id"] == "sorting-classic-seed404"
        and case["seed"] == 404
        and case["family"] == "classic"
        and case["updates"] == audit["completed_updates"] == 110749
        and case["actual_flops"] == audit["completed_flops"] == 5000039176863744
        and audit["requested_flops"] == 5_000_000_000_000_000
        and audit["skipped_optimizer_steps"] == 0
        and audit["cached_quality"]["all_1000_label_bound"] is True
        and audit["failure"]["native_proof_absent"] is True,
        "SORTING404_PINNED_NATIVE",
    )
    native = {
        "scope": "HOST_STRUCTURAL_ONLY_NOT_CLOUD_NATIVE_PROOF",
        "audit": audit["native_structural_verification"],
        "completed_updates": audit["completed_updates"],
        "completed_flops": audit["completed_flops"],
        "zero_skips_scope": audit["zero_skips_scope"],
        "ranks": audit["ranks"],
        "periodic_publication": audit["hf"],
    }
    return {
        "case": case,
        "native": native,
        "quality": audit["cached_quality"],
        "qualification": qualification(),
    }


def prepare(terminal, candidate, output):
    terminal, candidate, output = map(Path, (terminal, candidate, output))
    require(digest(terminal / "INDEX.JSON") == INDEX_SHA, "SORTING404_INDEX")
    original = read(terminal / "INDEX.JSON")["files"]
    names = [name for name in original if name.startswith("rawoutputs/")]
    require(len(names) == 24, "SORTING404_RAW_FILES")
    for name in [*names, *EVIDENCE]:
        require(pin(terminal / name) == original[name], "SORTING404_EVIDENCE_BYTES")
    package = candidate / "payload/normal-package"
    authenticate_package(package, REMAINING_FOUR)
    archive = candidate.parents[1] / "source-final/normal-package.zip.bin"
    require(digest(archive) == selected(REMAINING_FOUR)["archive"], "SORTING404_SOURCE_ARCHIVE")
    require(not output.exists(), "SORTING404_CREATE_ONLY")
    output.mkdir(parents=True)
    payload = output / "payload"
    shutil.copytree(terminal / ("rawoutputs/normal-" + NONCE), payload)
    provenance = payload / "posttraining-provenance"
    provenance.mkdir()
    for name in EVIDENCE:
        shutil.copyfile(terminal / name, provenance / name)
    shutil.copyfile(terminal / "INDEX.JSON", provenance / "terminal-index.json")
    shutil.copyfile(
        terminal / "rawoutputs" / ("seed202-pilot-" + NONCE[:16] + ".log"),
        provenance / "original-provider.log",
    )
    for name, source in {
        "normal-source-identity.json": package / "normal-source-identity.json",
        "scientific-wheel.whl": package / "compare_attention-0.1.0-py3-none-any.whl",
        "original-source.zip.bin": archive,
        "original-normal-plan.json": candidate / "payload/unified/normal-plan.json",
    }.items():
        shutil.copyfile(source, provenance / name)
    values = facts(payload)
    save(payload / "host-structural-audit.json", values["native"])
    save(payload / "posttraining-qualification.json", values["qualification"])
    subject = {
        "schema": SCOPE,
        "prefix": PREFIX,
        "step": 110749,
        "payload": str(payload.resolve()),
        "files": {
            p.relative_to(payload).as_posix(): pin(p) for p in payload.rglob("*") if p.is_file()
        },
        **values,
        "terminal_index_sha256": INDEX_SHA,
        "publisher_sha256": digest(__file__),
        "next_launch_authorized": False,
    }
    validate_payload(subject)
    save(output / "publication-subject.json", subject)
    return subject


def validate_payload(subject):
    require(
        subject["schema"] == SCOPE
        and subject["prefix"] == PREFIX
        and subject["step"] == 110749
        and subject["terminal_index_sha256"] == INDEX_SHA
        and subject["publisher_sha256"] == digest(__file__)
        and subject["next_launch_authorized"] is False,
        "SORTING404_SUBJECT",
    )
    root = Path(subject["payload"])
    paths = list(root.rglob("*"))
    require(
        not root.is_symlink()
        and not root.is_junction()
        and all(not p.is_symlink() and not p.is_junction() for p in paths),
        "SORTING404_LINK",
    )
    require(
        {p.relative_to(root).as_posix() for p in paths if p.is_file()} == set(subject["files"]),
        "SORTING404_FILES",
    )
    for name, expected in subject["files"].items():
        part = PurePosixPath(name)
        require(
            not part.is_absolute()
            and ".." not in part.parts
            and "\\" not in name
            and ":" not in name,
            "SORTING404_PATH",
        )
        require(pin(root / name) == expected, "SORTING404_BYTES")
    provenance = root / "posttraining-provenance"
    require(digest(provenance / "terminal-index.json") == INDEX_SHA, "SORTING404_INDEX")
    original = read(provenance / "terminal-index.json")["files"]
    required = {}
    for name, expected in original.items():
        if name.startswith("rawoutputs/normal-" + NONCE + "/"):
            required[name.removeprefix("rawoutputs/normal-" + NONCE + "/")] = expected
    required["posttraining-provenance/original-provider.log"] = original[
        "rawoutputs/seed202-pilot-" + NONCE[:16] + ".log"
    ]
    for name in EVIDENCE:
        required["posttraining-provenance/" + name] = original[name]
    for name, expected in required.items():
        require(subject["files"].get(name) == expected, "SORTING404_ORIGINAL_BYTES")
    require(
        digest(provenance / "original-source.zip.bin") == selected(REMAINING_FOUR)["archive"]
        and digest(provenance / "normal-source-identity.json") == selected(REMAINING_FOUR)["source"]
        and digest(provenance / "scientific-wheel.whl")
        == read(provenance / "normal-source-identity.json")["wheel_sha256"]
        and digest(provenance / "original-normal-plan.json")
        == "c02cc765bfab3779791c9784852ad1829ff7613e42c97e6ad84a404d24977752",
        "SORTING404_SOURCE",
    )
    values = facts(root)
    require(
        all(subject[key] == value for key, value in values.items())
        and read(root / "host-structural-audit.json") == values["native"]
        and read(root / "posttraining-qualification.json") == values["qualification"],
        "SORTING404_PINNED_FACTS",
    )
    extras = {
        "posttraining-provenance/terminal-index.json",
        "posttraining-provenance/normal-source-identity.json",
        "posttraining-provenance/scientific-wheel.whl",
        "posttraining-provenance/original-source.zip.bin",
        "posttraining-provenance/original-normal-plan.json",
        "host-structural-audit.json",
        "posttraining-qualification.json",
    }
    require(set(subject["files"]) == set(required) | extras, "SORTING404_FILES")
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
        "SORTING404_TECHNICAL_REVIEW",
    )


def preflight(subject, transport):
    """GET the actual private repository; no upload or ClearML operations."""

    def operation():
        from huggingface_hub import HfApi

        token, repo = transport.values["HF_TOKEN"], transport.values["HF_REPO"]
        api = HfApi(token=token)
        info = api.repo_info(repo, repo_type="model", token=token)
        require(info.private is True, "SORTING404_PRIVATE")
        listing = api.list_repo_files(repo, repo_type="model", revision=info.sha, token=token)
        require(not any(name.startswith(PREFIX + "/") for name in listing), "SORTING404_PRESENT")
        require(
            any(name.startswith("normal-training/" + NONCE + "/periodic/") for name in listing),
            "SORTING404_ORIGINAL_REPOSITORY",
        )
        return {
            "private": True,
            "revision": info.sha,
            "prefix": subject["prefix"],
            "matching_final_files": 0,
        }

    return transport.invoke("sorting404_final_preflight_get", operation)


def acceptance(subject, subject_path, review_path, final):
    return {
        "schema": "QUALIFIED_COMPLETED_SORTING404_ACCEPTANCE_V1",
        "status": "ACCEPTED_COMPLETED_NORMAL_WITH_MISSING_CLOUD_PROOF_QUALIFICATION",
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


def execute(subject_path, review_path, private, *, revision=None):
    """Single durable-before-upload write; an uncertain ACK permits GET only."""
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "SORTING404_HOST_MASK")
    subject_path = Path(subject_path)
    subject = read(subject_path)
    reviewed(subject, read(review_path))
    files = validate_payload(subject)
    from quality_inherit import leaf
    from quality_recovery import RecoveryPublisher
    from quality_services import HFTransport

    transport = HFTransport(leaf("services").credentials(private, rank=0))
    output = subject_path.parent
    publisher = RecoveryPublisher(output / "hf-final.jsonl", PREFIX, transport)
    require(not (output / "qualified-acceptance.json").exists(), "SORTING404_ALREADY_ACCEPTED")
    lock = output / "single-writer.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        records = publisher.records()
        if not records:
            require(revision is None, "SORTING404_RECONCILE_WITHOUT_INTENT")
            save(output / "final-absence-before-write.json", preflight(subject, transport))
            save(
                output / "reviewed-write-intent.json",
                {
                    "subject_sha256": digest(subject_path),
                    "review_sha256": digest(review_path),
                    "prefix": PREFIX,
                    "files": subject["files"],
                },
            )
            final = publisher.publish(subject["step"], files)
        else:
            require(
                records[-1]["files"] == subject["files"] and records[-1]["prefix"] == PREFIX,
                "SORTING404_JOURNAL_BINDING",
            )
            require(revision is not None, "SORTING404_UNKNOWN_GET_ONLY_REVISION_REQUIRED")
            final = publisher.reconcile(subject["step"], revision)
        result = acceptance(subject, subject_path, review_path, final)
        save(output / "qualified-acceptance.json", result)
        return result
    finally:
        lock.unlink()


def replay_completed(spec, recorded, inputs=None):
    require(set(spec) == {"subject", "review", "journal"}, "SORTING404_REPLAY_SCHEMA")
    subject = read(spec["subject"])
    reviewed(subject, read(spec["review"]))
    files = validate_payload(subject)
    if inputs is not None:
        paths = [Path(spec[key]).resolve() for key in spec]
        paths.extend(path.resolve() for path in files.values())
        require(all(str(path) in inputs["files"] for path in paths), "SORTING404_REPLAY_CLOSURE")
    import json

    records = [json.loads(line) for line in Path(spec["journal"]).read_text().splitlines()]
    require(bool(records) and records[-1]["state"] == "verified", "SORTING404_REPLAY_VERIFIED")
    final = records[-1]
    require(
        final["files"] == subject["files"]
        and final["prefix"] == PREFIX
        and final["step"] == subject["step"],
        "SORTING404_JOURNAL_BINDING",
    )
    expected = acceptance(subject, spec["subject"], spec["review"], final)
    require(canonical(expected) == canonical(recorded), "SORTING404_WHOLE_ACCEPTANCE")
    return expected


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("subject", "review", "private"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision")
    args = parser.parse_args(argv)
    execute(args.subject, args.review, args.private, revision=args.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
