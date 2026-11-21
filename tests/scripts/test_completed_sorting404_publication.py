"""Actual closed Sorting404 staging; fake reviews occur only inside synthetic tests."""

import copy
from pathlib import Path

import pytest

BASE = Path("E:/HermesArtifacts/compare-attention/remaining-seeds-sorting-mqar-v1")
TERMINAL = BASE / "sorting-classic404-terminal28"
CANDIDATE = BASE / "tracking-nonblocking-repair24/next-sorting-classic404/candidate"


@pytest.fixture
def staged(tmp_path):
    import completed_sorting404_publication as publisher

    output = tmp_path / "closure"
    subject = publisher.prepare(TERMINAL, CANDIDATE, output)
    return publisher, subject, output


def test_actual_completed_weights_stage_without_inventing_cloud_proof(staged):
    publisher, subject, output = staged
    assert subject["case"]["id"] == "sorting-classic-seed404"
    assert subject["case"]["updates"] == 110749
    assert subject["quality"]["generated_correct"] == 997
    assert subject["qualification"]["cloud_native_proof"] == "ABSENT_NOT_BACKFILLED"
    assert subject["qualification"]["historical_missing_module_name"] == "NOT_RETAINED"
    assert not (output / "payload/native-proof.json").exists()
    assert not (output / "qualified-acceptance.json").exists()
    assert publisher.validate_payload(subject)
    with pytest.raises(ValueError, match="TECHNICAL_REVIEW"):
        publisher.reviewed(subject, {"status": "PASS"})


@pytest.mark.parametrize("field", ["case", "native", "quality", "qualification"])
def test_subject_headers_cannot_override_pinned_science(staged, field):
    publisher, subject, _output = staged
    bad = copy.deepcopy(subject)
    bad[field] = {"forged": True}
    with pytest.raises(ValueError, match="PINNED_FACTS"):
        publisher.validate_payload(bad)


def test_cloud_proof_or_modified_saved_weights_are_rejected(staged):
    publisher, subject, output = staged
    (output / "payload/native-proof.json").write_text("{}")
    with pytest.raises(ValueError, match="FILES"):
        publisher.validate_payload(subject)
    (output / "payload/native-proof.json").unlink()
    (output / "payload/out/model.pt").write_bytes(b"not actual weights")
    with pytest.raises(ValueError, match="BYTES"):
        publisher.validate_payload(subject)


@pytest.mark.parametrize("unknown", [False, True])
def test_single_write_unknown_ack_get_only_and_whole_object_replay(staged, monkeypatch, unknown):
    import hashlib
    import json
    from types import SimpleNamespace

    import quality_inherit
    import quality_services

    publisher, subject, output = staged
    review_path = output / "SYNTHETIC-UNIT-REVIEW.json"
    review_path.write_text(
        json.dumps(
            {
                "scope": publisher.SCOPE,
                "status": "PASS",
                "independent": True,
                "reviewer": "SYNTHETIC_UNIT_FIXTURE_NOT_ACTUAL_PARENT_REVIEW",
                "subject_sha256": hashlib.sha256(publisher.canonical(subject).encode()).hexdigest(),
            }
        )
    )
    calls = []

    class Transport:
        def __init__(self, _values):
            pass

        def publish(self, _identity, _files):
            assert (output / "reviewed-write-intent.json").is_file()
            assert (
                json.loads((output / "hf-final.jsonl").read_text().splitlines()[-1])["state"]
                == "pending"
            )
            calls.append("publish")
            if unknown:
                raise RuntimeError("SYNTHETIC_UNKNOWN_ACK")
            return "a" * 40

        def read(self, identity, revision):
            calls.append("immutable_get")
            return {
                "private": True,
                "revision": revision,
                "files": {
                    name: (Path(subject["payload"]) / name).read_bytes()
                    for name in identity["files"]
                },
            }

    monkeypatch.setattr(
        quality_inherit, "leaf", lambda _: SimpleNamespace(credentials=lambda *_a, **_k: {})
    )
    monkeypatch.setattr(quality_services, "HFTransport", Transport)
    monkeypatch.setattr(
        publisher, "preflight", lambda *_: {"private": True, "matching_final_files": 0}
    )
    subject_path = output / "publication-subject.json"
    if unknown:
        with pytest.raises(RuntimeError, match="RECOVERY_PUBLICATION_UNKNOWN_STOP"):
            publisher.execute(subject_path, review_path, output / "not-read.env")
        with pytest.raises(ValueError, match="UNKNOWN_GET_ONLY_REVISION_REQUIRED"):
            publisher.execute(subject_path, review_path, output / "not-read.env")
        assert calls == ["publish"]
        result = publisher.execute(
            subject_path, review_path, output / "not-read.env", revision="a" * 40
        )
    else:
        result = publisher.execute(subject_path, review_path, output / "not-read.env")
    assert calls == ["publish", "immutable_get"]
    assert result["original_provider_status"] == "ERROR"
    assert result["ordinary_completion"] is False
    spec = {
        "subject": str(subject_path),
        "review": str(review_path),
        "journal": str(output / "hf-final.jsonl"),
    }
    assert publisher.replay_completed(spec, result) == result
    with pytest.raises(ValueError, match="WHOLE_ACCEPTANCE"):
        publisher.replay_completed(spec, dict(result, ordinary_completion=True))
