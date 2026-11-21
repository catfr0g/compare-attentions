"""Completed case publication stages genuine saved bytes, never finish/evaluate."""

import importlib
import json
from pathlib import Path

import pytest

TERMINAL = Path(
    "E:/HermesArtifacts/compare-attention/remaining-seeds-sorting-mqar-v1/mqar-delta303-terminal23"
)
CANDIDATE = Path(
    "E:/HermesArtifacts/compare-attention/remaining-seeds-sorting-mqar-v1/remaining-six-fixed-source13/next-mqar-delta303-17/candidate"
)


def test_completed_publication_requires_actual_parent_review(tmp_path):
    module = importlib.import_module("completed_normal_publication")
    subject = module.prepare(TERMINAL, CANDIDATE, tmp_path / "closure")
    assert subject["case"]["id"] == "mqar-k64-deltanet-seed303"
    assert subject["qualification"]["provider_status"] == "ERROR"
    assert subject["qualification"]["ordinary_completion_fabricated"] is False
    assert subject["case"]["updates"] == 11817
    assert subject["prefix"] == "normal-training/007d90ea4622493a8fe41959713a816a/final"
    assert subject["quality"]["per_query_correct"] == 59548
    assert subject["quality"]["whole_row_correct"] == 9
    with pytest.raises(ValueError, match="COMPLETED_PUBLICATION_TECHNICAL_REVIEW"):
        module.reviewed(subject, {"status": "PASS", "independent": True})
    assert not (tmp_path / "closure/hf-final.jsonl").exists()
    payload = tmp_path / "closure/payload"
    (payload / "out/model.pt").write_bytes(b"wrong weights")
    with pytest.raises(ValueError, match="COMPLETED_PUBLICATION_PAYLOAD_BYTES"):
        module.validate_payload(subject)


@pytest.mark.parametrize("unknown", [False, True])
def test_reviewed_publisher_durable_intent_and_no_blind_retry(monkeypatch, tmp_path, unknown):
    import hashlib
    from types import SimpleNamespace

    import quality_inherit
    import quality_services

    module = importlib.import_module("completed_normal_publication")
    output = tmp_path / "closure"
    subject = module.prepare(TERMINAL, CANDIDATE, output)
    review = {
        "scope": module.SCOPE,
        "status": "PASS",
        "independent": True,
        "reviewer": "SYNTHETIC_UNIT_FIXTURE_NOT_PRODUCTION_REVIEW",
        "subject_sha256": hashlib.sha256(module.canonical(subject).encode()).hexdigest(),
    }
    review_path = tmp_path / "synthetic-review.json"
    review_path.write_text(json.dumps(review))
    calls = []

    class Transport:
        def __init__(self, _):
            pass

        def publish(self, identity, files):
            assert (output / "reviewed-write-intent.json").is_file()
            assert (
                json.loads((output / "hf-final.jsonl").read_text().splitlines()[-1])["state"]
                == "pending"
            )
            calls.append("publish")
            if unknown:
                raise RuntimeError("uncertain synthetic ack")
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
        quality_inherit, "leaf", lambda _: SimpleNamespace(credentials=lambda *_args, **_kw: {})
    )
    monkeypatch.setattr(quality_services, "HFTransport", Transport)
    monkeypatch.setattr(
        module, "preflight", lambda *_: {"private": True, "matching_final_files": 0}
    )
    if unknown:
        with pytest.raises(RuntimeError, match="RECOVERY_PUBLICATION_UNKNOWN_STOP"):
            module.execute(
                output / "publication-subject.json", review_path, tmp_path / "not-read.env"
            )
        with pytest.raises(
            ValueError, match="COMPLETED_PUBLICATION_UNKNOWN_GET_ONLY_REVISION_REQUIRED"
        ):
            module.execute(
                output / "publication-subject.json", review_path, tmp_path / "not-read.env"
            )
        assert calls == ["publish"]
        assert not (output / "qualified-acceptance.json").exists()
        result = module.execute(
            output / "publication-subject.json",
            review_path,
            tmp_path / "not-read.env",
            revision="a" * 40,
        )
    else:
        result = module.execute(
            output / "publication-subject.json", review_path, tmp_path / "not-read.env"
        )
    assert calls == ["publish", "immutable_get"]
    assert result["original_provider_status"] == "ERROR"
    assert result["ordinary_completion"] is False
    assert result["final_publication"]["state"] == "verified"
    spec = {
        "subject": str(output / "publication-subject.json"),
        "review": str(review_path),
        "journal": str(output / "hf-final.jsonl"),
    }
    assert module.replay_completed(spec, result) == result
    wrong = dict(result, native={"updates": 1})
    with pytest.raises(ValueError, match="COMPLETED_PUBLICATION_WHOLE_ACCEPTANCE"):
        module.replay_completed(spec, wrong)
