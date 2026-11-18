"""Continuation prefix and restore evidence remain mandatory in ordinary acceptance."""

import importlib
import json
import shutil
import sys

import pytest
from test_seed202_mqar_builder import HISTORY, WORK


def _engineering_authority(root):
    from seed202_mqar_authority import MODE, bind_bytes, canonical, pin, validate

    mounted = root / "mounted-authority"
    mounted.mkdir()
    raw = b"ENGINEERING_ONLY_NO_SCIENTIFIC_CHECKPOINT"
    (mounted / "resume.zip.bin").write_bytes(raw)
    evidence = {
        "scope": "MQAR_EVIDENCE_ONLY_V1",
        "mode": MODE,
        "case": "mqar-k64-classic-seed202",
        "nonce": "4a" * 16,
        "mounted": {"resume.zip.bin": pin(raw)},
    }
    digest = pin(canonical(evidence))["sha256"]
    review = canonical(
        {
            "scope": "INDEPENDENT_EVIDENCE_REVIEW_V1",
            "evidence_sha256": digest,
            "decision": "PASS",
            "reviewer": "ENGINEERING_FIXTURE_ONLY",
        }
    )
    decision = canonical(
        {
            "scope": "EXECUTION_DECISION_V1",
            "evidence_sha256": digest,
            "review_sha256": pin(review)["sha256"],
            "decision": "ALLOW",
            "nonce": evidence["nonce"],
        }
    )
    bound = bind_bytes(mounted, evidence, review, decision)
    return {
        "mode": MODE,
        "case_id": evidence["case"],
        **bound,
        "mounted_authority_files": validate(mounted, **bound)["files"],
    }


def test_missing_real_restore_receipt_cannot_qualify_prefix(tmp_path):
    from build_seed202_mqar_continuation import stage_source

    source = stage_source(WORK / "parent", HISTORY, tmp_path / "source")
    old = {k: v for k, v in sys.modules.items() if k.startswith("seed202_pilot")}
    for name in old:
        del sys.modules[name]
    sys.path.insert(0, str(source))
    try:
        module = importlib.import_module("seed202_pilot.mqar_composite")
        provenance = importlib.import_module("seed202_pilot.mqar_continuation").provenance()
        root = tmp_path / "result"
        root.mkdir()
        requested = _engineering_authority(root)
        (root / "resume-provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
        with pytest.raises((ValueError, FileNotFoundError)):
            module.verify(root, {"id": "mqar-k64-classic-seed202"}, requested)
    finally:
        sys.path.remove(str(source))
        for name in list(sys.modules):
            if name.startswith("seed202_pilot"):
                del sys.modules[name]
        sys.modules.update(old)


def test_engineering_only_prefix_join_rejects_repeat_and_missing_rank(tmp_path):
    from build_seed202_mqar_continuation import original_files, stage_source

    source = stage_source(WORK / "parent", HISTORY, tmp_path / "source")
    old = {k: v for k, v in sys.modules.items() if k.startswith("seed202_pilot")}
    for name in old:
        del sys.modules[name]
    sys.path.insert(0, str(source))
    try:
        module = importlib.import_module("seed202_pilot.mqar_composite")
        provenance = importlib.import_module("seed202_pilot.mqar_continuation").provenance()
        fields = importlib.import_module("seed202_pilot.mqar_restore_gate").FIELDS
        original, pins = original_files()
        root = tmp_path / "ENGINEERING_ONLY_NO_TARGET_RESTORE_NO_OPTIMIZER"
        root.mkdir()
        requested = _engineering_authority(root)
        for name in pins:
            if name.startswith("diagnostic/"):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original / name, target)
        (root / "out").mkdir()
        (root / "resume-provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
        prefix = (original / "out/metrics.jsonl").read_text(encoding="utf-8")
        metrics = root / "out/metrics.jsonl"
        metrics.write_text(
            prefix + json.dumps({"type": "step", "loss": 1.0, "flops": 981 * 463856467968}) + "\n",
            encoding="utf-8",
        )
        for rank in (0, 1):
            (root / f"restore-equality-rank-{rank}.json").write_text(
                json.dumps(
                    {
                        "scope": "ACTUAL_TARGET_NATIVE_RESTORE_EQUALITY_BEFORE_981",
                        "rank": rank,
                        "step": 980,
                        "next_step": 981,
                        "fields": list(fields),
                        "native_restore_unchanged": True,
                        "rng_checked_after_history_replay": True,
                    }
                ),
                encoding="utf-8",
            )
        case = {"id": "mqar-k64-classic-seed202"}
        assert module.verify(root, case, requested)["original_status_preserved"] == "ERROR"
        saved = (root / "resume-provenance.json").read_bytes()
        (root / "resume-provenance.json").unlink()
        with pytest.raises(FileNotFoundError):
            module.verify(root, case, requested)
        (root / "resume-provenance.json").write_bytes(saved)
        metrics.write_text(
            prefix + json.dumps({"type": "step", "loss": 1.0, "flops": 980 * 463856467968}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="EXACT_FIRST_NEW_UPDATE_981"):
            module.verify(root, case, requested)
        metrics.write_text(
            prefix + json.dumps({"type": "step", "loss": 1.0, "flops": 981 * 463856467968}) + "\n",
            encoding="utf-8",
        )
        (root / "restore-equality-rank-1.json").unlink()
        with pytest.raises(FileNotFoundError):
            module.verify(root, case, requested)
    finally:
        sys.path.remove(str(source))
        for name in list(sys.modules):
            if name.startswith("seed202_pilot"):
                del sys.modules[name]
        sys.modules.update(old)
