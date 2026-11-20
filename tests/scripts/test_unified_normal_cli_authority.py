"""Real emitted CLI ingress ordering, with synthetic authority compatibility controls."""

import json
import os
from pathlib import Path

import pytest

REQUIRED = (
    "seal",
    "prepare",
    "probe",
    "publish",
    "publish-resume",
    "submit",
    "status",
    "collect",
    "services",
    "accept",
    "reconcile",
)


@pytest.fixture
def emitted(monkeypatch):
    selected = os.environ.get("UNIFIED_GENERATION_CANDIDATE")
    if not selected:
        pytest.skip("actual emitted candidate not selected")
    root = Path(selected)
    monkeypatch.syspath_prepend(str(root / "payload/source/src"))
    from seed202_pilot import cli

    assert Path(cli.__file__).resolve().is_relative_to(root.resolve())
    return root, cli


def observe_reads(monkeypatch):
    """Observe actual filesystem loads; neither the loader nor guard is replaced."""
    reads = []
    original = Path.read_bytes

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    return reads


@pytest.mark.parametrize("command", REQUIRED)
def test_public_ingress_denies_before_caller_paths(emitted, monkeypatch, tmp_path, command):
    import unified_normal_authority as authority
    from unified_normal_authority import require as actual_require

    root, cli = emitted
    denied = []

    def require(condition, code):
        if not condition:
            denied.append(code)
        return actual_require(condition, code)

    monkeypatch.setattr(authority, "require", require)
    monkeypatch.setattr(cli, "_provider", lambda *_: pytest.fail("PROVIDER_CONSTRUCTED"))
    paths = {name: tmp_path / (name + ".json") for name in ("plan", "review", "proof", "approval")}
    for path in paths.values():
        path.write_text(json.dumps({"candidate_root": str(root)}))
    reads = observe_reads(monkeypatch)
    argv = [command, "--candidate-dir", str(root), "--case", "mqar-k64-deltanet-seed202"]
    for name, path in paths.items():
        argv += ["--" + name, str(path)]
    assert cli.main(argv) == 1
    assert denied == ["UNIFIED_NORMAL_EXECUTION_AUTHORITY_REQUIRED"]
    assert not set(paths.values()).intersection(reads)


def synthetic_bound(root):
    """Private schema fixture, never actual source-plan review or execution authority."""
    from unified_normal_authority import pin, validate

    folder = root / "payload/unified"
    authority = folder / "authority"
    authority.mkdir(parents=True)
    plan = folder / "normal-plan.json"
    plan.write_text(
        json.dumps(
            {
                "cases": {"SYNTHETIC": {}},
                "engineering_only": False,
                "source_identity_sha256": "0" * 64,
            }
        )
    )
    common = {
        "case_id": "SYNTHETIC",
        "source_identity_sha256": "0" * 64,
        "plan_sha256": pin(plan)["sha256"],
    }
    review = authority / "independent-review.json"
    review.write_text(
        json.dumps(
            {
                **common,
                "status": "PASS",
                "independent": True,
                "reviewer": "SYNTHETIC_UNIT_FIXTURE_NOT_REAL_REVIEW",
            }
        )
    )
    decision = authority / "execution-decision.json"
    decision.write_text(
        json.dumps(
            {
                **common,
                "action": "NORMAL_FRESH_TRAINING",
                "approved": True,
                "review_sha256": pin(review)["sha256"],
            }
        )
    )
    (folder / "normal-authority.json").write_text(json.dumps(validate(plan, review, decision)))
    return folder


@pytest.mark.parametrize("command", REQUIRED)
def test_bound_ingress_reaches_real_existing_validator(emitted, monkeypatch, tmp_path, command):
    from seed202_pilot import seal

    _root, cli = emitted
    root = tmp_path / "SYNTHETIC-BOUND-NOT-A-LAUNCH-CANDIDATE"
    folder = synthetic_bound(root)
    (root / "candidate.json").write_text('{"status":"SYNTHETIC_INVALID_CANDIDATE"}')
    (root / "PLAN-UNAPPROVED.json").write_text("{}")
    paths = {name: tmp_path / (name + ".json") for name in ("plan", "review", "proof")}
    for path in paths.values():
        path.write_text(json.dumps({"candidate_root": str(root)}))
    errors = []
    original = seal._require

    def require(condition, code):
        if not condition:
            errors.append(code)
        return original(condition, code)

    monkeypatch.setattr(seal, "_require", require)
    monkeypatch.setattr(cli, "_provider", lambda *_: pytest.fail("PROVIDER_CONSTRUCTED"))
    reads = observe_reads(monkeypatch)
    argv = [command, "--candidate-dir", str(root)]
    for name, path in paths.items():
        argv += ["--" + name, str(path)]
    assert cli.main(argv) == 1
    if command == "seal":
        assert errors == ["PLAN_SCHEMA"]  # Real seal() consumed all actual loaded arguments.
        assert set(paths.values()).issubset(reads)
    else:
        assert root / "candidate.json" in reads  # Real _candidate(), not a stubbed dispatcher.
        assert not set(paths.values()).intersection(reads)
    authority_reads = [
        folder / "normal-authority.json",
        folder / "authority/independent-review.json",
        folder / "authority/execution-decision.json",
    ]
    assert all(path in reads for path in authority_reads)
    first_consumer = paths["plan"] if command == "seal" else root / "candidate.json"
    assert max(reads.index(path) for path in authority_reads) < reads.index(first_consumer)


@pytest.mark.parametrize("command", ["check", "make-plan", "next", "composite"])
def test_noauth_metadata_routes_keep_existing_authority_policy(emitted, command):
    from types import SimpleNamespace

    from unified_normal_authority import before_auth

    root, _cli = emitted
    before_auth(SimpleNamespace(command=command, candidate_dir=root))
