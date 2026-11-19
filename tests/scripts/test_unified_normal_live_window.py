"""Offline LIVE-window tracer: real public guards, Win32 leases and SDK encoding."""

import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path

import pytest


@pytest.fixture
def world(monkeypatch):
    logging.disable(logging.CRITICAL)
    root = Path(os.environ["UNIFIED_NORMAL_CANDIDATE"])
    for suffix in ("src", "tests/seed202_pilot", "tests/qualified_launch"):
        monkeypatch.syspath_prepend(str(root / "payload/source" / suffix))
    from fresh_sdk_fixture import FreshTransport
    from seed202_qualified_http_fixture import HttpFixture

    transport = FreshTransport(root, monkeypatch)
    transport.enable_mutations()
    wire = HttpFixture(transport, monkeypatch)
    yield root, transport, wire


def vectors(transport, case, plan, review, prepared, action):
    from seed202_pilot import cli

    captured = []
    original = cli.main
    # Capture the fixture's exact argv without running a command or changing guards.
    cli.main = lambda argv: captured.append(argv) or 0
    try:
        shared = ["--plan", plan, "--review", review, "--preparation", prepared]
        proof = transport.root / (action + "-live-proof.json")
        approval = transport.root / (action + "-live-approval.json")
        journal = Path(json.loads(plan.read_bytes())["journals"][action])
        transport.command(
            "probe", case, *shared, "--action", action, "--output", proof, journal=journal
        )
        transport.command(
            "seal",
            case,
            "--plan",
            plan,
            "--review",
            review,
            "--proof",
            proof,
            "--output",
            approval,
            journal=journal,
        )
        transport.command(action, case, *shared, "--approval", approval, journal=journal)
    finally:
        cli.main = original
    return captured


def setup(transport, root):
    from unified_normal_contract import descriptor

    _, request = descriptor(root)
    case = request["run"]["id"]
    plan = transport.root / "plan.json"
    prefix = [t for row in request["predecessors"][1:] for t in ("--acceptance", row["path"])]
    assert (
        transport.command(
            "make-plan", case, *prefix, "--output", plan, journal=transport.root / "journals"
        )
        == 0
    )
    value = json.loads(plan.read_bytes())
    review = transport.root / "SYNTHETIC-REVIEW-NOT-AUTHORITY.json"
    review.write_text(
        json.dumps(
            {
                "status": "PASS",
                "independent": True,
                "reviewer": "SYNTHETIC-TRANSPORT-NOT-REVIEW",
                "scope": "SEED202_PILOT_SINGLETON_FULL",
                "plan_sha256": hashlib.sha256(
                    json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    )
    return case, value, plan, review


@pytest.mark.parametrize("expired", [False, True])
def test_public_chain_keeps_immutable_work_before_live(
    world, monkeypatch, record_property, expired
):
    import time

    from seed202_pilot import cli, host_operator, prepared_files, seal
    from test_fresh_public_chain_integration06 import _prepare

    root, transport, wire = world
    case, plan_value, plan, review = setup(transport, root)
    clock = {"now": time.time(), "live": False, "charged": False}
    original_lease = prepared_files.lease
    acquisitions = []

    @contextlib.contextmanager
    def measured_lease(paths):
        acquisitions.append(clock["live"])
        with original_lease(paths) as pins:
            # Controlled cost at the unchanged full-hash boundary, not a fake _plan.
            if clock["live"] and not clock["charged"]:
                clock["now"] += 58.9076283
                clock["charged"] = True
            yield pins

    monkeypatch.setattr(prepared_files, "lease", measured_lease)
    monkeypatch.setattr(time, "time", lambda: clock["now"])
    actual_probe = host_operator._fresh_probe
    actual_main = cli.main
    phase_seconds = []
    live_return = []

    def timed_main(argv):
        started = time.perf_counter()
        result = actual_main(argv)
        phase_seconds.append((argv[0], time.perf_counter() - started))
        return result

    monkeypatch.setattr(cli, "main", timed_main)
    actual_quota = seal._quota
    guards = []

    def strict_quota(*args, **kwargs):
        try:
            return actual_quota(*args, **kwargs)
        except ValueError as error:
            if error.args == ("FRESH_ABSENCE_WINDOW",):
                guards.append("FRESH_ABSENCE_WINDOW")
            raise

    monkeypatch.setattr(seal, "_quota", strict_quota)
    ages = []

    def observed_probe(*args, **kwargs):
        result = actual_probe(*args, **kwargs)
        clock["now"] = result["absence_contract"]["started"] + (120.0 if expired else 63.2730298)
        clock["live"] = True
        ages.append(clock["now"] - result["absence_contract"]["started"])
        live_return.append(time.perf_counter())
        return result

    monkeypatch.setattr(host_operator, "_fresh_probe", observed_probe)
    for action in ("publish", "submit"):
        clock["live"] = False
        clock["charged"] = False
        prepared = _prepare(transport, case, plan_value, plan, review, action)
        first_return = len(live_return)
        commands = vectors(transport, case, plan, review, prepared, action)
        module = root / "payload/source/src/unified_normal_live.py"
        if module.exists():
            from unified_normal_live import run

            result = run(commands)
        else:
            result = 0
            for command in commands:
                result = cli.main(command)
                if result:
                    break
        if expired:
            assert result == 1 and transport.mutations == []
            assert guards == ["FRESH_ABSENCE_WINDOW"]
            assert not Path(commands[1][commands[1].index("--output") + 1]).exists()
            return
        assert result == 0, "POST_LIVE_IMMUTABLE_REPLAY_EXHAUSTED_STRICT_120_SECONDS"
        assert not clock["charged"], "full hashing was repeated after LIVE"
        assert all(age < 120 for age in ages)
        elapsed = time.perf_counter() - live_return[first_return]
        record_property(action + "_post_probe_wall_seconds", elapsed)
        assert 63.2730298 + elapsed < 120, "HOST_OVERHEAD_STILL_EXCEEDS_SAVED_MARGIN"
    assert [name for name, _ in transport.mutations if name != "transport"] == [
        "CreateDataset",
        "SaveKernel",
    ]
    assert {method for method, _ in wire.serialized} >= {"CreateDataset", "SaveKernel"}
    record_property("scope", "synthetic HTTP only; genuine whole predecessor replay; no cloud")
    record_property("ages", json.dumps(ages))
    record_property("full_hash_acquisitions", json.dumps(acquisitions))
    record_property("phase_wall_seconds", json.dumps(phase_seconds))


@pytest.fixture
def leased_file(monkeypatch):
    import uuid

    root = Path(os.environ["UNIFIED_NORMAL_CANDIDATE"])
    monkeypatch.syspath_prepend(str(root / "payload/source/src"))
    path = root.parent / "live-window-lease-fixtures" / uuid.uuid4().hex / "source.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original immutable source\n")
    return path


def test_retention_denies_replacement_and_rehashes_after_release(leased_file, monkeypatch):
    from seed202_pilot import prepared_files
    from unified_normal_live import retained_lease

    acquisitions = []
    actual = prepared_files.lease

    @contextlib.contextmanager
    def counted(paths):
        acquisitions.append(1)
        with actual(paths) as held:
            yield held

    monkeypatch.setattr(prepared_files, "lease", counted)
    with retained_lease([leased_file]) as first:
        with retained_lease([leased_file]) as nested:
            assert nested == first and len(acquisitions) == 1
            with pytest.raises(PermissionError):
                leased_file.write_bytes(b"replacement")
            with pytest.raises(PermissionError):
                leased_file.unlink()
    leased_file.write_bytes(b"replacement after release")
    with retained_lease([leased_file]) as changed:
        assert len(acquisitions) == 2 and changed != first
        assert (
            next(iter(changed.values()))["sha256"]
            == hashlib.sha256(b"replacement after release").hexdigest()
        )


def test_digest_reuses_only_held_authenticated_bytes(leased_file, monkeypatch):
    from seed202_pilot.storage import digest
    from unified_normal_live import retained_lease

    calls = []
    actual = hashlib.file_digest

    def counted(stream, algorithm):
        calls.append(1)
        return actual(stream, algorithm)

    monkeypatch.setattr(hashlib, "file_digest", counted)
    with retained_lease([leased_file]):
        assert digest(leased_file) == hashlib.sha256(leased_file.read_bytes()).hexdigest()
        assert len(calls) == 1, "immutable digest rehashed inside the held window"
    leased_file.write_bytes(b"changed after release")
    assert digest(leased_file) == hashlib.sha256(b"changed after release").hexdigest()
    assert len(calls) == 2


def test_root_closure_keeps_sibling_and_extra_file_guards(leased_file):
    from seed202_pilot.storage import digest
    from unified_normal_contract import verify_files
    from unified_normal_live import retained_lease

    sibling = leased_file.parent.with_name(leased_file.parent.name + "-sibling")
    sibling.mkdir()
    other = sibling / "source.py"
    other.write_bytes(b"separate root")
    files = {str(p): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in (leased_file, other)}
    value = {"files": files, "roots": [str(leased_file.parent), str(sibling)]}
    with retained_lease(files) as held:
        verify_files(value, held)
        unknown = leased_file.with_name("injected_import.py")
        unknown.write_bytes(b"unexpected import")
        with pytest.raises(ValueError, match="UNIFIED_IMPORT_ROOT_CLOSURE"):
            verify_files(value, held)
    # Differential check of the exact lexical projection used by the optimized
    # expected-set calculation, including Windows drive/case/boundary semantics.
    for name in ("E:/Root/a.py", "E:/Rooted/a.py", "e:/root/sub/a.py", "E:/Other/a.py"):
        for parent in ("E:/Root", "e:/root", "E:/", "E:/Other"):
            parts = tuple(os.path.normcase(p) for p in Path(name).parts)
            prefix = tuple(os.path.normcase(p) for p in Path(parent).parts)
            assert (parts[: len(prefix)] == prefix) == Path(name).is_relative_to(Path(parent))


def test_forged_boolean_is_not_a_lease(leased_file):
    from unified_normal_live import retained_lease

    with pytest.raises(TypeError):
        with retained_lease(True):
            pytest.fail("boolean granted a capability")
    with pytest.raises(TypeError):
        with retained_lease([leased_file], held=True):
            pytest.fail("caller boolean granted a capability")


def test_uncovered_path_is_fully_hashed(leased_file, monkeypatch):
    from seed202_pilot import prepared_files
    from unified_normal_live import retained_lease

    other = leased_file.with_name("other.py")
    other.write_bytes(b"new full byte identity")
    with retained_lease([leased_file]):
        actual = prepared_files.lease
        calls = []

        @contextlib.contextmanager
        def counted(paths):
            calls.append(paths)
            with actual(paths) as held:
                yield held

        monkeypatch.setattr(prepared_files, "lease", counted)
        with retained_lease([other]) as held:
            assert (
                calls
                and next(iter(held.values()))["sha256"]
                == hashlib.sha256(other.read_bytes()).hexdigest()
            )


@pytest.mark.parametrize("field", ["case", "plan", "review", "science-dir", "parent-archive"])
def test_chain_scope_mismatch_rejects_before_any_auth(leased_file, field, monkeypatch):
    from seed202_pilot import cli
    from unified_normal_live import run

    common = [
        "--candidate-dir",
        str(leased_file.parent),
        "--case",
        "case",
        "--parent-archive",
        "parent",
        "--private",
        "private",
        "--science-dir",
        "science",
        "--mqar-data",
        "data",
        "--lifetime",
        "lifetime",
        "--journal",
        "journal",
        "--plan",
        "plan",
        "--review",
        "review",
    ]
    commands = [
        [
            "probe",
            *common,
            "--preparation",
            "prepared",
            "--action",
            "publish",
            "--output",
            str(leased_file.parent / "proof"),
        ],
        [
            "seal",
            *common,
            "--proof",
            str(leased_file.parent / "proof"),
            "--output",
            str(leased_file.parent / "approval"),
        ],
        [
            "publish",
            *common,
            "--preparation",
            "prepared",
            "--approval",
            str(leased_file.parent / "approval"),
        ],
    ]
    commands[2][commands[2].index("--" + field) + 1] = "mismatched"
    calls = []
    monkeypatch.setattr(cli, "_provider", lambda *_: calls.append(1))
    with pytest.raises(ValueError, match="LIVE_SCOPE_MISMATCH"):
        run(commands)
    assert not calls


def test_process_only_credential_origins_are_not_artifact_leases(leased_file, monkeypatch):
    import types

    from seed202_pilot.preparation import paths

    root = Path(os.environ["UNIFIED_NORMAL_CANDIDATE"])
    credential = Path("C:/Users/Michael/Documents/projects/compare-attention/.env")
    actual = Path.read_bytes

    def no_private_read(path):
        assert path != credential, "engineering attempted a private credential read"
        return actual(path)

    monkeypatch.setattr(Path, "read_bytes", no_private_read)
    data = Path(
        "E:/HermesArtifacts/compare-attention/fp16-k64-small/"
        "272c2da7a9ef619711e4e92a1a98400ac0ed64b08f481d13752093d450820975/data"
    )
    args = types.SimpleNamespace(
        candidate_dir=root,
        case="mqar-k64-deltanet-seed202",
        mqar_data=data,
        auth_env=credential,
        private=credential,
    )
    selected = paths(args)
    assert str(credential) not in selected
    assert all(Path(p).drive == "E:" for p in selected)
