"""Public repaired-source generation contract; genuine inputs, NOAUTH only."""

import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def genuine_inputs(monkeypatch):
    path = os.environ.get("UNIFIED_GENERATION_INPUTS")
    if not path:
        pytest.skip("genuine NOAUTH generation inputs not selected")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    inputs = json.loads(Path(path).read_bytes())
    inputs["source_profile"] = "REPAIRED_NORMAL_707_V1"
    return inputs


def test_public_builder_authenticates_review_archive_before_generation(genuine_inputs, tmp_path):
    from unified_normal_operator import build

    archive = Path(os.environ["UNIFIED_GENERATION_ARCHIVE"])
    output = tmp_path / "public-unbound-generation"
    with pytest.raises(ValueError, match="^UNIFIED_GENERATION_REQUIRED$"):
        build(genuine_inputs, archive, output)
    package = output / "payload/normal-package"
    assert (package / "normal-source-identity.json").is_file()
    assert len([p for p in package.rglob("*") if p.is_file()]) == 110
    assert not (output / "candidate.json").exists()


def generation_inputs(inputs, root):
    """A pending engineering intent, never independent approval or execution authority."""
    inputs = dict(inputs)
    base = Path(os.environ["UNIFIED_GENERATION_INPUTS"]).parent.parent.parent
    consent = base / "normal-primary-cloud-resume01/human-authorization01.json"
    inputs["source_review"] = str(
        base / "independent-normal-primary-final-review01/INDEPENDENT-VERDICT.json"
    )
    generation = {
        "schema": "UNIFIED_NORMAL_GENERATION_INTENT_V1",
        "nonce": "a85c734b60de40299da351a422ea1fcb",
        "case_id": "mqar-k64-deltanet-seed202",
        "source_identity_sha256": (
            "707e957e95be410aed329235fbaedc72459d854accf1962c73349bc630f2b3d7"
        ),
        "authorization": {
            "path": str(consent),
            "sha256": "d2ae90adec3a5a22f58aae0537064a8991159bcbdc2a0c82306c581b992da823",
        },
    }
    path = root / "pending-generation-intent-not-execution-authority.json"
    path.write_text(json.dumps(generation))
    inputs["generation"] = str(path)
    return inputs


def test_public_request_rederives_case_from_actual_repaired_plan(genuine_inputs, tmp_path):
    from build_unified_normal_contract import build
    from unified_normal_contract import digest

    inputs = generation_inputs(genuine_inputs, tmp_path)
    output = tmp_path / "request-build"
    build(inputs, output)
    request = json.loads((output / "payload/unified/request.json").read_bytes())
    plan_path = output / "payload/unified/normal-plan.json"
    plan = json.loads(plan_path.read_bytes())
    case = request["run"]
    assert (
        case["source_identity_sha256"]
        == plan["source_identity_sha256"]
        == ("707e957e95be410aed329235fbaedc72459d854accf1962c73349bc630f2b3d7")
    )
    assert case["plan_sha256"] == digest(plan_path)
    old_case = json.loads(Path(inputs["normal_case"]).read_bytes())["case"]
    projected = {"source_identity_sha256", "plan_sha256", "runtime_sha256"}
    assert {k: v for k, v in case.items() if k not in projected} == {
        k: v for k, v in old_case.items() if k not in projected
    }
    assert case["runtime_sha256"] != "0" * 64
    assert (
        plan["generation"]["nonce"]
        == (json.loads(Path(inputs["generation"]).read_bytes())["nonce"])
    )
    receipt = json.loads((output / "derive-case-receipt.json").read_bytes())
    assert receipt["returncode"] == 0
    assert receipt["selected_source"] == str(Path(inputs["normal_package"]) / "src")
    assert receipt["model_or_training_executed"] is False


def test_public_operator_binds_one_fresh_namespace_everywhere(genuine_inputs, tmp_path):
    import ast
    import zipfile

    from unified_normal_authority import before_auth, binding
    from unified_normal_operator import build
    from unified_normal_source import operator_profile

    inputs = generation_inputs(genuine_inputs, tmp_path)
    output = tmp_path / "operator"
    build(inputs, Path(os.environ["UNIFIED_GENERATION_ARCHIVE"]), output)
    payload = output / "payload"
    request = json.loads((payload / "unified/request.json").read_bytes())
    plan = json.loads((output / "PLAN-UNAPPROVED.json").read_bytes())
    deployment = next(d for d in plan["deployments"] if d["case_id"] == request["run"]["id"])
    nonce = json.loads(Path(inputs["generation"]).read_bytes())["nonce"]
    assert deployment["nonce"] == nonce
    assert deployment["dataset_target"] == "zimike/sorting-mqar-seed202-" + nonce[:16] + "/1"
    assert deployment["kernel_target"] == "zimike/seed202-pilot-" + nonce[:16]
    result = operator_profile(output, deployment)
    assert result["normal_profile"] == "REPAIRED_NORMAL_707_V1"
    assert result["normal_service_prefix"] == "normal-training/" + nonce
    assert result["normal_authority_bound"] is False
    absence_source = (payload / "source/src/seed202_pilot/host_services.py").read_text()
    assert '"REPAIRED_NORMAL_707_V1"' in absence_source
    deploy = payload / "deploy" / request["run"]["id"]
    metadata = json.loads((deploy / "kernel/kernel-metadata.json").read_bytes())
    assert metadata["id"] == deployment["kernel_target"]
    assert metadata["dataset_sources"][0] == deployment["dataset_target"]
    dataset = json.loads((deploy / "dataset/dataset-metadata.json").read_bytes())
    assert dataset["id"] + "/1" == deployment["dataset_target"]
    tree = ast.parse((deploy / "kernel/kernel.py").read_bytes())
    saved = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "BINDING" for t in n.targets)
    )
    assert saved["nonce"] == nonce
    assert (
        saved["operator_owner"] + "/" + saved["operator_slug"] + "/1"
        == deployment["dataset_target"]
    )
    assert saved["normal_source_sha256"] == request["run"]["source_identity_sha256"]
    with zipfile.ZipFile(payload / "worker/seed202-pilot.zip.bin") as worker:
        assert (
            worker.read("normal-plan.json") == (payload / "unified/normal-plan.json").read_bytes()
        )
        assert (
            worker.read("normal-package/normal-source-identity.json")
            == (payload / "normal-package/normal-source-identity.json").read_bytes()
        )
        assert not any(n.startswith("authority/") for n in worker.namelist())
    assert binding(output)["normal_authority_bound"] is False
    import types

    for command in ("publish", "submit"):
        with pytest.raises(ValueError, match="UNIFIED_NORMAL_EXECUTION_AUTHORITY_REQUIRED"):
            before_auth(types.SimpleNamespace(command=command, candidate_dir=output))


@pytest.mark.parametrize("command", ["prepare", "seal", "probe"])
def test_unbound_generation_stops_before_any_preparation(genuine_inputs, tmp_path, command):
    import types

    from unified_normal_authority import SCHEMA, before_auth

    root = tmp_path / "unbound-no-authority"
    path = root / "payload/unified/normal-authority.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {"schema": SCHEMA, "status": "UNBOUND_REVIEW_AND_DECISION_REQUIRED", "files": {}}
        )
    )
    with pytest.raises(ValueError, match="^UNIFIED_NORMAL_EXECUTION_AUTHORITY_REQUIRED$"):
        before_auth(types.SimpleNamespace(command=command, candidate_dir=root))


def test_public_unpack_rejects_zip_symlink(genuine_inputs, tmp_path):
    import zipfile

    from unified_normal_contract import digest
    from unified_normal_operator import unpack

    archive = tmp_path / "SYNTHETIC-UNSAFE-ARCHIVE.zip"
    entry = zipfile.ZipInfo("normal-package/symlink")
    entry.create_system = 3
    entry.external_attr = 0o120777 << 16
    with zipfile.ZipFile(archive, "x") as saved:
        saved.writestr(entry, "some-target")
    with pytest.raises(ValueError, match="^UNIFIED_SAFE_ARCHIVE$"):
        unpack(archive, tmp_path / "never-accepted", digest(archive), "normal-package/")


@pytest.mark.parametrize(
    "nonce", ["9ff571a0588c96318a1c0de76198db84", "4a2060304e6b47220000000000000000"]
)
def test_generation_denies_reused_historical_namespaces(genuine_inputs, tmp_path, nonce):
    from unified_normal_generation import intent_value
    from unified_normal_profiles import REPAIRED

    inputs = generation_inputs(genuine_inputs, tmp_path)
    value = json.loads(Path(inputs["generation"]).read_bytes())
    value["nonce"] = nonce
    with pytest.raises(ValueError, match="^UNIFIED_GENERATION_FRESH_NONCE$"):
        intent_value(value, REPAIRED)


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("case_id", "mqar-k64-deltanet-seed303", "SCOPE"),
        ("case_id", "sorting-deltanet-seed202", "SCOPE"),
        ("case_id", "mqar-k64-classic-seed202", "SCOPE"),
        ("source_identity_sha256", "0" * 64, "SCOPE"),
        ("nonce", "0" * 32, "FRESH_NONCE"),
        ("nonce", "8043dc515cd194e8f34777d51522a3b8", "FRESH_NONCE"),
        ("nonce", "8043dc515cd194e80000000000000000", "FRESH_NONCE"),
        ("nonce", "not-a-nonce", "FRESH_NONCE"),
        ("nonce", True, "FRESH_NONCE"),
        ("schema", "RETRY_ALL_FUTURE_CASES", "SCHEMA"),
        ("authorization", {"path": "NOT_READ", "sha256": "0" * 64}, "HUMAN_PROVENANCE"),
    ],
)
def test_generation_denies_cross_scope_intents(genuine_inputs, tmp_path, field, value, code):
    from unified_normal_generation import intent_value
    from unified_normal_profiles import REPAIRED

    inputs = generation_inputs(genuine_inputs, tmp_path)
    intent = json.loads(Path(inputs["generation"]).read_bytes())
    intent[field] = value
    with pytest.raises(ValueError, match="^UNIFIED_GENERATION_" + code + "$"):
        intent_value(intent, REPAIRED)


@pytest.fixture
def actual_candidate(monkeypatch):
    name = os.environ.get("UNIFIED_GENERATION_CANDIDATE")
    if not name:
        pytest.skip("actual final NOAUTH candidate not selected")
    root = Path(name)
    monkeypatch.syspath_prepend(str(root / "payload/source/src"))
    return root


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("source_identity_sha256", "0" * 64, "COMPLETE_RECIPE"),
        ("world_size", 1, "COMPLETE_RECIPE"),
        ("local_batch", 64, "COMPLETE_RECIPE"),
        ("rank_step_flops", 1, "CADENCE_AND_IDENTITY"),
        ("seed", 303, "CADENCE_AND_IDENTITY"),
    ],
)
def test_actual_profile_denies_case_drift(actual_candidate, tmp_path, field, value, code):
    import shutil

    from unified_normal_source import operator_profile

    root = tmp_path / "SYNTHETIC-METADATA-ONLY-NOT-AUTHORITY"
    folder = root / "payload/unified"
    folder.mkdir(parents=True)
    original = actual_candidate / "payload/unified"
    for name in ("normal-plan.json", "normal-authority.json"):
        shutil.copyfile(original / name, folder / name)
    request = json.loads((original / "request.json").read_bytes())
    request["run"][field] = value
    (folder / "request.json").write_text(json.dumps(request))
    case_id = request["run"]["id"]
    configs = root / "payload/configs"
    configs.mkdir()
    shutil.copyfile(
        actual_candidate / "payload/configs" / (case_id + ".yaml"), configs / (case_id + ".yaml")
    )
    plan = json.loads((actual_candidate / "PLAN-UNAPPROVED.json").read_bytes())
    deployment = next(d for d in plan["deployments"] if d["case_id"] == case_id)
    with pytest.raises(ValueError, match="^UNIFIED_NORMAL_" + code + "$"):
        operator_profile(root, deployment)


def test_generated_seal_denies_unbound_before_proof_or_auth(actual_candidate):
    from seed202_pilot.seal import seal

    with pytest.raises(ValueError, match="^UNIFIED_NORMAL_EXECUTION_AUTHORITY_REQUIRED$"):
        seal({"candidate_root": str(actual_candidate)}, {}, {})


@pytest.mark.parametrize(
    "case",
    [
        "mqar-k64-deltanet-seed303",
        "mqar-k64-deltanet-seed404",
        "sorting-deltanet-seed202",
        "mqar-k64-classic-seed202",
    ],
)
def test_actual_public_planner_denies_future_cases(actual_candidate, tmp_path, case):
    from seed202_pilot.seal import singleton_plan

    with pytest.raises(ValueError, match="^UNIFIED_REQUESTED_CASE_ONLY$"):
        singleton_plan(case, actual_candidate, tmp_path / "never-created-journal", [])
    assert not (tmp_path / "never-created-journal").exists()


def test_same_generation_rebuild_has_byte_equal_review_subject(genuine_inputs, tmp_path):
    from build_unified_normal_contract import build

    inputs = generation_inputs(genuine_inputs, tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    build(inputs, first)
    build(inputs, second)
    subject = "payload/unified/normal-plan.json"
    assert (first / subject).read_bytes() == (second / subject).read_bytes()
    assert json.loads((first / "derived-case.json").read_bytes()) == (
        json.loads((second / "derived-case.json").read_bytes())
    )


def test_actual_repaired_profile_reaches_ordinary_authority_guard(actual_candidate, genuine_inputs):
    from unified_normal_ordinary import replay

    spec = {
        "profile": "REPAIRED_NORMAL_707_V1",
        "source_request": str(actual_candidate / "payload/unified/request.json"),
        "result": "UNREACHED",
        "data": "UNREACHED",
        "snapshot": "UNREACHED",
        "index": "UNREACHED",
    }
    with pytest.raises(ValueError, match="^UNIFIED_ORDINARY_SOURCE_AUTHORITY$"):
        replay(spec, {}, {})


def test_actual_repaired_dispatch_reaches_ordinary_authority_guard(
    actual_candidate, genuine_inputs, tmp_path
):
    import hashlib

    from unified_normal_replay import replay

    recorded = tmp_path / "SYNTHETIC-OBSERVATION-NOT-ACCEPTANCE.json"
    recorded.write_text(json.dumps({"case": {"id": "mqar-k64-deltanet-seed202"}}))
    sha = hashlib.sha256(recorded.read_bytes()).hexdigest()
    request = json.loads((actual_candidate / "payload/unified/request.json").read_bytes())
    request["predecessors"] = [
        {
            "case_id": "mqar-k64-deltanet-seed202",
            "policy": "AUTHENTIC_HOST_NATIVE_V2_FULL_REPLAY",
            "path": str(recorded),
            "sha256": sha,
            "replay": {
                "profile": "REPAIRED_NORMAL_707_V1",
                "source_request": str(actual_candidate / "payload/unified/request.json"),
                "result": "UNREACHED",
                "data": "UNREACHED",
                "snapshot": "UNREACHED",
                "index": "UNREACHED",
            },
        }
    ]
    request["files"][str(recorded)] = {"bytes": recorded.stat().st_size, "sha256": sha}
    path = tmp_path / "private-dispatch-fixture/payload/unified/request.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="^UNIFIED_ORDINARY_SOURCE_AUTHORITY$"):
        replay(path, "mqar-k64-deltanet-seed202")


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("missing", "ROOT_CLOSURE"),
        ("extra", "ROOT_CLOSURE"),
        ("tamper", "SOURCE_BYTES"),
    ],
)
def test_profile_rejects_nonclosed_source(actual_candidate, tmp_path, mutation, code):
    import shutil

    from unified_normal_profiles import REPAIRED, authenticate_package

    root = tmp_path / "private-source-integrity-fixture"
    shutil.copytree(actual_candidate / "payload/normal-package", root)
    if mutation == "missing":
        (root / "src/normal_training_plan.py").unlink()
    elif mutation == "extra":
        (root / "src/UNREVIEWED.py").write_text("raise RuntimeError('UNREVIEWED')\n")
    else:
        p = root / "src/normal_training_plan.py"
        p.write_bytes(p.read_bytes() + b"\n# SYNTHETIC TAMPER\n")
    with pytest.raises(ValueError, match="^UNIFIED_NORMAL_" + code + "$"):
        authenticate_package(root, REPAIRED)


def test_profile_rejects_unreviewed_release(genuine_inputs):
    from unified_normal_profiles import selected

    with pytest.raises(ValueError, match="^UNIFIED_SUPPORTED_SOURCE_PROFILE$"):
        selected("ARBITRARY_SOURCE_OVERRIDE")


def test_profile_rejects_wrong_source_identity(actual_candidate):
    from unified_normal_profiles import LEGACY, authenticate_package

    with pytest.raises(ValueError, match="^UNIFIED_NORMAL_IDENTITY$"):
        authenticate_package(actual_candidate / "payload/normal-package", LEGACY)


@pytest.mark.parametrize("member", ["../escape", "/absolute", "C:/native", "bad\\\\name"])
def test_public_unpack_rejects_unsafe_members(genuine_inputs, tmp_path, member):
    import zipfile

    from unified_normal_contract import digest
    from unified_normal_operator import unpack

    archive = tmp_path / "SYNTHETIC-UNSAFE-MEMBERS.zip"
    with zipfile.ZipFile(archive, "x") as saved:
        entry = zipfile.ZipInfo("SYNTHETIC")
        entry.filename = member
        entry.orig_filename = member
        saved.writestr(entry, "SYNTHETIC")
    with pytest.raises(ValueError, match="^UNIFIED_SAFE_ARCHIVE$"):
        unpack(archive, tmp_path / "never-created", digest(archive))


def test_ordinary_cannot_downgrade_new_profile(actual_candidate, genuine_inputs):
    from unified_normal_ordinary import replay

    spec = {
        "profile": "REPAIRED_NORMAL_V1",
        "source_request": str(actual_candidate / "payload/unified/request.json"),
        "result": "UNREACHED",
        "data": "UNREACHED",
        "snapshot": "UNREACHED",
        "index": "UNREACHED",
    }
    with pytest.raises(ValueError, match="^UNIFIED_ORDINARY_REQUEST_PROFILE$"):
        replay(spec, {}, {})
