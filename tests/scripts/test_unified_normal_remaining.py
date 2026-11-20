"""Finite direct-user group; metadata only, no scientific execution or authority."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))


def test_remaining_group_is_exact_ordered_eight():
    from normal_remaining_scope import CASE_IDS, case_rows

    expected = tuple(
        f"{task}-{family}-seed{seed}"
        for seed in (303, 404)
        for task in ("sorting", "mqar-k64")
        for family in ("classic", "deltanet")
    )
    assert CASE_IDS == expected
    assert len(case_rows()) == 8
    for row, expected_id in zip(case_rows(), expected, strict=True):
        assert row["case_id"] == expected_id
        assert row["case_deadline_seconds"] == 28800
        assert row["requested_flops"] == 5_000_000_000_000_000
        assert row["global_batch"] == 2 * row["local_batch"] == 64
        assert row["learning_rate"] == 0.001 and row["lr_schedule"] == "constant"
        assert row["total_full_events"] == 11 and row["periodic_events"] == 10
        assert row["test_split_permitted"] is False


def test_remaining_generation_does_not_admit_other_cases_before_read(monkeypatch):
    from unified_normal_generation import intent_value
    from unified_normal_profiles import REMAINING, selected

    value = {
        "schema": "UNIFIED_NORMAL_GENERATION_INTENT_V1",
        "nonce": "23ad41c16ec140fe890d41d658f1c788",
        "case_id": "mqar-k64-deltanet-seed505",
        "source_identity_sha256": selected(REMAINING)["source"],
        "authorization": {"path": "MUST_NOT_READ", "sha256": "0" * 64},
    }
    with pytest.raises(ValueError, match="^UNIFIED_GENERATION_SCOPE$"):
        intent_value(value, REMAINING)


@pytest.mark.parametrize(
    "case", ["sorting-classic-seed202", "mqar-k64-deltanet-seed303", "ruler-classic-seed303"]
)
def test_new_group_does_not_expand_old_profile(case):
    from unified_normal_generation import intent_value
    from unified_normal_profiles import REPAIRED, selected

    value = {
        "schema": "UNIFIED_NORMAL_GENERATION_INTENT_V1",
        "nonce": "23ad41c16ec140fe890d41d658f1c788",
        "case_id": case,
        "source_identity_sha256": selected(REPAIRED)["source"],
        "authorization": {"path": "MUST_NOT_READ", "sha256": "0" * 64},
    }
    with pytest.raises(ValueError, match="^UNIFIED_GENERATION_SCOPE$"):
        intent_value(value, REPAIRED)


def test_first_fresh_prefix_is_empty_and_later_is_exact():
    from normal_remaining_scope import CASE_IDS, require_prefix

    assert require_prefix(CASE_IDS[0], []) == 0
    assert require_prefix(CASE_IDS[1], [CASE_IDS[0]]) == 1
    for case_id, prefix in (
        (CASE_IDS[0], ["mqar-k64-deltanet-seed202"]),
        (CASE_IDS[1], []),
        (CASE_IDS[2], [CASE_IDS[1], CASE_IDS[0]]),
    ):
        with pytest.raises(ValueError, match="^REMAINING_ACCEPTED_PREFIX$"):
            require_prefix(case_id, prefix)


def test_source_release_is_finite_and_typed_review_not_historical_review():
    from unified_normal_profiles import REMAINING, selected

    release = selected(REMAINING)
    assert release["review_policy"] == "EXTERNAL_TYPED_NORMAL_SOURCE_AND_PLAN"
    assert len(release["source"]) == len(release["archive"]) == 64
    assert release["prefix"] == "normal-package/"
    assert release["source"] != selected("REPAIRED_NORMAL_707_V1")["source"]


def test_actual_user_authorization_all_eight():
    from normal_remaining_scope import AUTHORIZATION_SHA256, CASE_IDS
    from unified_normal_generation import intent_value
    from unified_normal_profiles import REMAINING, selected

    path = Path(
        "E:/HermesArtifacts/compare-attention/remaining-seeds-sorting-mqar-v1/authorization01.json"
    )
    if not path.is_file():
        pytest.skip("actual private nonsecret user permission not present")
    for index, case_id in enumerate(CASE_IDS):
        value = {
            "schema": "UNIFIED_NORMAL_GENERATION_INTENT_V1",
            "nonce": f"a73e50811bc144fa9051bbccaadd{index:04x}",
            "case_id": case_id,
            "source_identity_sha256": selected(REMAINING)["source"],
            "authorization": {"path": str(path), "sha256": AUTHORIZATION_SHA256},
        }
        assert intent_value(value, REMAINING) == value


def test_forged_remaining_permission_is_rejected_before_read(tmp_path):
    from unified_normal_generation import intent_value
    from unified_normal_profiles import REMAINING, selected

    value = {
        "schema": "UNIFIED_NORMAL_GENERATION_INTENT_V1",
        "nonce": "23ad41c16ec140fe890d41d658f1c788",
        "case_id": "sorting-classic-seed303",
        "source_identity_sha256": selected(REMAINING)["source"],
        "authorization": {"path": str(tmp_path / "absent.json"), "sha256": "0" * 64},
    }
    with pytest.raises(ValueError, match="^UNIFIED_GENERATION_HUMAN_PROVENANCE$"):
        intent_value(value, REMAINING)


def test_generated_finite_contract_and_seed_guard(tmp_path):
    import ast
    import shutil

    from normal_remaining_scope import CASE_IDS
    from unified_normal_remaining import integrate_group

    old = Path("E:/HermesArtifacts/compare-attention/unb08/payload/source/src/seed202_pilot")
    if not old.is_dir():
        pytest.skip("real distributed host source not present")
    root = tmp_path / "emitted/seed202_pilot"
    shutil.copytree(old, root)
    integrate_group(root)
    scope = {}
    exec(compile((root / "contract.py").read_bytes(), str(root / "contract.py"), "exec"), scope)
    assert tuple(case.case_id for case in scope["cases"]()) == CASE_IDS
    assert scope["require_authorized_case"](CASE_IDS[0]).seed == 303
    with pytest.raises(ValueError, match="^UNAUTHORIZED_PILOT_CASE$"):
        scope["require_authorized_case"]("mqar-k64-deltanet-seed202")
    fresh = (root / "fresh_mode.py").read_text()
    assert 'deployment.get("seed") == case.seed' in fresh
    ast.parse(fresh)
    seal = (root / "seal.py").read_text()
    assert "transition(candidate, case_id)" in seal
    assert "verify_qualified_predecessor(" not in seal


def test_remaining_public_constructor_is_not_legacy_input_schema(tmp_path):
    from build_unified_normal_contract import build
    from unified_normal_profiles import REMAINING

    with pytest.raises(ValueError, match="^REMAINING_BUILD_INPUTS$"):
        build({"source_profile": REMAINING}, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


@pytest.fixture
def actual_packet():
    root = Path(
        "E:/HermesArtifacts/compare-attention/remaining-seeds-sorting-mqar-v1/launch-ready02"
    )
    if not (root / "first-singleton03.json").is_file():
        pytest.skip("actual private production plan packet not present")
    return root


def test_actual_eight_are_production_plans_and_full_native_grids(actual_packet):
    import hashlib
    import json

    from normal_remaining_scope import CASE_IDS
    from unified_normal_profiles import REMAINING, selected

    queue = json.loads((actual_packet / "QUEUE.json").read_bytes())["cases"]
    assert tuple(row["case_id"] for row in queue) == CASE_IDS
    for row in queue:
        plan_path = Path(row["normal_plan"])
        plan = json.loads(plan_path.read_bytes())
        case = json.loads(Path(row["normal_case"]).read_bytes())["case"]
        recipe = json.loads(Path(row["recipe"]).read_bytes())
        assert plan["engineering_only"] is case["engineering_only"] is False
        assert plan["purpose"] == "fresh_normal_training"
        assert plan["source_identity_sha256"] == selected(REMAINING)["source"]
        assert (
            hashlib.sha256(plan_path.read_bytes()).hexdigest()
            == case["plan_sha256"]
            == row["normal_plan_raw_sha256"]
        )
        assert tuple(plan["cases"]) == (row["case_id"],)
        assert case["checkpoint_steps"] == case["validation_steps"]
        assert len(case["checkpoint_steps"]) == 11
        assert case["checkpoint_steps"][-1] == case["updates"]
        assert recipe["schedule"]["seed"] in (303, 404)
        assert recipe["schedule"]["compile"] is True
        assert recipe["schedule"]["dtype"] == "float16"
        assert recipe["schedule"]["early_stopping_patience"] == 0
        assert recipe["schedule"]["log_every_steps"] == 1
        assert recipe["schedule"]["batch_size"] == 64
        assert recipe["optim"]["learning_rate"] == 0.001
        assert recipe["optim"]["lr_schedule"] == "constant"
        assert row["required_seconds"] == 30600


def test_first_actual_public_plan_is_fresh_empty_prefix_and_fixed_source(actual_packet):
    import ast
    import json
    import zipfile

    from unified_normal_profiles import REMAINING, selected

    plan = json.loads((actual_packet / "first-singleton03.json").read_bytes())
    candidate = Path(plan["candidate_root"])
    request = json.loads((candidate / "payload/unified/request.json").read_bytes())
    assert plan["case_id"] == "sorting-classic-seed303"
    assert plan["predecessors"] == request["predecessors"] == []
    assert plan["qualified_predecessor"]["schema"] == "DIRECT_AUTHORIZED_REMAINING_SERIES_V1"
    assert plan["qualified_predecessor"]["case_index"] == 0
    assert plan["qualified_predecessor"]["historical_completion_fabricated"] is False
    assert plan["source_authority"]["fresh"]["normal_authority_bound"] is False
    deploy = candidate / "payload/deploy" / plan["case_id"]
    tree = ast.parse((deploy / "kernel/kernel.py").read_bytes())
    saved = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "BINDING" for t in n.targets)
    )
    assert saved["case_id"] == plan["case_id"]
    assert saved["kernel_seconds"] == 30600
    assert saved["normal_source_sha256"] == selected(REMAINING)["source"]
    assert saved["nonce"] == plan["nonce"]
    assert (
        saved["operator_owner"] + "/" + saved["operator_slug"] + "/1"
        == plan["deployment"]["dataset_target"]
    )
    metadata = json.loads((deploy / "kernel/kernel-metadata.json").read_bytes())
    assert metadata["dataset_sources"] == [
        plan["deployment"]["dataset_target"],
        "zimike/sorting-quality-input-fcff1b5daf574b70/1",
        "zimike/k64-private-20261109a1b2/1",
    ]
    with zipfile.ZipFile(candidate / "payload/worker/seed202-pilot.zip.bin") as worker:
        assert (
            worker.read("normal-plan.json")
            == (candidate / "payload/unified/normal-plan.json").read_bytes()
        )
        assert not any(n.startswith("authority/") for n in worker.namelist())
        assert (
            worker.read("normal-package/normal_remaining_scope.py")
            == (actual_packet / "normal-package/normal_remaining_scope.py").read_bytes()
        )


@pytest.mark.parametrize("command", ["prepare", "probe", "seal", "publish", "submit", "reconcile"])
def test_actual_unbound_packet_stops_before_paths_or_credentials(actual_packet, command):
    from types import SimpleNamespace

    from unified_normal_authority import before_auth

    with pytest.raises(ValueError, match="^UNIFIED_NORMAL_EXECUTION_AUTHORITY_REQUIRED$"):
        before_auth(
            SimpleNamespace(command=command, candidate_dir=actual_packet / "first-candidate03")
        )


def test_later_public_request_cannot_use_empty_or_old_prefix(actual_packet, tmp_path):
    import json

    from build_unified_normal_contract import build

    inputs = json.loads((actual_packet / "cases/sorting-deltanet-seed303/inputs.json").read_bytes())
    with pytest.raises(ValueError, match="^REMAINING_ACCEPTED_PREFIX$"):
        build(inputs, tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


@pytest.mark.parametrize("mutation", ["nonce", "engineering", "source", "recipe", "data"])
def test_frozen_subject_denies_new_nonce_budget_or_recipe(actual_packet, tmp_path, mutation):
    import json

    from unified_normal_remaining import require_frozen_plan

    path = actual_packet / "cases/sorting-classic-seed303/normal-plan.json"
    require_frozen_plan("sorting-classic-seed303", path)
    value = json.loads(path.read_bytes())
    case = value["cases"]["sorting-classic-seed303"]
    if mutation == "nonce":
        value["generation"]["nonce"] = "a" * 32
    elif mutation == "engineering":
        value["engineering_only"] = True
    elif mutation == "source":
        value["source_identity_sha256"] = "0" * 64
    elif mutation == "recipe":
        case["recipe_sha256"] = "0" * 64
    else:
        case["data_files"]["train_inputs.npy"]["sha256"] = "0" * 64
    altered = tmp_path / "SYNTHETIC_ALTERED_PLAN.json"
    altered.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="^REMAINING_FROZEN_PLAN$"):
        require_frozen_plan("sorting-classic-seed303", altered)


@pytest.mark.parametrize(
    "case_id",
    [
        f"{task}-{family}-seed{seed}"
        for seed in (303, 404)
        for task in ("sorting", "mqar-k64")
        for family in ("classic", "deltanet")
    ],
)
def test_eight_deployments_select_only_fresh_template(actual_packet, tmp_path, case_id):
    import json

    from unified_normal_remaining import deployment

    payload = tmp_path / "SYNTHETIC_METADATA_ONLY"
    rows = [
        {"case_id": "old-sorting-fresh", "task": "sorting", "mode": "FRESH_TWO_STAGE"},
        {"case_id": "old-mqar-continuation", "task": "mqar-k64", "mode": "MQAR_CLASSIC_STEP980_V1"},
        {"case_id": "old-mqar-fresh", "task": "mqar-k64", "mode": "FRESH_TWO_STAGE"},
    ]
    for row in rows:
        root = payload / "deploy" / row["case_id"]
        (root / "dataset").mkdir(parents=True)
        (root / "kernel").mkdir()
        (root / "dataset/dataset-metadata.json").write_text(
            json.dumps({"id": "SYNTHETIC", "title": "SYNTHETIC"})
        )
        (root / "kernel/kernel-metadata.json").write_text(
            json.dumps({"id": "SYNTHETIC", "title": "SYNTHETIC"})
        )
        (root / "kernel/kernel.py").write_text(
            'BINDING = {"case_id": "SYNTHETIC", "kernel_seconds": 30600}\n'
        )
        if row["mode"] != "FRESH_TWO_STAGE":
            (root / "resume-input").mkdir()
            (root / "resume-input/SYNTHETIC_NOT_WEIGHTS.txt").write_text("MUST_NOT_INHERIT")
    case_dir = actual_packet / "cases" / case_id
    run = json.loads((case_dir / "normal-case.json").read_bytes())["case"]
    generation = json.loads((case_dir / "generation.json").read_bytes())
    plan = {"deployments": rows}
    result = deployment(payload, plan, run, generation)
    assert result["mode"] == "FRESH_TWO_STAGE"
    assert result["seed"] == run["seed"]
    assert result["case_id"] == run["id"]
    assert not (payload / "deploy" / case_id / "resume-input").exists()
    assert "resume_input_target" not in result
