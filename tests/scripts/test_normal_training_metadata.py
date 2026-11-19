"""CPU metadata-only future recipes, native precision and fail-closed next boundary."""

import json

import build_normal_training_lifecycle as builder
import normal_training_native as native
import pytest
from normal_training_acceptance import next_witness
from normal_training_plan import derive, execution_authority

from tests.scripts.test_normal_training_closure import PARENT, REPO, engineering_inputs


def test_native_full_scaler_schema_is_not_identity_precision_metadata():
    payload = {
        "precision": {
            "optimizer_steps": 3,
            "skipped_optimizer_steps": 0,
            "scaler": {
                "scale": 128.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 2000,
                "_growth_tracker": 3,
            },
        }
    }
    native.validate_precision(payload, {"schedule": {"amp_max_scale": 128.0}}, 3)
    payload["precision"]["scaler"]["_growth_tracker"] = 2
    with pytest.raises(ValueError, match="NORMAL_FULL_SCALER"):
        native.validate_precision(payload, {"schedule": {"amp_max_scale": 128.0}}, 3)


@pytest.mark.parametrize("seed", [202, 303, 404])
@pytest.mark.parametrize("policy", ["constant", "warmup_cosine"])
@pytest.mark.parametrize("task", ["mqar", "sorting"])
@pytest.mark.parametrize("family", ["classic", "deltanet"])
def test_frozen_recipe_declares_task_family_seed_without_kv_task_key(
    tmp_path, seed, policy, task, family
):
    package = builder.build_package(PARENT, REPO, tmp_path / "normal")
    case_id, recipe, plan, _, data = engineering_inputs(
        tmp_path, package, seed=seed, policy=policy, task=task, family=family
    )
    case = derive(package, plan, case_id, recipe, data)
    assert case["family"] == family and case["seed"] == seed
    assert case["config"]["optim"]["lr_schedule"] == policy
    assert case["task"] == ("sorting" if task == "sorting" else "kv_retrieval")
    assert case["updates"] == 3
    if task == "mqar":
        assert "task" not in case["config"]["dataset"]
    if family == "classic":
        with pytest.raises(ValueError, match="NORMAL_ENGINEERING_NOT_AUTHORITY"):
            execution_authority(
                case, tmp_path / "not-created-review.json", tmp_path / "not-created-decision.json"
            )
    assert not (tmp_path / "not-created-review.json").exists()


def test_engineering_receipt_cannot_admit_next_scientific_case():
    with pytest.raises(ValueError, match="NORMAL_TARGET_ACCEPTANCE_REQUIRED"):
        next_witness({"scope": "ENGINEERING_CPU_SDK_FIXTURE_ONLY"}, {}, {})


def test_old_resume_authority_is_rejected_before_missing_review_is_read(tmp_path):
    package = builder.build_package(PARENT, REPO, tmp_path / "normal")
    case_id, recipe, plan, _, data = engineering_inputs(tmp_path, package)
    value = json.loads(plan.read_bytes())
    value["purpose"] = "resume_10780"
    plan.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="NORMAL_PLAN_PURPOSE"):
        derive(package, plan, case_id, recipe, data)
