"""Finite Delta404 scheduling on synthetic children; no scientific execution."""

import pytest


@pytest.mark.parametrize(
    ("policy", "case_id", "deadline"),
    [
        ("RULER_CLASSIC404_2H_V1", "RulerClassic404", 7200),
        ("RULER_DELTANET404_4H_V1", "RulerDeltanet404", 14400),
    ],
)
def test_real_setup_owned_child(tmp_path, monkeypatch, policy, case_id, deadline):
    import hashlib
    import json
    import sys
    import zipfile

    import normal_training_launch as launch

    # Synthetic closed package; intercept only process creation, never run science.
    identity = {"files": {}, "operational_time_policy": policy}
    raw = json.dumps(identity).encode()
    (tmp_path / "normal-source-identity.json").write_bytes(raw)
    plan = tmp_path / "normal-plan.json"
    plan.write_text(
        json.dumps(
            {"source_identity_sha256": hashlib.sha256(raw).hexdigest(), "engineering_only": False}
        )
    )
    with zipfile.ZipFile(tmp_path / "compare_attention-0.1.0-py3-none-any.whl", "w") as wheel:
        wheel.writestr("SYNTHETIC.txt", "No native code")
    monkeypatch.setattr(launch, "__file__", str(tmp_path / "setup.py"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(launch, "validated_driver_environment", lambda *_: {})
    observed = []

    class OwnedChild:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def wait(self, timeout):
            observed.append(timeout)
            return 0

    monkeypatch.setattr(launch.subprocess, "Popen", OwnedChild)
    for phase, expected in (("worker", deadline), ("finish", 1800)):
        assert (
            launch.main(
                [
                    "--uv",
                    sys.executable,
                    "--python",
                    sys.executable,
                    "--",
                    "--plan",
                    str(plan),
                    "--phase",
                    phase,
                    "--case",
                    case_id,
                ]
            )
            == 0
        )
        assert observed[-1] == expected


def test_delta_policy_rejects_cross_case_and_engineering():
    from normal_training_launch import execution_timeout

    identity = {"operational_time_policy": "RULER_DELTANET404_4H_V1"}
    assert execution_timeout(identity, False, "RulerDeltanet404", "worker") == 14400
    assert execution_timeout(identity, False, "RulerDeltanet404", "finish") == 1800
    for case in (None, "RulerClassic404", "RulerDeltanet303", "unknown"):
        with pytest.raises(ValueError, match="^NORMAL_OPERATIONAL_TIME_SCOPE$"):
            execution_timeout(identity, False, case, "worker")
    with pytest.raises(ValueError, match="^NORMAL_OPERATIONAL_TIME_SCOPE$"):
        execution_timeout(identity, True, "RulerDeltanet404", "worker")
    assert execution_timeout({}, True, None, "worker") == 120
    assert execution_timeout({}, False, "RulerDeltanet202", "worker") == 28800


def test_delta_source_builder_freezes_new_finite_policy(tmp_path):
    import hashlib
    import json
    from pathlib import Path

    from build_ruler_training_request import build

    root = Path("E:/HermesArtifacts/compare-attention/ruler-variable-tracking-three-seeds-v1")
    row = json.loads((root / "preparation02/six-case-pricing.json").read_bytes())["cases"][-1]
    recipe = Path(row["recipe"])
    assert hashlib.sha256(recipe.read_bytes()).hexdigest() == row["recipe_sha256"]
    old = root / "operational-time-policy53/source02"
    runtime = json.loads((old / "normal-plan.json").read_bytes())["cases"]["RulerClassic404"][
        "runtime_sha256"
    ]
    output = build(
        root.parent
        / "tiny-shakespeare-three-seeds-v1/provider04/first-classic202-candidate03"
        / "payload/normal-package",
        Path(__file__).parents[2],
        old / "normal-package/compare_attention-0.1.0-py3-none-any.whl",
        recipe,
        old / "data/ruler",
        tmp_path / "source",
        "RulerDeltanet404",
        runtime,
        operational_time_policy="RULER_DELTANET404_4H_V1",
    )
    identity = json.loads((output / "normal-package/normal-source-identity.json").read_bytes())
    assert identity["scope"] == "RULER_DELTANET404_OPERATIONAL_TIME_V1"
    assert identity["operational_time_policy"] == "RULER_DELTANET404_4H_V1"
    case = json.loads((output / "derived-case.json").read_bytes())["case"]
    config_path = Path(row["derived_config"])
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == row["derived_config_sha256"]
    original = json.loads(config_path.read_bytes())
    assert {k: v for k, v in case["config"].items() if k != "data"} == {
        k: v for k, v in original.items() if k != "data"
    }
    assert (case["updates"], case["step_flops"], case["actual_flops"]) == (
        90944,
        54978936832,
        5000004431249408,
    )


def test_delta_future_profile_rows_and_cross_bindings():
    import json
    from pathlib import Path

    from normal_ruler_scope import deltanet404_case_rows
    from unified_normal_generation import intent_value
    from unified_normal_profiles import RULER, RULER_DELTA_TIME, RULER_TIME, selected

    root = Path("E:/HermesArtifacts/compare-attention/ruler-variable-tracking-three-seeds-v1")
    value = json.loads((root / "next-classic404-unbound48/generation.json").read_bytes())
    value.update(
        case_id="RulerDeltanet404", source_identity_sha256=selected(RULER_DELTA_TIME)["source"]
    )
    assert intent_value(value, RULER_DELTA_TIME) == value
    assert [row["case_deadline_seconds"] for row in deltanet404_case_rows()] == [28800] * 4 + [
        7200,
        14400,
    ]
    for case_id in ("RulerClassic404", "RulerDeltanet303", "RulerClassic202", "unknown"):
        with pytest.raises(ValueError, match="^RULER_OPERATIONAL_TIME_SCOPE$"):
            intent_value({**value, "case_id": case_id}, RULER_DELTA_TIME)
    for profile in (RULER, RULER_TIME):
        with pytest.raises(ValueError, match="^UNIFIED_GENERATION_SCOPE$"):
            intent_value(
                {**value, "source_identity_sha256": selected(profile)["source"]}, RULER_DELTA_TIME
            )


def test_delta_adapter_emits_shared_outer_limit(tmp_path):
    import ast
    from pathlib import Path

    from normal_ruler_scope import AUTHORIZATION_SHA256
    from unified_normal_profiles import RULER_DELTA_TIME, independent_adapter, selected

    template = tmp_path / "deploy/template"
    (template / "kernel").mkdir(parents=True)
    (template / "dataset").mkdir()
    (template / "kernel/kernel.py").write_text("BINDING = {}\n")
    for kind in ("kernel", "dataset"):
        (template / kind / (kind + "-metadata.json")).write_text("{}")
    plan = {
        "deployments": [{"case_id": "template", "task": "sorting", "mode": "FRESH_TWO_STAGE"}],
        "host_acceptance": {},
    }
    root = Path("E:/HermesArtifacts/compare-attention/ruler-variable-tracking-three-seeds-v1")
    generation = {
        "schema": "UNIFIED_NORMAL_GENERATION_INTENT_V1",
        "nonce": "9e498a7757e841d29c12a2a798a78347",
        "case_id": "RulerDeltanet404",
        "source_identity_sha256": selected(RULER_DELTA_TIME)["source"],
        "authorization": {
            "path": str(root / "USER-AUTHORIZATION01.json"),
            "sha256": AUTHORIZATION_SHA256,
        },
    }
    run = {
        "id": "RulerDeltanet404",
        "task": "ruler_variable_tracking",
        "seed": 404,
        "family": "deltanet",
        "config": {"dataset": {"n_val": 1000}},
    }
    result = independent_adapter(RULER_DELTA_TIME).deployment(tmp_path, plan, run, generation)
    assert (result["case_deadline_seconds"], result["acceptance_tail_seconds"]) == (14400, 1800)
    binding = ast.literal_eval(
        ast.parse((tmp_path / "deploy/RulerDeltanet404/kernel/kernel.py").read_text()).body[0].value
    )
    assert binding["kernel_seconds"] == 16200
    assert binding["case_id"] == "RulerDeltanet404"


def test_delta_accepts_authentic_classic404_timing_predecessor_only():
    import copy
    import json
    from pathlib import Path

    from unified_normal_ruler import validate_request

    root = Path("E:/HermesArtifacts/compare-attention/ruler-variable-tracking-three-seeds-v1")
    rows = json.loads(
        (root / "operational-time-policy53/boundary02/inputs-unbound.json").read_bytes()
    )["predecessors"]
    rows.append(
        {
            "case_id": "RulerClassic404",
            "policy": "AUTHENTIC_HOST_NATIVE_V2_FULL_REPLAY",
            "replay": {"profile": "RULER_CLASSIC404_OPERATIONAL_TIME_V1"},
        }
    )
    run = json.loads(
        (root / "deltanet404-operational-time63/source/derived-case.json").read_bytes()
    )["case"]
    value = {"run": run, "predecessors": rows}
    validate_request(value)
    bad = copy.deepcopy(value)
    bad["predecessors"][-1]["replay"]["profile"] = "RULER_INDEPENDENT_CURRENT_V1"
    with pytest.raises(ValueError, match="^RULER_ORDINARY_PREFIX_REQUIRED$"):
        validate_request(bad)
    for key, changed in (("seed", 303), ("optim", {"weight_decay": 0.0})):
        bad = copy.deepcopy(value)
        if key == "seed":
            bad["run"]["config"]["schedule"][key] = changed
        else:
            bad["run"]["config"][key] = changed
        with pytest.raises(ValueError, match="^RULER_FROZEN_SCIENTIFIC_CONFIG$"):
            validate_request(bad)


def test_timeout_rejects_malformed_policy_with_typed_guard():
    from normal_training_launch import execution_timeout

    for policy in ({}, [], True, 14400, "unreviewed"):
        with pytest.raises(ValueError, match="^NORMAL_OPERATIONAL_TIME_POLICY$"):
            execution_timeout(
                {"operational_time_policy": policy}, False, "RulerDeltanet404", "worker"
            )
