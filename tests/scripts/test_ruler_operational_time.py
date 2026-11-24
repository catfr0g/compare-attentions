"""Future-only operational limits; no scientific execution or provider calls."""


def test_setup_main_passes_timeout_to_owned_child(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys
    import zipfile

    import normal_training_launch as launch

    # Synthetic closed package; intercept only process creation, never run science.
    identity = {"files": {}, "operational_time_policy": "RULER_CLASSIC404_2H_V1"}
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
    for phase, expected in (("worker", 7200), ("finish", 1800)):
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
                    "RulerClassic404",
                ]
            )
            == 0
        )
        assert observed[-1] == expected


def test_unchanged_real_quota_guard_has_exact_two_hour_threshold():
    import datetime
    import math
    from pathlib import Path
    from types import SimpleNamespace

    import pytest

    path = Path(
        "E:/HermesArtifacts/compare-attention/ruler-variable-tracking-three-seeds-v1/next-classic404-bound49/candidate/payload/source/src/seed202_pilot/preflight.py"
    )
    lines = path.read_text().splitlines()[165:183]
    code = compile("\n".join(line[4:] for line in lines), str(path), "exec")

    def require(ok, guard):
        if not ok:
            raise ValueError(guard)

    def check(available, reserved=0, deadline=7200):
        quota = SimpleNamespace(
            total_time_allowed=datetime.timedelta(seconds=available + reserved),
            time_used=datetime.timedelta(),
            time_reserved=datetime.timedelta(seconds=reserved),
        )
        scope = {
            "provider": SimpleNamespace(
                api=SimpleNamespace(quota_view=lambda: SimpleNamespace(gpu_quota=quota))
            ),
            "math": math,
            "deployment": {"case_deadline_seconds": deadline, "acceptance_tail_seconds": 1800},
            "_require": require,
        }
        exec(code, scope)
        return scope

    assert check(11250)["needed"] == 9000
    assert check(11250)["planning"] == 9000
    assert check(36824.961)["planning"] > 9000
    for available, reserved, deadline in (
        (11249, 0, 7200),
        (11250, 1, 7200),
        (36824.961, 0, 28800),
    ):
        with pytest.raises(ValueError, match="^QUOTA_RESERVE$"):
            check(available, reserved, deadline)


def test_source_builder_freezes_policy_without_changing_science(tmp_path):
    import json
    from pathlib import Path

    from build_ruler_training_request import build

    root = Path("E:/HermesArtifacts/compare-attention/ruler-variable-tracking-three-seeds-v1")
    parent = (
        root.parent
        / "tiny-shakespeare-three-seeds-v1/provider04/first-classic202-candidate03"
        / "payload/normal-package"
    )
    source = root / "next-classic404-unbound48/source"
    runtime = json.loads((source / "normal-plan.json").read_bytes())["cases"]["RulerClassic404"][
        "runtime_sha256"
    ]
    output = build(
        parent,
        Path(__file__).parents[2],
        source / "normal-package/compare_attention-0.1.0-py3-none-any.whl",
        source / "recipe.yaml",
        source / "data/ruler",
        tmp_path / "source",
        "RulerClassic404",
        runtime,
        operational_time_policy=True,
    )
    identity = json.loads((output / "normal-package/normal-source-identity.json").read_bytes())
    assert identity["operational_time_policy"] == "RULER_CLASSIC404_2H_V1"
    assert (output / "normal-package/compare_attention-0.1.0-py3-none-any.whl").read_bytes() == (
        source / "normal-package/compare_attention-0.1.0-py3-none-any.whl"
    ).read_bytes()
    old = json.loads((source / "derived-case.json").read_bytes())["case"]
    new = json.loads((output / "derived-case.json").read_bytes())["case"]
    assert {k: v for k, v in old["config"].items() if k != "data"} == {
        k: v for k, v in new["config"].items() if k != "data"
    }


def test_future_adapter_emits_matching_planning_and_kernel_limits(tmp_path):
    import ast
    import json

    from unified_normal_ruler_time import deployment

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
    from pathlib import Path

    from normal_ruler_scope import AUTHORIZATION_SHA256
    from unified_normal_profiles import RULER_TIME, selected

    root = Path("E:/HermesArtifacts/compare-attention/ruler-variable-tracking-three-seeds-v1")
    generation = {
        "schema": "UNIFIED_NORMAL_GENERATION_INTENT_V1",
        "nonce": "e99b1d94e65c44dcb5dd4f335692d499",
        "case_id": "RulerClassic404",
        "source_identity_sha256": selected(RULER_TIME)["source"],
        "authorization": {
            "path": str(root / "USER-AUTHORIZATION01.json"),
            "sha256": AUTHORIZATION_SHA256,
        },
    }
    run = {
        "id": "RulerClassic404",
        "task": "ruler_variable_tracking",
        "seed": 404,
        "family": "classic",
        "config": {"dataset": {"n_val": 1000}},
    }
    result = deployment(tmp_path, plan, run, generation)
    assert result["case_deadline_seconds"] == 7200
    assert result["acceptance_tail_seconds"] == 1800
    assert result["source_profile"] == RULER_TIME
    binding = ast.literal_eval(
        ast.parse((tmp_path / "deploy/RulerClassic404/kernel/kernel.py").read_text()).body[0].value
    )
    assert binding["kernel_seconds"] == 9000
    assert binding["case_id"] == "RulerClassic404"
    assert (
        json.loads((tmp_path / "deploy/RulerClassic404/kernel/kernel-metadata.json").read_bytes())[
            "id"
        ]
        == result["kernel_target"]
    )


def test_future_profile_rejects_other_cases_and_wrong_source_or_authorization():
    import copy
    import json
    from pathlib import Path

    import pytest
    from unified_normal_generation import intent_value
    from unified_normal_profiles import RULER, RULER_TIME, selected

    root = Path("E:/HermesArtifacts/compare-attention/ruler-variable-tracking-three-seeds-v1")
    value = json.loads((root / "next-classic404-unbound48/generation.json").read_bytes())
    value["source_identity_sha256"] = selected(RULER_TIME)["source"]
    assert intent_value(value, RULER_TIME) == value
    for case_id in ("RulerClassic202", "RulerDeltanet404", "TinyClassic404", "unknown"):
        bad = {**value, "case_id": case_id}
        with pytest.raises(ValueError, match="^RULER_OPERATIONAL_TIME_SCOPE$"):
            intent_value(bad, RULER_TIME)
    with pytest.raises(ValueError, match="^UNIFIED_GENERATION_SCOPE$"):
        intent_value({**value, "source_identity_sha256": selected(RULER)["source"]}, RULER_TIME)
    bad = copy.deepcopy(value)
    bad["authorization"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="^UNIFIED_GENERATION_HUMAN_PROVENANCE$"):
        intent_value(bad, RULER_TIME)


def test_future_classic404_limit_preserves_historical_rows():
    import normal_ruler_scope as scope

    historical = scope.case_rows()
    current = scope.operational_case_rows()
    assert [row["case_deadline_seconds"] for row in historical] == [28800] * 6
    assert [row["case_deadline_seconds"] for row in current] == [28800] * 4 + [7200, 28800]
    for old, new in zip(historical, current, strict=True):
        assert {k: v for k, v in old.items() if k != "case_deadline_seconds"} == {
            k: v for k, v in new.items() if k != "case_deadline_seconds"
        }


def test_setup_uses_finite_authenticated_worker_deadline():
    from normal_training_launch import execution_timeout

    policy = {"operational_time_policy": "RULER_CLASSIC404_2H_V1"}
    assert execution_timeout(policy, False, "RulerClassic404", "worker") == 7200
    assert execution_timeout(policy, False, "RulerClassic404", "finish") == 1800
    assert execution_timeout(policy, False, "RulerClassic404", "accept") == 28800
    assert execution_timeout({}, False, "RulerClassic404", "worker") == 28800
    assert execution_timeout({}, True, "synthetic", "worker") == 120
    import pytest

    for case in ("RulerClassic202", "RulerDeltanet404", "invalid"):
        with pytest.raises(ValueError, match="^NORMAL_OPERATIONAL_TIME_SCOPE$"):
            execution_timeout(policy, False, case, "worker")
    with pytest.raises(ValueError, match="^NORMAL_OPERATIONAL_TIME_POLICY$"):
        execution_timeout(
            {"operational_time_policy": "arbitrary"}, False, "RulerClassic404", "worker"
        )
