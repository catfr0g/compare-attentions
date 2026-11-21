"""Fresh saved NORMAL worker -> public parser/train -> host acceptance (CPU only)."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import build_normal_training_lifecycle as builder
import pytest

PARENT = Path(
    "E:/HermesArtifacts/compare-attention/sorting-mqar-seed202-pilot-v1/"
    "mqar-classic-resume-repair-v1/mount-postreview-real03/"
    "sorting-mqar-seed202-pilot-v1.zip"
)
REPO = Path(__file__).resolve().parents[2]


def test_fresh_normal_package_has_saved_worker_and_new_source_identity(tmp_path):
    package = builder.build_package(PARENT, REPO, tmp_path / "normal")
    identity = json.loads((package / "normal-source-identity.json").read_bytes())
    assert identity["scope"] == "FUTURE_NORMAL_TRAINING_LIFECYCLE_NOT_OLD_RESUME"
    assert identity["original_wheel_sha256"] == (
        "512cb5b073e47deca36d2bfb41f2d183e0711b8817b9a0aa18dbd2e07591b2d3"
    )
    assert identity["wheel_sha256"] != identity["original_wheel_sha256"]
    assert (package / "worker.py").is_file()
    assert (package / "setup.py").is_file()
    assert (package / "src/normal_training_worker.py").is_file()
    assert (package / "src/normal_training_acceptance.py").is_file()
    assert (package / "src/training_completion.py").read_bytes() == (
        REPO / "scripts/training_completion.py"
    ).read_bytes()
    assert not list(package.rglob("*.npy"))


def engineering_inputs(
    tmp_path, package, *, seed=303, policy="warmup_cosine", task="mqar", family="classic"
):
    from normal_training_plan import digest

    from tests.scripts.test_training_lifecycle import setup

    config = setup(tmp_path, seed=seed, policy=policy, task=task)
    if family != "classic":
        from compare_attention.train.flops import estimate_step

        config = config.model_copy(
            update={
                "model": config.model.model_copy(
                    update={"attention": family, "n_head": 1, "n_embd": 16}
                ),
                "schedule": config.schedule.model_copy(
                    update={"dtype": "float16", "compile": True, "amp_max_scale": 128.0}
                ),
            }
        )
        config = config.model_copy(
            update={
                "schedule": config.schedule.model_copy(
                    update={"train_flops": 3 * estimate_step(config).total}
                )
            }
        )
    config = config.model_copy(
        update={"schedule": config.schedule.model_copy(update={"log_every_steps": 1})}
    )
    recipe = tmp_path / "fresh-recipe.yaml"
    recipe.write_text(
        json.dumps(
            {
                "data": str(config.data),
                "dataset_task": "sorting" if task == "sorting" else "kv_retrieval",
                "distributed": True,
                "tracker": "clearml",
                "model": config.model.model_dump(mode="json"),
                "schedule": config.schedule.model_dump(mode="json"),
                "optim": config.optim.model_dump(mode="json"),
            }
        )
    )
    case_id = f"fresh-{task}-{family}-{seed}-{policy}"
    plan = tmp_path / "future-plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "normal-training-plan-v1",
                "purpose": "fresh_normal_training",
                "engineering_only": family == "classic",
                "source_identity_sha256": digest(package / "normal-source-identity.json"),
                "cases": {
                    case_id: {
                        "seed": seed,
                        "task": "sorting" if task == "sorting" else "kv_retrieval",
                        "family": family,
                        "recipe_sha256": digest(recipe),
                        "runtime_sha256": "0" * 64,
                        "data_files": {
                            p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
                            for p in config.data.iterdir()
                            if p.is_file()
                        },
                    }
                },
            }
        )
    )
    private = tmp_path / "fixture-only.env"
    private.write_text(
        "\n".join(
            [
                "CLEARML_API_HOST=https://fixture-api.invalid",
                "CLEARML_WEB_HOST=https://fixture-web.invalid",
                "CLEARML_FILES_HOST=https://fixture-files.invalid",
                "CLEARML_API_ACCESS_KEY=fixture-access",
                "CLEARML_API_SECRET_KEY=fixture-secret",
                "HF_TOKEN=hf_FIXTURE_ONLY",
                "HF_REPO=fixture/repo",
            ]
        )
    )
    return case_id, recipe, plan, private, config.data


def saved_phase(package, inputs, output, phase, *, expected_code=0):
    case_id, recipe, plan, private, data = inputs
    command = [
        shutil.which("uv"),
        "run",
        "--offline",
        "--no-project",
        "--no-sync",
        "--python",
        sys.executable,
        "python",
        "-B",
        str(package / "setup.py"),
        "--uv",
        shutil.which("uv"),
        "--python",
        sys.executable,
        "--engineering-bootstrap",
        str(Path(__file__).with_name("normal_training_http_fixture.py")),
        "--",
        "--package",
        str(package),
        "--plan",
        str(plan),
        "--recipe",
        str(recipe),
        "--data",
        str(data),
        "--case",
        case_id,
        "--prefix",
        "normal-training/ENGINEERING_ONLY/" + case_id,
        "--private",
        str(private),
        "--output",
        str(output),
        "--phase",
        phase,
    ]
    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper()
        in {
            "SYSTEMROOT",
            "WINDIR",
            "PATH",
            "PATHEXT",
            "COMSPEC",
            "OS",
            "USERNAME",
            "USER",
            "LOGNAME",
            "HOME",
            "USERPROFILE",
            "APPDATA",
            "LOCALAPPDATA",
            "NUMBER_OF_PROCESSORS",
            "PROCESSOR_ARCHITECTURE",
            "TMPDIR",
            "TEMP",
            "TMP",
            "UV_CACHE_DIR",
        }
    }
    env.update(
        CUDA_VISIBLE_DEVICES="-1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHON_DOTENV_DISABLED="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    log = output.parent / (phase + ".log")
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            env=env,
            cwd=package,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
    assert result.returncode == expected_code, log.read_text(encoding="utf-8", errors="replace")
    return log.read_text(encoding="utf-8", errors="replace")


@pytest.mark.parametrize("seed", [202, 303, 404])
@pytest.mark.parametrize("policy", ["constant", "warmup_cosine"])
@pytest.mark.parametrize("task", ["mqar", "sorting"])
def test_actual_saved_uv_worker_public_training_service_and_host_acceptance(
    tmp_path, seed, policy, task, monkeypatch
):
    package = builder.build_package(PARENT, REPO, tmp_path / "normal")
    inputs = engineering_inputs(tmp_path, package, seed=seed, policy=policy, task=task)
    output = tmp_path / "run"
    for phase in ("worker", "finish", "accept"):
        saved_phase(package, inputs, output, phase)
    accepted = json.loads((tmp_path / "run-normal-accepted.json").read_bytes())
    assert accepted["status"] == "ACCEPTED_NORMAL_LIFECYCLE"
    assert accepted["scope"] == "ENGINEERING_CPU_SDK_FIXTURE_ONLY"
    assert accepted["case"]["seed"] == seed
    assert accepted["native"]["updates"] == 3
    assert accepted["training"]["expected_learning_rate_events"] == (
        0 if policy == "constant" else 3
    )
    assert accepted["source"]["source_rank_records"] == 2
    assert accepted["next_launch_authorized"] is False
    state = json.loads((tmp_path / "http/state.json").read_bytes())
    calls = state["calls"]
    assert sum("/commit/" in row[1] for row in calls) == 3
    assert any("events.get_task_events" in row[1] for row in calls)
    assert sum("/resolve/" in row[1] for row in calls) > 12
    # These exact public SDK discovery endpoints intentionally carry no token.
    # Private reads and all writes still traverse strict transport assertions.
    assert all(row[2] for row in calls if row[1] not in {"/updates", "/api/agent-harnesses"})
    import torch
    from normal_training_acceptance import verify_native_service
    from normal_training_native import verify_native

    raw_service = json.loads((output / "out/training-unsampled-readback.json").read_bytes())
    verify_native_service(output, raw_service)
    import copy

    corrupted = copy.deepcopy(raw_service)
    next(
        row for row in corrupted["events"] if row["metric"] == "train" and row["variant"] == "loss"
    )["value"] += 0.5
    with pytest.raises(ValueError, match="NORMAL_NATIVE_SERVICE_VALUES"):
        verify_native_service(output, corrupted)

    monkeypatch.syspath_prepend(str(package / "inherited/operator"))
    rng = torch.get_rng_state().clone()
    assert verify_native(output / "out", accepted["case"]) == accepted["native"]
    assert torch.equal(rng, torch.get_rng_state()), "structural audit changed CPU RNG"
    metrics = output / "out/metrics.jsonl"
    raw = metrics.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    rows[-1]["quality"]["val_loss"] += 1
    try:
        metrics.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        with pytest.raises(ValueError, match="NORMAL_NATIVE_QUALITY"):
            verify_native(output / "out", accepted["case"])
    finally:
        metrics.write_bytes(raw)


@pytest.mark.parametrize(
    "mode,phase",
    [
        ("result_log", "clearml_finish"),
        ("flush", "clearml_flush"),
        ("close", "clearml_close"),
        ("readback", "clearml_readback"),
        ("receipt_write", "clearml_raw_receipt_write"),
        ("identity", "identity_write"),
        ("diagnostic", "clearml_readback"),
    ],
)
def test_saved_gloo_service_primary_is_safe_in_json_and_console(tmp_path, mode, phase):
    package = builder.build_package(PARENT, REPO, tmp_path / "normal")
    inputs = engineering_inputs(tmp_path, package)
    (tmp_path / "failure-mode.json").write_text(json.dumps({"mode": mode}))
    output = tmp_path / "run"
    if mode != "identity":
        console = saved_phase(package, inputs, output, "worker", expected_code=0)
        diagnostics = json.loads((output / "training-diagnostics.json").read_bytes())
        assert diagnostics["warnings"]
        assert all(row["diagnostic"]["status"] == "warning" for row in diagnostics["warnings"])
        assert not (output / "lifecycle-failure.json").exists()
        assert json.loads((output / "out/run_identity.json").read_bytes())["status"] == "completed"
        all_new_json = "\n".join(path.read_text() for path in output.rglob("*.json"))
        for canary in ("CANARY_PRIVATE", "CANARY_SECONDARY", "private.invalid", "https://private"):
            assert canary not in all_new_json
            assert canary not in console
        return
    console = saved_phase(package, inputs, output, "worker", expected_code=1)
    report = json.loads((output / "lifecycle-failure.json").read_bytes())
    assert report["exception_class"] == "LookupError"
    assert report["phase"] == phase
    assert report["frames"][-1]["function"] == "fail"
    assert "LookupError" in console and phase in console
    all_new_json = "\n".join(path.read_text() for path in output.rglob("*.json"))
    for canary in ("CANARY_PRIVATE", "CANARY_SECONDARY", "private.invalid", "https://private"):
        assert canary not in all_new_json
        assert canary not in console
    assert not (tmp_path / "run-normal-accepted.json").exists()
