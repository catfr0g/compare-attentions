"""Saved new main -> exact NORMAL setup -> UV/Gloo -> native/service host acceptance."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tests.scripts.test_normal_training_closure import engineering_inputs


def archive(root, output):
    files = {
        p.relative_to(root).as_posix(): {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in root.rglob("*")
        if p.is_file()
    }
    (root / "manifest.json").write_text(json.dumps({"files": files}))
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as saved:
        for p in root.rglob("*"):
            if p.is_file():
                saved.write(p, p.relative_to(root).as_posix())
    return hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize("layout", ["owner", "flat"])
def test_saved_main_real_normal_child_and_host_acceptance(tmp_path, monkeypatch, layout):
    candidate = Path(os.environ["UNIFIED_NORMAL_CANDIDATE"])
    source = candidate / "payload/deploy/mqar-k64-deltanet-seed202/kernel/kernel.py"
    spec = importlib.util.spec_from_file_location("saved_unified_main", source)
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    worker = tmp_path / "worker"
    package = worker / "normal-package"
    shutil.copytree(candidate / "payload/normal-package", package)
    shutil.copyfile(
        candidate / "payload/source/src/unified_normal_task.py", worker / "normal-task.py"
    )
    inputs = engineering_inputs(tmp_path, package, seed=303, policy="constant")
    case_id, recipe, plan, private, data = inputs
    shutil.copyfile(recipe, worker / "recipe.yaml")
    shutil.copyfile(plan, worker / "normal-plan.json")
    mount = tmp_path / "input"

    def mounted(slug):
        path = mount / "datasets/zimike" / slug if layout == "owner" else mount / slug
        path.mkdir(parents=True)
        return path

    operator = mounted(kernel.BINDING["operator_slug"])
    operator_sha = archive(worker, operator / "seed202-pilot.zip.bin")
    science = tmp_path / "science"
    science.mkdir()
    with zipfile.ZipFile(candidate / "payload/inherited/sorting-science.zip.bin") as original:
        for name in original.namelist():
            if name.endswith(".py"):
                path = science / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(original.read(name))
    # The source/environment setup seam is external to NORMAL, and executes no
    # installs in this CPU fixture. The selected child below is NOT mocked.
    science_archive = mounted("sorting-quality-input-fcff1b5daf574b70") / "scientific.zip.bin"
    science_sha = archive(science, science_archive)
    mounted_data = mounted("ca-kv-1c660aa5f54d88ea9e0c1ce4")
    for p in data.iterdir():
        shutil.copyfile(p, mounted_data / p.name)
    mounted_private = mounted("k64-private-20261109a1b2") / "private.env"
    shutil.copyfile(private, mounted_private)
    phases = []
    monkeypatch.setattr(kernel, "driver_directory", lambda: "CPU_FIXTURE_NOT_GPU")
    monkeypatch.setattr(kernel, "inspect_hardware", lambda _driver: phases.append("hardware"))

    def prepared(*_args):
        return shutil.which("uv"), sys.executable, dict(os.environ), {"engineering_only": True}

    real_run = subprocess.run

    def selected(command, env, cwd, timeout):
        assert phases[0] == "hardware"
        phase = command[command.index("--phase") + 1]
        phases.append(phase)
        if phase != "task":
            assert "setup.py" == Path(command[command.index("-B") + 1]).name
            split = command.index("--", command.index("--uv"))
            command = [
                *command[:split],
                "--engineering-bootstrap",
                str(Path(__file__).with_name("normal_training_http_fixture.py")),
                *command[split:],
            ]
        else:
            assert "normal-task.py" == Path(command[command.index("-B") + 1]).name
        log = tmp_path / (layout + "-" + phase + ".log")
        with log.open("w") as stream:
            result = real_run(
                command,
                env=env,
                cwd=cwd,
                timeout=min(timeout, 180),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        assert result.returncode == 0, log.read_text()

    original_extract = kernel.extract_archive

    def extracted_bootstrap(*args):
        directory = original_extract(*args)
        if directory.name == "sorting-science-fcff1b5daf574b70":
            monkeypatch.syspath_prepend(str(directory / "tools"))
            monkeypatch.syspath_prepend(str(directory / "operator/inherited"))
            for name in ("process_control", "session", "runtime", "sorting_runtime", "tiny_uv"):
                monkeypatch.delitem(sys.modules, name, raising=False)
            session = importlib.import_module("session")
            process_control = importlib.import_module("process_control")
            monkeypatch.setattr(session, "setup", prepared)
            monkeypatch.setattr(process_control, "bounded_run", selected)
        return directory

    monkeypatch.setattr(kernel, "extract_archive", extracted_bootstrap)
    kernel.BINDING.update(operator_sha256=operator_sha, science_sha256=science_sha, case_id=case_id)
    output = tmp_path / "working"
    output.mkdir()
    kernel.main(mount, output, tmp_path / "extracted")
    run = output / ("normal-" + kernel.BINDING["nonce"])
    extracted = tmp_path / "extracted" / ("normal-" + kernel.BINDING["nonce"])
    final_inputs = (
        case_id,
        extracted / "recipe.yaml",
        extracted / "normal-plan.json",
        mounted_private,
        mounted_data,
    )
    host_env = dict(os.environ)
    for key in (
        "HF_HOME",
        "CLEARML_CACHE_DIR",
        "TORCH_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "XDG_CACHE_HOME",
        "UV_CACHE_DIR",
    ):
        host_env[key] = str(Path(os.environ["TMPDIR"]) / "unified-host-test-caches" / key.lower())
    http_state = mounted_private.parent / "http/state.json"
    before = json.loads(http_state.read_bytes())
    for phase in ("services", "accept"):
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
            str(Path(__file__).with_name("unified_normal_http_host.py")),
            str(candidate / "payload/source/src/unified_normal_host.py"),
            "--package",
            str(extracted / "normal-package"),
            "--recipe",
            str(final_inputs[1]),
            "--plan",
            str(final_inputs[2]),
            "--private",
            str(mounted_private),
            "--data",
            str(mounted_data),
            "--result",
            str(run),
            "--case",
            case_id,
            "--phase",
            phase,
            "--snapshot",
            str(tmp_path / "immutable-host-readbacks"),
            "--output",
            str(tmp_path / ("host-" + phase + ".json")),
        ]
        result = real_run(
            command, env=host_env, capture_output=True, text=True, timeout=180, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
    accepted = json.loads((tmp_path / "host-accept.json").read_bytes())
    assert accepted["status"] == "ACCEPTED_UNIFIED_NORMAL"
    assert accepted["publication"]["fresh_sdk_readback"] is True
    after = json.loads(http_state.read_bytes())
    assert len(before["tasks"]) == len(after["tasks"])
    assert sum("/commit/" in row[1] for row in before["calls"]) == sum(
        "/commit/" in row[1] for row in after["calls"]
    )
    from normal_training_acceptance import next_witness

    with pytest.raises(ValueError, match="NORMAL_TARGET_ACCEPTANCE_REQUIRED"):
        next_witness(accepted["lifecycle"], {}, {})
    assert phases == ["hardware", "worker", "task", "finish"]
    assert accepted["source"]["wheel_sha256"] == (
        "a1e1db146ac0fef3583edbc6bee1d41ed1e83dffe85702c09667d34c96047e90"
    )
    task = json.loads((run / "task-engineering.json").read_bytes())
    assert task["scientific_acceptance"] is False and task["task_evaluation_executed"] is False
    assert accepted["scope"] == "ENGINEERING_CPU_SDK_FIXTURE_ONLY"
    assert accepted["native"]["updates"] == 3 and not accepted["next_launch_authorized"]
