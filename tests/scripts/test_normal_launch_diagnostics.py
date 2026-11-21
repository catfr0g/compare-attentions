"""Tiny stdlib children at the emitted NORMAL supervisor; never scientific execution."""

import ast
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
import unified_normal_render as render

BASE = Path("E:/HermesArtifacts/compare-attention")
FROZEN = BASE / "unb02/payload"
REPO = Path(__file__).resolve().parents[2]
UV = [
    shutil.which("uv"),
    "run",
    "--offline",
    "--no-sync",
    "--no-project",
    "--python",
    sys.executable,
    "python",
    "-B",
]
SECRET = (
    "planted_private_token_78291 https://private.invalid/?token=78291 Authorization: Bearer 78291"
)


def pin(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def save_archive(root, target):
    files = {
        p.relative_to(root).as_posix(): pin(p.read_bytes())
        for p in root.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    (root / "manifest.json").write_text(json.dumps({"files": files}))
    with zipfile.ZipFile(target, "x") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
    return pin(target.read_bytes())["sha256"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def saved_entry_source():
    tree = ast.parse((REPO / "scripts/build_normal_training_lifecycle.py").read_bytes())
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_write"
        and len(n.args) == 3
        and isinstance(n.args[1], ast.Constant)
        and n.args[1].value == "worker.py"
    )
    return ast.literal_eval(call.args[2])


def tiny_kernel(
    tmp_path,
    monkeypatch,
    *,
    failing_phase="worker",
    hard_exit=False,
    pre_main_error=False,
    setup_failure=False,
    layout="flat",
    task="mqar-k64",
):
    """Real emitted kernel/main, source checks, setup.py, UV and DEVNULL supervisor."""
    deployment = {
        "case_id": "mqar-k64-deltanet-seed202",
        "worker_sha256": "0" * 64,
        "dataset_target": "zimike/synthetic-diagnostics-1111111111111111/1",
        "source_profile": "REPAIRED_NORMAL_V1",
        "nonce": "1" * 32,
        "task": task,
    }
    source = render.render({}, deployment, FROZEN)
    path = tmp_path / "kernel.py"
    path.write_text(source, encoding="utf8")
    kernel = load("tiny_saved_kernel", path)
    release = tmp_path / "release"
    package = release / "normal-package"
    (package / "src").mkdir(parents=True)
    shutil.copyfile(REPO / "scripts/normal_training_launch.py", package / "setup.py")
    for name in ("normal_launch_diagnostics", "normal_driver_environment", "normal_torchrun_entry"):
        owner = REPO / "scripts" / (name + ".py")
        if owner.exists():
            shutil.copyfile(owner, package / (name + ".py"))
    with zipfile.ZipFile(package / "compare_attention-0.1.0-py3-none-any.whl", "x"):
        pass
    # Stdlib-only children: torchrun may orchestrate them, never construct a model.
    child = "import os,sys\ndef main():\n    assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'\n"
    if hard_exit:
        child += (
            "    assert os.environ['WORLD_SIZE']=='2'\n"
            "    if os.environ['RANK']=='0':\n        os._exit(23)\n"
            "    import time\n    time.sleep(30)\n"
        )
    elif failing_phase == "worker":
        child += (
            "    assert os.environ['WORLD_SIZE']=='2'\n"
            "    if os.environ['RANK']=='0':\n        raise ValueError(" + repr(SECRET) + ")\n"
            "    raise RuntimeError(" + repr(SECRET) + ")\n"
        )
    elif failing_phase == "finish":
        child += "    raise ValueError(" + repr(SECRET) + ")\n"
    else:
        child += "    return 0\n"
    if pre_main_error:
        child = "raise ModuleNotFoundError(" + repr(SECRET) + ")\n"
    (package / "src/normal_training_worker.py").write_text(child)
    (package / "worker.py").write_bytes(saved_entry_source())
    files = {
        p.relative_to(package).as_posix(): pin(p.read_bytes())
        for p in package.rglob("*")
        if p.is_file()
    }
    identity = package / "normal-source-identity.json"
    identity.write_text(json.dumps({"files": files}))
    (release / "normal-plan.json").write_text(
        json.dumps(
            {
                "engineering_only": True,
                "source_identity_sha256": pin(identity.read_bytes())["sha256"],
            }
        )
    )
    if setup_failure:
        (release / "normal-plan.json").write_text(
            json.dumps({"engineering_only": True, "source_identity_sha256": "0" * 64})
        )
    (release / "recipe.yaml").write_text("{}")
    shutil.copyfile(REPO / "scripts/unified_normal_task.py", release / "normal-task.py")
    science = tmp_path / "science"
    bootstrap = science / "operator/inherited"
    tools = science / "tools"
    bootstrap.mkdir(parents=True)
    tools.mkdir()
    old = FROZEN / "normal-package/inherited/operator/inherited"
    shutil.copyfile(old / "process_control.py", bootstrap / "process_control.py")
    for name in ("session", "runtime"):
        shutil.copyfile(old / (name + ".py"), bootstrap / (name + ".py"))
    for name in ("tiny_uv", "sorting_runtime"):
        shutil.copyfile(old / "tools" / (name + ".py"), tools / (name + ".py"))
    manifest = {
        p.relative_to(science).as_posix(): pin(p.read_bytes())
        for p in science.rglob("*")
        if p.is_file()
    }
    (science / "manifest.json").write_text(json.dumps({"files": manifest}))
    mount = tmp_path / "mount"
    mount.mkdir()

    def mounted(slug):
        root = mount / "datasets/zimike" / slug if layout == "owner" else mount / slug
        root.mkdir(parents=True)
        return root

    archive = mounted(kernel.BINDING["operator_slug"]) / "seed202-pilot.zip.bin"
    science_archive = mounted("sorting-quality-input-fcff1b5daf574b70") / "scientific.zip.bin"
    kernel.BINDING.update(
        operator_sha256=save_archive(release, archive),
        science_sha256=save_archive(science, science_archive),
        normal_source_sha256=pin(identity.read_bytes())["sha256"],
    )
    data = mounted("ca-kv-1c660aa5f54d88ea9e0c1ce4") / "meta.json"
    data.write_text("{}")
    private = mounted("k64-private-20261109a1b2") / "private.env"
    private.write_text("SYNTHETIC_ONLY=1\n")

    monkeypatch.setattr(kernel, "driver_directory", lambda: None)
    monkeypatch.setattr(kernel, "inspect_hardware", lambda _driver: None)
    extract = kernel.extract_archive

    def extracted(*args):
        root = extract(*args)
        if root.name == "sorting-science-fcff1b5daf574b70":
            monkeypatch.syspath_prepend(str(root / "tools"))
            monkeypatch.syspath_prepend(str(root / "operator/inherited"))
            for name in ("session", "runtime", "process_control", "sorting_runtime", "tiny_uv"):
                monkeypatch.delitem(sys.modules, name, raising=False)
            import process_control
            import session

            monkeypatch.setattr(
                session,
                "setup",
                lambda *_args: (shutil.which("uv"), sys.executable, dict(os.environ), {}),
            )
            if os.name == "nt":

                def exited(process):
                    assert process.wait(timeout=10) != 0

                monkeypatch.setattr(process_control, "stop_group", exited)
            if failing_phase in ("finish", "task"):
                # Retain genuine finish supervisor but omit preceding scientific worker/task.
                real = process_control.bounded_run

                def selected(command, env, cwd, timeout):
                    if command[command.index("--phase") + 1] == failing_phase:
                        if failing_phase == "task":
                            command += ["--unknown", SECRET]
                        return real(command, env, cwd, timeout)

                monkeypatch.setattr(process_control, "bounded_run", selected)
        return root

    monkeypatch.setattr(kernel, "extract_archive", extracted)
    return kernel


def test_setup_primary_survives_real_supervisor(tmp_path, monkeypatch):
    kernel = tiny_kernel(tmp_path, monkeypatch, setup_failure=True)
    working = tmp_path / "working"
    with pytest.raises(subprocess.CalledProcessError) as caught:
        kernel.main(tmp_path / "mount", working, tmp_path / "extracted")
    assert caught.value.returncode == 1
    sink = working / ("normal-" + kernel.BINDING["nonce"]) / ".launch-diagnostics"
    records = [json.loads(p.read_bytes()) for p in sink.glob("*.json")]
    primary = next(
        (
            r
            for r in records
            if r.get("schema") == "normal-launch-primary-v1" and r.get("stage") == "setup"
        ),
        None,
    )
    assert primary is not None, "setup primary was lost at genuine DEVNULL supervisor"
    assert primary["phase"] == "worker"
    assert primary["exception_class"] == "ValueError"
    assert primary["category"] == "NORMAL_SETUP_NEW_SOURCE_BINDING"
    assert any(f["basename"] == "setup.py" and f["function"] == "main" for f in primary["frames"])
    assert getattr(caught.value, "_normal_primary", None) == primary
    assert SECRET not in json.dumps(records)


def test_kernel_console_projection_retains_safe_child_primary(tmp_path, monkeypatch):
    kernel = tiny_kernel(tmp_path, monkeypatch, setup_failure=True)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        kernel.main(tmp_path / "mount", tmp_path / "working", tmp_path / "extracted")
    public = kernel.public_error(caught.value, "entry")
    assert public["primary"]["category"] == "NORMAL_SETUP_NEW_SOURCE_BINDING"
    assert public["primary"]["phase"] == "worker"
    assert public["returncode"] == 1
    assert SECRET not in json.dumps(public)


def test_setup_bootstrap_failure_is_durable_before_worker(tmp_path, monkeypatch):
    kernel = tiny_kernel(tmp_path, monkeypatch)
    original = kernel.extract_archive

    def extracted(*args):
        root = original(*args)
        if root.name == "sorting-science-fcff1b5daf574b70":
            import session

            def fail(*_args):
                raise ModuleNotFoundError(SECRET)

            monkeypatch.setattr(session, "setup", fail)
        return root

    monkeypatch.setattr(kernel, "extract_archive", extracted)
    with pytest.raises(ModuleNotFoundError):
        kernel.main(tmp_path / "mount", tmp_path / "working", tmp_path / "extracted")
    sink = tmp_path / "working" / ("normal-" + kernel.BINDING["nonce"]) / ".launch-diagnostics"
    primary = [
        json.loads(p.read_bytes())
        for p in sink.glob("*.json")
        if "exception_class" in json.loads(p.read_bytes())
    ]
    assert len(primary) == 1
    assert primary[0]["phase"] == "setup" and primary[0]["stage"] == "bootstrap"
    assert primary[0]["exception_class"] == "ModuleNotFoundError"
    assert SECRET not in json.dumps(primary)


@pytest.mark.parametrize(
    "change",
    [
        {"exception_class": "planted_secret_identifier"},
        {"category": "planted_secret_identifier"},
        {"frames": [{"basename": "setup.py", "function": "planted_secret_identifier", "line": 1}]},
        {"frames": [{"basename": "private_token.py", "function": "main", "line": 1}]},
        {"frames": [{"basename": "setup.py", "function": "main", "line": True}]},
        {"phase": "finish"},
        {"rank": 1},
        {"source": "b" * 64},
        {"case": "other"},
        {"schema": "other"},
        {"message": SECRET},
        {"returncode": True},
        {"exception_class": [SECRET]},
        {"rank": False},
    ],
)
def test_supervisor_rejects_untrusted_sidecar(tmp_path, change):
    import normal_launch_diagnostics as diag

    context = diag.NormalDiagnostics(tmp_path, "worker", "a" * 64, "mqar-k64-deltanet-seed202")
    value = context.report(ValueError(), "pre-worker", 0)
    value.update(change)
    context.path("pre-worker", 0).write_text(json.dumps(value))

    def fail():
        raise subprocess.CalledProcessError(1, ["unused", SECRET])

    with pytest.raises(subprocess.CalledProcessError) as caught:
        context.supervise(fail)
    primary = caught.value._normal_primary
    assert primary["stage"] == "supervisor" and primary["exception_class"] == "CalledProcessError"
    assert primary["returncode"] == 1
    assert SECRET not in json.dumps(primary)


@pytest.mark.parametrize("raw", [b"{broken", b" " * 8193, b"[]"])
def test_bounded_sidecar_parse_failures_keep_outer_code(tmp_path, raw):
    import normal_launch_diagnostics as diag

    context = diag.NormalDiagnostics(tmp_path, "finish", "a" * 64, "mqar-k64-deltanet-seed202")
    context.path("finish", None).write_bytes(raw)

    def fail():
        raise subprocess.CalledProcessError(1, [SECRET])

    with pytest.raises(subprocess.CalledProcessError) as caught:
        context.supervise(fail)
    assert caught.value.returncode == 1
    assert caught.value._normal_primary["phase"] == "finish"
    assert caught.value._normal_primary["stage"] == "supervisor"


def test_bad_attached_record_cannot_mask_primary(tmp_path):
    import normal_launch_diagnostics as diag

    context = diag.NormalDiagnostics(tmp_path, "worker", "a" * 64, "mqar-k64-deltanet-seed202")
    error = RuntimeError(SECRET)
    error._normal_primary = {"phase": [SECRET]}

    def fail():
        raise error

    with pytest.raises(RuntimeError) as caught:
        context.capture(fail, "pre-worker")
    assert caught.value is error
    assert caught.value._normal_primary["exception_class"] == "RuntimeError"
    assert SECRET not in json.dumps(caught.value._normal_primary)


def test_missing_diagnostic_sink_does_not_mask_original(tmp_path):
    import normal_launch_diagnostics as diag

    context = diag.NormalDiagnostics(
        tmp_path / "absent", "worker", "a" * 64, "mqar-k64-deltanet-seed202"
    )
    error = ValueError(SECRET)

    def fail():
        raise error

    with pytest.raises(ValueError) as caught:
        context.capture(fail, "pre-worker")
    assert caught.value is error
    assert caught.value._normal_primary["exception_class"] == "ValueError"
    assert not context.root.exists()


def test_existing_scope_is_rejected_without_overwrite(tmp_path, monkeypatch):
    kernel = tiny_kernel(tmp_path, monkeypatch)
    root = tmp_path / "working" / ("normal-" + kernel.BINDING["nonce"]) / ".launch-diagnostics"
    root.mkdir(parents=True)
    old = root / "worker-setup-parent.json"
    old.write_bytes(b"PREEXISTING_IMMUTABLE_RECORD")
    with pytest.raises(FileExistsError):
        kernel.main(tmp_path / "mount", tmp_path / "working", tmp_path / "extracted")
    assert old.read_bytes() == b"PREEXISTING_IMMUTABLE_RECORD"
    assert not (tmp_path / "extracted").exists()


def test_coordinated_failure_retains_original_lifecycle_phase(tmp_path):
    from types import SimpleNamespace

    import normal_launch_diagnostics as diag

    from compare_attention.train.lifecycle import failure_phase, observe_failures

    observed = []
    error = None
    namespace = {}
    exec(
        compile(
            "def finish():\n    raise ValueError('" + SECRET + "')\n",
            "training_completion.py",
            "exec",
        ),
        namespace,
    )
    try:
        with observe_failures(SimpleNamespace(record_failure=observed.append), rank=0):
            try:
                with failure_phase("clearml_readback"):
                    namespace["finish"]()
            except ValueError:
                raise RuntimeError("coordinated secondary " + SECRET) from None
    except RuntimeError as caught:
        error = caught
    assert observed[0]["exception_class"] == "ValueError"
    assert observed[0]["phase"] == "clearml_readback"
    context = diag.NormalDiagnostics(tmp_path, "worker", "a" * 64, "mqar-k64-deltanet-seed202")

    def fail():
        raise error

    with pytest.raises(RuntimeError) as caught:
        context.capture(fail, "worker", 0)
    primary = context.read("worker", 0)
    assert primary["exception_class"] == "ValueError"
    assert primary["primary_phase"] == "clearml_readback"
    assert primary["phase"] == "worker" and primary["rank"] == 0
    assert primary["frames"][-1]["basename"] == "training_completion.py"
    assert SECRET not in json.dumps(primary)
    assert caught.value is error


def test_rank_zero_primary_wins_over_rank_one_preflight(tmp_path):
    import normal_launch_diagnostics as diag

    context = diag.NormalDiagnostics(tmp_path, "worker", "a" * 64, "mqar-k64-deltanet-seed202")
    primary = context.report(ValueError(), "worker", 0)
    context.persist(primary)
    coordinated = context.report(RuntimeError(), "pre-worker", 1)
    context.persist(coordinated)

    def fail():
        raise subprocess.CalledProcessError(1, [SECRET])

    with pytest.raises(subprocess.CalledProcessError) as caught:
        context.supervise(fail)
    assert caught.value._normal_primary == primary
    assert context.read("pre-worker", 1) == coordinated


def test_supervisor_stage_is_durable_before_real_child(tmp_path):
    import normal_launch_diagnostics as diag

    context = diag.NormalDiagnostics(tmp_path, "finish", "a" * 64, "mqar-k64-deltanet-seed202")
    marker = tmp_path / "finish-entered-setup.json"
    child = (
        "import json; from pathlib import Path; p=json.loads(Path("
        + repr(str(marker))
        + ").read_bytes()); assert p=="
        + repr(
            {
                "schema": "normal-launch-stage-v1",
                "source": "a" * 64,
                "case": "mqar-k64-deltanet-seed202",
                "phase": "finish",
                "stage": "setup",
                "callsite": "bounded_run",
            }
        )
    )

    def operation():
        result = subprocess.run(
            UV + ["-c", child],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=dict(os.environ),
            timeout=30,
        )
        assert result.returncode == 0
        return 17

    assert context.supervise(operation) == 17
    assert not list(tmp_path.glob("*-parent.json"))
