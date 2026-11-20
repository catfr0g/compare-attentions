"""Tiny CPU tests of emitted public entry and diagnostic transport rejection."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import normal_launch_diagnostics as diag
import pytest

from tests.scripts.test_normal_launch_diagnostics import (
    REPO,
    SECRET,
    UV,
    load,
    pin,
    saved_entry_source,
)


def test_early_saved_entry_keeps_dependency_import_primary(tmp_path, monkeypatch):
    # Use the builder's actual saved entry literal, with a stdlib-only import rejection.
    raw = saved_entry_source()
    (tmp_path / "worker.py").write_bytes(raw)
    (tmp_path / "src").mkdir()
    shutil.copyfile(
        REPO / "scripts/normal_launch_diagnostics.py", tmp_path / "normal_launch_diagnostics.py"
    )
    (tmp_path / "src/normal_training_worker.py").write_text(
        "raise ModuleNotFoundError(" + repr(SECRET) + ")\n"
    )
    root = tmp_path / "diagnostics"
    root.mkdir()
    context = diag.NormalDiagnostics(root, "worker", "a" * 64, "mqar-k64-deltanet-seed202")
    env = dict(os.environ, **context.environment())
    result = subprocess.run(
        UV + [str(tmp_path / "worker.py")], env=env, capture_output=True, timeout=30
    )
    assert result.returncode == 1
    primary = context.read("pre-worker", None)
    assert primary is not None, "dependency import failed before worker diagnostic handler"
    assert primary["exception_class"] == "ModuleNotFoundError"
    assert any(
        f["basename"] == "normal_training_worker.py" and f["function"] == "<module>"
        for f in primary["frames"]
    )
    assert SECRET.encode() not in result.stdout + result.stderr
    assert json.loads(result.stdout)["exception_class"] == "ModuleNotFoundError"


def driver_setup(tmp_path, monkeypatch, incoming=None, *, target=False, invalidate=False):
    """Actual inherited producer -> emitted setup sanitizer -> selected metadata child."""
    from tests.scripts.test_normal_launch_diagnostics import FROZEN

    package = tmp_path / "package"
    (package / "inherited/tools").mkdir(parents=True)
    (package / "src").mkdir()
    for name, entry_name in (
        ("normal_training_launch", "setup"),
        ("normal_launch_diagnostics", "normal_launch_diagnostics"),
        ("normal_driver_environment", "normal_driver_environment"),
    ):
        source = REPO / "scripts" / (name + ".py")
        if source.exists():
            shutil.copyfile(source, package / (entry_name + ".py"))
    driver = tmp_path / "validated-driver"
    driver.mkdir()
    validator = package / "inherited/tools/study_environment.py"
    validator.write_text(
        "from pathlib import Path\ndef driver_directory():\n    p=Path("
        + repr(str(driver))
        + ")\n    return str(p) if p.is_dir() else None\n"
    )
    # Synthetic metadata validator, NOT proof of Linux UID/mode/library qualification.
    namespace = load("synthetic_driver_validator", validator)
    monkeypatch.setitem(sys.modules, "study_environment", namespace)
    runtime = load(
        "real_inherited_runtime", FROZEN / "normal-package/inherited/operator/inherited/runtime.py"
    )
    produced = runtime.child_environment(tmp_path / "caches", cpu=False)
    assert produced["LD_LIBRARY_PATH"] == str(driver)
    if invalidate:
        driver.rmdir()
    import zipfile

    with zipfile.ZipFile(package / "compare_attention-0.1.0-py3-none-any.whl", "x"):
        pass
    (package / "worker.py").write_text("# not selected\n")
    files = {
        p.relative_to(package).as_posix(): pin(p.read_bytes())
        for p in package.rglob("*")
        if p.is_file()
    }
    identity = package / "normal-source-identity.json"
    identity.write_text(json.dumps({"files": files}))
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "engineering_only": not target,
                "source_identity_sha256": pin(identity.read_bytes())["sha256"],
            }
        )
    )
    output = tmp_path / "selected.json"
    child = tmp_path / "metadata_child.py"
    child.write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'\n"
        "assert 'torch' not in sys.modules\nPath("
        + repr(str(output))
        + ").write_text(json.dumps({k:os.environ.get(k) for k in "
        "('LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','HF_TOKEN',"
        "'DYLD_LIBRARY_PATH','HTTP_PROXY')}))\n"
    )
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = produced["LD_LIBRARY_PATH"] if incoming is None else incoming
    env.update(HF_TOKEN=SECRET, DYLD_LIBRARY_PATH=SECRET, HTTP_PROXY=SECRET)
    sink = tmp_path / "diagnostics"
    sink.mkdir()
    env.update(
        diag.NormalDiagnostics(sink, "finish", "a" * 64, "mqar-k64-deltanet-seed202").environment()
    )
    command = UV + [
        str(package / "setup.py"),
        "--uv",
        shutil.which("uv"),
        "--python",
        sys.executable,
        "--engineering-bootstrap",
        str(child),
        "--",
        "--plan",
        str(plan),
        "--phase",
        "finish",
    ]
    return subprocess.run(command, env=env, capture_output=True, timeout=30), output, driver


def test_validated_driver_reaches_actual_selected_child(tmp_path, monkeypatch):
    result, output, driver = driver_setup(tmp_path, monkeypatch)
    assert result.returncode == 0
    actual = json.loads(output.read_bytes())
    assert actual == {
        "LD_LIBRARY_PATH": str(driver),
        "CUDA_VISIBLE_DEVICES": "-1",
        "HF_TOKEN": None,
        "DYLD_LIBRARY_PATH": None,
        "HTTP_PROXY": None,
    }


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "/tmp/unvalidated",
        "/usr/local/nvidia/lib64:/tmp/injected",
        "/usr/local/nvidia/lib64\nprivate",
        "relative",
    ],
)
def test_unvalidated_driver_never_reaches_selected_child(tmp_path, monkeypatch, bad):
    result, output, _driver = driver_setup(tmp_path, monkeypatch, incoming=bad)
    assert result.returncode == 1
    assert not output.exists()
    assert SECRET.encode() not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "where", ["parser", "derive", "authority", "runtime", "namespace", "credentials", "body"]
)
def test_actual_worker_preflight_and_body_keep_primary(tmp_path, monkeypatch, capsys, where):
    from types import SimpleNamespace

    import normal_training_worker as worker

    root = tmp_path / "diagnostics"
    root.mkdir()
    context = diag.NormalDiagnostics(root, "worker", "a" * 64, "mqar-k64-deltanet-seed202")
    for key, value in context.environment().items():
        monkeypatch.setenv(key, value)
    args = []
    for name in ("package", "plan", "recipe", "data", "private", "output"):
        args += ["--" + name, str(tmp_path / name)]
    args += [
        "--case",
        "mqar-k64-deltanet-seed202",
        "--prefix",
        "normal-training/fixture",
        "--phase",
        "worker",
    ]
    if where == "parser":
        args += ["--unknown", SECRET]

    def fail(*_args, **_kwargs):
        raise ValueError(SECRET)

    monkeypatch.setattr(
        worker, "derive", fail if where == "derive" else lambda *_: {"engineering_only": False}
    )
    monkeypatch.setattr(
        worker, "execution_authority", fail if where == "authority" else lambda *_: None
    )
    monkeypatch.setattr(worker, "worker", fail)
    if where == "namespace":
        args[args.index("--prefix") + 1] = "bad-prefix"
    services = SimpleNamespace(
        credentials=fail if where == "credentials" else lambda *_args, **_kwargs: {}
    )
    runtime = SimpleNamespace(guard=fail if where == "runtime" else lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "quality_inherit",
        SimpleNamespace(leaf=lambda name: services if name == "services" else runtime),
    )
    code = diag.normal_cli(lambda: worker.main(args), "pre-worker")
    assert code == (2 if where == "parser" else 1)
    stage = "worker" if where == "body" else "pre-worker"
    primary = context.read(stage, None)
    assert primary is not None
    assert primary["exception_class"] == ("SystemExit" if where == "parser" else "ValueError")
    assert primary["phase"] == "worker" and primary["stage"] == stage
    if where == "namespace":
        assert primary["category"] == "NORMAL_DISTINCT_NAMESPACE"
    console = capsys.readouterr()
    assert SECRET not in console.out + console.err
    assert json.loads(console.out)["exception_class"] == primary["exception_class"]


@pytest.mark.parametrize("layout", ["flat", "owner"])
@pytest.mark.parametrize("task", ["mqar-k64", "sorting"])
def test_saved_finish_primary_crosses_actual_outer_wrapper(tmp_path, monkeypatch, layout, task):
    import ast
    import contextlib
    import io
    import logging

    from tests.scripts.test_normal_launch_diagnostics import tiny_kernel

    kernel = tiny_kernel(tmp_path, monkeypatch, failing_phase="finish", layout=layout, task=task)
    tree = ast.parse(Path(kernel.__file__).read_bytes())
    wrapper = ast.fix_missing_locations(ast.Module(body=[tree.body[-1]], type_ignores=[]))
    namespace = dict(kernel.__dict__)
    namespace.update(
        __name__="__main__",
        main=lambda: kernel.main(tmp_path / "mount", tmp_path / "working", tmp_path / "extracted"),
    )
    stdout = io.StringIO()
    previous = logging.root.manager.disable
    try:
        with contextlib.redirect_stdout(stdout), pytest.raises(SystemExit) as caught:
            exec(compile(wrapper, kernel.__file__, "exec"), namespace)
    finally:
        logging.disable(previous)
    assert caught.value.code == 1
    report = json.loads(stdout.getvalue())
    assert report["exception_class"] == "CalledProcessError" and report["returncode"] == 1
    primary = report["primary"]
    assert primary["exception_class"] == "ValueError" and primary["phase"] == "finish"
    assert primary["stage"] == "pre-worker"
    assert primary["frames"][-1]["basename"] == "normal_training_worker.py"
    assert primary["frames"][-1]["function"] == "main"
    assert SECRET not in stdout.getvalue()
    root = tmp_path / "working" / ("normal-" + kernel.BINDING["nonce"]) / ".launch-diagnostics"
    assert json.loads((root / "setup-entered-bootstrap.json").read_bytes())["callsite"] == "setup"
    assert json.loads((root / "finish-entered-setup.json").read_bytes())["phase"] == "finish"


def test_malformed_diagnostic_channel_does_not_mask_original(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NORMAL_DIAGNOSTIC_ROOT", str(tmp_path))
    monkeypatch.setenv("NORMAL_DIAGNOSTIC_PHASE", "invalid")
    monkeypatch.setenv("NORMAL_DIAGNOSTIC_SOURCE", SECRET)
    monkeypatch.setenv("NORMAL_DIAGNOSTIC_CASE", SECRET)

    def fail():
        raise ValueError(SECRET)

    assert diag.normal_cli(fail, "pre-worker") == 1
    assert not list(tmp_path.iterdir())
    console = capsys.readouterr()
    assert SECRET not in console.out + console.err


@pytest.mark.parametrize(
    "target,invalidate,category",
    [
        (True, False, "NORMAL_TARGET_MASK_FORBIDDEN"),
        (False, True, "NORMAL_VALIDATED_DRIVER_REQUIRED"),
    ],
)
def test_revalidation_and_host_mask_reject_before_selected_child(
    tmp_path, monkeypatch, target, invalidate, category
):
    result, output, _driver = driver_setup(
        tmp_path, monkeypatch, target=target, invalidate=invalidate
    )
    assert result.returncode == 1 and not output.exists()
    report = json.loads(result.stdout)
    assert report["category"] == category and report["exception_class"] == "ValueError"
    assert SECRET.encode() not in result.stdout + result.stderr


def test_kernel_task_import_layout_consumes_shared_helper(tmp_path):
    import ast

    import build_normal_training_lifecycle as builder
    import unified_normal_render

    from tests.scripts.test_normal_training_closure import PARENT

    package = builder.build_package(PARENT, REPO, tmp_path / "package")
    tree = ast.parse(unified_normal_render.MAIN)
    assignment = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Subscript)
            and isinstance(t.slice, ast.Constant)
            and t.slice.value == "PYTHONPATH"
            for t in n.targets
        )
    )
    scope = {"env": {}, "os": os, "package": package}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])),
            "actual-kernel-env.py",
            "exec",
        ),
        scope,
    )
    env = dict(os.environ, **scope["env"])
    result = subprocess.run(
        UV
        + [
            "-c",
            "import sys,normal_training_worker,normal_launch_diagnostics; "
            "assert 'torch' not in sys.modules",
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, "actual normal-task import layout cannot consume shared helper"


def test_emitted_kernel_keeps_formatted_module_under_500_lines():
    import unified_normal_render

    from tests.scripts.test_normal_launch_diagnostics import FROZEN

    source = unified_normal_render.render(
        {},
        {
            "case_id": "mqar-k64-deltanet-seed202",
            "worker_sha256": "0" * 64,
            "source_profile": "REPAIRED_NORMAL_V1",
            "nonce": "1" * 32,
            "task": "mqar-k64",
        },
        FROZEN,
    )
    result = subprocess.run(
        UV + ["-m", "ruff", "format", "--stdin-filename", "kernel.py", "-"],
        input=source,
        text=True,
        capture_output=True,
        timeout=30,
        env=dict(os.environ),
    )
    assert result.returncode == 0
    assert len(result.stdout.splitlines()) < 500


def test_selected_torchrun_cpu_children_keep_rank_primaries(tmp_path, monkeypatch):
    from tests.scripts.test_normal_launch_diagnostics import tiny_kernel

    kernel = tiny_kernel(tmp_path, monkeypatch, failing_phase="worker")
    with pytest.raises(subprocess.CalledProcessError) as caught:
        kernel.main(tmp_path / "mount", tmp_path / "working", tmp_path / "extracted")
    assert caught.value.returncode == 1
    root = tmp_path / "working" / ("normal-" + kernel.BINDING["nonce"]) / ".launch-diagnostics"
    context = diag.NormalDiagnostics(
        root, "worker", kernel.BINDING["normal_source_sha256"], kernel.BINDING["case_id"]
    )
    primary = context.read("pre-worker", 0)
    coordinated = context.read("pre-worker", 1)
    assert primary["exception_class"] == "ValueError" and primary["rank"] == 0
    assert coordinated["exception_class"] == "RuntimeError" and coordinated["rank"] == 1
    assert caught.value._normal_primary == primary
    assert not (root / "task-entered-task.json").exists()
    assert SECRET not in json.dumps([primary, coordinated])


def test_exception_diagnostic_attribute_failure_never_masks_primary(tmp_path):
    class PrivateError(RuntimeError):
        @property
        def _normal_primary(self):
            raise OSError(SECRET)

    context = diag.NormalDiagnostics(tmp_path, "worker", "a" * 64, "mqar-k64-deltanet-seed202")
    error = PrivateError(SECRET)

    def fail():
        raise error

    with pytest.raises(PrivateError) as caught:
        context.capture(fail, "pre-worker")
    assert caught.value is error


def test_saved_task_parser_primary_crosses_supervisor(tmp_path, monkeypatch):
    from tests.scripts.test_normal_launch_diagnostics import tiny_kernel

    kernel = tiny_kernel(tmp_path, monkeypatch, failing_phase="task")
    with pytest.raises(subprocess.CalledProcessError) as caught:
        kernel.main(tmp_path / "mount", tmp_path / "working", tmp_path / "extracted")
    assert caught.value.returncode == 2
    report = kernel.public_error(caught.value, "entry")
    primary = report["primary"]
    assert primary["phase"] == "task" and primary["stage"] == "task"
    assert primary["exception_class"] == "SystemExit" and primary["returncode"] == 2
    assert SECRET not in json.dumps(report)
