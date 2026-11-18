"""Frozen saved main, real safe mounts and selected UV parser; CPU boundary only."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
import uuid
import zipfile
from pathlib import Path

import pytest


def test_real_saved_hardware_guard_accepts_parent_wrong_torch_and_rejects_topology(monkeypatch):
    candidate = Path(os.environ["MQAR_TEST_CANDIDATE"])
    scope = {"__name__": "ENGINEERING_HARDWARE_METADATA_ONLY"}
    exec(
        compile(
            (candidate / "payload/deploy/mqar-k64-classic-seed202/kernel/kernel.py").read_bytes(),
            "saved-main.py",
            "exec",
        ),
        scope,
    )
    devices = {"count": 2, "cap": (7, 5)}
    fake = types.SimpleNamespace(
        __version__="WRONG_PARENT_TORCH_ALLOWED_BEFORE_SETUP",
        version=types.SimpleNamespace(cuda="WRONG_PARENT_CUDA"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: devices["count"],
            get_device_name=lambda _i: "Tesla T4",
            get_device_capability=lambda _i: devices["cap"],
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    scope["platform"] = types.SimpleNamespace(
        system=lambda: "Linux", python_version=lambda: "3.13.16"
    )
    scope["os"] = types.SimpleNamespace(environ={})
    diagnostic = {}
    result = scope["inspect_hardware"]("/usr/local/nvidia/lib64", diagnostic=diagnostic)
    assert result["visible_count"] == 2
    assert diagnostic["torch_version"] == fake.__version__
    devices["count"] = 1
    with pytest.raises(RuntimeError, match="HARDWARE_EXACT_TWO_T4_REQUIRED"):
        scope["inspect_hardware"]("/usr/local/nvidia/lib64")
    devices.update(count=2, cap=(8, 0))
    with pytest.raises(RuntimeError, match="HARDWARE_T4_SM75_REQUIRED"):
        scope["inspect_hardware"]("/usr/local/nvidia/lib64")
    devices["cap"] = (7, 5)
    scope["os"].environ["CUDA_VISIBLE_DEVICES"] = "-1"
    with pytest.raises(ValueError, match="UNMASKED_VALIDATED_DRIVER_REQUIRED"):
        scope["inspect_hardware"]("/usr/local/nvidia/lib64")
    # The selected interpreter retains the strict package/build predicate that the
    # parent hardware inspection intentionally does not impose before setup.
    with zipfile.ZipFile(candidate / "payload/inherited/sorting-science.zip.bin") as archive:
        runtime_raw = archive.read("operator/inherited/runtime.py")
    worker_scope = {"__name__": "ENGINEERING_SELECTED_RUNTIME_METADATA_ONLY"}
    exec(compile(runtime_raw, "runtime.py", "exec"), worker_scope)
    worker_scope["platform"] = scope["platform"]
    worker_scope["os"] = types.SimpleNamespace(environ={})
    fake.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(version=lambda: 91900))
    with pytest.raises(RuntimeError, match="PINNED_TORCH_RUNTIME_REQUIRED"):
        worker_scope["guard"]()
    fake.__version__, fake.version.cuda = "2.11.0+cu130", "13.0"
    versions = {
        "flash-linear-attention": "0.6.0",
        "fla-core": "0.6.0",
        "triton": "3.6.0",
        "clearml": "2.1.12",
    }
    worker_scope["importlib"] = types.SimpleNamespace(
        metadata=types.SimpleNamespace(version=lambda package: versions[package])
    )
    assert worker_scope["guard"]()["visible_count"] == 2
    versions["triton"] = "UNAPPROVED"
    with pytest.raises(RuntimeError, match="PINNED_RUNTIME_PACKAGE_REQUIRED"):
        worker_scope["guard"]()


def test_actual_import_shadow_and_postimport_inventory_guards(tmp_path, monkeypatch):
    from seed202_mqar_authority import check_imports, import_inventory

    release, science = tmp_path / "release", tmp_path / "science"
    release.mkdir()
    science.mkdir()
    item = release / "module.py"
    item.write_bytes(b"# authenticated\n")
    before = import_inventory(release, science)
    monkeypatch.setitem(
        sys.modules,
        "seed202_pilot.ENGINEERING_SHADOW",
        types.SimpleNamespace(__file__=str(tmp_path / "shadow.py")),
    )
    # An ambient package cache must fail even when every extracted byte is intact.
    with pytest.raises(ValueError, match="UNEXPECTED_IMPORT_SHADOW"):
        check_imports(release, science, before)
    item.write_bytes(b"# changed during warm import\n")
    with pytest.raises(ValueError, match="POSTIMPORT_INVENTORY_DRIFT"):
        check_imports(release, science, before)


@pytest.mark.parametrize("layout", ["flat", "owner"])
def test_frozen_saved_main_mounts_checkpoint_before_selected_uv_metadata_boundary(
    tmp_path, monkeypatch, layout, record_property
):
    from seed202_mqar_binding_fixture import engineering_candidate

    candidate = Path(os.environ["MQAR_TEST_CANDIDATE"])
    tested_archive = candidate / "sorting-mqar-seed202-pilot-v1.zip"
    with tested_archive.open("rb") as stream:
        tested_archive_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    root = candidate.parent / ("saved-main-engineering-" + uuid.uuid4().hex)
    candidate = engineering_candidate(candidate, root / "fixture")
    deploy = candidate / "payload/deploy/mqar-k64-classic-seed202"
    mount, working, package = [root / name for name in ("mount", "working", "package")]
    mount.mkdir(parents=True)
    if layout == "owner":
        mount_base = mount / "datasets/zimike"
        mount_base.mkdir(parents=True)
    else:
        mount_base = mount
    source = (deploy / "kernel/kernel.py").read_text(encoding="utf-8")
    scope = {"__name__": "ENGINEERING_CPU_METADATA_ONLY"}
    exec(compile(source, "frozen-mqar-kernel.py", "exec"), scope)
    binding = scope["BINDING"]
    for slug, filename, path in (
        (
            binding["operator_slug"],
            "seed202-pilot.zip.bin",
            deploy / "dataset/seed202-pilot.zip.bin",
        ),
        (binding["resume_slug"], "resume.zip.bin", deploy / "resume-input/resume.zip.bin"),
        (
            "sorting-quality-input-fcff1b5daf574b70",
            "scientific.zip.bin",
            candidate / "payload/inherited/sorting-science.zip.bin",
        ),
    ):
        folder = mount_base / slug
        folder.mkdir()
        shutil.copyfile(path, folder / filename)
    for slug, filename, raw in (
        (
            "ca-kv-1c660aa5f54d88ea9e0c1ce4",
            "meta.json",
            b'{"format_version":2,"name":"kv_retrieval","ignore_index":-1}',
        ),
        ("k64-private-20261109a1b2", "private.env", b"SYNTHETIC_ONLY=1\n"),
    ):
        folder = mount_base / slug
        folder.mkdir()
        (folder / filename).write_bytes(raw)
    for name in (
        "evidence.json",
        "independent-review.json",
        "execution-decision.json",
        "authority-inventory.json",
    ):
        shutil.copyfile(deploy / "resume-input" / name, mount_base / binding["resume_slug"] / name)
    events = []

    # Supply only measured metadata. Keep the saved inspect_hardware/hardware code
    # intact: a wrong parent Torch build must reach setup on the valid topology.
    def available():
        if not events:
            events.append("hardware")
        return True

    parent_torch = types.SimpleNamespace(
        __version__="WRONG_PARENT_TORCH",
        version=types.SimpleNamespace(cuda="WRONG_PARENT_CUDA"),
        cuda=types.SimpleNamespace(
            is_available=available,
            device_count=lambda: 2,
            get_device_name=lambda _i: "Tesla T4",
            get_device_capability=lambda _i: (7, 5),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", parent_torch)
    scope["driver_directory"] = lambda: "/usr/local/nvidia/lib64"
    scope["platform"] = types.SimpleNamespace(
        system=lambda: "Linux", python_version=lambda: "3.13.16"
    )
    scope["os"] = types.SimpleNamespace(
        **{
            **vars(os),
            "environ": {
                key: value for key, value in os.environ.items() if key != "CUDA_VISIBLE_DEVICES"
            },
        }
    )
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "quality_inherit.py").write_text(
        "def leaf(name):\n"
        " class Boundary:\n"
        "  def guard(self):\n"
        "   import os, sys\n"
        "   from pathlib import Path\n"
        "   sys.path.insert(0, os.environ['SEED202_EXTRACTED_WHEEL'])\n"
        "   from compare_attention.cli.train import parse_train_args\n"
        "   from compare_attention.train.config import TrainConfig\n"
        "   release = Path(sys.argv[sys.argv.index('--release') + 1])\n"
        "   data = Path(sys.argv[sys.argv.index('--mqar') + 1])\n"
        "   config_path = release / 'configs/mqar-k64-classic-seed202.yaml'\n"
        "   args = parse_train_args(['--config', str(config_path), '--data', str(data)])\n"
        "   config = TrainConfig(data=args.data, model=args.model,\n"
        "                        schedule=args.schedule, optim=args.optim)\n"
        "   assert config.schedule.seed == 202 and args.dataset_task == 'kv_retrieval'\n"
        "   print('ENGINEERING_CPU_PARSER_BOUNDARY_NO_TRAINING')\n"
        "   raise SystemExit(73)\n"
        " return Boundary()\n",
        encoding="utf-8",
    )
    # Cache the engineering runtime boundary before the real runner restores its
    # authenticated import roots. No candidate file or runtime predicate is edited.
    (shim / "sitecustomize.py").write_text("import quality_inherit\n", encoding="utf-8")

    def setup(_science, _output, _deadline):
        assert events == ["hardware"]
        restored_input = package / ("seed202-resume-" + binding["nonce"]) / "original"
        assert (restored_input / "out/checkpoints/step_00000980.pt").is_file()
        events.append("setup-engineering-no-install")
        return Path(shutil.which("uv")), Path(sys.executable), dict(os.environ), {}

    def bounded_run(command, env, cwd, _timeout):
        assert events == ["hardware", "setup-engineering-no-install"]
        original = Path(command[command.index("--resume-input") + 1])
        checkpoint = original / "out/checkpoints/step_00000980.pt"
        with checkpoint.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == (
                "e2289ced84240f1f81dfcaa254741457a7863ae9a40db6ab68f9cc021e904542"
            )
        env.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
        env["SEED202_EXTRACTED_WHEEL"] = os.environ["SEED202_EXTRACTED_WHEEL"]
        env["PYTHONPATH"] = str(shim) + os.pathsep + env["PYTHONPATH"]
        child = subprocess.run(
            command, env=env, cwd=cwd, capture_output=True, text=True, check=False
        )
        assert child.returncode == 73, child.stderr
        assert child.stdout.strip() == "ENGINEERING_CPU_PARSER_BOUNDARY_NO_TRAINING"
        trace = root / "ENGINEERING-UV-command.json"
        trace.write_text(
            json.dumps(
                {
                    "scope": "ENGINEERING_ONLY_CPU_NO_MODEL_OPTIMIZER_OR_RNG_RESTORE",
                    "layout": layout,
                    "command": command,
                    "exit_code": child.returncode,
                    "tested_input_archive_sha256": tested_archive_sha256,
                    "kernel_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "kernel": str(deploy / "kernel/kernel.py"),
                    "mask": env["CUDA_VISIBLE_DEVICES"],
                    "bytecode_disabled": env["PYTHONDONTWRITEBYTECODE"],
                    "stdout": child.stdout.strip(),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        record_property("uv_command_receipt", str(trace))
        events.append("selected-uv-public-parser")

    # Real main rejects cached modules from other runs; isolate this engineering process view.
    for name in tuple(sys.modules):
        if name.startswith(("seed202_pilot", "quality_")):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(
        sys.modules,
        "session",
        types.SimpleNamespace(
            setup=setup, __file__=str(package / "sorting-science-fcff1b5daf574b70/tools/session.py")
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "process_control",
        types.SimpleNamespace(
            bounded_run=bounded_run,
            __file__=str(package / "sorting-science-fcff1b5daf574b70/tools/process_control.py"),
        ),
    )
    previous = sys.path.copy()
    try:
        scope["main"](mount_root=mount, working=working, package_root=package)
        assert events[-1] == "selected-uv-public-parser"
    finally:
        sys.path[:] = previous
