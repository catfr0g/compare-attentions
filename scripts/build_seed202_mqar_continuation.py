"""Build source for a scoped continuation from authenticated immutable archives."""

from __future__ import annotations

import hashlib
import json
import os
import pprint
import shutil
import subprocess
import zipfile
from pathlib import Path

HISTORY_SHA = "65b027f84fa249528846db75edbe516e3e4213e84d3d181903dae068d35f4878"
SCIENCE_SHA = "71872615b0cfbe2406cd08762cc02c390c88a70b6bb5f4bb38d6571830c8a3bb"
EXECUTION = Path(
    "E:/HermesArtifacts/compare-attention/sorting-mqar-seed202-pilot-v1/"
    "remaining-three-after-tail-v1/qualified-launch-bridge-v1/execution-classic-20261118-v1"
)


def original_files():
    """Join all17 original case files to the separately authenticated full collector index."""
    index = EXECUTION / "failed-download-index-v1.json"
    if sha(index) != "464c032ad042342a720ee77da19a22683b8d1d70df74c29acc33479b58c41652":
        raise ValueError("ORIGINAL_INDEX_PIN")
    pins = json.loads(index.read_bytes())["inventory"]
    raw = EXECUTION / "failed-download-v1"
    (checkpoint,) = raw.rglob("step_00000980.pt")
    root = checkpoint.parent.parent.parent
    result = {}
    for path in root.rglob("*"):
        if path.is_file():
            pin = {"bytes": path.stat().st_size, "sha256": sha(path)}
            if path.is_symlink() or pins[path.relative_to(raw).as_posix()] != pin:
                raise ValueError("ORIGINAL_CASE_BYTES")
            result[path.relative_to(root).as_posix()] = pin
    if len(result) != 17:
        raise ValueError("ORIGINAL_CASE_INVENTORY")
    return root, result


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("EXACT_INHERITED_SOURCE_SEAM")
    return text.replace(before, after)


def stage_science_helpers(parent, destination):
    """Extract only authenticated Python helpers, without duplicating data arrays."""
    from pathlib import PurePosixPath

    archive = Path(parent) / "inherited/sorting-science.zip.bin"
    if sha(archive) != SCIENCE_SHA or Path(destination).exists():
        raise ValueError("AUTHENTIC_HELPERS_CREATE_ONLY")
    with zipfile.ZipFile(archive) as saved:
        if len(saved.namelist()) != len(set(saved.namelist())):
            raise ValueError("UNIQUE_SCIENCE_MEMBERS")
        for name in saved.namelist():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("SAFE_SCIENCE_MEMBERS")
            if name.startswith(("operator/", "tools/")) and name.endswith(".py"):
                target = Path(destination) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(saved.read(name))
    return Path(destination)


def stage_source(parent, history_archive, destination):
    """Create separate source bytes; never alter or hardlink inherited inputs."""
    parent, destination = Path(parent), Path(destination)
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
    ):
        raise ValueError("LOCAL_CPU_MASK_BEFORE_IMPORTS_REQUIRED")
    if destination.exists():
        raise ValueError("CREATE_ONLY_SOURCE")
    manifest = json.loads((parent / "manifest.json").read_bytes())["files"]
    for name, pin in manifest.items():
        if name.startswith("source/src/"):
            path = parent / name
            if (
                path.is_symlink()
                or path.stat().st_size != pin["bytes"]
                or sha(path) != pin["sha256"]
            ):
                raise ValueError("AUTHENTIC_PARENT_SOURCE")
    if sha(history_archive) != HISTORY_SHA:
        raise ValueError("AUTHENTIC_SUCCESSFUL_RESUME_ARCHIVE")
    shutil.copytree(parent / "source/src", destination)
    package = destination / "seed202_pilot"
    maintained = Path(__file__).parent
    from seed202_native_schema import repair_native

    native = package / "cloud_native.py"
    native_source = repair_native(native.read_bytes()).decode("utf-8")
    native_source = replace_once(
        native_source,
        "    from quality_inherit import native_checks",
        "    from seed202_pilot.mqar_native_checks import native_checks",
    )
    native.write_text(native_source, encoding="utf-8", newline="\n")
    for old, new in (
        ("seed202_native_schema", "native_schema"),
        ("seed202_mqar_render", "mqar_render"),
        ("seed202_mqar_checkpoint", "mqar_checkpoint"),
        ("seed202_mqar_restore_gate", "mqar_restore_gate"),
        ("seed202_mqar_continuation", "mqar_continuation"),
        ("seed202_mqar_resume_case", "mqar_resume_case"),
        ("seed202_mqar_composite", "mqar_composite"),
        ("seed202_mqar_command_scope", "mqar_command_scope"),
        ("seed202_mqar_host_paths", "mqar_host_paths"),
        ("seed202_mqar_native_checks", "mqar_native_checks"),
        ("seed202_mqar_authority", "mqar_authority"),
        ("seed202_mqar_publisher", "mqar_publisher"),
        ("seed202_mqar_package_guard", "mqar_package_guard"),
        ("seed202_mqar_worker_paths", "mqar_worker_paths"),
    ):
        shutil.copyfile(maintained / (old + ".py"), package / (new + ".py"))
    original_root, pins = original_files()
    shutil.copyfile(original_root / "out/metrics.jsonl", package / "mqar-prefix-metrics.jsonl")
    (package / "mqar_resume_pins.py").write_text(
        '"""Authenticated original MQAR ERROR case inventory."""\n\nFILES = '
        + pprint.pformat(pins, width=88, sort_dicts=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with zipfile.ZipFile(history_archive) as saved:
        pins = json.loads(saved.read("manifest.json"))["files"]

        def inherited(name):
            member = "source/src/seed202_pilot/" + name + ".py"
            raw = saved.read(member)
            if pins[member] != {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}:
                raise ValueError("AUTHENTIC_RESUME_MEMBER")
            return raw.decode("utf-8")

        runner = replace_once(
            inherited("cloud_runner"),
            'parser.add_argument("--resume-input", type=Path)',
            'parser.add_argument("--resume-input", type=Path, required=True)',
        )
        runner = replace_once(
            runner, '"seed202_pilot.cloud_entry"', '"seed202_pilot.mqar_resume_observer"'
        )
        entry = replace_once(
            inherited("cloud_entry"),
            'parser.add_argument("--resume-input", type=Path)',
            'parser.add_argument("--resume-input", type=Path, required=True)',
        )
        entry = entry.replace("seed202_pilot.cloud_case", "seed202_pilot.mqar_resume_case")
    observer = replace_once(
        (package / "cloud_observer.py").read_text(encoding="utf-8"),
        "from seed202_pilot.cloud_entry import main as entry",
        "from seed202_pilot.mqar_resume_entry import main as entry",
    )
    for name, text in (
        ("mqar_resume_runner", runner),
        ("mqar_resume_entry", entry),
        ("mqar_resume_observer", observer),
    ):
        if name != "mqar_resume_observer":
            text = replace_once(
                text,
                "    args = parser.parse_args(argv)",
                '    parser.add_argument("--authority-root", type=Path, required=True)\n'
                '    parser.add_argument("--evidence-sha256", required=True)\n'
                '    parser.add_argument("--authority-sha256", required=True)\n'
                "    args = parser.parse_args(argv)",
            )
            if name == "mqar_resume_runner":
                text = replace_once(
                    text,
                    '        leaf("process_control").bounded_run('
                    "command, env, args.science, timeout)",
                    '        command += ["--authority-root", str(args.authority_root),\n'
                    '                    "--evidence-sha256", args.evidence_sha256,\n'
                    '                    "--authority-sha256", args.authority_sha256]\n'
                    '        leaf("process_control").bounded_run('
                    "command, env, args.science, timeout)",
                )
                text = replace_once(
                    text,
                    "    args = parser.parse_args(argv)",
                    "    args = parser.parse_args(argv)\n"
                    "    from seed202_pilot.mqar_authority import validate_args\n"
                    "    validate_args(args)",
                )
            else:
                text = replace_once(
                    text,
                    "        from seed202_pilot.cloud_contract import derive_case\n",
                    "",
                )
                text = replace_once(
                    text,
                    '        return derive_case(args.release / "configs" / '
                    '(args.case + ".yaml"), data)',
                    "        from seed202_pilot.mqar_worker_paths import bind_case\n"
                    '        return bind_case(args.release / "configs" / (args.case + ".yaml"),\n'
                    "                         data, args.resume_input)",
                )
                text = replace_once(
                    text,
                    "    guard()\n    bind_wheel()",
                    "    guard()\n"
                    "    from seed202_pilot.mqar_package_guard import install\n"
                    "    install(args.science)\n"
                    "    from seed202_pilot.mqar_authority import validate_args\n"
                    "    validate_args(args)\n"
                    "    bind_wheel()",
                )
            text = replace_once(
                text,
                "    args = parser.parse_args(argv)",
                "    args = parser.parse_args(argv)\n"
                "    import re\n\n"
                '    if args.case != "mqar-k64-classic-seed202" or not re.fullmatch(\n'
                '        r"[0-9a-f]{32}", args.nonce\n'
                "    ):\n"
                '        raise ValueError("EXACT_MQAR_CONTINUATION_SCOPE")',
            )
        compile(text, name + ".py", "exec")
        (package / (name + ".py")).write_text(text, encoding="utf-8", newline="\n")
    from seed202_mqar_host_integration import integrate

    integrate(package)
    from seed202_mqar_native_integration import integrate as integrate_native

    integrate_native(package, original_root / "out/checkpoints/step_00000980.pt")
    path = package / "science.py"
    science = path.read_text(encoding="utf-8")
    science = replace_once(
        science,
        "    try:\n        from quality_freeze import verify_package",
        "    from seed202_pilot.mqar_package_guard import install, restore\n"
        "    compatibility = install(science_dir)\n"
        "    try:\n        from quality_freeze import verify_package",
    )
    science = replace_once(
        science,
        "    finally:\n        for name, module in list(sys.modules.items()):",
        "    finally:\n        restore(compatibility)\n"
        "        for name, module in list(sys.modules.items()):",
    )
    science = replace_once(
        science, 'compile(raw, module.__file__, "exec")', 'compile(raw, name + ".py", "exec")'
    )
    science = replace_once(
        science,
        "bind_native(native_module.verify_native)",
        'bind_native(vars(native_module)["verify_native"])',
    )
    path.write_text(science, encoding="utf-8", newline="\n")
    text = replace_once(
        path.read_text(encoding="utf-8"),
        "from seed202_pilot.host_paths import expected_data_paths, verify_expected_native",
        "from seed202_pilot.mqar_host_paths import expected_data_paths, verify_expected_native",
    )
    path.write_text(text, encoding="utf-8", newline="\n")
    field = (
        " " * 12 + '"raw_task_report_sha256": hashlib.sha256(canonical_bytes(report)).hexdigest(),'
    )
    text = replace_once(
        path.read_text(encoding="utf-8"),
        field,
        field + "\n            **__import__(\n"
        '                "seed202_pilot.mqar_composite", fromlist=["consume"]\n'
        "            ).consume(root, case, config_path, spec),",
    )
    path.write_text(text, encoding="utf-8", newline="\n")
    generated = [*package.glob("mqar_*.py"), package / "resume_router.py"]
    generated += [
        package / (name + ".py")
        for name in (
            "cloud_native",
            "fresh_mode",
            "host_operator",
            "preparation",
            "preflight",
            "seal",
            "cli",
            "science",
            "periodic",
        )
    ]
    result = subprocess.run(
        [
            shutil.which("uv"),
            "run",
            "--offline",
            "--no-sync",
            "python",
            "-B",
            "-m",
            "ruff",
            "check",
            "--select",
            "I",
            "--fix",
            *map(str, generated),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("GENERATED_SOURCE_IMPORT_ORDER_FAILED")
    result = subprocess.run(
        [
            shutil.which("uv"),
            "run",
            "--offline",
            "--no-sync",
            "python",
            "-B",
            "-m",
            "ruff",
            "format",
            *map(str, generated),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("GENERATED_SOURCE_FORMAT_FAILED")
    return destination
