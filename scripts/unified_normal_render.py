"""Render the saved NORMAL entry using authenticated existing bootstrap primitives."""

import ast
from pathlib import Path

from unified_normal_contract import digest, require
from unified_normal_source import service_names

MAIN = """def main(
    mount_root=Path("/kaggle/input"),
    working=Path("/kaggle/working"),
    package_root=Path("/kaggle/temp"),
):
    started = time.monotonic()
    driver = driver_directory()
    inspect_hardware(driver)
    operator_archive = resolve_mount(
        mount_root, BINDING["operator_owner"], BINDING["operator_slug"], "seed202-pilot.zip.bin"
    )
    release = extract_archive(
        operator_archive, package_root / ("normal-" + BINDING["nonce"]),
        BINDING["operator_sha256"],
    )
    science_archive = resolve_mount(
        mount_root, "zimike", "sorting-quality-input-fcff1b5daf574b70", "scientific.zip.bin"
    )
    science = extract_archive(
        science_archive, package_root / "sorting-science-fcff1b5daf574b70",
        BINDING["science_sha256"],
    )
    data = science / "data/sorting"
    if BINDING["task"] == "kv_retrieval":
        data = resolve_mount(
            mount_root, "zimike", "ca-kv-1c660aa5f54d88ea9e0c1ce4", "meta.json"
        ).parent
    package = release / "normal-package"
    identity = package / "normal-source-identity.json"
    if hashlib.sha256(identity.read_bytes()).hexdigest() != BINDING["normal_source_sha256"]:
        raise ValueError("NORMAL_SOURCE_BINDING")
    for name, pin in json.loads(identity.read_bytes())["files"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\\\" in name or ":" in name:
            raise ValueError("NORMAL_SOURCE_PATH")
        raw = (package / name).read_bytes()
        if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise ValueError("NORMAL_SOURCE_BYTES")
    plan = json.loads((release / "normal-plan.json").read_bytes())
    if plan["engineering_only"] is not True:
        authority = {}
        validator = release / "normal-authority.py"
        exec(compile(validator.read_bytes(), str(validator), "exec"), authority)
        authority["validate"](
            release / "normal-plan.json", release / "authority/independent-review.json",
            release / "authority/execution-decision.json",
        )
    elif os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("NORMAL_ENGINEERING_CPU_ONLY")
    # These are inherited, authenticated bootstrap modules, not scientific imports.
    sys.path[:0] = [str(science / "operator/inherited"), str(science / "tools")]
    from process_control import bounded_run
    from session import setup

    manifest = json.loads((science / "manifest.json").read_bytes())["files"]
    for name in ("process_control", "session", "runtime", "sorting_runtime", "tiny_uv"):
        module = __import__(name)
        expected = Path(module.__file__).resolve()
        if not expected.is_relative_to(science.resolve()) or expected.name != name + ".py":
            raise ValueError("NORMAL_BOOTSTRAP_IMPORT_ORIGIN")
        pin = manifest[expected.relative_to(science.resolve()).as_posix()]
        raw = expected.read_bytes()
        if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise ValueError("NORMAL_BOOTSTRAP_IMPORT_BYTES")

    output = working / ("normal-" + BINDING["nonce"])
    uv, python, env, _topology = setup(science, output, started + 600)
    private = resolve_mount(
        mount_root, "zimike", "k64-private-20261109a1b2", "private.env"
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([
        str(package / "src"), str(package / "inherited/operator"),
        str(package / "inherited/operator/inherited"), str(package / "inherited/tools"),
    ])
    for phase in ("worker", "finish"):
        command = [
            str(uv), "run", "--offline", "--no-project", "--no-sync", "--python", str(python),
            "python", "-B", str(package / "setup.py"),
            "--uv", str(uv), "--python", str(python), "--",
            "--package", str(package), "--plan", str(release / "normal-plan.json"),
            "--recipe", str(release / "recipe.yaml"), "--data", str(data),
            "--case", BINDING["case_id"], "--prefix", BINDING["normal_service_prefix"],
            "--private", str(private), "--output", str(output), "--phase", phase,
            "--review", str(release / "authority/independent-review.json"),
            "--decision", str(release / "authority/execution-decision.json"),
        ]
        bounded_run(command, env, package,
                    BINDING["kernel_seconds"] - (time.monotonic() - started))
        if phase == "worker":
            task = [
                str(uv), "run", "--offline", "--no-project", "--no-sync", "--python", str(python),
                "python", "-B", str(release / "normal-task.py"), "--phase", "task",
                "--package", str(package), "--plan", str(release / "normal-plan.json"),
                "--recipe", str(release / "recipe.yaml"), "--data", str(data),
                "--private", str(private), "--output", str(output), "--case", BINDING["case_id"],
                "--review", str(release / "authority/independent-review.json"),
                "--decision", str(release / "authority/execution-decision.json"),
            ]
            bounded_run(task, env, package,
                        BINDING["kernel_seconds"] - (time.monotonic() - started))
"""


def render(metadata, deployment, payload):
    """Preserve the old exact mount/archive/hardware guard; replace the actual entry."""
    del metadata
    old = Path(payload) / "deploy" / deployment["case_id"] / "kernel/kernel.py"
    text = old.read_text(encoding="utf-8")
    tree = ast.parse(text)
    assignment = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "BINDING" for t in n.targets)
    )
    bound = ast.literal_eval(assignment.value)
    bound.update(
        operator_sha256=deployment["worker_sha256"],
        normal_service_prefix=service_names(deployment)["normal_service_prefix"],
        normal_source_sha256="89b7e3ac5b42716e24d0d2a1994d13c32845e0c93a668f284e3cb341fe301cb8",
        task="kv_retrieval" if deployment["task"] == "mqar-k64" else "sorting",
    )
    before = ast.get_source_segment(text, assignment)
    text = text.replace(before, "BINDING = " + repr(bound), 1)
    function = next(
        n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    text = text.replace(ast.get_source_segment(text, function), MAIN.rstrip(), 1)
    require("seed202_pilot.cloud_runner" not in text, "UNIFIED_NO_OLD_RUNNER")
    require(
        digest(Path(payload) / "normal-package/normal-source-identity.json")
        == bound["normal_source_sha256"],
        "UNIFIED_RENDER_SOURCE",
    )
    compile(text, "unified-normal-kernel.py", "exec")
    return text
