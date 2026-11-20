"""Render the saved NORMAL entry using authenticated existing bootstrap primitives."""

import ast
import textwrap
from pathlib import Path

from unified_normal_contract import digest, require
from unified_normal_source import service_names

MAIN = """def normal_main(mount_root, working, package_root, started):
    global NormalDiagnostics
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
    import importlib.util

    diagnostic_spec = importlib.util.spec_from_file_location(
        "normal_launch_diagnostics", package / "normal_launch_diagnostics.py"
    )
    diagnostic_module = importlib.util.module_from_spec(diagnostic_spec)
    diagnostic_spec.loader.exec_module(diagnostic_module)
    NormalDiagnostics = diagnostic_module.NormalDiagnostics
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
    diagnostic_root = output / ".launch-diagnostics"
    NormalDiagnostics(
        diagnostic_root, "setup", BINDING["normal_source_sha256"], BINDING["case_id"]
    ).mark("bootstrap", "setup")
    uv, python, env, _topology = setup(science, output, started + 600)
    private = resolve_mount(
        mount_root, "zimike", "k64-private-20261109a1b2", "private.env"
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([
        str(package), str(package / "src"), str(package / "inherited/operator"),
        str(package / "inherited/operator/inherited"), str(package / "inherited/tools"),
    ])
    for phase in ("worker", "finish"):
        diagnostics = NormalDiagnostics(
            diagnostic_root, phase, BINDING["normal_source_sha256"], BINDING["case_id"]
        )
        env.update(diagnostics.environment())
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
        diagnostics.supervise(lambda command=command: bounded_run(
            command, env, package,
            BINDING["kernel_seconds"] - (time.monotonic() - started)))
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
            diagnostics = NormalDiagnostics(
                diagnostic_root, "task", BINDING["normal_source_sha256"], BINDING["case_id"]
            )
            env.update(diagnostics.environment())
            diagnostics.supervise(lambda task=task: bounded_run(
                task, env, package,
                BINDING["kernel_seconds"] - (time.monotonic() - started)), stage="task")


def main(
    mount_root=Path("/kaggle/input"),
    working=Path("/kaggle/working"),
    package_root=Path("/kaggle/temp"),
):
    started = time.monotonic()
    diagnostic_root = working / ("normal-" + BINDING["nonce"]) / ".launch-diagnostics"
    diagnostic_root.mkdir(parents=True, exist_ok=False)
    try:
        return normal_main(mount_root, working, package_root, started)
    except Exception as error:
        if "NormalDiagnostics" in globals():
            diagnostics = NormalDiagnostics(
                diagnostic_root, "setup", BINDING["normal_source_sha256"], BINDING["case_id"]
            )
            def propagate(error=error):
                raise error
            return diagnostics.capture(propagate, "bootstrap")
        # Before source extraction, use the existing finite standalone projection.
        try:
            with (diagnostic_root / "setup-bootstrap.json").open("x") as stream:
                json.dump(normal_outer_error(error, "setup"), stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            pass
        raise
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
        normal_source_sha256=digest(Path(payload) / "normal-package/normal-source-identity.json"),
        task="kv_retrieval" if deployment["task"] == "mqar-k64" else "sorting",
    )
    before = ast.get_source_segment(text, assignment)
    text = text.replace(before, "BINDING = " + repr(bound), 1)
    function = next(
        n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    text = text.replace(ast.get_source_segment(text, function), MAIN.rstrip(), 1)

    # Keep the existing finite standalone policy byte-value equivalent and readable
    # without expanding the generated launch module beyond the repository line cap.
    policies = [
        (ast.get_source_segment(text, node), node.targets[0].id, ast.literal_eval(node.value))
        for node in ast.parse(text).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"CLASSES", "CATEGORIES", "FILES", "FUNCTIONS"}
    ]
    for segment, name, values in policies:
        packed = textwrap.fill(" ".join(sorted(values)), width=88)
        text = text.replace(segment, name + ' = set("""' + packed + '""".split())', 1)

    text = text.replace(
        "def public_error(error, phase):", "def normal_outer_error(error, phase):", 1
    )

    boundary = """def public_error(error, phase):
    report = normal_outer_error(error, phase)
    primary = getattr(error, "_normal_primary", None)
    if (
        "NormalDiagnostics" in globals() and type(primary) is dict
        and type(primary.get("phase")) is str
        and primary["phase"] in ("worker", "finish", "task", "accept", "setup")
    ):
        context = NormalDiagnostics(
            ".", primary["phase"], BINDING["normal_source_sha256"], BINDING["case_id"]
        )
        if context.valid(primary, primary.get("stage"), primary.get("rank")):
            report["primary"] = primary
    return report


"""
    text = text.replace('if __name__ == "__main__":', boundary + 'if __name__ == "__main__":', 1)
    require("seed202_pilot.cloud_runner" not in text, "UNIFIED_NO_OLD_RUNNER")
    require(
        digest(Path(payload) / "normal-package/normal-source-identity.json")
        == bound["normal_source_sha256"],
        "UNIFIED_RENDER_SOURCE",
    )
    compile(text, "unified-normal-kernel.py", "exec")
    return text
