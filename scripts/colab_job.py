"""Small stdlib detached supervisor. No retry, scheduling or idle prevention."""

# Exact JSON ints reject bool; detached children must outlive this function.
# pylint: disable=unidiomatic-typecheck,consider-using-with

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PACKAGE = Path("/content/compare-attention-package")
VENV = Path("/content/compare-attention-venv")
RESULT = Path("/content/compare-attention-results/deltanet-full-seed42-run01")
ENV_FILE = Path("/content/compare-attention-private/kaggle-env.env")
PHASES = ("bootstrap", "setup", "run")


def clean_environment():
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(("HF_", "HUGGING", "CLEARML_", "KAGGLE_", "PIP_", "GOOGLE_"))
        and key not in ("PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE", "PYTHONINSPECT")
    }
    env.update(
        PYTHON_DOTENV_DISABLED="1",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        PIP_CONFIG_FILE=os.devnull,
        PIP_DISABLE_PIP_VERSION_CHECK="1",
    )
    return env


def command(phase, control):
    if phase == "bootstrap":
        return [sys.executable, "-B", str(Path(control) / "bootstrap.py")]
    if phase not in PHASES:
        raise ValueError("INVALID_PHASE")
    args = [
        sys.executable,
        "-B",
        str(PACKAGE / "tools/colab_runner.py"),
        "--package-dir",
        str(PACKAGE),
        "--venv",
        str(VENV),
        "--" + phase,
    ]
    if phase == "run":
        args += ["--family", "deltanet", "--result-dir", str(RESULT), "--env-file", str(ENV_FILE)]
    return args


def atomic(path, payload):
    temporary = path.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    # Retry only local atomic publication when a Windows reader holds the target.
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.01)


def status(control):
    """Project only typed, controlled status. Never return subprocess log content."""
    result = {}
    for phase in PHASES:
        path = Path(control) / (phase + ".json")
        if not path.exists():
            requested = (Path(control) / (phase + ".requested.json")).exists()
            result[phase] = {
                "phase": phase,
                "state": "unknown" if requested else "not_started",
                "exit_code": None,
                "pid": None,
            }
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (
            raw.get("state") not in ("running", "completed", "failed")
            or raw.get("phase") != phase
            or type(raw.get("pid")) is not int
            or not (raw.get("exit_code") is None or type(raw["exit_code"]) is int)
        ):
            raise ValueError("INVALID_JOB_STATUS")
        result[phase] = {key: raw[key] for key in ("phase", "state", "exit_code", "pid")}
    return result


def start(phase, control):
    """Exclusive marker precedes spawn; lost acknowledgement is NEVER retried."""
    if phase not in PHASES:
        raise ValueError("INVALID_PHASE")
    control = Path(control).resolve()
    control.mkdir(parents=True, exist_ok=True)
    if phase != "bootstrap":
        previous = PHASES[PHASES.index(phase) - 1]
        if status(control)[previous]["state"] != "completed":
            raise ValueError("PREREQUISITE_NOT_COMPLETED")
    marker = control / (phase + ".requested.json")
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"phase": phase, "state": "requested"}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    with open(os.devnull, "rb") as incoming, open(os.devnull, "ab") as outgoing:
        child = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "supervise",
                phase,
                "--control-dir",
                str(control),
            ],
            stdin=incoming,
            stdout=outgoing,
            stderr=outgoing,
            env=clean_environment(),
            start_new_session=True,
            cwd=control,
            close_fds=True,
        )
    # Do not overwrite the supervisor's state: child is its only writer.
    return {"phase": phase, "state": "requested", "pid": child.pid, "exit_code": None}


def supervise(phase, control):
    control = Path(control)
    state = {"phase": phase, "state": "running", "pid": os.getpid(), "exit_code": None}
    atomic(control / (phase + ".json"), state)
    code = 1
    try:
        with (control / (phase + ".log")).open("ab") as log:
            completed = subprocess.run(
                command(phase, control),
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                cwd=control,
                env=clean_environment(),
            )
            code = completed.returncode
        if phase == "run" and code == 0:
            receipt = RESULT.parent / (RESULT.name + ".hf-upload-receipt.json")
            if (
                not (RESULT / "colab-native-gate.json").is_file()
                or json.loads(receipt.read_text()).get("verified") is not True
            ):
                code = 1
    except Exception:  # pylint: disable=broad-exception-caught
        code = 1  # No original exception, provider output, key or URL enters status.
    state.update(state="completed" if code == 0 else "failed", exit_code=code)
    atomic(control / (phase + ".json"), state)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "supervise", "status"))
    parser.add_argument("phase", choices=PHASES, nargs="?")
    parser.add_argument("--control-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "status":
        print(json.dumps(status(args.control_dir)))
        return 0
    if args.phase is None:
        parser.error("phase required")
    if args.action == "supervise":
        return supervise(args.phase, args.control_dir)
    if not args.execute:
        print(json.dumps({"state": "plan", "phase": args.phase}))
        return 0
    print(json.dumps(start(args.phase, args.control_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
