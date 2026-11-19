"""Check real emitted NORMAL entrypoints, not just the maintained generator."""

import json
import os
import shutil
import subprocess
import sys

import build_normal_training_lifecycle as builder

from tests.scripts.test_normal_training_closure import PARENT, REPO


def test_every_emitted_normal_entrypoint_passes_quality(tmp_path):
    package = builder.build_package(PARENT, REPO, tmp_path / "normal")
    entrypoints = sorted(
        {
            path
            for pattern in ("worker.py", "setup.py", "kernel*.py")
            for path in package.glob(pattern)
        }
    )
    assert {package / "worker.py", package / "setup.py"} <= set(entrypoints)
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
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
        PYLINTHOME=str(tmp_path / "pylint-cache"),
        PYTHONPATH=os.pathsep.join(
            map(str, [package / "src", package / "inherited/operator", REPO / "src"])
        ),
    )
    base = [
        shutil.which("uv"),
        "run",
        "--offline",
        "--no-project",
        "--python",
        sys.executable,
        "python",
        "-B",
    ]
    config = str(REPO / "pyproject.toml")
    gates = {
        "ruff-check": ["-m", "ruff", "check", "--no-cache", "--config", config],
        "ruff-format": ["-m", "ruff", "format", "--check", "--no-cache", "--config", config],
        "pylint": ["-m", "pylint", "--rcfile", config, "--disable=all", "--enable=E,F"],
    }
    results = {}
    for name, arguments in gates.items():
        result = subprocess.run(
            base + arguments + list(map(str, entrypoints)),
            cwd=package,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        output = result.stdout + result.stderr
        (tmp_path / (name + ".log")).write_text(output, encoding="utf-8")
        results[name] = {"exit_code": result.returncode, "output": output}
    (tmp_path / "entrypoint-quality.json").write_text(
        json.dumps(
            {
                "entrypoints": [path.relative_to(package).as_posix() for path in entrypoints],
                "results": results,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    assert all(row["exit_code"] == 0 for row in results.values()), results
