"""Create-only Tiny worker review request using the existing NORMAL executor.

No provider API, review issuance, allocation, training, or credential loading.
The resulting request is deliberately UNBOUND until the parent reviews its exact
source/plan and supplies the existing NORMAL review and execution decision.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from normal_training_plan import canonical, digest, require
from unified_normal_profiles import REMAINING_THREE, authenticate_package

CASE_IDS = tuple(
    f"Tiny{family.title()}{seed}" for seed in (202, 303, 404) for family in ("classic", "deltanet")
)
WHEEL = "compare_attention-0.1.0-py3-none-any.whl"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def inventory(root):
    return {
        p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def wheel_sources(wheel, repository):
    """Authenticate every package Python member against the actual current source."""
    with zipfile.ZipFile(wheel) as saved:
        names = saved.namelist()
        require(len(names) == len(set(names)), "TINY_WHEEL_UNIQUE")
        for name in names:
            part = PurePosixPath(name)
            require(
                not part.is_absolute()
                and ".." not in part.parts
                and "\\" not in name
                and ":" not in name,
                "TINY_WHEEL_PATH",
            )
        members = {n for n in names if n.startswith("compare_attention/") and n.endswith(".py")}
        current = {
            p.relative_to(repository / "src").as_posix()
            for p in (repository / "src/compare_attention").rglob("*.py")
        }
        require(members == current, "TINY_WHEEL_SOURCE_SET")
        require(
            all(saved.read(n) == (repository / "src" / n).read_bytes() for n in members),
            "TINY_WHEEL_CURRENT_BYTES",
        )
    return len(members)


def derive_isolated(package, plan, case_id, recipe, data):
    """Actual emitted helper + wheel + public parser/data path, stopped before NN."""
    with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"], prefix="tiny-derive-") as tmp:
        with zipfile.ZipFile(package / WHEEL) as saved:
            saved.extractall(tmp)
        code = (
            "import sys,json;from pathlib import Path;"
            "sys.path[:0]=[sys.argv[1],str(Path(sys.argv[2])/'src')];"
            "import normal_training_plan as p;"
            "assert Path(p.__file__).resolve()=="
            "(Path(sys.argv[2])/'src/normal_training_plan.py').resolve();"
            "print(json.dumps(p.derive(*sys.argv[2:]),sort_keys=True))"
        )
        argv = [
            shutil.which("uv"),
            "run",
            "--offline",
            "--no-project",
            "--no-sync",
            "--python",
            sys.executable,
            "python",
            "-I",
            "-B",
            "-c",
            code,
            tmp,
            str(package),
            str(plan),
            case_id,
            str(recipe),
            str(data),
        ]
        child = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=180)
        require(child.returncode == 0, "TINY_EMITTED_DERIVE_REJECTED")
        return json.loads(child.stdout)


def build(parent, repository, wheel, recipe, data, output, case_id, runtime_sha256):
    """Freeze genuinely current science with the proven UV/two-rank/HF lifecycle."""
    parent, repository, wheel, recipe, data, output = map(
        lambda p: Path(p).resolve(), (parent, repository, wheel, recipe, data, output)
    )
    require(not output.exists(), "TINY_REQUEST_CREATE_ONLY")
    require(case_id in CASE_IDS and re.fullmatch(r"[0-9a-f]{64}", runtime_sha256), "TINY_CASE")
    authenticate_package(parent, REMAINING_THREE)
    members = wheel_sources(wheel, repository)
    output.mkdir(parents=True, exist_ok=False)
    package = output / "normal-package"
    shutil.copytree(parent, package)
    (package / "normal-source-identity.json").unlink()
    shutil.copyfile(wheel, package / WHEEL)
    for name in (
        "normal_training_plan",
        "normal_training_native",
        "normal_training_worker",
        "normal_training_acceptance",
        "normal_training_lm",
        "unified_normal_host",
        "training_completion",
        "training_diagnostics",
    ):
        shutil.copyfile(repository / "scripts" / (name + ".py"), package / "src" / (name + ".py"))
    shutil.copyfile(repository / "scripts/normal_training_launch.py", package / "setup.py")
    identity = {
        "schema": "normal-source-identity-v1",
        "scope": "TINY_INDEPENDENT_CURRENT_SCIENCE_V1",
        "parent_identity_sha256": digest(parent / "normal-source-identity.json"),
        "wheel_sha256": digest(wheel),
        "current_package_python_members": members,
        "scientific_source_modified": False,
        "tracking_policy": "AUXILIARY_NONBLOCKING_V1",
        "files": inventory(package),
    }
    save(package / "normal-source-identity.json", identity)
    shutil.copyfile(repository / "scripts/unified_normal_task.py", output / "normal-task.py")
    shutil.copyfile(
        repository / "scripts/unified_normal_authority.py", output / "normal-authority.py"
    )
    shutil.copyfile(recipe, output / "recipe.yaml")
    names = [
        "meta.json",
        "input.txt",
        *(
            f"{split}_{kind}.npy"
            for split in ("train", "val", "test")
            for kind in ("inputs", "labels")
        ),
    ]
    staged = output / "data/tiny"
    staged.mkdir(parents=True)
    for name in names:
        require((data / name).is_file() and not (data / name).is_symlink(), "TINY_DATA_FILE")
        shutil.copyfile(data / name, staged / name)
    # Recipes remain byte-identical. Runtime data paths are passed through --data.
    import yaml

    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    family = "classic" if "Classic" in case_id else "deltanet"
    seed = int(case_id[-3:])
    require(
        config["schedule"]["seed"] == seed and config["model"]["attention"] == family,
        "TINY_RECIPE_IDENTITY",
    )
    plan = {
        "schema": "normal-training-plan-v1",
        "purpose": "fresh_normal_training",
        "engineering_only": False,
        "source_identity_sha256": digest(package / "normal-source-identity.json"),
        "cases": {
            case_id: {
                "seed": seed,
                "task": "tiny_shakespeare_char",
                "family": family,
                "recipe_sha256": digest(recipe),
                "runtime_sha256": runtime_sha256,
                "data_files": inventory(staged),
            }
        },
    }
    save(output / "normal-plan.json", plan)
    case = derive_isolated(
        package, output / "normal-plan.json", case_id, output / "recipe.yaml", staged
    )
    save(output / "derived-case.json", {"case": case})
    fields = {
        "package": package,
        "plan": output / "normal-plan.json",
        "recipe": output / "recipe.yaml",
        "data": staged,
        "case": case_id,
        "prefix": "normal-training/tiny-" + case_id,
        "private": output / "PARENT-PRIVATE.env",
        "output": output / "PARENT-CLOUD-RESULT",
        "phase": "worker",
        "review": output / "authority/independent-review.json",
        "decision": output / "authority/execution-decision.json",
    }
    argv = [
        "uv",
        "run",
        "--offline",
        "--no-sync",
        "python",
        "-B",
        str(package / "setup.py"),
        "--uv",
        shutil.which("uv"),
        "--python",
        sys.executable,
        "--",
    ]
    argv += [token for name, value in fields.items() for token in ("--" + name, str(value))]
    save(
        output / "request.json",
        {
            "schema": "tiny-normal-worker-review-request-v1",
            "case": case,
            "source_identity_sha256": plan["source_identity_sha256"],
            "plan_sha256": digest(output / "normal-plan.json"),
            "entrypoint_argv": argv,
            "launch_authority": False,
            "status": "UNBOUND_PARENT_REVIEW_REQUIRED",
            "provider_mutations": 0,
            "required_next": [
                "parent source/plan review",
                "NORMAL execution decision",
                "Kaggle binding: fresh quota/privacy/namespace/seal; retained static lease",
            ],
        },
    )
    save(output / "manifest.json", {"files": inventory(output)})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("parent", "repository", "wheel", "recipe", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    args = parser.parse_args(argv)
    build(
        args.parent,
        args.repository,
        args.wheel,
        args.recipe,
        args.data,
        args.output,
        args.case,
        args.runtime_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
