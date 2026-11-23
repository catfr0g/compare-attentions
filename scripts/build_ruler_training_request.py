"""Create-only independent RULER source/request; never issues reviews or launches."""

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

from build_tiny_training_request import WHEEL, derive_isolated, inventory, save, wheel_sources
from normal_ruler_scope import CASE_IDS
from normal_training_plan import digest, require
from unified_normal_profiles import TINY, authenticate_package


def build(parent, repository, wheel, recipe, data, output, case_id, runtime_sha256):
    """Freeze genuinely current science with the proven UV/two-rank/HF lifecycle."""
    parent, repository, wheel, recipe, data, output = map(
        lambda p: Path(p).resolve(), (parent, repository, wheel, recipe, data, output)
    )
    require(not output.exists(), "RULER_REQUEST_CREATE_ONLY")
    require(case_id in CASE_IDS and re.fullmatch(r"[0-9a-f]{64}", runtime_sha256), "RULER_CASE")
    authenticate_package(parent, TINY)
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
        "normal_training_tracking",
        "unified_normal_host",
        "training_completion",
        "training_diagnostics",
    ):
        shutil.copyfile(repository / "scripts" / (name + ".py"), package / "src" / (name + ".py"))
    shutil.copyfile(repository / "scripts/normal_training_launch.py", package / "setup.py")
    identity = {
        "schema": "normal-source-identity-v1",
        "scope": "RULER_INDEPENDENT_CURRENT_SCIENCE_V1",
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
        *(
            f"{split}_{kind}.npy"
            for split in ("train", "val", "test")
            for kind in ("inputs", "labels")
        ),
    ]
    staged = output / "data/ruler"
    staged.mkdir(parents=True)
    for name in names:
        require((data / name).is_file() and not (data / name).is_symlink(), "RULER_DATA_FILE")
        shutil.copyfile(data / name, staged / name)
    # Recipes remain byte-identical. Runtime data paths are passed through --data.
    import yaml

    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    family = "classic" if "Classic" in case_id else "deltanet"
    seed = int(case_id[-3:])
    require(
        config["schedule"]["seed"] == seed and config["model"]["attention"] == family,
        "RULER_RECIPE_IDENTITY",
    )
    plan = {
        "schema": "normal-training-plan-v1",
        "purpose": "fresh_normal_training",
        "engineering_only": False,
        "source_identity_sha256": digest(package / "normal-source-identity.json"),
        "cases": {
            case_id: {
                "seed": seed,
                "task": "ruler_variable_tracking",
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
        "prefix": "normal-training/ruler-" + case_id,
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
            "schema": "ruler-normal-worker-review-request-v1",
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
    with zipfile.ZipFile(
        output / "ruler-normal-source.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name in inventory(package):
            archive.write(package / name, "normal-package/" + name)
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
