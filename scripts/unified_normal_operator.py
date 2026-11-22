"""Create a versioned operator from authenticated archives, never issue approval."""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

from build_unified_normal_contract import build as build_request
from unified_normal_contract import canonical, digest, read, require
from unified_normal_source import PROFILE

PARENT = "e10e7e693e5eeca90ba732d2644d8c00a303bd32533abe04b3789955c5b499f5"
NORMAL = "67861b9b2e62081c4b486ac918c0716c32257f3c5af5fe6b5bf01aa5bb62cf1c"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value) + "\n", encoding="utf-8", newline="\n")


def inventory(root):
    return {
        p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def unpack(archive, target, expected, prefix=""):
    require(digest(archive) == expected, "UNIFIED_ARCHIVE_AUTHORITY")
    with zipfile.ZipFile(archive) as saved:
        names = saved.namelist()
        require(len(names) == len(set(names)), "UNIFIED_UNIQUE_ARCHIVE")
        for name in names:
            part = PurePosixPath(name)
            require(
                not part.is_absolute()
                and ".." not in part.parts
                and "\\" not in name
                and ":" not in name
                and saved.getinfo(name).orig_filename == name
                and not saved.getinfo(name).is_dir()
                and (saved.getinfo(name).external_attr >> 16) & 0o170000 != 0o120000,
                "UNIFIED_SAFE_ARCHIVE",
            )
            if not name.startswith(prefix):
                continue
            path = target / name.removeprefix(prefix)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(saved.read(name))


def replace(path, before, after):
    text = path.read_text(encoding="utf-8")
    require(text.count(before) == 1, "UNIFIED_EXACT_INTERFACE_" + path.name)
    path.write_text(text.replace(before, after), encoding="utf-8", newline="\n")


def integrate(source, profile_name=PROFILE):
    """Every audited consumer imports the same typed contract, without monkeypatching."""
    for path in source.glob("*.py"):
        if path.name == "qualified_bridge.py":
            continue
        text = path.read_text(encoding="utf-8")
        text = text.replace(
            "seed202_pilot.qualified_bridge import", "unified_normal_contract import"
        )
        path.write_text(text, encoding="utf-8", newline="\n")
    replace(
        source / "cli.py",
        '    try:\n        if args.command == "composite":',
        "    try:\n"
        "        from unified_normal_authority import before_auth\n\n"
        "        before_auth(args)\n"
        '        if args.command == "composite":',
    )
    seal = source / "seal.py"
    replace(
        seal,
        '    _require(type(plan) is dict and plan.get("schema_version") == 2, "PLAN_SCHEMA")\n',
        "    from unified_normal_preflight import authenticated_plan\n\n"
        "    if authenticated_plan(plan):\n        return\n"
        '    _require(type(plan) is dict and plan.get("schema_version") == 2, "PLAN_SCHEMA")\n',
    )
    text = seal.read_text(encoding="utf-8")
    start = text.index(
        "            and (\n", text.index('    predecessors = proof["predecessor_evidence"]')
    )
    end = text.index("            for row in predecessors", start)
    text = text[:start] + "            and valid_evidence_row(row)\n" + text[end:]
    text = text.replace(
        '    predecessors = proof["predecessor_evidence"]',
        "    from unified_normal_contract import valid_evidence_row\n\n"
        '    predecessors = proof["predecessor_evidence"]',
    )
    seal.write_text(text, encoding="utf-8", newline="\n")
    replace(
        seal,
        '    """Bind one reviewed singleton to a fresh provider proof; issue no effects."""\n',
        '    """Bind one reviewed singleton to a fresh provider proof; issue no effects."""\n'
        "    from unified_normal_authority import require_bound\n\n"
        '    require_bound(plan["candidate_root"])\n',
    )
    replace(
        seal,
        "    case = require_authorized_case(case_id)\n"
        "    candidate = Path(candidate_root).resolve()",
        "    from unified_normal_contract import descriptor as requested_run\n\n"
        "    candidate = Path(candidate_root).resolve()\n"
        "    _request_path, request = requested_run(candidate)\n"
        '    _require(case_id == request["run"]["id"], "UNIFIED_REQUESTED_CASE_ONLY")\n'
        "    case = require_authorized_case(case_id)",
    )
    replace(
        source / "fresh_mode.py",
        '        "requested_versioned_inputs": wanted,\n',
        '        "requested_versioned_inputs": wanted,\n'
        '        **__import__("unified_normal_source", '
        'fromlist=["operator_profile"]).operator_profile(\n'
        "            candidate, deployment\n        ),\n",
    )
    replace(
        source / "qualified_imports.py",
        "        verify_files(inputs)\n        check(local)\n"
        "        try:\n            yield\n        finally:\n            check(local)",
        "        from unified_normal_contract import held_inputs\n\n"
        "        with held_inputs(candidate, local):\n"
        "            check(local)\n            try:\n                yield\n"
        "            finally:\n                check(local)",
    )
    replace(
        source / "qualified_imports.py",
        "    from unified_normal_contract import descriptor, verify_files",
        "    from unified_normal_contract import descriptor",
    )
    replace(
        source / "qualified_imports.py",
        "    from seed202_pilot.prepared_files import check, lease",
        "    from seed202_pilot.prepared_files import check\n"
        "    from unified_normal_live import retained_lease as lease",
    )
    replace(
        source / "preparation.py",
        "from seed202_pilot.prepared_files import check, lease, safe_path, safe_root",
        "from seed202_pilot.prepared_files import check, safe_path, safe_root\n"
        "from unified_normal_live import retained_lease as lease",
    )
    replace(
        source / "storage.py",
        'def digest(path: Path) -> str:\n    with Path(path).open("rb") as stream:',
        "def digest(path: Path) -> str:\n"
        "    from unified_normal_live import held_digest\n\n"
        "    authenticated = held_digest(path)\n"
        "    if authenticated is not None:\n"
        "        return authenticated\n"
        '    with Path(path).open("rb") as stream:',
    )
    path = source / "host_operator.py"
    text = path.read_text(encoding="utf-8")
    start = text.index('    if args.command == "services":', text.index("def _execute("))
    end = text.index('    if args.command == "reconcile":', start)
    text = (
        text[:start] + '    if args.command in {"services", "accept"}:\n'
        "        from unified_normal_host import run_public\n\n"
        "        return run_public(args, approval, provider, metadata, source)\n" + text[end:]
    )
    path.write_text(text, encoding="utf-8", newline="\n")
    replace(
        source / "qualified_imports.py",
        "    candidate = Path(args.candidate_dir).resolve()",
        "    from unified_normal_host import validate_arguments\n"
        "    from unified_normal_authority import before_auth\n\n"
        "    before_auth(args)\n"
        "    validate_arguments(args)\n"
        "    candidate = Path(args.candidate_dir).resolve()",
    )
    replace(
        source / "host_services.py",
        '    def read() -> None:\n        with services.service_env(values, "track"):',
        "    def read() -> None:\n"
        "        from unified_normal_source import service_names\n\n"
        '        names = service_names({"source_profile": '
        + repr(profile_name)
        + ', "nonce": nonce})\n'
        '        with services.service_env(values, "track"):',
    )
    replace(
        source / "host_services.py",
        "            task_pattern = (\n"
        '                r"^seed202-" + re.escape(nonce) + "-" '
        '+ re.escape(case_id) + r"(?:-task-val)?$"\n'
        "            )",
        '            task_pattern = names["normal_task_pattern"]',
    )
    replace(
        source / "host_services.py",
        '            prefix = "seed202-pilot-v1/" + nonce + "/" + case_id + "/"',
        '            prefix = names["normal_service_prefix"] + "/"',
    )
    # Compatibility names are explicit re-exports, never another policy registry.
    (source / "qualified_bridge.py").write_text(
        '"""Compatibility exports of the single versioned predecessor verifier."""\n\n'
        + "from unified_normal_contract import (\n"
        + "".join(
            "    " + name + " as " + name + ",\n"
            for name in (
                "binding",
                "descriptor",
                "evidence_paths",
                "expected_evidence",
                "next_qualified_case",
                "proof_rows",
                "replay_predecessor",
                "verify_files",
            )
        )
        + ")\n",
        encoding="utf-8",
        newline="\n",
    )


def format_generated(paths):
    """Apply the real formatter before source hashes and the final archive are frozen."""
    base = [
        "uv",
        "run",
        "--offline",
        "--no-project",
        "--no-sync",
        "--python",
        sys.executable,
        "python",
        "-B",
        "-m",
        "ruff",
    ]
    config = str(Path(__file__).resolve().parents[1] / "pyproject.toml")
    for command in (["format"], ["check", "--fix"], ["format"]):
        subprocess.run(
            [*base, *command, "--config", config, *map(str, paths)], check=True, capture_output=True
        )


def freeze(candidate):
    payload = candidate / "payload"
    manifest = payload / "manifest.json"
    manifest.unlink(missing_ok=True)
    files = inventory(payload)
    save(
        manifest,
        {"schema_version": 1, "scope": "COMPLETE_SEED202_PILOT_REVIEW_CANDIDATE", "files": files},
    )
    archive = candidate / "sorting-mqar-seed202-pilot-v1.zip"
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as saved:
        for name in [*files, "manifest.json"]:
            saved.write(payload / name, name)
    shutil.copyfile(payload / "PLAN-UNAPPROVED.json", candidate / "PLAN-UNAPPROVED.json")
    save(
        candidate / "candidate.json",
        {
            "status": "COMPLETE_FROZEN_CANDIDATE_REVIEW_REQUIRED",
            "qualification": "SOURCE_FREEZE_NOT_TEST_COMPLETION_OR_LAUNCH_AUTHORITY",
            "sha256": digest(archive),
            "members": len(files) + 1,
            "cloud_mutations": 0,
            "independent_review": "REQUIRED_NOT_SELF_ISSUED",
            "launch_approval": "NOT_ISSUED",
        },
    )
    with zipfile.ZipFile(archive) as saved:
        require(set(saved.namelist()) == set(files) | {"manifest.json"}, "UNIFIED_READBACK_SET")
        for name in saved.namelist():
            require(saved.read(name) == (payload / name).read_bytes(), "UNIFIED_READBACK_BYTES")


def build(inputs, normal_archive, destination):
    """Assemble a new host and actual NORMAL worker, preserving all old releases."""
    from unified_normal_render import render

    destination = Path(destination).resolve()
    require(not destination.exists(), "UNIFIED_OPERATOR_CREATE_ONLY")
    parent = Path(inputs["sorting_candidate"]) / "sorting-mqar-seed202-pilot-v1.zip"
    unpack(parent, destination / "payload", PARENT)
    normal = destination / "payload/normal-package"
    from unified_normal_profiles import REPAIRED, TINY, authenticate_package, is_remaining_profile
    from unified_normal_profiles import selected as source_profile

    profile_name = inputs.get("source_profile", PROFILE)
    release = source_profile(profile_name)
    unpack(normal_archive, normal, release["archive"], release["prefix"])
    authenticate_package(normal, profile_name)
    finite = profile_name == TINY or is_remaining_profile(profile_name)
    if profile_name == REPAIRED or finite:
        require("generation" in inputs, "UNIFIED_GENERATION_REQUIRED")
    request_inputs = {
        k: v
        for k, v in inputs.items()
        if k
        not in (
            {"normal_review", "normal_decision"}
            | (
                {"normal_plan", "normal_recipe"}
                if profile_name != REPAIRED and not finite
                else set()
            )
        )
    }
    request_inputs["normal_package"] = str(normal)
    packet = destination / "request-build"
    build_request(request_inputs, packet)
    payload = destination / "payload"
    shutil.copytree(packet / "payload/unified", payload / "unified")
    request = read(payload / "unified/request.json")
    # Source copies are pinned after their actual distribution layout is established.
    request["roots"].remove(str(packet / "source"))
    request["files"] = {
        k: v for k, v in request["files"].items() if not Path(k).is_relative_to(packet / "source")
    }
    host = payload / "source/src"
    for name in (
        "unified_normal_contract",
        "completed_normal_publication",
        "completed_sorting404_publication",
        "training_diagnostics",
        "unified_normal_live",
        "unified_normal_preflight",
        "unified_normal_replay",
        "unified_normal_source",
        "unified_normal_profiles",
        "unified_normal_generation",
        "unified_normal_task",
        "unified_normal_host",
        "unified_normal_host_view",
        "unified_normal_authority",
        "unified_normal_publications",
        "unified_normal_ordinary",
        "unified_normal_remaining",
        "normal_remaining_scope",
        "normal_fresh_restart_scope",
        "normal_tiny_scope",
        "normal_training_lm",
        "unified_normal_tiny",
    ):
        shutil.copyfile(Path(__file__).with_name(name + ".py"), host / (name + ".py"))
    integrate(host / "seed202_pilot", profile_name)
    if profile_name == TINY:
        from unified_normal_tiny import integrate_group

        integrate_group(host / "seed202_pilot")
    elif is_remaining_profile(profile_name):
        from unified_normal_remaining import integrate_group

        integrate_group(host / "seed202_pilot")
    format_generated(
        [
            host / "seed202_pilot" / (name + ".py")
            for name in (
                "seal",
                "fresh_mode",
                "host_operator",
                "host_services",
                "audited_publish",
                "preparation",
                "storage",
                "cli",
                "qualified_imports",
                "qualified_bridge",
                "contract",
            )
        ]
    )
    plan = read(payload / "PLAN-UNAPPROVED.json")
    selected = request["run"]
    if profile_name == TINY:
        from unified_normal_generation import intent
        from unified_normal_tiny import deployment as tiny_deployment

        deployment = tiny_deployment(
            payload, plan, selected, intent(inputs["generation"], profile_name)
        )
    elif is_remaining_profile(profile_name):
        from unified_normal_generation import intent
        from unified_normal_remaining import deployment as fresh_deployment

        deployment = fresh_deployment(
            payload, plan, selected, intent(inputs["generation"], profile_name), profile_name
        )
    else:
        require(selected["id"] == "mqar-k64-deltanet-seed202", "UNIFIED_CURRENT_AUTHORIZATION")
        deployment = next(r for r in plan["deployments"] if r["case_id"] == selected["id"])
    deployment["source_profile"] = profile_name
    if profile_name == REPAIRED:
        from unified_normal_generation import apply_namespace, intent

        apply_namespace(payload, deployment, intent(inputs["generation"], profile_name))
    recipe = payload / "configs" / (selected["id"] + ".yaml")
    shutil.copyfile(inputs["normal_recipe"], recipe)
    deployment["config_sha256"] = digest(recipe)
    normal_plan = {
        "schema": "normal-training-plan-v1",
        "purpose": "fresh_normal_training",
        "engineering_only": False,
        "source_identity_sha256": release["source"],
        "cases": {
            selected["id"]: {
                "seed": selected["seed"],
                "task": selected["task"],
                "family": selected["family"],
                "recipe_sha256": digest(recipe),
                "runtime_sha256": selected["runtime_sha256"],
                "data_files": read(inputs["normal_plan"])["cases"][selected["id"]]["data_files"],
            }
        },
    }
    if profile_name != REPAIRED and not finite:
        save(payload / "unified/normal-plan.json", normal_plan)
    from unified_normal_authority import SCHEMA, validate

    authority = {"schema": SCHEMA, "status": "UNBOUND_REVIEW_AND_DECISION_REQUIRED", "files": {}}
    require(
        ("normal_review" in inputs) == ("normal_decision" in inputs),
        "UNIFIED_AUTHORITY_BOTH_INPUTS",
    )
    if "normal_review" in inputs:
        authority = validate(
            payload / "unified/normal-plan.json", inputs["normal_review"], inputs["normal_decision"]
        )
        target = payload / "unified/authority"
        target.mkdir()
        for key, name in (
            ("normal_review", "independent-review.json"),
            ("normal_decision", "execution-decision.json"),
        ):
            shutil.copyfile(inputs[key], target / name)
            p = Path(inputs[key]).resolve()
            request["files"][str(p)] = {"bytes": p.stat().st_size, "sha256": digest(p)}
    save(payload / "unified/normal-authority.json", authority)
    selected["plan_sha256"] = digest(payload / "unified/normal-plan.json")
    deployment["normal_plan_sha256"] = selected["plan_sha256"]
    save(payload / "unified/request.json", request)
    worker_root = destination / "worker-build"
    shutil.copytree(normal, worker_root / "normal-package")
    shutil.copyfile(host / "unified_normal_task.py", worker_root / "normal-task.py")
    shutil.copyfile(host / "unified_normal_authority.py", worker_root / "normal-authority.py")
    if "normal_review" in inputs:
        shutil.copytree(payload / "unified/authority", worker_root / "authority")
    shutil.copyfile(recipe, worker_root / "recipe.yaml")
    shutil.copyfile(payload / "unified/normal-plan.json", worker_root / "normal-plan.json")
    if profile_name == TINY:
        shutil.copytree(selected["config"]["data"], worker_root / "data/tiny")
    save(worker_root / "manifest.json", {"files": inventory(worker_root)})
    worker = payload / "worker/seed202-pilot.zip.bin"
    with zipfile.ZipFile(worker, "w", compression=zipfile.ZIP_DEFLATED) as saved:
        for name in inventory(worker_root):
            saved.write(worker_root / name, name)
    deploy = payload / "deploy" / selected["id"]
    shutil.copyfile(worker, deploy / "dataset/seed202-pilot.zip.bin")
    deployment["worker_sha256"] = digest(worker)
    source = render(read(deploy / "kernel/kernel-metadata.json"), deployment, payload)
    source = source.replace("# fmt: off\n", "")
    (deploy / "kernel/kernel.py").write_text(source, encoding="utf-8", newline="\n")
    format_generated([deploy / "kernel/kernel.py"])
    deployment["kernel_source_sha256"] = digest(deploy / "kernel/kernel.py")
    plan["source_profile"] = profile_name
    save(payload / "PLAN-UNAPPROVED.json", plan)
    repository = Path(__file__).resolve().parents[1]
    maintained = [
        *repository.joinpath("scripts").glob("unified_normal_*.py"),
        *repository.joinpath("scripts").glob("normal_*scope.py"),
        repository / "scripts/build_unified_normal_contract.py",
        *repository.joinpath("docs").glob("unified-normal-*.md"),
        *repository.joinpath("tests/scripts").glob("*unified_normal*.py"),
        *repository.joinpath("tests/scripts").glob("*tiny*.py"),
    ]
    for original in maintained:
        target = payload / "source/maintained" / original.relative_to(repository)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    freeze(destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--normal-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(build(read(args.inputs), args.normal_archive, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
