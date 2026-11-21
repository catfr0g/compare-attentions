"""Bounded remaining-series metadata integration into the existing NORMAL launcher."""

import ast
import hashlib
import shutil
from pathlib import Path

from normal_remaining_scope import CASE_IDS, case_rows, permission, require_prefix
from unified_normal_contract import canonical, descriptor, digest, read, require
from unified_normal_profiles import (
    FRESH_RESTART,
    REMAINING,
    REMAINING_FOUR,
    REMAINING_SIX,
    authenticate_package,
    is_remaining_profile,
    selected,
)

# Exact eight actual nonengineering review subjects, including recipe/data/runtime,
# source, caller nonce and original permission. Not approvals or future overrides.
PLAN_PINS = {
    "sorting-classic-seed303": "75d7320381dcdd6875ccefbe7e47ab7317b5c48a94123896d9074313babbae10",
    "sorting-deltanet-seed303": "f04e501ddf38fdc21b423dbc086cccc591b597381d5f23ca4884345da2138e9c",
    "mqar-k64-classic-seed303": "450acdfef4e0e5429237883b4c5a0705b7f6e5f9a889e6a4edf28eb438e696f2",
    "mqar-k64-deltanet-seed303": "fa6da27fd6a407a7594123a70641c6cd14e5a5fd3462ff0bd069986aaaff6ccb",
    "sorting-classic-seed404": "666a1b1bc7b720a7d302f5e61dc54ecf956df0b78c226c99995fe3fc49c3b3e0",
    "sorting-deltanet-seed404": "ddc7480ff9143d09b537f1c465dd331f9fec3373b96ce9c6977f467d2e739ec1",
    "mqar-k64-classic-seed404": "ac6b3c617055c2a54f630a30211e3470e0a3c37a17246ac56d68ee059b076ae7",
    "mqar-k64-deltanet-seed404": "75806539c9953a9bef64235495862726269f31bbc872eb4222a80874f4274bc3",
}

# New source115 subjects for only the original six unstarted fresh cases.
# Historical d7 and the separately authorized Delta303 restart stay immutable.
SIX_PLAN_PINS = {
    "mqar-k64-classic-seed303": "e05197c1f90656796e84e6beed90116655e4648f16af401f8f189d26094a69c9",
    "mqar-k64-deltanet-seed303": "c8ed3d2396374fb29091e97922f3b7fb7033cec080609d9ef3a1c6e5276e2fa6",
    "sorting-classic-seed404": "c8f89f0222deb046d834a99df51ca818d135f9e4e361218236818a76fde53571",
    "sorting-deltanet-seed404": "7c46d9c56285cc18b88cf656a0026eab83962249511377b3071d7f1398afdb5e",
    "mqar-k64-classic-seed404": "a28a2965f6bea46a652fa02569ce3854b9859dd5837d1b37d3bfadfe83526121",
    "mqar-k64-deltanet-seed404": "edb3609e9f298983efbf5bc52cc46c717ae73ca79cf17174e389e7133e148af9",
}


FOUR_PLAN_PINS = {
    "sorting-classic-seed404": "c02cc765bfab3779791c9784852ad1829ff7613e42c97e6ad84a404d24977752",
    "sorting-deltanet-seed404": "2996b3c1d3d65d13f29e602bd1d04d45e0136c75498d86adb1395b09602a8cd3",
    "mqar-k64-classic-seed404": "fdc76b8bec1541923c9e9993e273652fcf4bd2ddc352a05db71e69629b461876",
    "mqar-k64-deltanet-seed404": "9cfcc86510d246f4ae8d6f0437df0a669e76021f15d295d6ff009d79993eaf5c",
}


def require_frozen_plan(case_id, path, profile=REMAINING):
    """The bounded release cannot be repurposed for a different recipe or nonce."""
    require_plan_pin(case_id, digest(path), profile)


def require_plan_pin(case_id, pin, profile):
    if profile == REMAINING_FOUR:
        require(
            case_id in CASE_IDS[4:] and pin == FOUR_PLAN_PINS.get(case_id),
            "REMAINING_FOUR_FROZEN_PLAN",
        )
    elif profile == REMAINING_SIX:
        require(
            case_id in CASE_IDS[2:] and case_id in SIX_PLAN_PINS and pin == SIX_PLAN_PINS[case_id],
            "REMAINING_SIX_FROZEN_PLAN",
        )
    elif profile == FRESH_RESTART:
        from normal_fresh_restart_scope import frozen_plan

        frozen_plan(case_id, pin)
    else:
        require(
            profile == REMAINING and case_id in PLAN_PINS and pin == PLAN_PINS[case_id],
            "REMAINING_FROZEN_PLAN",
        )


def integrate_group(source):
    """Emit finite selectors only; preserve all SDK, lifetime and scientific guards."""
    from unified_normal_operator import replace

    path = Path(source) / "contract.py"
    text = path.read_text(encoding="utf-8")
    node = next(
        n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "cases"
    )
    segment = ast.get_source_segment(text, node)
    replacement = (
        "def cases() -> tuple[CaseSpec, ...]:\n"
        '    """Exact directly authorized group; old pilot permission is unchanged."""\n'
        "    from normal_remaining_scope import case_specs\n\n"
        "    return case_specs(CaseSpec)"
    )
    path.write_text(text.replace(segment, replacement, 1), encoding="utf-8", newline="\n")
    replace(
        source / "fresh_mode.py",
        'deployment.get("seed") == 202',
        'deployment.get("seed") == case.seed',
    )
    replace(
        source / "seal.py",
        "    from seed202_pilot.transition import verify_qualified_predecessor\n\n"
        "    transition = verify_qualified_predecessor(\n"
        '        candidate / "payload/predecessor/qualified-acceptance.json"\n    )',
        "    from unified_normal_remaining import transition\n\n"
        "    transition = transition(candidate, case_id)",
    )


def transition(candidate, case_id):
    """A fresh user-authorized group, not fabricated ordinary seed202 completion."""
    from unified_normal_generation import intent_value

    _path, request = descriptor(candidate)
    plan = read(Path(candidate) / "payload/unified/normal-plan.json")
    profile = request["normal"]["profile"]
    generation = intent_value(plan["generation"], profile)
    require_frozen_plan(case_id, Path(candidate) / "payload/unified/normal-plan.json", profile)
    require(case_id == generation["case_id"] == request["run"]["id"], "REMAINING_TRANSITION_CASE")
    index = require_prefix(case_id, [r["case_id"] for r in request["predecessors"]])
    return {
        "schema": "DIRECT_AUTHORIZED_REMAINING_SERIES_V1",
        "authorization_sha256": generation["authorization"]["sha256"],
        "case_order": list(CASE_IDS),
        "case_index": index,
        "accepted_prefix": list(CASE_IDS[:index]),
        "historical_acceptances_modified": False,
        "historical_completion_fabricated": False,
        "transition_permission": "EXPLICIT_USER_PERMISSION_EXACT_EIGHT_SEQUENTIAL",
        **(
            {
                "fresh_restart_consent_sha256": generation["restart"]["sha256"],
                "failed_kernel_preserved": 146268145,
                "attempt_limit": 1,
                "resume_permitted": False,
                "stop_on_new_failure": True,
            }
            if profile == FRESH_RESTART
            else {}
        ),
    }


def prefix_policies(rows, profile):
    """Only the repaired four-case release can consume the truthful case3 qualification."""
    from unified_normal_contract import Policy

    if profile == REMAINING_FOUR:
        require(
            all(
                row["policy"]
                == (Policy.COMPLETED_NORMAL if row["case_id"] == CASE_IDS[3] else Policy.ORDINARY)
                for row in rows
            ),
            "REMAINING_QUALIFIED_CASE3_PREFIX_REQUIRED",
        )
    else:
        require(
            all(row["policy"] == Policy.ORDINARY for row in rows),
            "REMAINING_ORDINARY_PREFIX_REQUIRED",
        )


def build_request(inputs, output):
    """Snapshot only the new group inputs; require genuine ordinary prefix rows."""
    from build_unified_normal_contract import pin, save
    from unified_normal_generation import intent, prepare_case
    from unified_normal_source import RUNTIME_RECORD, WHEEL

    require(
        set(inputs)
        == {
            "sorting_candidate",
            "normal_package",
            "normal_case",
            "runtime_record",
            "source_profile",
            "generation",
            "normal_plan",
            "normal_recipe",
            "predecessors",
        },
        "REMAINING_BUILD_INPUTS",
    )
    profile = inputs["source_profile"]
    require(is_remaining_profile(profile), "REMAINING_SOURCE_PROFILE")
    generation = intent(inputs["generation"], profile)
    rows = inputs["predecessors"]
    require(type(rows) is list, "REMAINING_PREFIX_SCHEMA")
    for row in rows:
        require(
            type(row) is dict and set(row) == {"case_id", "policy", "path", "sha256", "replay"},
            "REMAINING_ORDINARY_PREFIX_REQUIRED",
        )
    prefix_policies(rows, profile)
    require_prefix(generation["case_id"], [row["case_id"] for row in rows])
    require_frozen_plan(generation["case_id"], inputs["normal_plan"], profile)
    output = Path(output).resolve()
    require(not output.exists(), "UNIFIED_CREATE_ONLY")
    normal = Path(inputs["normal_package"])
    authenticate_package(normal, profile)
    runtime_path = Path(inputs["runtime_record"])
    require(digest(runtime_path) == RUNTIME_RECORD, "UNIFIED_AUTHENTIC_RUNTIME_REFERENCE")
    fields = ["completed", 0, "acceptance", "timing", "metadata", "execution_runtime"]
    runtime = read(runtime_path)
    for field in fields:
        runtime = runtime[field]
    runtime_sha = hashlib.sha256(canonical(runtime).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    case = prepare_case(inputs, output, runtime_sha)
    source = output / "source"
    source.mkdir()
    for original in Path(__file__).parent.glob("unified_normal_*.py"):
        shutil.copyfile(original, source / original.name)
    for helper in (
        "normal_remaining_scope",
        "normal_fresh_restart_scope",
        "completed_normal_publication",
        "training_diagnostics",
    ):
        shutil.copyfile(Path(__file__).with_name(helper + ".py"), source / (helper + ".py"))
    files, roots = {}, set()

    def include(path):
        path = Path(path).resolve()
        require(path.is_file() and not path.is_symlink() and not path.is_junction(), "UNIFIED_FILE")
        require(
            path.suffix.lower() not in {".env", ".pem", ".key", ".dpapi"},
            "UNIFIED_NO_PRIVATE_BYTES",
        )
        files[str(path)] = pin(path)

    def tree(root):
        root = Path(root).resolve()
        require(root.is_dir() and not root.is_symlink() and not root.is_junction(), "UNIFIED_ROOT")
        roots.add(str(root))
        for path in root.rglob("*"):
            require(not path.is_symlink() and not path.is_junction(), "UNIFIED_LINK")
            if path.is_file():
                include(path)

    for key in ("normal_case", "runtime_record", "generation", "normal_plan", "normal_recipe"):
        include(inputs[key])
    include(generation["authorization"]["path"])
    if profile == FRESH_RESTART:
        include(generation["restart"]["path"])
    tree(normal)
    tree(source)
    data = Path(case["config"]["data"]).resolve()
    roots.add(str(data))
    pins = read(output / "payload/unified/normal-plan.json")["cases"][case["id"]]["data_files"]
    # Genuine derive just verified every data pin. Reuse those authenticated values
    # here; the existing immutable lease still rechecks all bytes before effects.
    for name, expected in pins.items():
        path = data / name
        require(path.is_file() and path.stat().st_size == expected["bytes"], "REMAINING_DATA_PIN")
        files[str(path)] = expected
    for path in data.iterdir():
        if path.is_file() and path.name not in pins:
            include(path)
    for row in rows:
        spec = row["replay"]
        if row["policy"] == "QUALIFIED_MQAR_DELTA303_COMPLETED_NATIVE_V1":
            from completed_normal_publication import replay_completed

            require(
                profile == REMAINING_FOUR and row["case_id"] == CASE_IDS[3],
                "REMAINING_QUALIFIED_CASE3_SCOPE",
            )
            replay_completed(spec, read(row["path"]))
            tree(Path(spec["subject"]).parent)
            include(spec["review"])
            include(row["path"])
            require(digest(row["path"]) == row["sha256"], "REMAINING_PREFIX_PIN")
            continue
        require(
            set(spec) == {"profile", "source_request", "result", "data", "snapshot", "index"}
            and is_remaining_profile(spec["profile"]),
            "REMAINING_PREFIX_REPLAY_SCHEMA",
        )
        previous = Path(spec["source_request"]).parents[2]
        for root in (previous, Path(spec["result"]).parent, spec["data"], spec["snapshot"]):
            tree(root)
        # Ordinary replay requires explicit roots as well as the complete file pins.
        # These subtrees were already inventoried above; do not reread their bytes.
        roots.update(
            str(path.resolve())
            for path in (
                previous / "payload",
                previous / "payload/normal-package",
                previous / "payload/source/src",
                Path(spec["result"]),
            )
        )
        include(row["path"])
        include(spec["index"])
        require(digest(row["path"]) == row["sha256"], "REMAINING_PREFIX_PIN")
    save(
        output / "payload/unified/request.json",
        {
            "schema": "UNIFIED_NORMAL_REQUEST_V1",
            "run": case,
            "predecessors": rows,
            "normal": {
                "profile": profile,
                "package": str(normal),
                "source_identity_sha256": selected(profile)["source"],
                "wheel_sha256": WHEEL,
                "runtime_record": str(runtime_path),
                "runtime_record_sha256": RUNTIME_RECORD,
                "runtime_path": fields,
                "runtime_sha256": runtime_sha,
            },
            "science": read(
                Path(inputs["sorting_candidate"]) / "payload/qualified-launch/bridge.json"
            )["science"],
            "roots": sorted(roots),
            "files": files,
        },
    )
    return output


def deployment(payload, plan, run, generation, profile=REMAINING):
    """Construct a genuinely fresh deployment, never inherit resume or approvals."""
    from unified_normal_generation import intent_value, namespace
    from unified_normal_operator import save

    intent_value(generation, profile)
    permission(generation["authorization"])
    index = CASE_IDS.index(run["id"])
    row = case_rows()[index]
    require(
        (row["seed"], row["family"], row["task"] == "sorting")
        == (run["seed"], run["family"], run["task"] == "sorting"),
        "REMAINING_DEPLOYMENT_CASE",
    )
    template = next(
        item
        for item in plan["deployments"]
        if item["task"] == row["task"] and item.get("mode") == "FRESH_TWO_STAGE"
    )
    old = Path(payload) / "deploy" / template["case_id"]
    root = Path(payload) / "deploy" / run["id"]
    shutil.copytree(old, root)
    targets = namespace(generation["nonce"])
    result = {
        "case_id": run["id"],
        "task": row["task"],
        "family": row["family"],
        "seed": row["seed"],
        "mode": "FRESH_TWO_STAGE",
        "source_profile": profile,
        "nonce": generation["nonce"],
        **targets,
        "order": index,
        "requested_flops": row["requested_flops"],
        "case_deadline_seconds": row["case_deadline_seconds"],
        "acceptance_tail_seconds": 1800,
        "policy": "STOPALL_ANYFAIL",
        "predecessor_barrier": "REPLAY_AUTHENTIC_HOST_NATIVE_V2_FULL_BEFORE_MUTATION",
        "requires_acceptance_prefix": list(CASE_IDS[:index]),
    }
    dataset = read(root / "dataset/dataset-metadata.json")
    dataset.update(
        id=targets["dataset_target"].removesuffix("/1"),
        title=targets["dataset_target"].split("/")[1],
    )
    kernel = read(root / "kernel/kernel-metadata.json")
    kernel.update(
        id=targets["kernel_target"],
        title=targets["kernel_target"].split("/")[1],
        dataset_sources=[
            targets["dataset_target"],
            "zimike/sorting-quality-input-fcff1b5daf574b70/1",
            *(["zimike/ca-kv-1c660aa5f54d88ea9e0c1ce4/1"] if row["task"] == "mqar-k64" else []),
            "zimike/k64-private-20261109a1b2/1",
        ],
    )
    save(root / "dataset/dataset-metadata.json", dataset)
    save(root / "kernel/kernel-metadata.json", kernel)
    path = root / "kernel/kernel.py"
    text = path.read_text(encoding="utf-8")
    assignment = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "BINDING" for t in n.targets)
    )
    bound = ast.literal_eval(assignment.value)
    bound.update(case_id=run["id"], kernel_seconds=30600)
    text = text.replace(ast.get_source_segment(text, assignment), "BINDING = " + repr(bound), 1)
    path.write_text(text, encoding="utf-8", newline="\n")
    result["kernel_metadata_sha256"] = digest(root / "kernel/kernel-metadata.json")
    plan.update(case_order=list(CASE_IDS), deployments=[result])
    return result
