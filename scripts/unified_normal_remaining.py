"""Bounded remaining-series metadata integration into the existing NORMAL launcher."""

import ast
import hashlib
import shutil
from pathlib import Path

from normal_remaining_scope import CASE_IDS, case_rows, permission, require_prefix
from unified_normal_contract import canonical, descriptor, digest, read, require
from unified_normal_profiles import REMAINING, authenticate_package, selected

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


def require_frozen_plan(case_id, path):
    """The bounded release cannot be repurposed for a different recipe or nonce."""
    require(case_id in PLAN_PINS and digest(path) == PLAN_PINS[case_id], "REMAINING_FROZEN_PLAN")


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
    generation = intent_value(plan["generation"], REMAINING)
    require_frozen_plan(case_id, Path(candidate) / "payload/unified/normal-plan.json")
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
    }


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
    require(inputs["source_profile"] == REMAINING, "REMAINING_SOURCE_PROFILE")
    generation = intent(inputs["generation"], REMAINING)
    rows = inputs["predecessors"]
    require(type(rows) is list, "REMAINING_PREFIX_SCHEMA")
    for row in rows:
        require(
            type(row) is dict
            and set(row) == {"case_id", "policy", "path", "sha256", "replay"}
            and row["policy"] == "AUTHENTIC_HOST_NATIVE_V2_FULL_REPLAY",
            "REMAINING_ORDINARY_PREFIX_REQUIRED",
        )
    require_prefix(generation["case_id"], [row["case_id"] for row in rows])
    require_frozen_plan(generation["case_id"], inputs["normal_plan"])
    output = Path(output).resolve()
    require(not output.exists(), "UNIFIED_CREATE_ONLY")
    normal = Path(inputs["normal_package"])
    authenticate_package(normal, REMAINING)
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
    shutil.copyfile(
        Path(__file__).with_name("normal_remaining_scope.py"), source / "normal_remaining_scope.py"
    )
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
        require(
            set(spec) == {"profile", "source_request", "result", "data", "snapshot", "index"}
            and spec["profile"] == REMAINING,
            "REMAINING_PREFIX_REPLAY_SCHEMA",
        )
        previous = Path(spec["source_request"]).parents[2]
        for root in (previous, Path(spec["result"]).parent, spec["data"], spec["snapshot"]):
            tree(root)
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
                "profile": REMAINING,
                "package": str(normal),
                "source_identity_sha256": selected(REMAINING)["source"],
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


def deployment(payload, plan, run, generation):
    """Construct a genuinely fresh deployment, never inherit resume or approvals."""
    from unified_normal_generation import namespace
    from unified_normal_operator import save

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
        "source_profile": REMAINING,
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
