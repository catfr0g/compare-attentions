"""Tiny-only adapter to the existing frozen NORMAL operator and fresh guards."""

import ast
import hashlib
import shutil
from pathlib import Path

from normal_tiny_scope import CASE_IDS, case_rows, require_prefix
from unified_normal_contract import descriptor, digest, read, require
from unified_normal_profiles import TINY, authenticate_package, selected


def validate_request(value):
    """Independent prefix; no qualified historical exceptions in this new series."""
    require_prefix(value["run"]["id"], [row["case_id"] for row in value["predecessors"]])
    require(
        all(
            row["policy"] == "AUTHENTIC_HOST_NATIVE_V2_FULL_REPLAY"
            and row["replay"]["profile"] == TINY
            for row in value["predecessors"]
        ),
        "TINY_ORDINARY_PREFIX_REQUIRED",
    )


def transition(candidate, case_id):
    from unified_normal_generation import intent_value

    _path, request = descriptor(candidate)
    plan = read(Path(candidate) / "payload/unified/normal-plan.json")
    generation = intent_value(plan["generation"], TINY)
    validate_request(request)
    require(case_id == request["run"]["id"] == generation["case_id"], "TINY_TRANSITION_CASE")
    return {
        "schema": "DIRECT_AUTHORIZED_TINY_SERIES_V1",
        "authorization_sha256": generation["authorization"]["sha256"],
        "case_order": list(CASE_IDS),
        "case_index": CASE_IDS.index(case_id),
        "accepted_prefix": [row["case_id"] for row in request["predecessors"]],
        "historical_acceptances_modified": False,
        "transition_permission": "EXPLICIT_USER_PERMISSION_EXACT_SIX_SEQUENTIAL",
    }


def integrate_group(source):
    """Change only finite selectors in the NEW emitted host, not old namespaces."""
    from unified_normal_operator import replace
    from unified_normal_remaining import integrate_group as integrate_remaining

    integrate_remaining(source)
    replace(
        source / "contract.py",
        "from normal_remaining_scope import case_specs",
        "from normal_tiny_scope import case_specs",
    )
    replace(
        source / "seal.py",
        "from unified_normal_remaining import transition",
        "from unified_normal_tiny import transition",
    )


def build_request(inputs, output):
    """Use the public NORMAL derive and genuine runtime, without old case evidence."""
    from build_unified_normal_contract import pin, save
    from unified_normal_generation import intent, prepare_case
    from unified_normal_source import RUNTIME_RECORD

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
        "TINY_BUILD_INPUTS",
    )
    require(inputs["source_profile"] == TINY, "TINY_PROFILE")
    generation = intent(inputs["generation"], TINY)
    rows = inputs["predecessors"]
    require(type(rows) is list, "TINY_PREFIX_SCHEMA")
    for row in rows:
        require(
            type(row) is dict and set(row) == {"case_id", "policy", "path", "sha256", "replay"},
            "TINY_PREFIX_SCHEMA",
        )
    validate_request({"run": {"id": generation["case_id"]}, "predecessors": rows})
    output = Path(output).resolve()
    require(not output.exists(), "UNIFIED_CREATE_ONLY")
    normal = Path(inputs["normal_package"])
    authenticate_package(normal, TINY)
    runtime_path = Path(inputs["runtime_record"])
    require(digest(runtime_path) == RUNTIME_RECORD, "UNIFIED_AUTHENTIC_RUNTIME_REFERENCE")
    fields = ["completed", 0, "acceptance", "timing", "metadata", "execution_runtime"]
    runtime = read(runtime_path)
    for field in fields:
        runtime = runtime[field]
    from unified_normal_contract import canonical

    runtime_sha = hashlib.sha256(canonical(runtime).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    case = prepare_case(inputs, output, runtime_sha)
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

    source = output / "source"
    source.mkdir()
    for original in Path(__file__).parent.glob("unified_normal_*.py"):
        shutil.copyfile(original, source / original.name)
    for name in (
        "normal_remaining_scope",
        "normal_fresh_restart_scope",
        "normal_tiny_scope",
        "normal_training_lm",
        "completed_normal_publication",
        "completed_sorting404_publication",
        "training_diagnostics",
    ):
        shutil.copyfile(Path(__file__).with_name(name + ".py"), source / (name + ".py"))
    for key in ("normal_case", "runtime_record", "generation", "normal_plan", "normal_recipe"):
        include(inputs[key])
    include(generation["authorization"]["path"])
    tree(normal)
    tree(source)
    tree(case["config"]["data"])
    for row in rows:
        spec = row["replay"]
        require(
            set(spec) == {"profile", "source_request", "result", "data", "snapshot", "index"}
            and spec["profile"] == TINY,
            "TINY_PREFIX_REPLAY_SCHEMA",
        )
        previous = Path(spec["source_request"]).parents[2]
        for root in (previous, Path(spec["result"]).parent, spec["data"], spec["snapshot"]):
            tree(root)
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
        require(digest(row["path"]) == row["sha256"], "TINY_PREFIX_PIN")
    save(
        output / "payload/unified/request.json",
        {
            "schema": "UNIFIED_NORMAL_REQUEST_V1",
            "run": case,
            "predecessors": rows,
            "normal": {
                "profile": TINY,
                "package": str(normal),
                "source_identity_sha256": selected(TINY)["source"],
                "wheel_sha256": selected(TINY)["wheel"],
                "runtime_record": str(runtime_path),
                "runtime_record_sha256": RUNTIME_RECORD,
                "runtime_path": fields,
                "runtime_sha256": runtime_sha,
            },
            # Historical science here authenticates bootstrap/runtime only, NOT Tiny data.
            "science": read(
                Path(inputs["sorting_candidate"]) / "payload/qualified-launch/bridge.json"
            )["science"],
            "roots": sorted(roots),
            "files": files,
        },
    )
    return output


def deployment(payload, plan, run, generation):
    """Reuse the proven fresh two-stage target pattern with new Tiny-owned nonces."""
    from unified_normal_generation import intent_value, namespace
    from unified_normal_operator import save

    intent_value(generation, TINY)
    index = CASE_IDS.index(run["id"])
    row = case_rows()[index]
    require(
        (run["task"], run["seed"], run["family"]) == (row["task"], row["seed"], row["family"]),
        "TINY_DEPLOYMENT_CASE",
    )
    template = next(
        item
        for item in plan["deployments"]
        if item["task"] == "sorting" and item.get("mode") == "FRESH_TWO_STAGE"
    )
    root = Path(payload) / "deploy" / run["id"]
    shutil.copytree(Path(payload) / "deploy" / template["case_id"], root)
    targets = namespace(generation["nonce"])
    result = {
        "case_id": run["id"],
        "task": row["task"],
        "family": row["family"],
        "seed": row["seed"],
        "mode": "FRESH_TWO_STAGE",
        "source_profile": TINY,
        "nonce": generation["nonce"],
        **targets,
        "order": index,
        "requested_flops": row["requested_flops"],
        "case_deadline_seconds": 28800,
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
            "zimike/k64-private-20261109a1b2/1",
        ],
    )
    save(root / "dataset/dataset-metadata.json", dataset)
    save(root / "kernel/kernel-metadata.json", kernel)
    path = root / "kernel/kernel.py"
    text = path.read_text(encoding="utf-8")
    node = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "BINDING" for t in n.targets)
    )
    bound = ast.literal_eval(node.value)
    bound.update(case_id=run["id"], kernel_seconds=30600)
    path.write_text(
        text.replace(ast.get_source_segment(text, node), "BINDING = " + repr(bound), 1),
        encoding="utf-8",
        newline="\n",
    )
    result["kernel_metadata_sha256"] = digest(root / "kernel/kernel-metadata.json")
    plan.update(
        case_order=list(CASE_IDS),
        deployments=[result],
        scope="TINY_INDEPENDENT_SIX_CASES_V1",
        seeds_authorized=[202, 303, 404],
        seeds_forbidden_pending_user_decision=[],
        requested_total_flops=30_000_000_000_000_000,
    )
    plan["host_acceptance"].update(
        fresh_clearml_training_and_inference=False,
        clearml_policy="AUXILIARY_NONBLOCKING_V1",
        task_all_validation_rows=run["config"]["dataset"]["n_val"],
        task_metric="last_full_budget_validation_character_cross_entropy",
        extra_inference_permitted=False,
        test_scored=False,
    )
    return result
