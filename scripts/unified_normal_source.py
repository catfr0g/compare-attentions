"""Explicit repaired-NORMAL source profile; old evidence retains its own wheel."""

from pathlib import Path

from unified_normal_contract import canonical, descriptor, digest, read, require
from unified_normal_profiles import PROFILES, REPAIRED, is_remaining_profile
from unified_normal_profiles import selected as source_profile

SOURCE = "89b7e3ac5b42716e24d0d2a1994d13c32845e0c93a668f284e3cb341fe301cb8"
WHEEL = "a1e1db146ac0fef3583edbc6bee1d41ed1e83dffe85702c09667d34c96047e90"
PROFILE = "REPAIRED_NORMAL_V1"
RUNTIME_RECORD = "0806af0c9dac27b5e25be168ccddeb884f069a5e8bc11d7eb47855a48043781f"


def service_names(deployment):
    """One namespace declaration shared by saved main, proof and absence consumer."""
    import re

    require(
        deployment["source_profile"] in PROFILES
        and type(deployment["nonce"]) is str
        and re.fullmatch(r"[0-9a-f]{32}", deployment["nonce"]),
        "UNIFIED_NORMAL_SERVICE_DECLARATION",
    )
    prefix = "normal-training/" + deployment["nonce"]
    return {
        "normal_service_prefix": prefix,
        "normal_task_pattern": "^" + re.escape(prefix.replace("/", "-")) + "(?:-task-val)?$",
    }


def operator_profile(candidate, deployment):
    """Join the real saved NORMAL plan/recipe to the immutable requested run."""
    result = profile(candidate, deployment)
    _request_path, request = descriptor(candidate)
    root = Path(candidate) / "payload"
    expected_source = source_profile(request["normal"]["profile"])["source"]
    plan_path = root / "unified/normal-plan.json"
    plan = read(plan_path)
    case = request["run"]
    require(
        deployment["case_id"] == case["id"]
        and plan["schema"] == "normal-training-plan-v1"
        and plan["purpose"] == "fresh_normal_training"
        and plan["engineering_only"] is False
        and plan["source_identity_sha256"] == expected_source
        and set(plan["cases"]) == {case["id"]}
        and digest(plan_path) == deployment["normal_plan_sha256"] == case["plan_sha256"],
        "UNIFIED_NORMAL_PLAN_SCOPE",
    )
    recipe = read(root / "configs" / (case["id"] + ".yaml"))
    require(
        all(case["config"][name] == recipe[name] for name in ("model", "schedule", "optim"))
        and case["source_identity_sha256"] == expected_source
        and case["engineering_only"] is False
        and case["world_size"] == 2
        and case["global_batch"] == recipe["schedule"]["batch_size"]
        and 2 * case["local_batch"] == case["global_batch"]
        and case["updates"]
        == (recipe["schedule"]["train_flops"] + case["step_flops"] - 1) // case["step_flops"]
        and case["actual_flops"] == case["updates"] * case["step_flops"],
        "UNIFIED_NORMAL_COMPLETE_RECIPE",
    )
    points = list(
        range(
            recipe["schedule"]["val_every_steps"],
            case["updates"],
            recipe["schedule"]["val_every_steps"],
        )
    ) + [case["updates"]]
    require(
        case["checkpoint_steps"] == case["validation_steps"] == points
        and 2 * case["rank_step_flops"] == case["step_flops"]
        and case["task"] == recipe["dataset_task"]
        and case["family"] == recipe["model"]["attention"]
        and case["seed"] == recipe["schedule"]["seed"],
        "UNIFIED_NORMAL_CADENCE_AND_IDENTITY",
    )
    selected = plan["cases"][case["id"]]
    data = Path(case["config"]["data"]).resolve()
    declared_files = {
        Path(name).name: pin
        for name, pin in request["files"].items()
        if Path(name).parent.resolve() == data and Path(name).name != "dataset-metadata.json"
    }
    require(
        str(data) in request["roots"]
        and selected["data_files"] == declared_files
        and set(declared_files)
        == {
            "meta.json",
            *(["data-manifest.json"] if case["task"] == "kv_retrieval" else []),
            "train_inputs.npy",
            "train_labels.npy",
            "val_inputs.npy",
            "val_labels.npy",
            "test_inputs.npy",
            "test_labels.npy",
        },
        "UNIFIED_NORMAL_REQUESTED_DATA_BINDING",
    )
    require(
        selected["runtime_sha256"] == result["normal_runtime_sha256"]
        and selected["task"] == case["task"]
        and selected["family"] == case["family"]
        and selected["seed"] == case["seed"]
        and selected["recipe_sha256"] == digest(root / "configs" / (case["id"] + ".yaml")),
        "UNIFIED_NORMAL_RECIPE_SCOPE",
    )
    from unified_normal_authority import binding as authority_binding

    if request["normal"]["profile"] == REPAIRED or is_remaining_profile(
        request["normal"]["profile"]
    ):
        from unified_normal_generation import intent_value

        generation = intent_value(plan["generation"], request["normal"]["profile"])
        require(
            deployment["nonce"] == generation["nonce"],
            "UNIFIED_GENERATION_NAMESPACE_BINDING",
        )

    return {
        **result,
        "normal_plan_sha256": digest(plan_path),
        **authority_binding(candidate),
        **service_names(deployment),
    }


def profile(candidate, deployment):
    """Exact explicit profile, including selected-child setup and genuine runtime."""
    _request_path, request = descriptor(candidate)
    value = request["normal"]
    require(value.get("profile") in PROFILES, "UNIFIED_EXPLICIT_NORMAL_PROFILE")
    profile_name = value["profile"]
    expected_source = source_profile(profile_name)["source"]
    require(
        set(value)
        == {
            "profile",
            "package",
            "source_identity_sha256",
            "wheel_sha256",
            "runtime_record",
            "runtime_record_sha256",
            "runtime_path",
            "runtime_sha256",
        }
        and deployment["source_profile"] == profile_name
        and value["source_identity_sha256"] == expected_source
        and value["wheel_sha256"] == WHEEL,
        "UNIFIED_EXPLICIT_NORMAL_PROFILE",
    )
    root = Path(value["package"])
    require(
        digest(root / "normal-source-identity.json") == expected_source, "UNIFIED_NORMAL_IDENTITY"
    )
    identity = read(root / "normal-source-identity.json")
    require(
        str(root.resolve()) in request["roots"]
        and {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        == set(identity["files"]) | {"normal-source-identity.json"}
        and all(not p.is_symlink() and not p.is_junction() for p in root.rglob("*")),
        "UNIFIED_NORMAL_ROOT_CLOSURE",
    )
    require(identity["wheel_sha256"] == WHEEL, "UNIFIED_NORMAL_WHEEL_SCOPE")
    for name, pin in identity["files"].items():
        require(
            not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and "\\" not in name
            and ":" not in name,
            "UNIFIED_NORMAL_SAFE_SOURCE",
        )
        path = root / name
        require(
            path.is_file()
            and not path.is_symlink()
            and not path.is_junction()
            and path.stat().st_size == pin["bytes"]
            and digest(path) == pin["sha256"],
            "UNIFIED_NORMAL_SOURCE_BYTES",
        )
    require(
        value["runtime_record_sha256"] == RUNTIME_RECORD
        and digest(value["runtime_record"]) == RUNTIME_RECORD,
        "UNIFIED_GENUINE_T4_REFERENCE",
    )
    saved = read(value["runtime_record"])
    for key in value["runtime_path"]:
        saved = saved[key]
    import hashlib

    require(
        hashlib.sha256(canonical(saved).encode()).hexdigest() == value["runtime_sha256"],
        "UNIFIED_RUNTIME_REFERENCE_VALUE",
    )
    require(
        saved["python"] == "3.13.16"
        and saved["torch"] == "2.11.0+cu130"
        and saved["cuda"] == "13.0"
        and saved["cudnn"] == 91900,
        "UNIFIED_RUNTIME_TARGET",
    )
    return {
        "normal_source_identity_sha256": expected_source,
        "normal_wheel_sha256": WHEEL,
        "normal_runtime_sha256": value["runtime_sha256"],
        "normal_profile": profile_name,
    }
