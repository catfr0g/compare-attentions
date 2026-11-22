"""Explicit pending generation, genuine public case derivation and namespace binding."""

import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from unified_normal_contract import canonical, digest, read, require
from unified_normal_profiles import (
    FRESH_RESTART,
    REMAINING_FOUR,
    REMAINING_SIX,
    REMAINING_THREE,
    REPAIRED,
    TINY,
    authenticate_review,
    is_remaining_profile,
    selected,
)

CONSENT = "d2ae90adec3a5a22f58aae0537064a8991159bcbdc2a0c82306c581b992da823"
OLD_NONCE = "8043dc515cd194e8f34777d51522a3b8"
CASE = "mqar-k64-deltanet-seed202"
SCHEMA = "UNIFIED_NORMAL_GENERATION_INTENT_V1"


def _save(path, value, *, create_only=True):
    """Durable writes only in the public builder's new candidate output tree."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if create_only else "w", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def intent(path, profile_name):
    """Pending human provenance, never a technical review, retry permit or seal."""
    return intent_value(read(path), profile_name)


def intent_value(value, profile_name):
    """Validate the identical typed subject in builder and downstream plan consumer."""
    require(
        type(value) is dict
        and set(value)
        == (
            {"schema", "nonce", "case_id", "source_identity_sha256", "authorization"}
            | ({"restart"} if profile_name == FRESH_RESTART else set())
        )
        and value["schema"] == SCHEMA,
        "UNIFIED_GENERATION_SCHEMA",
    )
    from normal_remaining_scope import CASE_IDS

    if profile_name == TINY:
        from normal_tiny_scope import CASE_IDS  # independent finite group

    if profile_name == REMAINING_THREE:
        require(value["case_id"] in CASE_IDS[5:], "REMAINING_THREE_CASE_SCOPE")
    if profile_name == REMAINING_FOUR:
        require(value["case_id"] in CASE_IDS[4:], "REMAINING_FOUR_CASE_SCOPE")
    if profile_name == REMAINING_SIX:
        require(value["case_id"] in CASE_IDS[2:], "REMAINING_SIX_CASE_SCOPE")
    require(
        (
            (profile_name == REPAIRED and value["case_id"] == CASE)
            or (is_remaining_profile(profile_name) and value["case_id"] in CASE_IDS)
            or (profile_name == TINY and value["case_id"] in CASE_IDS)
        )
        and value["source_identity_sha256"] == selected(profile_name)["source"],
        "UNIFIED_GENERATION_SCOPE",
    )
    require(
        type(value["nonce"]) is str
        and re.fullmatch(r"[0-9a-f]{32}", value["nonce"])
        and value["nonce"] != OLD_NONCE
        and value["nonce"][:16] not in {OLD_NONCE[:16], "9ff571a0588c9631", "4a2060304e6b4722"}
        and value["nonce"] != "0" * 32,
        "UNIFIED_GENERATION_FRESH_NONCE",
    )
    authorization = value["authorization"]
    if profile_name == TINY:
        from normal_tiny_scope import permission

        permission(authorization)
        return value
    if is_remaining_profile(profile_name):
        from normal_remaining_scope import permission

        permission(authorization)
        if profile_name == FRESH_RESTART:
            from normal_fresh_restart_scope import permission as restart_permission

            restart_permission(value)
        return value
    require(
        type(authorization) is dict
        and set(authorization) == {"path", "sha256"}
        and authorization["sha256"] == CONSENT
        and digest(authorization["path"]) == CONSENT,
        "UNIFIED_GENERATION_HUMAN_PROVENANCE",
    )
    consent = read(authorization["path"])
    require(
        consent["case_id"] == value["case_id"]
        and consent["reviewed_normal_source_identity"] == value["source_identity_sha256"]
        and consent["historical_kernel"]["numeric_id"] == 146133822
        and consent["historical_kernel"]["version"] == 1
        and consent["historical_kernel"]["status"] == "ERROR"
        and consent["historical_kernel"]["preserve_original"] is True
        and consent["not_a_technical_review_or_live_proof_or_seal"] is True,
        "UNIFIED_GENERATION_STOP_HISTORY",
    )
    return value


def prepare_case(inputs, output, runtime_sha256):
    """Re-derive through authenticated NORMAL derive, not by relabelling old case."""
    name = inputs["source_profile"]
    generation = intent(inputs["generation"], name)
    if name != TINY and not is_remaining_profile(name):
        authenticate_review(inputs["source_review"], name)
    # The new release is frozen UNBOUND; its source+plan review is required by
    # the existing typed NORMAL authority before preparation or credentials.
    case_id = generation["case_id"]
    predecessor = read(inputs["normal_plan"])
    require(set(predecessor["cases"]) == {case_id}, "UNIFIED_GENERATION_ONE_CASE")
    normal_plan = {
        "schema": "normal-training-plan-v1",
        "purpose": "fresh_normal_training",
        "engineering_only": False,
        "source_identity_sha256": selected(name)["source"],
        "generation": generation,
        "cases": {case_id: {**predecessor["cases"][case_id], "runtime_sha256": runtime_sha256}},
    }
    require(
        normal_plan["cases"][case_id]["recipe_sha256"] == digest(inputs["normal_recipe"]),
        "UNIFIED_GENERATION_RECIPE",
    )
    plan_path = output / "payload/unified/normal-plan.json"
    _save(plan_path, normal_plan)
    package = Path(inputs["normal_package"])
    data = Path(read(inputs["normal_case"])["case"]["config"]["data"])
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "UNIFIED_DERIVE_CPU_ONLY")
    with tempfile.TemporaryDirectory(prefix="normal-plan-wheel-", dir=os.environ["TMPDIR"]) as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(package / "compare_attention-0.1.0-py3-none-any.whl") as saved:
            names = saved.namelist()
            require(len(names) == len(set(names)), "UNIFIED_DERIVE_UNIQUE_WHEEL")
            for member in saved.infolist():
                part = PurePosixPath(member.filename)
                require(
                    not part.is_absolute()
                    and ".." not in part.parts
                    and "\\" not in member.filename
                    and ":" not in member.filename
                    and (member.external_attr >> 16) & 0o170000 != 0o120000,
                    "UNIFIED_DERIVE_SAFE_WHEEL",
                )
            saved.extractall(root)
        code = (
            "import sys,json;from pathlib import Path;"
            "sys.path[:0]=[sys.argv[1],str(Path(sys.argv[2])/'src')];"
            "import normal_training_plan as producer;"
            "assert Path(producer.__file__).resolve()=="
            "(Path(sys.argv[2])/'src/normal_training_plan.py').resolve();"
            "case=producer.derive(*sys.argv[2:7]);"
            "print(json.dumps(case,sort_keys=True,separators=(',',':')))"
        )
        argv = [
            "uv",
            "run",
            "--offline",
            "--no-project",
            "--no-sync",
            "--python",
            sys.executable,
            "python",
            "-B",
            "-c",
            code,
            str(root),
            str(package),
            str(plan_path),
            case_id,
            str(inputs["normal_recipe"]),
            str(data),
        ]
        child = subprocess.run(argv, capture_output=True, text=True, timeout=180, check=False)
        _save(
            output / "derive-case-receipt.json",
            {
                "argv": argv,
                "returncode": child.returncode,
                "stdout": child.stdout,
                "stderr": child.stderr,
                "selected_source": str(package / "src"),
                "model_or_training_executed": False,
            },
        )
        require(child.returncode == 0, "UNIFIED_GENUINE_CASE_DERIVATION")
        import json

        case = json.loads(child.stdout)
    old = read(inputs["normal_case"])["case"]
    bindings = {"source_identity_sha256", "plan_sha256", "runtime_sha256"}
    require(
        {k: v for k, v in case.items() if k not in bindings}
        == {k: v for k, v in old.items() if k not in bindings},
        "UNIFIED_GENERATION_SCIENTIFIC_EQUALITY",
    )
    _save(output / "derived-case.json", {"case": case})
    return case


def namespace(nonce):
    """One deterministic finite target naming contract, independent of case seed."""
    return {
        "dataset_target": "zimike/sorting-mqar-seed202-" + nonce[:16] + "/1",
        "kernel_target": "zimike/seed202-pilot-" + nonce[:16],
    }


def apply_namespace(payload, deployment, generation):
    """Retarget only a new create-only candidate; historical objects are inputs only."""
    require(
        deployment["case_id"] == CASE
        and deployment["nonce"] == OLD_NONCE
        and deployment["dataset_target"] == namespace(OLD_NONCE)["dataset_target"]
        and deployment["kernel_target"] == namespace(OLD_NONCE)["kernel_target"],
        "UNIFIED_GENERATION_HISTORICAL_PARENT",
    )
    nonce = generation["nonce"]
    targets = namespace(nonce)
    root = Path(payload) / "deploy" / CASE
    dataset_path = root / "dataset/dataset-metadata.json"
    kernel_path = root / "kernel/kernel-metadata.json"
    dataset, kernel = read(dataset_path), read(kernel_path)
    require(
        dataset["id"] + "/1" == deployment["dataset_target"]
        and kernel["id"] == deployment["kernel_target"]
        and kernel["dataset_sources"].count(deployment["dataset_target"]) == 1,
        "UNIFIED_GENERATION_PARENT_METADATA",
    )
    kernel["dataset_sources"] = [
        targets["dataset_target"] if p == deployment["dataset_target"] else p
        for p in kernel["dataset_sources"]
    ]
    dataset.update(
        id=targets["dataset_target"].removesuffix("/1"),
        title=targets["dataset_target"].split("/")[1],
    )
    kernel.update(id=targets["kernel_target"], title=targets["kernel_target"].split("/")[1])
    _save(dataset_path, dataset, create_only=False)
    _save(kernel_path, kernel, create_only=False)
    deployment.update(**targets, nonce=nonce, kernel_metadata_sha256=digest(kernel_path))
