"""Exact future NORMAL recipes and independent source/review bindings, never approval."""

import hashlib
import json
import re
from pathlib import Path


def require(condition, code):
    if not condition:
        raise ValueError(code)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def create_json(path, value):
    from quality_recovery import create_json as durable_create

    durable_create(path, value)


def source_identity(package):
    """Authenticate all import roots before a service imports or credentials load."""
    package = Path(package).resolve()
    identity = json.loads((package / "normal-source-identity.json").read_bytes())
    require(identity["schema"] == "normal-source-identity-v1", "NORMAL_SOURCE_SCHEMA")
    for name, pin in identity["files"].items():
        require(
            not name.startswith("/")
            and ".." not in name.split("/")
            and "\\" not in name
            and ":" not in name,
            "NORMAL_SOURCE_PATH",
        )
        path = package / name
        require(not path.is_symlink() and path.is_file(), "NORMAL_SOURCE_FILE")
        require(
            path.stat().st_size == pin["bytes"] and digest(path) == pin["sha256"],
            "NORMAL_SOURCE_BYTES",
        )
    return identity


def derive(package, plan_path, case_id, recipe, data):
    """Reprice one complete public CLI recipe, preserving every numerical option."""
    from compare_attention.cli.train import parse_train_args
    from compare_attention.train.config import TrainConfig
    from compare_attention.train.flops import estimate_step

    package, recipe, data = map(Path, (package, recipe, data))
    source_identity(package)
    plan = json.loads(Path(plan_path).read_bytes())
    require(plan["schema"] == "normal-training-plan-v1", "NORMAL_PLAN_SCHEMA")
    require(plan["purpose"] == "fresh_normal_training", "NORMAL_PLAN_PURPOSE")
    require(
        plan["source_identity_sha256"] == digest(package / "normal-source-identity.json"),
        "NORMAL_NEW_SOURCE_BINDING",
    )
    require(type(plan["engineering_only"]) is bool, "NORMAL_PLAN_SCOPE")
    require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", case_id)), "NORMAL_CASE_ID")
    selected = plan["cases"][case_id]
    require(digest(recipe) == selected["recipe_sha256"], "NORMAL_RECIPE_BYTES")
    require(type(selected["seed"]) is int, "NORMAL_DECLARED_SEED")
    require(selected["family"] in {"classic", "deltanet"}, "NORMAL_FAMILY")
    require(
        selected["task"]
        in {"sorting", "kv_retrieval", "tiny_shakespeare_char", "ruler_variable_tracking"},
        "NORMAL_TASK",
    )
    expected_files = set(selected["data_files"])
    require(
        {"meta.json", "train_inputs.npy", "train_labels.npy", "val_inputs.npy", "val_labels.npy"}
        | (
            ({"input.txt"} if selected["task"] == "tiny_shakespeare_char" else set())
            | (
                {"test_inputs.npy", "test_labels.npy"}
                if selected["task"] in {"tiny_shakespeare_char", "ruler_variable_tracking"}
                else set()
            )
        )
        <= expected_files,
        "NORMAL_FULL_DATA_PINS",
    )
    for name, pin in selected["data_files"].items():
        require(
            Path(name).name == name and "/" not in name and "\\" not in name, "NORMAL_DATA_PATH"
        )
        path = data / name
        require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == pin["bytes"]
            and digest(path) == pin["sha256"],
            "NORMAL_DATA_BYTES",
        )
    args = parse_train_args(["--config", str(recipe), "--data", str(data)])
    require(
        args.dataset_task == selected["task"]
        and args.distributed
        and args.tracker == "clearml"
        and args.resume_from is None
        and args.hf_repo is None,
        "NORMAL_PUBLIC_PARSER",
    )
    config = TrainConfig(data=args.data, model=args.model, schedule=args.schedule, optim=args.optim)
    require(
        config.schedule.seed == selected["seed"] and config.model.attention == selected["family"],
        "NORMAL_RECIPE_TASK_SEED_FAMILY",
    )
    require(
        config.schedule.early_stopping_patience == 0
        and config.schedule.checkpoint_keep_last == 2
        and config.schedule.checkpoint_every_steps == config.schedule.val_every_steps
        and config.schedule.log_every_steps == 1,
        "NORMAL_FULL_TRACE_RECIPE",
    )
    saved = config.model_dump(mode="json")
    if selected["task"] == "sorting":
        require(saved["dataset"].get("task") == "sorting", "NORMAL_SORTING_SCHEMA")
    elif selected["task"] == "tiny_shakespeare_char":
        from compare_attention.data.tiny_shakespeare import CharacterLMConfig, verify_integrity

        dataset = CharacterLMConfig.model_validate(saved["dataset"], strict=True)
        verify_integrity(data, dataset)
    elif selected["task"] == "ruler_variable_tracking":
        from compare_attention.data.tracking import RulerConfig, verify_integrity

        dataset = RulerConfig.model_validate(saved["dataset"], strict=True)
        verify_integrity(data, dataset)
    else:
        from compare_attention.data.config import KVRetrievalConfig

        require(
            set(saved["dataset"]) == set(KVRetrievalConfig.model_fields), "NORMAL_KV_TYPED_SCHEMA"
        )
        KVRetrievalConfig.model_validate(saved["dataset"], strict=True)
    cost = estimate_step(config)
    local = config.model_copy(
        update={
            "schedule": config.schedule.model_copy(
                update={"batch_size": config.schedule.batch_size // 2}
            )
        }
    )
    rank_cost = estimate_step(local)
    require(
        config.schedule.batch_size % 2 == 0 and 2 * rank_cost.total == cost.total,
        "NORMAL_RANK_BUDGET",
    )
    updates = (config.schedule.train_flops + cost.total - 1) // cost.total
    points = list(
        range(config.schedule.val_every_steps, updates, config.schedule.val_every_steps)
    ) + [updates]
    if plan["engineering_only"]:
        require(
            updates <= 4
            and config.dataset.n_train <= 32
            and config.model.vocab_size <= 64
            and config.model.n_embd <= 32
            and not config.schedule.compile
            and config.schedule.dtype == "float32",
            "NORMAL_ENGINEERING_TINY_ONLY",
        )
    return {
        "id": case_id,
        "task": selected["task"],
        "family": selected["family"],
        "seed": selected["seed"],
        "config": saved,
        "updates": updates,
        "checkpoint_steps": points,
        "validation_steps": points,
        "step_flops": cost.total,
        "rank_step_flops": rank_cost.total,
        "accounting": cost.to_dict(),
        "rank_accounting": rank_cost.to_dict(),
        "actual_flops": updates * cost.total,
        "global_batch": config.schedule.batch_size,
        "local_batch": local.schedule.batch_size,
        "world_size": 2,
        "engineering_only": plan["engineering_only"],
        "plan_sha256": digest(plan_path),
        "source_identity_sha256": plan["source_identity_sha256"],
        "runtime_sha256": selected["runtime_sha256"],
    }


def execution_authority(case, review_path, decision_path):
    """Old resume approvals and an engineering fixture can never authorize GPU work."""
    require(not case["engineering_only"], "NORMAL_ENGINEERING_NOT_AUTHORITY")
    review, decision = (
        json.loads(Path(path).read_bytes()) for path in (review_path, decision_path)
    )
    for value in (review, decision):
        require(
            value["source_identity_sha256"] == case["source_identity_sha256"]
            and value["plan_sha256"] == case["plan_sha256"]
            and value["case_id"] == case["id"],
            "NORMAL_FRESH_AUTHORITY_BINDING",
        )
    require(
        review["status"] == "PASS" and review["independent"] is True, "NORMAL_INDEPENDENT_REVIEW"
    )
    require(
        decision["action"] == "NORMAL_FRESH_TRAINING"
        and decision["approved"] is True
        and decision["review_sha256"] == digest(review_path),
        "NORMAL_NEW_EXECUTION_DECISION",
    )
