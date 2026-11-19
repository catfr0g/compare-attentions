"""Exactly one authentic public inference CLI; no manufactured native identity."""

import json
import os
import shutil
from pathlib import Path

from mqar_final_tail_native import digest, require


def create_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def execute_once(source, model_config, data, output, start_task, *, evaluator=None):
    """Execute original public inference once with the pinned policy."""
    from inference_guard import audit

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(source)
    before = digest(source)
    create_json(
        output / "evaluation-intent.json",
        {
            "source_model_sha256": before,
            "split": "val",
            "rows": 1000,
            "batch_size": 32,
            "optimizer_updates_permitted": 0,
            "state": "EXACTLY_ONCE_NEVER_REEVALUATE",
        },
    )
    task_id = start_task()
    require(type(task_id) is str and bool(task_id), "INFERENCE_TASK_BEFORE_MODEL")
    import torch

    torch.manual_seed(202)
    require(torch.initial_seed() == 202, "INFERENCE_EFFECTIVE_SEED")
    create_json(output / "inference-task.json", {"task_id": task_id})
    standalone = output.parent / (output.name + "-standalone")
    standalone.mkdir(exist_ok=False)
    shutil.copyfile(source, standalone / "model.pt")
    create_json(standalone / "model.yaml", {"model": model_config})
    require(digest(standalone / "model.pt") == before, "STANDALONE_BYTES")
    command = [
        "--checkpoint",
        str(standalone / "model.pt"),
        "--config",
        str(standalone / "model.yaml"),
        "--data",
        str(data),
        "--split",
        "val",
        "--batch-size",
        "32",
        "--device",
        "cuda",
        "--dtype",
        "float16",
        "--output",
        str(output / "mqar-val.json"),
    ]
    from mqar_final_tail_evaluate import evaluate_public

    require(evaluator is None, "PUBLIC_SCORER_CANNOT_BE_REPLACED")
    with audit() as proof:
        evaluate_public(command, data=data, split="val", records=output / "mqar-val-records.json")
    require(proof["models_loaded"] == 1, "ONE_MODEL_LOAD")
    require(digest(source) == digest(standalone / "model.pt") == before, "WEIGHTS_MUTATED")
    create_json(
        output / "no-optimization.json",
        {
            **proof,
            "training_flops": 0,
            "scorer_invocations": 1,
            "scorer": "compare_attention.cli.evaluate.main",
            "standalone_provenance": None,
            "source_model_sha256": before,
        },
    )
    return proof
