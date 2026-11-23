"""RULER cloud-native observation and pure cached scoring; no training replacement."""

import hashlib
import json
from pathlib import Path

from normal_training_plan import canonical, digest, require

TRACKING_SCOPE = "NATIVE_CLOUD_GENERATION_NO_SEPARATE_INFERENCE_SERVICE"


def read_json(path):
    """Do not normalize duplicate/nonfinite cached evidence into an accepted object."""

    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "NORMAL_RULER_JSON_DUPLICATE")
            result[key] = value
        return result

    def nonfinite(_value):
        raise ValueError("NORMAL_RULER_JSON_NONFINITE")

    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=nonfinite)


def service_scope():
    """Service absence is a protocol choice, never verified inference telemetry."""
    return {
        "scope": TRACKING_SCOPE,
        "remote_mutations": 0,
        "diagnostic": {"status": "not_required", "authority": "AUXILIARY_ONLY"},
    }


def cloud_task(package, root, case, data):
    """CLOUD only: load final native FP32 weights, generate VAL once, save evidence."""
    import os

    import numpy as np
    import torch
    from normal_training_native import verify_native
    from normal_training_plan import create_json
    from normal_training_worker import imported_sources

    from compare_attention.data.tracking import RulerConfig
    from compare_attention.evaluation_checkpoint import load_evaluation_model

    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" and torch.cuda.is_available(),
        "NORMAL_RULER_CLOUD_ONLY",
    )
    root, data = Path(root), Path(data)
    proof = verify_native(root / "out", case)
    config = RulerConfig.model_validate(case["config"]["dataset"], strict=True)
    model, loaded = load_evaluation_model(root / "out/model.pt")
    identity = read_json(root / "out/run_identity.json")
    require(
        loaded["format"] == "state_dict"
        and loaded["step"] is None
        and loaded["parameters_sha256"] == proof["final_sha256"]
        and loaded["provenance"] == identity
        and identity["status"] == "completed"
        and identity["config"] == case["config"],
        "NORMAL_RULER_LOADER_JOIN",
    )
    x, y = (
        np.load(data / f"val_{kind}.npy", mmap_mode="r", allow_pickle=False)
        for kind in ("inputs", "labels")
    )
    from compare_attention.data.tracking import validate_arrays

    require(
        canonical(read_json(data / "meta.json")) == canonical(config.model_dump(mode="json")),
        "NORMAL_RULER_METADATA",
    )
    for kind in ("inputs", "labels"):
        name = f"val_{kind}.npy"
        require(digest(data / name) == config.sha256[name], "NORMAL_RULER_VAL_BYTES")
    validate_arrays(x, y, config)
    model.to("cuda")
    metrics, records = observe_generation(
        model, x, y, config, device="cuda", dtype="float16", batch_size=32
    )
    report = cached_report(root, case, data, proof, records)
    require(canonical(metrics) == canonical(report["metrics"]), "NORMAL_RULER_NATIVE_CACHE_JOIN")
    create_json(root / "task-val.json", report)
    create_json(root / "task-source.json", imported_sources(package))
    create_json(root / "task-clearml-readback.json", service_scope())
    return report


def observe_generation(
    model, inputs, labels, config, *, batch_size=32, device="cpu", dtype="float32"
):
    """Observe exactly one unchanged native generation, never replace its outputs."""
    import numpy as np
    import torch

    from compare_attention.tracking_evaluation import generated_metrics

    predictions = np.empty((len(inputs), config.answer_length), dtype=np.int64)
    calls = 0

    def observer(_model, args, output):
        nonlocal calls
        batch, position = divmod(calls, config.answer_length)
        start = batch * batch_size
        size = min(batch_size, len(inputs) - start)
        (prefix,) = args
        require(
            size > 0 and tuple(prefix.shape) == (size, config.prompt_length + position),
            "NORMAL_RULER_OBSERVER_ORDER",
        )
        expected = np.concatenate(
            (
                inputs[start : start + size, : config.prompt_length],
                predictions[start : start + size, :position],
            ),
            axis=1,
        )
        require(
            np.array_equal(prefix.detach().cpu().numpy(), expected),
            "NORMAL_RULER_OBSERVER_FEEDBACK",
        )
        require(
            tuple(output.shape) == (*prefix.shape, config.vocab_size), "NORMAL_RULER_OBSERVER_VOCAB"
        )
        # Leave nonfinite rejection to the unchanged native evaluator.
        last = output[:, -1, :]
        if bool(torch.isfinite(last).all()):
            predictions[start : start + size, position] = last.argmax(-1).detach().cpu().numpy()
        calls += 1
        return None

    handle = model.register_forward_hook(observer)
    try:
        metrics = generated_metrics(
            model, inputs, labels, config, batch_size=batch_size, device=device, dtype=dtype
        )
    finally:
        handle.remove()
    require(
        calls == config.answer_length * ((len(inputs) + batch_size - 1) // batch_size),
        "NORMAL_RULER_OBSERVER_CALLS",
    )
    records = [
        {"row": row, "prediction": prediction.tolist()}
        for row, prediction in enumerate(predictions)
    ]
    require(
        canonical(metrics) == canonical(score_predictions(inputs, labels, config, records)),
        "NORMAL_RULER_NATIVE_CACHE_JOIN",
    )
    return metrics, records


def score_predictions(inputs, labels, config, records):
    """Native formulas over already saved tokens, including prompt-edge link scoring."""
    import numpy as np

    from compare_attention.data.tracking import oracle

    require(type(records) is list and len(records) == len(inputs) > 0, "NORMAL_RULER_RECORD_ROWS")
    for index, row in enumerate(records):
        require(
            type(row) is dict
            and set(row) == {"row", "prediction"}
            and type(row["row"]) is int
            and row["row"] == index
            and type(row["prediction"]) is list
            and len(row["prediction"]) == config.answer_length
            and all(
                type(token) is int and 0 <= token < config.vocab_size for token in row["prediction"]
            ),
            "NORMAL_RULER_RECORD_ORDER",
        )
    predictions = np.array([row["prediction"] for row in records], dtype=np.int64)
    truth = labels[:, config.prompt_length - 1 : config.prompt_length - 1 + config.answer_length]
    correct = predictions == truth
    valid = (predictions >= 20) & (predictions < 20 + config.n_vars)
    links = 0
    for prompt, predicted in zip(inputs[:, : config.prompt_length], predictions, strict=True):
        _, details = oracle(prompt, config)
        edges = details["edges"]
        links += sum(
            edges.get(int(a)) == int(b) for a, b in zip(predicted[:-1], predicted[1:], strict=True)
        )
    total = len(inputs) * (config.answer_length - 1)
    return {
        "task": config.task,
        "generated_rows": len(inputs),
        "generated_positions": int(predictions.size),
        "generated_correct_positions": int(correct.sum()),
        "generated_position_accuracy": float(correct.mean()),
        "generation_policy": "prompt_only_full_vocabulary_greedy_fixed_task_steps_no_eos",
        "generation_cost_scope": "inference_only_excluded_from_training_flops_and_throughput",
        "primary_metrics": ["generated_chain_exact_match"],
        "generated_chain_exact_match": float(correct.all(1).mean()),
        "generated_chain_correct": int(correct.all(1).sum()),
        "generated_valid_variable_predictions": int(valid.sum()),
        "generated_valid_variable_fraction": float(valid.mean()),
        "generated_links_correct": links,
        "generated_links_total": total,
        "generated_correct_link_fraction": links / total if total else None,
    }


def cached_report(root, case, data, native, records):
    """Reconstruct the entire report without loading weights or executing a model."""
    import numpy as np

    from compare_attention.data.tracking import RulerConfig, validate_arrays

    root, data = Path(root), Path(data)
    require(case["task"] == "ruler_variable_tracking", "NORMAL_RULER_TASK")
    config = RulerConfig.model_validate(case["config"]["dataset"], strict=True)
    require(
        canonical(read_json(data / "meta.json")) == canonical(config.model_dump(mode="json")),
        "NORMAL_RULER_METADATA",
    )
    for kind in ("inputs", "labels"):
        name = f"val_{kind}.npy"
        require(digest(data / name) == config.sha256[name], "NORMAL_RULER_VAL_BYTES")
    x, y = (
        np.load(data / f"val_{kind}.npy", mmap_mode="r", allow_pickle=False)
        for kind in ("inputs", "labels")
    )
    require(x.shape == (config.n_val, config.input_seq_len), "NORMAL_RULER_VAL_SHAPE")
    validate_arrays(x, y, config)
    identity = read_json(root / "out/run_identity.json")
    require(
        identity["status"] == "completed"
        and identity["stop_reason"] == "budget_exhausted"
        and canonical(identity["config"]) == canonical(case["config"])
        and native["updates"] == case["updates"]
        and native["final_sha256"] == identity["final_parameters_sha256"],
        "NORMAL_RULER_LAST_ITERATE_JOIN",
    )
    return {
        "schema": "normal-native-ruler-quality-v1",
        "task": case["task"],
        "split": "val",
        "test_scored": False,
        "evidence_scope": "TARGET_GPU_RESULT",
        "selection": "last_full_budget_iterate",
        "step": case["updates"],
        "flops": case["actual_flops"],
        "label_alignment": config.label_alignment,
        "ignore_index": config.ignore_index,
        "checkpoint": {
            "weights_file_sha256": digest(root / "out/model.pt"),
            "parameters_sha256": native["final_sha256"],
            "provenance": identity,
            "step": case["updates"],
        },
        "data": {
            "meta_sha256": digest(data / "meta.json"),
            "inputs_sha256": digest(data / "val_inputs.npy"),
            "labels_sha256": digest(data / "val_labels.npy"),
            "config": config.model_dump(mode="json"),
            "rows": config.n_val,
            "prompt_length": config.prompt_length,
            "answer_length": config.answer_length,
            "input_seq_len": config.input_seq_len,
        },
        "config_sha256": hashlib.sha256(canonical(case["config"]).encode()).hexdigest(),
        "metrics": score_predictions(x, y, config, records),
        "generated_records": records,
        "source": {
            "source_identity_sha256": case["source_identity_sha256"],
            "plan_sha256": case["plan_sha256"],
        },
    }


def accept_cached(package, root, case, data):
    """Hard native proof, whole report equality and actual scientific import bytes."""
    import zipfile

    from normal_training_native import verify_native

    root, package = Path(root), Path(package)
    native = verify_native(root / "out", case)
    report = read_json(root / "task-val.json")
    expected = cached_report(root, case, data, native, report["generated_records"])
    require(canonical(report) == canonical(expected), "NORMAL_RULER_CACHED_REPORT_JOIN")
    source = read_json(root / "task-source.json")
    wheel = package / "compare_attention-0.1.0-py3-none-any.whl"
    require(
        set(source) == {"wheel_sha256", "imported_modules", "no_training_method_replacement"}
        and source["no_training_method_replacement"] is True
        and source["wheel_sha256"] == digest(wheel)
        and {
            "compare_attention.evaluation_checkpoint",
            "compare_attention.tracking_evaluation",
            "compare_attention.models." + case["family"],
        }
        <= set(source["imported_modules"]),
        "NORMAL_RULER_SOURCE",
    )
    with zipfile.ZipFile(wheel) as saved:
        for name, row in source["imported_modules"].items():
            base = name.replace(".", "/")
            require(
                name.startswith("compare_attention.") or name == "compare_attention",
                "NORMAL_RULER_SOURCE_MODULE",
            )
            require(
                set(row) == {"member", "sha256"}
                and row["member"] in {base + ".py", base + "/__init__.py"}
                and row["sha256"] == hashlib.sha256(saved.read(row["member"])).hexdigest(),
                "NORMAL_RULER_SOURCE_BYTES",
            )
    return {
        "status": "ACCEPTED",
        "case": case,
        "final_sha256": native["final_sha256"],
        "report_sha256": digest(root / "task-val.json"),
        "wheel_sha256": digest(wheel),
        "next_launch_authorized": False,
    }
