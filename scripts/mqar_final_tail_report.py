"""Task-specific production VAL closure, separate from CPU engineering scores."""

import json
import math
from pathlib import Path

from mqar_final_tail_metrics import close_records
from mqar_final_tail_native import FINAL, MODEL_SHA, canonical, digest, require

ZERO = {
    "optimizer_constructions",
    "optimizer_updates",
    "backward_calls",
    "checkpoint_rng_restores",
    "checkpoint_loads",
    "training_calls",
    "training_flops",
}


def no_optimization(proof):
    require(all(type(proof.get(k)) is int and proof[k] == 0 for k in ZERO), "NO_OPTIMIZATION")
    require(proof["scorer_invocations"] == proof["models_loaded"] == 1, "ONE_PUBLIC_SCORER_MODEL")
    require(
        proof["loaded_parameters_before"] == proof["loaded_parameters_after"] == [FINAL],
        "ORIGINAL_PARAMETERS_UNCHANGED",
    )
    require(
        proof["source_model_sha256"] == MODEL_SHA and proof["standalone_provenance"] is None,
        "STANDALONE_WEIGHT_SOURCE",
    )


def verify_runtime(value):
    require(
        value["platform"] == "Linux"
        and value["python"] == "3.13.16"
        and value["torch"] == "2.11.0+cu130"
        and value["cuda"] == "13.0"
        and value["cudnn"] == 91900
        and value["triton"] == "3.6.0"
        and value["gpu"]["name"] == "Tesla T4"
        and value["gpu"]["index"] == 0
        and value["gpu"]["compute_capability"] == [7, 5],
        "ACTUAL_TARGET_RUNTIME",
    )


def verify_report(report, data, records):
    from mqar_final_tail_data import MQAR_DATA

    from compare_attention.data.store import load_config, load_split

    require(
        set(report)
        == {
            "schema_version",
            "loss",
            "accuracy",
            "accuracy_percent",
            "correct",
            "supervised_count",
            "rows",
            "split",
            "batch_size",
            "device",
            "inference_dtype",
            "parameter_dtype",
            "model_config",
            "runtime",
            "checkpoint",
            "data",
        },
        "EXACT_MQAR_PUBLIC_REPORT_SCHEMA",
    )
    require(
        report["schema_version"] == 1
        and report["split"] == "val"
        and report["rows"] == 1000
        and report["batch_size"] == 32
        and report["device"] == "cuda"
        and report["inference_dtype"] == "float16"
        and report["parameter_dtype"] == "float32",
        "FULL_VAL_TARGET_POLICY",
    )
    verify_runtime(report["runtime"])
    require(
        report["checkpoint"]["sha256"] == MODEL_SHA
        and report["checkpoint"]["parameters_sha256"] == FINAL
        and report["checkpoint"]["provenance"] is None
        and report["checkpoint"]["format"] == "state_dict"
        and report["checkpoint"]["step"] is None,
        "PUBLIC_LOADER_TRUTHFUL_NULL",
    )
    data = Path(data)
    names = ("meta.json", "val_inputs.npy", "val_labels.npy")
    for name in names:
        require(
            digest(data / name) == MQAR_DATA["files"][name]["sha256"]
            and (data / name).stat().st_size == MQAR_DATA["files"][name]["bytes"],
            "VAL_BYTES",
        )
    cfg = load_config(data)
    require(
        canonical(cfg.model_dump(mode="json"))
        == canonical(MQAR_DATA["spec"])
        == canonical(report["data"]["config"]),
        "DATA_CONFIG_METADATA_DOMAIN",
    )
    require(report["data"]["sha256"] == {n: digest(data / n) for n in names}, "PUBLIC_DATA_BINDING")
    arrays = load_split(data, "val", mmap=True)
    require(arrays.labels.shape == (1000, 512), "VAL_BANK_SHAPE")
    require(
        all(0 <= p < cfg.vocab_size for row in records for p in row["predictions"]),
        "PREDICTIONS_IN_ORIGINAL_VOCABULARY",
    )
    closed = close_records(records, arrays.labels)
    require(
        closed["rows"] == 1000
        and closed["supervised_count"] == 64000
        and all(len(r["positions"]) == 64 for r in records),
        "FULL64000_QUERY_TARGETS",
    )
    for key in ("rows", "correct", "supervised_count", "accuracy"):
        require(
            type(report[key]) is type(closed[key]) and report[key] == closed[key], "EXACT_METRICS"
        )
    require(
        type(report["loss"]) in (float, int)
        and math.isfinite(report["loss"])
        and math.isclose(report["loss"], closed["loss"], rel_tol=2e-5, abs_tol=2e-6)
        and report["accuracy_percent"] == 100 * closed["accuracy"],
        "QUERY_CE_CLOSURE",
    )
    expected_model = {
        "vocab_size": 8192,
        "block_size": 512,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "dropout": 0.0,
        "bias": True,
        "attention": "classic",
    }
    require(
        canonical(report["model_config"]) == canonical(expected_model), "EXACT_OLD_MODEL_CONFIG"
    )
    return closed


def load_report(folder, data):
    folder = Path(folder)
    report = json.loads((folder / "mqar-val.json").read_bytes())
    records = json.loads((folder / "mqar-val-records.json").read_bytes())
    return report, verify_report(report, data, records)


def quality_metrics(report):
    return {
        "val_query_ce": report["loss"],
        "val_query_accuracy": report["accuracy"],
        "val_query_correct": report["correct"],
        "val_query_targets": report["supervised_count"],
        "val_rows": report["rows"],
    }
