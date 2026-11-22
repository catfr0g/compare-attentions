"""Native last-iterate validation CE/PPL binding; no model creation or inference."""

import hashlib
import json
import math
from pathlib import Path

from normal_training_plan import canonical, digest, require


def cached_quality(root, case, data, native):
    """Call only after verify_native: its FULL/history/last-weight join is mandatory.

    This serializes the unchanged trainer's masked-target aggregate. It does not
    recompute a loss, choose a best checkpoint, or evaluate the TEST arrays.
    """
    import numpy as np

    from compare_attention.data.tiny_shakespeare import CharacterLMConfig, verify_integrity

    root, data = Path(root), Path(data)
    require(case["task"] == "tiny_shakespeare_char", "NORMAL_LM_TASK")
    dataset = CharacterLMConfig.model_validate(case["config"]["dataset"], strict=True)
    require(
        canonical(json.loads((data / "meta.json").read_bytes()))
        == canonical(dataset.model_dump(mode="json")),
        "NORMAL_LM_METADATA",
    )
    verify_integrity(data, dataset)
    identity = json.loads((root / "out/run_identity.json").read_bytes())
    rows = [json.loads(line) for line in (root / "out/metrics.jsonl").read_text().splitlines()]
    result = rows[-1]
    vals = [row for row in rows if row["type"] == "validation"]
    require(
        identity["status"] == "completed"
        and identity["stop_reason"] == "budget_exhausted"
        and canonical(identity["config"]) == canonical(case["config"])
        and result["type"] == "result"
        and result["provenance"] == identity
        and result["config"] == case["config"]
        and native["updates"] == result["steps"] == case["updates"]
        and result["train_flops"] == vals[-1]["flops"] == case["actual_flops"]
        and native["final_sha256"] == identity["final_parameters_sha256"],
        "NORMAL_LM_LAST_ITERATE_JOIN",
    )
    ce = vals[-1]["loss"]
    require(
        type(ce) in (int, float)
        and math.isfinite(ce)
        and 0 <= ce <= 700
        and ce == result["quality"]["val_loss"],
        "NORMAL_LM_FINITE_CE",
    )
    labels = np.load(data / "val_labels.npy", mmap_mode="r", allow_pickle=False)
    split = dataset.splits["val"]
    require(
        labels.dtype == np.int64
        and labels.shape == (dataset.n_val, dataset.input_seq_len)
        and int((labels != dataset.ignore_index).sum()) == split["supervised_targets"]
        and int((labels == dataset.ignore_index).sum()) == split["masked_padding"],
        "NORMAL_LM_MASK_DENOMINATOR",
    )
    return {
        "schema": "normal-native-character-quality-v1",
        "task": case["task"],
        "split": "val",
        "test_scored": False,
        "selection": "last_full_budget_iterate",
        "step": case["updates"],
        "flops": case["actual_flops"],
        "character_cross_entropy": ce,
        "perplexity": math.exp(ce),
        "supervised_targets": split["supervised_targets"],
        "masked_padding": split["masked_padding"],
        "rows": dataset.n_val,
        "label_alignment": dataset.label_alignment,
        "ignore_index": dataset.ignore_index,
        "meta_sha256": digest(data / "meta.json"),
        "labels_sha256": digest(data / "val_labels.npy"),
        "inputs_sha256": digest(data / "val_inputs.npy"),
        "weights_file_sha256": digest(root / "out/model.pt"),
        "final_parameters_sha256": native["final_sha256"],
        "config_sha256": hashlib.sha256(canonical(case["config"]).encode()).hexdigest(),
        "native_aggregation": "masked_target_cross_entropy",
        "additional_forward_executed": False,
    }


def accept_cached(package, root, case, data):
    """Fresh hard native replay precedes comparison with the saved quality bytes."""
    from normal_training_native import verify_native

    native = verify_native(Path(root) / "out", case)
    report = cached_quality(root, case, data, native)
    require(
        canonical(json.loads((Path(root) / "task-val.json").read_bytes())) == canonical(report),
        "NORMAL_LM_CACHED_REPORT_JOIN",
    )
    return {
        "status": "ACCEPTED",
        "case": case,
        "final_sha256": native["final_sha256"],
        "report_sha256": digest(Path(root) / "task-val.json"),
        "wheel_sha256": digest(Path(package) / "compare_attention-0.1.0-py3-none-any.whl"),
        "next_launch_authorized": False,
    }
