"""Observe actual frozen public MQAR scorer; never substitute labels for predictions."""

import json
import math
from pathlib import Path

from mqar_final_tail_metrics import close_records, require


def evaluate_public(argv, *, data, split, records):
    """Public parser/loader/forward unchanged; scoped hook retains logits' query outputs."""
    import numpy as np
    import torch
    from torch.nn import functional as functional

    from compare_attention import evaluation
    from compare_attention.cli.evaluate import main
    from compare_attention.data.store import load_split

    records = Path(records)
    if records.exists():
        raise FileExistsError("CREATE_ONLY_QUERY_RECORDS")
    require(split == "val", "RECOVERY_VAL_ONLY_TEST_UNUSED")
    require("--split" in argv and argv[argv.index("--split") + 1] == split, "PUBLIC_SPLIT")
    output = Path(argv[argv.index("--output") + 1])
    if output.exists():
        raise FileExistsError("CREATE_ONLY_PUBLIC_REPORT")
    arrays = load_split(data, split, mmap=True)
    original = evaluation.load_evaluation_model
    observed, handles, loads = [], [], []

    def capture(_model, args, logits):
        start = len(observed)
        end = start + len(args[0])
        expected = torch.from_numpy(np.array(arrays.inputs[start:end], copy=True))
        require(torch.equal(args[0].detach().cpu(), expected), "FORWARD_INPUT_ORDER")
        labels = torch.from_numpy(np.array(arrays.labels[start:end], copy=True)).to(logits.device)
        require(logits.ndim == 3 and logits.shape[:2] == labels.shape, "LOGIT_SHAPE")
        require(bool(torch.isfinite(logits).all()), "FINITE_LOGITS")
        mask = labels != -1
        nll = functional.cross_entropy(logits.float()[mask], labels[mask], reduction="none")
        predictions = logits.argmax(-1)
        offset = 0
        for i in range(len(labels)):
            positions = torch.where(mask[i])[0]
            count = len(positions)
            observed.append(
                {
                    "row": start + i,
                    "positions": positions.cpu().tolist(),
                    "targets": labels[i][positions].cpu().tolist(),
                    "predictions": predictions[i][positions].cpu().tolist(),
                    "nll": nll[offset : offset + count].cpu().tolist(),
                }
            )
            offset += count

    def load(*args, **kwargs):
        require(not loads, "EXACTLY_ONE_PUBLIC_MODEL")
        value = original(*args, **kwargs)
        loads.append(value[1])
        handles.append(value[0].register_forward_hook(capture))
        return value

    evaluation.load_evaluation_model = load
    try:
        require(main(argv) == 0, "PUBLIC_EVALUATOR_RETURN")
    finally:
        evaluation.load_evaluation_model = original
        for handle in handles:
            handle.remove()
    report = json.loads(output.read_bytes())
    closed = close_records(observed, arrays.labels)
    require(len(loads) == 1 and loads[0]["provenance"] is None, "NULL_PUBLIC_PROVENANCE")
    require(not any(k.startswith("generated") for k in report), "MQAR_NONGENERATIVE")
    require(
        all(report[k] == closed[k] for k in ("rows", "correct", "supervised_count", "accuracy")),
        "PUBLIC_METRICS_CLOSURE",
    )
    # FP32 per-query NLL versus frozen public summed autocast loss: explicit numerical domain.
    require(
        math.isclose(report["loss"], closed["loss"], rel_tol=2e-5, abs_tol=2e-6),
        "PUBLIC_NLL_CLOSURE",
    )
    with records.open("x", encoding="utf-8") as stream:
        json.dump(observed, stream, allow_nan=False)
        stream.flush()
        __import__("os").fsync(stream.fileno())
    require(json.loads(records.read_bytes()) == observed, "QUERY_RECORD_READBACK")
    return report, observed
