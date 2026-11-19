"""Strict same-position MQAR records; no model, generation or oracle scorer."""

import math


def require(condition, code):
    if not condition:
        raise ValueError(code)


def close_records(records, labels):
    """Close typed chronological predictions to independently supplied immutable labels."""
    import numpy as np

    require(labels.dtype == np.int64 and labels.ndim == 2, "LABEL_ARRAY")
    require(type(records) is list and len(records) == len(labels), "FULL_ROW_COVERAGE")
    count, correct, exact, losses = 0, 0, 0, []
    fields = {"row", "positions", "targets", "predictions", "nll"}
    for index, (row, y) in enumerate(zip(records, labels, strict=True)):
        require(type(row) is dict and set(row) == fields, "RECORD_FIELDS")
        require(type(row["row"]) is int and row["row"] == index, "ROW_IDENTITY")
        positions = np.flatnonzero(y != -1).tolist()
        require(bool(positions), "ROW_WITHOUT_QUERY")
        for name in ("positions", "targets", "predictions"):
            require(
                type(row[name]) is list and all(type(x) is int for x in row[name]),
                "TYPED_QUERY_RECORD",
            )
        require(row["positions"] == positions, "SAME_POSITION_MASK")
        require(row["targets"] == y[y != -1].tolist(), "EXACT_LABELS")
        require(
            len(row["predictions"]) == len(row["nll"]) == len(positions)
            and all(x >= 0 for x in row["predictions"]),
            "QUERY_COVERAGE",
        )
        require(
            type(row["nll"]) is list
            and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in row["nll"]),
            "FINITE_QUERY_NLL",
        )
        hits = sum(p == t for p, t in zip(row["predictions"], row["targets"], strict=True))
        count += len(positions)
        correct += hits
        exact += hits == len(positions)
        losses.extend(row["nll"])
    require(count > 0 and math.isfinite(math.fsum(losses)), "AGGREGATE_NLL")
    return {
        "rows": len(records),
        "supervised_count": count,
        "correct": correct,
        "accuracy": correct / count,
        "loss": math.fsum(losses) / count,
        "whole_example_correct": exact,
        "whole_example_exact_match": exact / len(records),
    }
