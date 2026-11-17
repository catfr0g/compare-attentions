"""Unrestricted permutation prediction scoring, independent of batch partitioning."""

import numpy as np


def permutation_metrics(predicted, targets, *, allowed_token_ids):
    """Source primary EM and micro token accuracy; secondary metrics never replace them.

    Masks may vary by row. Empty rows match source EM (vacuous correctness);
    rows with no remaining labels are excluded from exclude-first EM.
    """
    # Each explicitly named count is useful evidence, not redundant model state.
    # pylint: disable=too-many-locals
    predicted, targets = np.asarray(predicted), np.asarray(targets)
    if predicted.ndim != 2 or predicted.shape != targets.shape or not targets.size:
        raise ValueError("predictions and targets require matching nonempty 2D shapes")
    mask = targets != -1
    match = predicted == targets
    count = int(mask.sum())
    correct = int((match & mask).sum())
    sequence_correct = int((match | ~mask).all(axis=1).sum())
    first = mask & (np.cumsum(mask, axis=1) == 1)
    after = mask & ~first
    after_rows = after.any(axis=1)
    after_correct = int((match & after).sum())
    after_count = int(after.sum())
    last = mask & (np.cumsum(mask[:, ::-1], axis=1)[:, ::-1] == 1)
    last_count = int(last.sum())
    position_count = mask.sum(axis=0)
    position_correct = (mask & match).sum(axis=0)
    return {
        "acc": sequence_correct / len(targets),
        "token_acc": correct / count if count else 0.0,
        "sequence_correct": sequence_correct,
        "sequence_count": len(targets),
        "exclude_first_token_acc": after_correct / after_count if after_count else 0.0,
        "exclude_first_acc": (
            float((match | ~after).all(axis=1)[after_rows].mean()) if after_rows.any() else 0.0
        ),
        "exclude_first_sequence_count": int(after_rows.sum()),
        "exclude_first_supervised_count": after_count,
        "last_accuracy": int((last & match).sum()) / last_count if last_count else 0.0,
        "last_supervised_count": last_count,
        "outside_group_predictions": int((mask & ~np.isin(predicted, allowed_token_ids)).sum()),
        "pad_predictions": int((mask & (predicted == 0)).sum()),
        "position_supervised_count": position_count.tolist(),
        "position_correct": position_correct.tolist(),
        "position_accuracy": [
            int(c) / int(n) if n else None
            for c, n in zip(position_correct, position_count, strict=True)
        ],
        "decode_policy": "unrestricted_full_vocab_argmax_no_group_mask",
    }
