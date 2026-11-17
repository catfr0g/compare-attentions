"""Sorting inference and independent structural, multiset and lexicographic closure."""

from collections import Counter

import numpy as np

from compare_attention.data.sorting import encode_items, oracle, parse_items
from compare_attention.tracking_evaluation import greedy_generate


def _fraction(good, mask):
    total = int(np.count_nonzero(mask))
    correct = int(np.count_nonzero(good & mask))
    return {"count": total, "correct": correct, "accuracy": correct / total if total else None}


def score_predictions(prompts, predictions, config):
    """Close the actual single-call predictions against an independent prompt parser."""
    rows = len(prompts)
    cap = config.n_items * (config.item_len + 1)
    if predictions.dtype != np.int64 or predictions.shape != (rows, cap) or rows == 0:
        raise ValueError("sorting predictions require nonempty int64 [rows, declared cap]")
    truth, original = zip(*(oracle(p, config) for p in prompts), strict=True)
    truth = np.stack(truth)
    syntax, multiset, ascending = (np.zeros(rows, dtype=bool) for _ in range(3))
    eos = predictions == config.eos_id
    has_eos = eos.any(1)
    lengths = np.where(has_eos, eos.argmax(1) + 1, cap)
    for index, row in enumerate(predictions):
        try:
            parsed = parse_items(row[: lengths[index]], config, answer=True)
            if np.any(row[lengths[index] :] != config.pad_id):
                continue
        except ValueError:
            continue
        syntax[index] = True
        multiset[index] = Counter(parsed) == Counter(original[index])
        ascending[index] = parsed == sorted(parsed)
    correct = (predictions == truth).all(1) & has_eos & syntax & multiset & ascending
    mask = np.ones(rows, dtype=bool)
    slices = {}
    for prefix in (1, 2):
        collision = np.array([len({i[:prefix] for i in items}) < len(items) for items in original])
        slices[f"prefix{prefix}_collision"] = _fraction(correct, collision)
        slices[f"prefix{prefix}_no_collision"] = _fraction(correct, ~collision)
    return {
        "task": "sorting",
        "primary_metrics": ["generated_answer_exact_match"],
        "generated_rows": rows,
        "generated_positions": rows * cap,
        "generated_answer_correct": int(correct.sum()),
        "generated_answer_exact_match": float(correct.mean()),
        "generated_correct_positions": int((predictions == truth).sum()),
        "generated_position_accuracy": float((predictions == truth).mean()),
        "syntax_valid": _fraction(syntax, mask),
        "multiset_preserved": _fraction(multiset, mask),
        "ascending_order": _fraction(ascending, mask),
        "illegal_outputs": _fraction(~syntax, mask),
        "slices": slices,
        "generation_policy": "prompt_only_full_vocab_greedy_predicted_eos_declared_cap",
        "generation_cost_scope": "inference_only_excluded_from_training_flops_and_throughput",
        "generated_records": [
            {
                "row": i,
                "prediction": predictions[i].tolist(),
                "generated_length": int(lengths[i]),
                "predicted_eos": bool(has_eos[i]),
                "correct": bool(correct[i]),
                "syntax_valid": bool(syntax[i]),
                "multiset_preserved": bool(multiset[i]),
                "ascending_order": bool(ascending[i]),
            }
            for i in range(rows)
        ],
    }


def generated_metrics(model, inputs, _labels, config, **kwargs):
    """One decoding call: never use labels or teacher-forced tails for generation."""
    prompts = inputs[:, : config.prompt_length]
    cap = config.n_items * (config.item_len + 1)
    predictions = greedy_generate(
        model, prompts, steps=cap, eos_token_id=config.eos_id, pad_token_id=config.pad_id, **kwargs
    )
    return score_predictions(prompts, predictions, config)


def baselines(train_labels, test_inputs, config):
    """TRAIN-only per-position mode; all denominators come from the frozen TEST."""
    constant = np.array(
        [np.bincount(train_labels[:, p], minlength=8).argmax() for p in range(39, 79)],
        dtype=np.int64,
    )
    truth, originals = zip(*(oracle(row[:40], config) for row in test_inputs), strict=True)
    truth = np.stack(truth)
    controls = {"train_position_constant": np.tile(constant, (len(truth), 1))}
    for name, prefix in (("input_copy", 0), ("stable_first1", 1), ("stable_first2", 2)):
        controls[name] = np.stack(
            [
                encode_items(items if not prefix else sorted(items, key=lambda x: x[:prefix]))
                for items in originals
            ]
        )
    return {
        name: {
            "correct": int((prediction == truth).all(1).sum()),
            "count": len(truth),
            "accuracy": float((prediction == truth).all(1).mean()),
        }
        for name, prediction in controls.items()
    }
