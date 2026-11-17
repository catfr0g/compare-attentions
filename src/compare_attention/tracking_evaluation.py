"""Free-running answer metrics; never feed teacher-forced answer tokens into generation."""

from contextlib import nullcontext

import numpy as np
import torch

from compare_attention.data.tracking import RulerConfig, oracle


def greedy_generate(
    model,
    prompts,
    *,
    steps,
    batch_size=32,
    device="cpu",
    dtype="float32",
    eos_token_id=None,
    pad_token_id=None,
):
    """Full-vocab fixed-step decoding; length comes from task parameters, not labels."""
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps < 1
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("positive integer generation steps/batch size required")
    if (
        prompts.dtype != np.int64
        or prompts.ndim != 2
        or not prompts.size
        or prompts.shape[1] + steps - 1 > model.config.block_size
    ):
        raise ValueError("nonempty int64 prompts must fit model context")
    if np.any(prompts < 0) or np.any(prompts >= model.config.vocab_size):
        raise ValueError("prompt outside vocabulary")
    if device not in {"cpu", "cuda"} or dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("unsupported device/dtype")
    if device == "cpu" and (dtype != "float32" or model.config.attention == "deltanet"):
        raise ValueError("CPU supports classic float32 only; DeltaNet requires CUDA")
    if eos_token_id is not None:
        for token in (eos_token_id, pad_token_id):
            if type(token) is not int or not 0 <= token < model.config.vocab_size:
                raise ValueError("EOS/pad must be integer vocabulary IDs")
        if eos_token_id == pad_token_id:
            raise ValueError("EOS must differ from pad")
    elif pad_token_id is not None:
        raise ValueError("pad requires EOS stopping")
    result = np.full((len(prompts), steps), pad_token_id or 0, dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            prefix = torch.from_numpy(np.array(prompts[start : start + batch_size], copy=True)).to(
                device
            )
            finished = torch.zeros(len(prefix), dtype=torch.bool, device=device)
            for position in range(steps):
                context = (
                    nullcontext()
                    if dtype == "float32"
                    else torch.autocast("cuda", dtype=getattr(torch, dtype))
                )
                with context:
                    logits = model(prefix)[:, -1, :]
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("nonfinite generated logits")
                prediction = logits.argmax(-1)
                if eos_token_id is not None:
                    prediction = torch.where(finished, pad_token_id, prediction)
                    finished |= prediction == eos_token_id
                result[start : start + len(prefix), position] = prediction.cpu().numpy()
                if eos_token_id is not None and bool(finished.all()):
                    break
                if position + 1 < steps:
                    prefix = torch.cat((prefix, prediction[:, None]), dim=1)
    return result


def _fraction(correct, mask):
    count = int(np.count_nonzero(mask))
    matched = int(np.count_nonzero(correct & mask))
    return {"count": count, "correct": matched, "accuracy": matched / count if count else None}


def generated_metrics(model, inputs, labels, config, **kwargs):
    """Only prompt tokens enter greedy_generate; all labels are scoring-only."""
    # pylint: disable=too-many-locals
    prompts = inputs[:, : config.prompt_length]
    steps = config.num_hops + 1 if isinstance(config, RulerConfig) else 1
    predictions = greedy_generate(model, prompts, steps=steps, **kwargs)
    truth = labels[:, config.prompt_length - 1 : config.prompt_length - 1 + steps]
    correct = predictions == truth
    rows = len(inputs)
    metrics = {
        "task": config.task,
        "generated_rows": rows,
        "generated_positions": int(predictions.size),
        "generated_correct_positions": int(correct.sum()),
        "generated_position_accuracy": float(correct.mean()),
        "generation_policy": "prompt_only_full_vocabulary_greedy_fixed_task_steps_no_eos",
        "generation_cost_scope": "inference_only_excluded_from_training_flops_and_throughput",
    }
    if isinstance(config, RulerConfig):
        links_correct = 0
        for prompt, predicted in zip(prompts, predictions, strict=True):
            _, details = oracle(prompt, config)
            edges = details["edges"]
            links_correct += sum(
                edges.get(int(a)) == int(b)
                for a, b in zip(predicted[:-1], predicted[1:], strict=True)
            )
        valid = (predictions >= 20) & (predictions < 20 + config.n_vars)
        links_total = rows * (steps - 1)
        metrics.update(
            primary_metrics=["generated_chain_exact_match"],
            generated_chain_exact_match=float(correct.all(1).mean()),
            generated_chain_correct=int(correct.all(1).sum()),
            generated_valid_variable_predictions=int(valid.sum()),
            generated_valid_variable_fraction=float(valid.mean()),
            generated_links_correct=links_correct,
            generated_links_total=links_total,
            generated_correct_link_fraction=links_correct / links_total if links_total else None,
        )
    else:
        details = [oracle(prompt, config)[1] for prompt in prompts]
        moved = np.array([row["address_moved"] for row in details], dtype=bool)
        changed = np.array([row["bit_changed"] for row in details], dtype=bool)
        good = correct[:, 0]
        masks = {
            "address_moved": moved,
            "address_unchanged": ~moved,
            "bit_changed": changed,
            "bit_unchanged": ~changed,
            "moved_and_bit_changed": moved & changed,
            "moved_and_bit_unchanged": moved & ~changed,
            "answer_bit_0": truth[:, 0] == 4,
            "answer_bit_1": truth[:, 0] == 5,
        }
        metrics.update(
            primary_metrics=["bit_accuracy"],
            bit_accuracy=float(good.mean()),
            bit_correct=int(good.sum()),
            outside_answer_class=int(np.count_nonzero((predictions != 4) & (predictions != 5))),
            slices={name: _fraction(good, mask) for name, mask in masks.items()},
        )
    return metrics
