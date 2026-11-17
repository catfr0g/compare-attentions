"""Read-only checkpoint evaluation; labels are aligned at the same position."""

# Explicit validation predicates and metric accumulators keep the scoring contract visible.
# pylint: disable=too-many-locals,too-many-branches,too-many-boolean-expressions,too-many-statements

import hashlib
import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from compare_attention.data.permutation import PermutationConfig
from compare_attention.data.permutation import validate_arrays as validate_permutation_arrays
from compare_attention.data.selective_count import SelectiveCountConfig, validate_arrays
from compare_attention.data.sorting import SortingConfig
from compare_attention.data.sorting import validate_arrays as validate_sorting_arrays
from compare_attention.data.store import load_config, load_split
from compare_attention.data.tiny_shakespeare import CharacterLMConfig
from compare_attention.data.tracking import RulerConfig, StateRecallConfig
from compare_attention.data.tracking import validate_arrays as validate_tracking_arrays
from compare_attention.evaluation_checkpoint import load_evaluation_model
from compare_attention.permutation_metrics import permutation_metrics
from compare_attention.sorting_evaluation import generated_metrics as sorting_generated_metrics
from compare_attention.tracking_evaluation import generated_metrics
from compare_attention.train.identity import runtime_metadata


def evaluate_model(
    model, inputs, labels, *, batch_size=32, device="cpu", dtype="float32", task_config=None
):
    """Aggregate NLL and accuracy by supervised tokens, including the tail batch."""
    if device not in {"cpu", "cuda"} or dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("unsupported device/dtype")
    if device == "cpu" and (dtype != "float32" or model.config.attention == "deltanet"):
        raise ValueError("CPU supports classic float32 evaluation only; DeltaNet requires CUDA")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if device == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA bfloat16 is unsupported")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if (
        inputs.dtype != np.int64
        or labels.dtype != np.int64
        or inputs.ndim != 2
        or inputs.shape != labels.shape
        or not inputs.size
        or inputs.shape[1] > model.config.block_size
    ):
        raise ValueError("inputs/labels must be matching nonempty int64 [rows, sequence] arrays")
    if np.any(inputs < 0) or np.any(inputs >= model.config.vocab_size):
        raise ValueError("inputs outside model vocabulary")
    if np.any(labels < -1) or np.any(labels >= model.config.vocab_size):
        raise ValueError("labels outside model vocabulary or ignore_index=-1")
    if not np.any(labels != -1):
        raise ValueError("no supervised targets")
    count_task = isinstance(task_config, SelectiveCountConfig)
    tracking_task = isinstance(task_config, (RulerConfig, StateRecallConfig))
    sorting_task = isinstance(task_config, SortingConfig)
    if sorting_task:
        validate_sorting_arrays(inputs, labels, task_config)
    if tracking_task:
        validate_tracking_arrays(inputs, labels, task_config)
    if count_task:
        validate_arrays(inputs, labels, task_config)
    permutation_task = isinstance(task_config, PermutationConfig)
    if permutation_task:
        validate_permutation_arrays(inputs, labels, task_config)
    permutation_predictions = []
    count_errors = {"absolute": 0, "raw_absolute": 0, "bias": 0, "outside": 0}
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            x = torch.from_numpy(np.array(inputs[start : start + batch_size], copy=True)).to(device)
            y = torch.from_numpy(np.array(labels[start : start + batch_size], copy=True)).to(device)
            context = (
                torch.autocast("cuda", dtype=getattr(torch, dtype))
                if dtype != "float32"
                else nullcontext()
            )
            with context:
                logits = model(x)
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("nonfinite evaluation logits")
                loss = F.cross_entropy(
                    logits.flatten(0, 1), y.flatten(), ignore_index=-1, reduction="sum"
                )
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite evaluation loss")
            mask = y != -1
            if count_task:
                raw = logits.argmax(-1)[mask] - task_config.class_low
                guess = raw.clamp(0, task_config.max_count)
                true = y[mask] - task_config.class_low
                count_errors["absolute"] += int((guess - true).abs().sum().item())
                count_errors["raw_absolute"] += int((raw - true).abs().sum().item())
                count_errors["bias"] += int((guess - true).sum().item())
                count_errors["outside"] += int(((raw < 0) | (raw > task_config.max_count)).sum())
            if permutation_task:
                permutation_predictions.append(logits.argmax(-1).cpu().numpy())
            total_loss += loss.item()
            correct += int(((logits.argmax(-1) == y) & mask).sum().item())
            count += int(mask.sum().item())
    if not math.isfinite(total_loss):
        raise ValueError("nonfinite aggregate loss")
    metrics = {
        "loss": total_loss / count,
        "accuracy": correct / count,
        "accuracy_percent": 100 * correct / count,
        "correct": correct,
        "supervised_count": count,
        "rows": len(inputs),
    }
    if count_task:
        metrics.update(
            task=task_config.task,
            primary_metrics=["accuracy", "count_nmae"],
            count_mae=count_errors["absolute"] / count,
            count_raw_mae=count_errors["raw_absolute"] / count,
            count_bias=count_errors["bias"] / count,
            count_nmae=count_errors["absolute"] / count / task_config.metric_normalizer,
            count_metric_normalizer=task_config.metric_normalizer,
            count_outside_class_range=count_errors["outside"],
            count_decode_policy="full_vocab_argmax_minus_class_low_clip_0_max_count_for_nmae_only",
        )
    if permutation_task:
        metrics.update(
            permutation_metrics(
                np.concatenate(permutation_predictions),
                labels,
                allowed_token_ids=task_config.allowed_token_ids,
            ),
            task=task_config.task,
            group=task_config.group,
            primary_metrics=["acc", "token_acc"],
            legacy_accuracy_alias="token_acc",
        )
    if tracking_task or sorting_task:
        metrics.update(
            teacher_forced_ce_nats_per_token=metrics["loss"],
            teacher_forced_token_accuracy=metrics["accuracy"],
            teacher_forced_correct=metrics["correct"],
            teacher_forced_supervised_count=metrics["supervised_count"],
            legacy_accuracy_alias="teacher_forced_token_accuracy",
        )
        metrics.update(
            (sorting_generated_metrics if sorting_task else generated_metrics)(
                model,
                inputs,
                labels,
                task_config,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        )
    return metrics


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate_checkpoint(
    checkpoint, data, *, split="test", batch_size=32, device="auto", dtype="auto", config=None
):
    data = Path(data)
    model, loaded = load_evaluation_model(checkpoint, config)
    checkpoint = loaded["path"]
    model_config = model.config
    identity = loaded["provenance"]
    fingerprint = loaded["parameters_sha256"]
    dataset_config = load_config(data)
    arrays = load_split(data, split, mmap=True)
    counts = {
        "train": dataset_config.n_train,
        "val": dataset_config.n_val,
        "test": dataset_config.n_test,
    }
    expected_shape = (counts[split], dataset_config.input_seq_len)
    if arrays.inputs.shape != expected_shape or arrays.labels.shape != expected_shape:
        raise ValueError("selected split shape/count differs from dataset metadata")
    if dataset_config.vocab_size != model_config.vocab_size:
        raise ValueError("dataset vocabulary differs from model vocabulary")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    dtype = ("float16" if device == "cuda" else "float32") if dtype == "auto" else dtype
    model.to(device)
    metrics = evaluate_model(
        model,
        arrays.inputs,
        arrays.labels,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        task_config=dataset_config,
    )
    if isinstance(dataset_config, CharacterLMConfig):
        metrics.update(
            task=dataset_config.task,
            bits_per_character=metrics["loss"] / math.log(2),
            primary_metrics=["loss", "bits_per_character"],
            secondary_metrics=["accuracy"],
        )
    return {
        "schema_version": 1,
        **metrics,
        "split": split,
        "batch_size": batch_size,
        "device": device,
        "inference_dtype": dtype,
        "parameter_dtype": "float32",
        "model_config": model_config.model_dump(),
        "runtime": runtime_metadata(),
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": file_sha256(checkpoint),
            "parameters_sha256": fingerprint,
            "provenance": identity,
            "format": loaded["format"],
            "step": loaded["step"],
        },
        "data": {
            "path": str(data.resolve()),
            "config": json.loads(dataset_config.model_dump_json()),
            "sha256": {
                name: file_sha256(data / name)
                for name in ("meta.json", f"{split}_inputs.npy", f"{split}_labels.npy")
            },
        },
    }
