"""Literal Query Count labels, with a versioned fixed-pool protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from compare_attention.data.tiny_shakespeare import sha256

SOURCE_COMMIT = "444bf8466d26b39ea5df90ae0016c62e53392b1f"
SOURCE = "On-Point-RND/transformer-architecture-generalization"
NAME = "selective_count"


class SelectiveCountConfig(BaseModel):
    """Uncompressed source token IDs; exactly one last-position count label."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    name: Literal["selective_count"] = NAME
    format_version: Literal[1] = 1
    task: Literal["selective_count_classification"] = "selective_count_classification"
    label_alignment: Literal["single_last_input_position"] = "single_last_input_position"
    dtype: Literal["int64"] = "int64"
    ignore_index: Literal[-1] = -1
    pad_id: Literal[0] = 0
    symbol_low: Literal[1] = 1
    n_symbols: int = Field(default=32, ge=2, strict=True)
    input_seq_len: int = Field(default=512, ge=1, strict=True)
    max_count: int = Field(ge=1, strict=True)
    class_low: int = Field(ge=3, strict=True)
    vocab_size: int = Field(ge=4, strict=True)
    input_symbol_range: list[int]
    output_class_range: list[int]
    metric_normalizer: float = Field(gt=0, allow_inf_nan=False)
    n_train: int = Field(default=100_000, ge=1, strict=True)
    n_val: int = Field(default=1000, ge=1, strict=True)
    n_test: int = Field(default=1000, ge=1, strict=True)
    seed: int = Field(default=1337, ge=0, strict=True)
    split_seeds: dict[str, int]
    source_repository: Literal["On-Point-RND/transformer-architecture-generalization"] = SOURCE
    source_commit: Literal["444bf8466d26b39ea5df90ae0016c62e53392b1f"] = SOURCE_COMMIT
    source_defaults: dict[str, int] = Field(
        default_factory=lambda: {"n_symbols": 32, "count_ratio": 10, "seq_len": 320}
    )
    training_stream: Literal["ordered_rows_cyclic_no_shuffle"] = "ordered_rows_cyclic_no_shuffle"
    generation: Literal["independent_split_default_rng_unique_global_prompts"] = (
        "independent_split_default_rng_unique_global_prompts"
    )
    sha256: dict[str, str]

    @model_validator(mode="after")
    def _contract(self):
        if any(
            (
                self.class_low != self.symbol_low + self.n_symbols,
                self.max_count < self.input_seq_len,
                self.vocab_size != self.class_low + self.max_count + 1,
                self.metric_normalizer != self.input_seq_len / self.n_symbols + 1,
                self.input_symbol_range != [1, self.n_symbols],
                self.output_class_range != [self.class_low, self.class_low + self.max_count],
            )
        ):
            raise ValueError("inconsistent source vocabulary/count metric contract")
        names = {f"{s}_{k}.npy" for s in ("train", "val", "test") for k in ("inputs", "labels")}
        if self.sha256.keys() != names or self.split_seeds.keys() != {"train", "val", "test"}:
            raise ValueError("exact array manifest and split seeds required")
        return self


def encode_labels(inputs: np.ndarray, *, n_symbols: int = 32) -> np.ndarray:
    """Count the LAST query symbol in the ENTIRE input, including itself."""
    labels = np.full(inputs.shape, -1, dtype=np.int64)
    labels[:, -1] = 1 + n_symbols + np.count_nonzero(inputs == inputs[:, -1:], axis=1)
    return labels


def validate_arrays(inputs, labels, config: SelectiveCountConfig) -> None:
    """Reject wrong masks, leaked answer tokens, invalid symbols and corrupt oracles."""
    if any(
        (
            inputs.dtype != np.int64,
            labels.dtype != np.int64,
            inputs.ndim != 2,
            inputs.shape != labels.shape,
            inputs.size == 0,
        )
    ):
        raise ValueError("count arrays require matching nonempty int64 shapes")
    if inputs.shape[1] != config.input_seq_len:
        raise ValueError("count input sequence length mismatch")
    if np.any(inputs < config.symbol_low) or np.any(inputs >= config.class_low):
        raise ValueError("count inputs must contain only source symbols, never PAD/classes")
    if np.any(labels[:, :-1] != -1) or np.any(labels[:, -1] == -1):
        raise ValueError("count requires exactly one last-position label")
    if not np.array_equal(labels, encode_labels(inputs, n_symbols=config.n_symbols)):
        raise ValueError("count labels disagree with oracle")


def verify_integrity(directory: Path, config: SelectiveCountConfig) -> None:
    """Hash all arrays and validate all records before pricing or execution."""
    for name, digest in config.sha256.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"count dataset integrity mismatch: {name}")
    for split in ("train", "val", "test"):
        x, y = (
            np.load(directory / f"{split}_{kind}.npy", mmap_mode="r", allow_pickle=False)
            for kind in ("inputs", "labels")
        )
        if x.shape != (getattr(config, f"n_{split}"), config.input_seq_len):
            raise ValueError("count metadata split shape mismatch")
        validate_arrays(x, y, config)


def prepare_selective_count(
    output: Path,
    *,
    context=512,
    n_symbols=32,
    seed=1337,
    n_train=100_000,
    n_val=1000,
    n_test=1000,
) -> SelectiveCountConfig:
    """Create only a fresh directory, with independent deterministic split RNGs.

    Each row uses the source's one-row integers call (not a shifted LM).
    Unlike its online/default stream or with-replacement pool sampling, rows
    here are globally unique, split-seeded, and consumed cyclically in order.
    """
    # pylint: disable=too-many-locals
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    split_seeds = dict(
        zip(
            ("train", "val", "test"),
            map(int, np.random.SeedSequence(seed).generate_state(3)),
            strict=True,
        )
    )
    names = {f"{s}_{k}.npy": "" for s in split_seeds for k in ("inputs", "labels")}
    config = SelectiveCountConfig(
        input_seq_len=context,
        n_symbols=n_symbols,
        max_count=context,
        class_low=1 + n_symbols,
        vocab_size=n_symbols + context + 2,
        input_symbol_range=[1, n_symbols],
        output_class_range=[1 + n_symbols, 1 + n_symbols + context],
        metric_normalizer=context / n_symbols + 1,
        seed=seed,
        split_seeds=split_seeds,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        sha256=names,
    )
    output.mkdir(parents=True)
    seen = set()
    for split, split_seed in split_seeds.items():
        rng = np.random.default_rng(split_seed)
        x = np.empty((getattr(config, f"n_{split}"), context), dtype=np.int64)
        for target in x:
            for _ in range(10_000):
                symbols = 1 + rng.integers(0, n_symbols, size=context, dtype=np.int64)
                key = hashlib.sha256(symbols.tobytes()).digest()
                if key not in seen:
                    seen.add(key)
                    target[:] = symbols
                    break
            else:
                raise ValueError("not enough distinct count prompts")
        y = encode_labels(x, n_symbols=n_symbols)
        for kind, array in (("inputs", x), ("labels", y)):
            name = f"{split}_{kind}.npy"
            np.save(output / name, array, allow_pickle=False)
            names[name] = sha256(output / name)
    config = config.model_copy(update={"sha256": names})
    (output / "meta.json").write_text(
        json.dumps(config.model_dump(), indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return config
