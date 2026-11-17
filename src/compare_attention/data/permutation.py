"""Pinned S5/C5 right-composition task with a declared fixed-length pool adaptation."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from compare_attention.data.tiny_shakespeare import sha256

ALL_PERMUTATIONS = np.asarray(list(itertools.permutations(range(5))), dtype=np.int64)
PERMUTATION_INDEX = {tuple(p): i for i, p in enumerate(ALL_PERMUTATIONS)}
CYCLIC_PERMUTATIONS = [PERMUTATION_INDEX[tuple((np.arange(5) + shift) % 5)] for shift in range(5)]


class PermutationConfig(BaseModel):
    """Full source vocabulary, dense same-position supervision and immutable split identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    name: Literal["permutation"] = "permutation"
    format_version: Literal[1] = 1
    task: Literal["permutation_running_product"] = "permutation_running_product"
    group: Literal["c5", "s5"]
    label_alignment: Literal["same_position_running_right_product"] = (
        "same_position_running_right_product"
    )
    dtype: Literal["int64"] = "int64"
    ignore_index: Literal[-1] = -1
    pad_id: Literal[0] = 0
    perm_low: Literal[1] = 1
    vocab_size: Literal[121] = 121
    input_seq_len: int = Field(default=32, ge=1, strict=True)
    allowed_token_ids: list[int]
    n_train: int = Field(default=100_000, ge=1, strict=True)
    n_val: int = Field(default=1000, ge=1, strict=True)
    n_test: int = Field(default=1000, ge=1, strict=True)
    seed: int = Field(default=1337, ge=0, strict=True)
    split_seeds: dict[str, int]
    rejected_duplicates: dict[str, int]
    source_repository: Literal["On-Point-RND/transformer-architecture-generalization"] = (
        "On-Point-RND/transformer-architecture-generalization"
    )
    source_commit: Literal["444bf8466d26b39ea5df90ae0016c62e53392b1f"] = (
        "444bf8466d26b39ea5df90ae0016c62e53392b1f"
    )
    source_sha256: Literal["27d20f29e4ad2741897b8b17fa6d3b50e1b66e3c8514b81d30c9cf3433edc464"] = (
        "27d20f29e4ad2741897b8b17fa6d3b50e1b66e3c8514b81d30c9cf3433edc464"
    )
    source_default_length_range: tuple[Literal[4], Literal[33]] = (4, 33)
    generation: Literal["independent_split_default_rng_unique_global_prompts"] = (
        "independent_split_default_rng_unique_global_prompts"
    )
    training_stream: Literal["ordered_rows_cyclic_no_shuffle"] = "ordered_rows_cyclic_no_shuffle"
    sha256: dict[str, str]

    @model_validator(mode="after")
    def _contract(self):
        expected = (
            [i + 1 for i in CYCLIC_PERMUTATIONS] if self.group == "c5" else list(range(1, 121))
        )
        if self.allowed_token_ids != expected:
            raise ValueError("inconsistent group token IDs; full vocabulary is never compressed")
        names = {f"{s}_{k}.npy" for s in ("train", "val", "test") for k in ("inputs", "labels")}
        if self.sha256.keys() != names or any(
            field.keys() != {"train", "val", "test"}
            for field in (self.split_seeds, self.rejected_duplicates)
        ):
            raise ValueError("exact array manifest, split seeds and rejection counts required")
        if any(
            v < 0 for field in (self.split_seeds, self.rejected_duplicates) for v in field.values()
        ):
            raise ValueError("negative split seed/rejection count")
        return self


def encode_labels(inputs: np.ndarray) -> np.ndarray:
    """Compose new factors on the RIGHT: (a o b)[j] = a[b[j]]."""
    state = np.tile(np.arange(5), (len(inputs), 1))
    labels = np.empty_like(inputs)
    for position in range(inputs.shape[1]):
        move = ALL_PERMUTATIONS[inputs[:, position] - 1]
        state = np.take_along_axis(state, move, axis=1)
        labels[:, position] = [1 + PERMUTATION_INDEX[tuple(p)] for p in state]
    return labels


def validate_arrays(inputs, labels, config: PermutationConfig) -> None:
    """Validate every factor, dense label, same-position oracle and actual length."""
    if any(
        (
            inputs.dtype != np.int64,
            labels.dtype != np.int64,
            inputs.ndim != 2,
            inputs.shape != labels.shape,
            not inputs.size,
        )
    ):
        raise ValueError("permutation arrays require matching nonempty int64 sequence shapes")
    if inputs.shape[1] != config.input_seq_len:
        raise ValueError("permutation arrays require matching nonempty int64 sequence shapes")
    if not np.isin(inputs, config.allowed_token_ids).all():
        raise ValueError("permutation inputs contain PAD or factors outside selected group")
    if not np.array_equal(labels, encode_labels(inputs)):
        raise ValueError("permutation labels disagree with same-position right-product oracle")


def verify_integrity(directory: Path, config: PermutationConfig) -> None:
    """Hash and validate all prepared arrays before training or pricing."""
    for name, digest in config.sha256.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"permutation dataset integrity mismatch: {name}")
    for split in ("train", "val", "test"):
        x, y = (
            np.load(directory / f"{split}_{kind}.npy", mmap_mode="r", allow_pickle=False)
            for kind in ("inputs", "labels")
        )
        if x.shape != (getattr(config, f"n_{split}"), config.input_seq_len):
            raise ValueError("permutation metadata split shape mismatch")
        validate_arrays(x, y, config)


def prepare_permutation(
    output: Path, *, group: str, context=32, seed=1337, n_train=100_000, n_val=1000, n_test=1000
) -> PermutationConfig:
    """Create a fresh pool using the pinned source's exact one-row RNG call.

    Fixed length is source-parameter-supported, NOT its uniform 4..32 default.
    Splits use independent RNGs and prompts are globally unique within this group.
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
    allowed = CYCLIC_PERMUTATIONS if group == "c5" else list(range(120))
    names = {f"{s}_{k}.npy": "" for s in split_seeds for k in ("inputs", "labels")}
    rejections = dict.fromkeys(split_seeds, 0)
    config = PermutationConfig(
        group=group,
        input_seq_len=context,
        seed=seed,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        allowed_token_ids=[i + 1 for i in allowed],
        split_seeds=split_seeds,
        rejected_duplicates=rejections,
        sha256=names,
    )
    if n_train + n_val + n_test > len(allowed) ** context:
        raise ValueError("not enough distinct permutation prompts")
    output.mkdir(parents=True)
    seen = set()
    for split, split_seed in split_seeds.items():
        rng = np.random.default_rng(split_seed)
        x = np.empty((getattr(config, f"n_{split}"), context), dtype=np.int64)
        for target in x:
            for _ in range(10_000):
                factors = 1 + rng.choice(allowed, size=context, replace=True)
                key = hashlib.sha256(factors.tobytes()).digest()
                if key not in seen:
                    seen.add(key)
                    target[:] = factors
                    break
                rejections[split] += 1
            else:
                raise ValueError("not enough distinct permutation prompts")
        y = encode_labels(x)
        for kind, array in (("inputs", x), ("labels", y)):
            name = f"{split}_{kind}.npy"
            np.save(output / name, array, allow_pickle=False)
            names[name] = sha256(output / name)
    config = config.model_copy(update={"sha256": names, "rejected_duplicates": rejections})
    (output / "meta.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return config
