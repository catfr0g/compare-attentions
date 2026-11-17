"""Literal pinned StringSortingTask adapted to split-seeded unique cyclic pools."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from compare_attention.data.sorting_source.sorting import StringSortingTask
from compare_attention.data.tiny_shakespeare import sha256

SOURCE_COMMIT = "30922c4100992a245a23103a790ad737ea8ede4c"
SOURCE_HASHES = {
    "base.py": "c9088836a4af2407f30b307d4f6d4c149b04bda7e4fabc39e159b02b5bc67e1a",
    "sorting.py": "91316c07c34e9efde2a53f0771cf38e67df7095db4da2b05c2ecf69e3b2e63e7",
    "registry_source.py": "146141c0d7811fb50a776c1fa366a7effd6f067c6166011732118d03ff887e02",
}
SPLITS = ("train", "val", "test")


def split_seeds(seed):
    return dict(zip(SPLITS, map(int, np.random.SeedSequence(seed).generate_state(3)), strict=True))


class SortingConfig(BaseModel):
    """Separate typed contract; existing TrackingConfig and its pins stay unchanged."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    name: Literal["sorting"] = "sorting"
    task: Literal["sorting"] = "sorting"
    format_version: Literal[1] = 1
    dtype: Literal["int64"] = "int64"
    ignore_index: Literal[-1] = -1
    pad_id: Literal[0] = 0
    eos_id: Literal[3] = 3
    n_items: Literal[8] = 8
    item_len: Literal[4] = 4
    alphabet: Literal["0123"] = "0123"
    duplicates: bool = True
    descending: bool = False
    prompt_length: Literal[40] = 40
    answer_length: Literal[40] = 40
    input_seq_len: Literal[128] = 128
    vocab_size: Literal[8] = 8
    label_alignment: Literal["answer_only_shifted_next_token"] = "answer_only_shifted_next_token"
    padding_policy: Literal["right_pad_to_128_for_delta_chunk64"] = (
        "right_pad_to_128_for_delta_chunk64"
    )
    n_train: int = Field(default=100_000, ge=1)
    n_val: int = Field(default=1000, ge=1)
    n_test: int = Field(default=1000, ge=1)
    seed: int = Field(default=1337, ge=0)
    split_seeds: dict[str, int]
    source_repository: Literal["On-Point-RND/transformer-architecture-generalization"] = (
        "On-Point-RND/transformer-architecture-generalization"
    )
    source_commit: Literal["30922c4100992a245a23103a790ad737ea8ede4c"] = SOURCE_COMMIT
    source_sha256: dict[str, str] = Field(default_factory=lambda: dict(SOURCE_HASHES))
    training_stream: Literal["ordered_rows_cyclic_no_shuffle"] = "ordered_rows_cyclic_no_shuffle"
    generation: Literal["independent_split_default_rng_unique_global_prompts"] = (
        "independent_split_default_rng_unique_global_prompts"
    )
    sha256: dict[str, str]

    @model_validator(mode="before")
    @classmethod
    def _literal_types(cls, raw):
        if isinstance(raw, dict):
            for key in (
                "format_version",
                "ignore_index",
                "pad_id",
                "eos_id",
                "n_items",
                "item_len",
                "prompt_length",
                "answer_length",
                "input_seq_len",
                "vocab_size",
            ):
                if key in raw and type(raw[key]) is not int:
                    raise ValueError("literal integer fields require actual integers")
        return raw

    @model_validator(mode="after")
    def _pins(self):
        names = {f"{s}_{k}.npy" for s in SPLITS for k in ("inputs", "labels")}
        if self.sha256.keys() != names or self.split_seeds != split_seeds(self.seed):
            raise ValueError("exact array manifest and derived split seeds required")
        if self.source_sha256 != SOURCE_HASHES or not self.duplicates or self.descending:
            raise ValueError("sorting source pins/parameters mismatch")
        return self


def verify_source():
    for name, expected in SOURCE_HASHES.items():
        if sha256(Path(__file__).with_name("sorting_source") / name) != expected:
            raise ValueError(f"sorting vendor source hash mismatch: {name}")


def source_task(config, seed):
    return StringSortingTask(
        n_items=config.n_items,
        item_len=config.item_len,
        alphabet=config.alphabet,
        duplicates=config.duplicates,
        descending=config.descending,
        seed=seed,
    )


def parse_items(sequence, config, *, answer=False):
    """Independent strict parser; no calls into the vendor encoder or sorter."""
    tokens = list(map(int, sequence))
    terminal = config.eos_id if answer else 1
    if len(tokens) != config.answer_length or tokens[-1] != terminal:
        raise ValueError("sorting structure/terminal mismatch")
    groups = []
    current = []
    for token in tokens[:-1]:
        if token == 2:
            if len(current) != config.item_len:
                raise ValueError("sorting item length mismatch")
            groups.append(tuple(current))
            current = []
        elif 4 <= token < config.vocab_size:
            current.append(token)
        else:
            raise ValueError("illegal sorting character")
    groups.append(tuple(current))
    if len(groups) != config.n_items or any(len(g) != config.item_len for g in groups):
        raise ValueError("sorting item count/length mismatch")
    return groups


def encode_items(items, terminal=3):
    return np.asarray(
        [t for i, item in enumerate(items) for t in ((*item, 2) if i < len(items) - 1 else item)]
        + [terminal],
        dtype=np.int64,
    )


def oracle(prompt, config):
    items = parse_items(prompt, config)
    ordered = sorted(items)
    assert Counter(items) == Counter(ordered)
    return encode_items(ordered), items


def validate_arrays(inputs, labels, config):
    if (
        inputs.dtype != np.int64
        or labels.dtype != np.int64
        or inputs.ndim != 2
        or inputs.shape != labels.shape
        or not inputs.size
        or inputs.shape[1] != config.input_seq_len
    ):
        raise ValueError("sorting arrays require matching nonempty int64 shapes")
    for x, y in zip(inputs, labels, strict=True):
        answer, _ = oracle(x[:40], config)
        if not np.array_equal(y[39:79], answer) or np.any(y[:39] != -1) or np.any(y[79:] != -1):
            raise ValueError("sorting answer-only labels disagree with oracle")
        if not np.array_equal(x[40:79], answer[:-1]) or np.any(x[79:] != 0):
            raise ValueError("sorting shifted input/right-padding mismatch")


def _unique_item(generator, seen):
    # pylint: disable=protected-access
    for _ in range(10_000):
        item = generator._sample_one()
        key = hashlib.sha256(item.prompt.tobytes()).digest()
        if key not in seen:
            seen.add(key)
            return item
    raise ValueError("not enough unique sorting prompts")


def verify_integrity(directory, config, *, source_parity=True):
    """Every row: independent oracle, disjointness and literal generator/collator replay."""
    directory = Path(directory)
    verify_source()
    for name, expected in config.sha256.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"sorting dataset integrity mismatch: {name}")
    seen, source_seen = set(), set()
    for split, seed in config.split_seeds.items():
        x, y = (
            np.load(directory / f"{split}_{k}.npy", mmap_mode="r", allow_pickle=False)
            for k in ("inputs", "labels")
        )
        if x.shape != (getattr(config, f"n_{split}"), 128):
            raise ValueError("sorting metadata shape mismatch")
        validate_arrays(x, y, config)
        generator = source_task(config, seed)
        for row, target in zip(x, y, strict=True):
            key = hashlib.sha256(row[:40].tobytes()).digest()
            if key in seen:
                raise ValueError("sorting prompts must be globally unique")
            seen.add(key)
            if source_parity:
                item = _unique_item(generator, source_seen)
                sx, sy = generator.collate([item], config.input_seq_len)
                if not np.array_equal(row, sx[0]) or not np.array_equal(target, sy[0]):
                    raise ValueError("sorting literal generator/collator parity mismatch")


def prepare_sorting(output, *, seed=1337, n_train=100_000, n_val=1000, n_test=1000):
    """New unique fixed pools; upstream online replacement sampling is not claimed."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    verify_source()
    hashes = {f"{s}_{k}.npy": "" for s in SPLITS for k in ("inputs", "labels")}
    config = SortingConfig(
        seed=seed,
        split_seeds=split_seeds(seed),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        sha256=hashes,
    )
    output.mkdir(parents=True)
    seen = set()
    for split, split_seed in config.split_seeds.items():
        generator = source_task(config, split_seed)
        shape = (getattr(config, f"n_{split}"), 128)
        x, y = np.empty(shape, dtype=np.int64), np.empty(shape, dtype=np.int64)
        for index in range(shape[0]):
            item = _unique_item(generator, seen)
            x[index : index + 1], y[index : index + 1] = generator.collate([item], 128)
        validate_arrays(x, y, config)
        for kind, array in (("inputs", x), ("labels", y)):
            name = f"{split}_{kind}.npy"
            np.save(output / name, array, allow_pickle=False)
            hashes[name] = sha256(output / name)
    config = config.model_copy(update={"sha256": hashes})
    (output / "meta.json").write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return config
