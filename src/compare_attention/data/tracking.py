"""Pinned literal generators/collation with qualified split-seeded fixed pools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from compare_attention.data.tiny_shakespeare import sha256
from compare_attention.data.tracking_oracles import ruler_oracle, state_oracle
from compare_attention.data.tracking_source.ruler import RulerTask
from compare_attention.data.tracking_source.state_based_recall import StateBasedRecallTask

SOURCE_COMMIT = "30922c4100992a245a23103a790ad737ea8ede4c"
SOURCE_HASHES = {
    "base.py": "c9088836a4af2407f30b307d4f6d4c149b04bda7e4fabc39e159b02b5bc67e1a",
    "ruler.py": "563b147e29fa72206a53ea3e67fb2c304de6509012eddde43bb5459980fec5a5",
    "state_based_recall.py": "be199124e735b0a4f5f7ad23dfeaa5f774700b546b9d2ab469b492de48fbe1b4",
}
TASKS = ("ruler_variable_tracking", "state_based_recall")
SPLITS = ("train", "val", "test")


class TrackingConfig(BaseModel):
    """Only new task metadata; no fabricated KV/Count/group fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    format_version: Literal[1] = 1
    dtype: Literal["int64"] = "int64"
    ignore_index: Literal[-1] = -1
    pad_id: Literal[0] = 0
    label_alignment: Literal["answer_only_shifted_next_token"] = "answer_only_shifted_next_token"
    input_seq_len: int = Field(ge=1, strict=True)
    padding_policy: Literal["none", "right_pad_to_128_for_delta_chunk64"] = "none"
    prompt_length: int = Field(ge=1, strict=True)
    answer_length: int = Field(ge=1, strict=True)
    vocab_size: int = Field(ge=1, strict=True)
    n_train: int = Field(default=100_000, ge=1, strict=True)
    n_val: int = Field(default=1000, ge=1, strict=True)
    n_test: int = Field(default=1000, ge=1, strict=True)
    seed: int = Field(default=1337, ge=0, strict=True)
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

    @model_validator(mode="after")
    def _common_contract(self):
        names = {f"{s}_{k}.npy" for s in SPLITS for k in ("inputs", "labels")}
        seeds = dict(
            zip(SPLITS, map(int, np.random.SeedSequence(self.seed).generate_state(3)), strict=True)
        )
        if self.sha256.keys() != names or self.split_seeds != seeds:
            raise ValueError("exact array manifest and derived split seeds required")
        if self.source_sha256 != SOURCE_HASHES:
            raise ValueError("source pins mismatch")
        actual_length = self.prompt_length + self.answer_length - 1
        expected_length = actual_length if self.padding_policy == "none" else 128
        if self.input_seq_len != expected_length or actual_length > self.input_seq_len:
            raise ValueError("shifted answer-only length mismatch")
        return self


class RulerConfig(TrackingConfig):
    """Four interleaved chains, queried chain zero; no EOS or padding."""

    name: Literal["ruler_variable_tracking"] = TASKS[0]
    task: Literal["ruler_variable_tracking"] = TASKS[0]
    n_vars: int = Field(default=64, ge=1, strict=True)
    v_card: int = Field(default=32, ge=1, strict=True)
    num_chains: int = Field(default=4, ge=1, strict=True)
    num_hops: int = Field(default=4, ge=0, strict=True)
    n_filler: int = Field(default=32, ge=0, strict=True)

    @model_validator(mode="after")
    def _ruler_contract(self):
        if (
            self.answer_length != self.num_hops + 1
            or self.prompt_length != 4 * self.num_chains * self.answer_length + self.n_filler + 2
            or self.vocab_size != 20 + self.n_vars + self.v_card
            or self.num_chains * self.answer_length > self.n_vars
            or self.num_chains > self.v_card
        ):
            raise ValueError("RULER source shape/vocabulary mismatch")
        return self


class StateRecallConfig(TrackingConfig):
    """Five pointers; bit tokens 4/5; full-vocabulary answer argmax."""

    name: Literal["state_based_recall"] = TASKS[1]
    task: Literal["state_based_recall"] = TASKS[1]
    n_bits: int = Field(default=32, ge=5, strict=True)
    max_bits: int = Field(default=32, ge=5, strict=True)
    n_swaps: int = Field(default=4, ge=0, strict=True)

    @model_validator(mode="after")
    def _state_contract(self):
        if (
            self.answer_length != 1
            or self.prompt_length != self.n_bits + 14 + 2 * self.n_swaps
            or self.vocab_size != 11 + self.max_bits
            or self.max_bits < self.n_bits
        ):
            raise ValueError("state recall source shape/vocabulary mismatch")
        return self


def source_task(config, seed):
    """The actual source classes (including source RNG calls and collate)."""
    if isinstance(config, RulerConfig):
        return RulerTask(
            **{
                k: getattr(config, k)
                for k in ("n_vars", "v_card", "num_chains", "num_hops", "n_filler")
            },
            seed=seed,
        )
    return StateBasedRecallTask(
        n_bits=config.n_bits, max_bits=config.max_bits, n_swaps=config.n_swaps, seed=seed
    )


def oracle(prompt, config):
    """Task selection stays explicit; Count semantics/metrics are never inherited."""
    return (ruler_oracle if isinstance(config, RulerConfig) else state_oracle)(prompt, config)


def validate_arrays(inputs, labels, config):
    """Validate EVERY prompt, complete answer span and legitimate teacher-forced prefix."""
    if (
        inputs.dtype != np.int64
        or labels.dtype != np.int64
        or inputs.ndim != 2
        or inputs.shape != labels.shape
        or not inputs.size
        or inputs.shape[1] != config.input_seq_len
    ):
        raise ValueError("tracking arrays require matching nonempty int64 shapes")
    if np.any(inputs < 0) or np.any(inputs >= config.vocab_size):
        raise ValueError("tracking input outside vocabulary")
    boundary = config.prompt_length - 1
    if np.any(labels[:, :boundary] != -1):
        raise ValueError("tracking requires answer-only labels")
    for x, y in zip(inputs, labels, strict=True):
        answer, _ = oracle(x[: config.prompt_length], config)
        end = boundary + config.answer_length
        if not np.array_equal(y[boundary:end], answer) or np.any(y[end:] != -1):
            raise ValueError("tracking answer labels disagree with prompt oracle")
        if not np.array_equal(x[config.prompt_length : end], answer[:-1]) or np.any(x[end:] != 0):
            raise ValueError("tracking teacher-forced answer prefix mismatch")


def verify_integrity(directory, config):
    """Reject hash/shape/oracle corruption and duplicated prompts across all splits."""
    directory = Path(directory)
    for name, digest in config.sha256.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"tracking dataset integrity mismatch: {name}")
    seen = set()
    for split in SPLITS:
        x, y = (
            np.load(directory / f"{split}_{kind}.npy", mmap_mode="r", allow_pickle=False)
            for kind in ("inputs", "labels")
        )
        if x.shape != (getattr(config, f"n_{split}"), config.input_seq_len):
            raise ValueError("tracking metadata split shape mismatch")
        validate_arrays(x, y, config)
        for row in x:
            key = hashlib.sha256(row[: config.prompt_length].tobytes()).digest()
            if key in seen:
                raise ValueError("tracking prompts must be globally unique")
            seen.add(key)


def prepare_tracking(
    output,
    *,
    task,
    seed=1337,
    n_train=100_000,
    n_val=1000,
    n_test=1000,
    parameters=None,
    pad_to_128=False,
):
    """Create a new fixed-pool adaptation, NOT upstream online/replacement training."""
    # pylint: disable=too-many-locals,protected-access
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if task not in TASKS:
        raise ValueError("unsupported tracking task")
    params = dict(parameters or {})
    config_type = RulerConfig if task == TASKS[0] else StateRecallConfig
    defaults = (
        {"n_vars": 64, "v_card": 32, "num_chains": 4, "num_hops": 4, "n_filler": 32}
        if task == TASKS[0]
        else {"n_bits": 32, "max_bits": 32, "n_swaps": 4}
    )
    if params.keys() - defaults.keys():
        raise ValueError("unknown tracking source parameter")
    defaults.update(params)
    if task == TASKS[0]:
        answer = defaults["num_hops"] + 1
        prompt = 4 * defaults["num_chains"] * answer + defaults["n_filler"] + 2
        vocab = 20 + defaults["n_vars"] + defaults["v_card"]
    else:
        answer, prompt = 1, defaults["n_bits"] + 14 + 2 * defaults["n_swaps"]
        vocab = 11 + defaults["max_bits"]
    split_seeds = dict(
        zip(SPLITS, map(int, np.random.SeedSequence(seed).generate_state(3)), strict=True)
    )
    hashes = {f"{s}_{k}.npy": "" for s in SPLITS for k in ("inputs", "labels")}
    if pad_to_128 and task != TASKS[0]:
        raise ValueError("chunk padding is only proposed for RULER")
    config = config_type(
        **defaults,
        input_seq_len=128 if pad_to_128 else prompt + answer - 1,
        padding_policy="right_pad_to_128_for_delta_chunk64" if pad_to_128 else "none",
        prompt_length=prompt,
        answer_length=answer,
        vocab_size=vocab,
        seed=seed,
        split_seeds=split_seeds,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        sha256=hashes,
    )
    output.mkdir(parents=True)
    seen = set()
    for split, split_seed in split_seeds.items():
        generator = source_task(config, split_seed)
        shape = (getattr(config, f"n_{split}"), config.input_seq_len)
        x, y = np.empty(shape, dtype=np.int64), np.empty(shape, dtype=np.int64)
        for index in range(shape[0]):
            for _ in range(10_000):
                item = generator._sample_one()
                key = hashlib.sha256(item.prompt.tobytes()).digest()
                if key not in seen:
                    seen.add(key)
                    break
            else:
                raise ValueError("not enough unique tracking prompts")
            x[index : index + 1], y[index : index + 1] = generator.collate(
                [item], config.input_seq_len
            )
        validate_arrays(x, y, config)
        for kind, array in (("inputs", x), ("labels", y)):
            name = f"{split}_{kind}.npy"
            np.save(output / name, array, allow_pickle=False)
            hashes[name] = sha256(output / name)
    config = config.model_copy(update={"sha256": hashes})
    (output / "meta.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return config
