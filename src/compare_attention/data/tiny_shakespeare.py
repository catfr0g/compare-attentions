"""Pinned Tiny Shakespeare bytes → disjoint, same-position character LM arrays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

SOURCE_COMMIT = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/karpathy/char-rnn/{SOURCE_COMMIT}"
    "/data/tinyshakespeare/input.txt"
)
SOURCE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
NAME = "tiny_shakespeare_char"


class CharacterLMConfig(BaseModel):
    """Task-specific metadata; never impersonates key–value retrieval."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    name: Literal["tiny_shakespeare_char"] = NAME
    format_version: Literal[1] = 1
    task: Literal["character_language_modeling"] = "character_language_modeling"
    label_alignment: Literal["same_position_next_character"] = "same_position_next_character"
    dtype: Literal["int64"] = "int64"
    ignore_index: Literal[-1] = -1
    input_seq_len: int = Field(ge=2)
    vocab_size: int = Field(ge=1)
    n_train: int = Field(ge=1)
    n_val: int = Field(ge=1)
    n_test: int = Field(ge=1)
    alphabet: list[str]
    source_sha256: str
    source_url: str
    source_commit: str
    source_characters: int = Field(ge=40)
    source_bytes: int = Field(ge=40)
    splits: dict[str, dict]
    sha256: dict[str, str]
    window_policy: Literal["stride_context_pad_tail_mask_labels"] = (
        "stride_context_pad_tail_mask_labels"
    )
    training_stream: Literal["ordered_rows_cyclic_no_shuffle"] = "ordered_rows_cyclic_no_shuffle"

    @model_validator(mode="after")
    def _alphabet(self):
        if self.alphabet != sorted(set(self.alphabet)) or len(self.alphabet) != self.vocab_size:
            raise ValueError("alphabet must be sorted unique vocabulary")
        if any(len(char) != 1 for char in self.alphabet):
            raise ValueError("alphabet entries must be single Unicode characters")
        names = {"input.txt"} | {
            f"{split}_{kind}.npy"
            for split in ("train", "val", "test")
            for kind in ("inputs", "labels")
        }
        if self.sha256.keys() != names or self.sha256["input.txt"] != self.source_sha256:
            raise ValueError("character dataset integrity manifest must contain exact data files")
        return self


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_integrity(directory: Path, config: CharacterLMConfig) -> None:
    """Bind source and all arrays before pricing, training or public evaluation."""
    for name, expected in config.sha256.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"character dataset integrity mismatch: {name}")


def _windows(tokens: np.ndarray, context: int) -> tuple[np.ndarray, np.ndarray]:
    targets = len(tokens) - 1
    rows = (targets + context - 1) // context
    x = np.zeros((rows, context), dtype=np.int64)
    y = np.full((rows, context), -1, dtype=np.int64)
    for row, start in enumerate(range(0, targets, context)):
        count = min(context, targets - start)
        x[row, :count] = tokens[start : start + count]
        y[row, :count] = tokens[start + 1 : start + count + 1]
    return x, y


def prepare_tiny_shakespeare(
    source: Path,
    output: Path,
    *,
    context: int = 512,
    expected_sha256: str = SOURCE_SHA256,
) -> CharacterLMConfig:
    """Create only a new directory; no normalization, overwrite, UNK or pickle."""
    # A test fixture may supply an explicit digest; public CLI always requires the pinned source.
    # pylint: disable=too-many-locals
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("source SHA256 does not match expected bytes")
    text = raw.decode("utf-8", errors="strict")
    if context < 2 or len(text) < 40:
        raise ValueError("context >= 2 and source >= 40 characters required")
    boundaries = [0, len(text) * 90 // 100, len(text) * 95 // 100, len(text)]
    alphabet = sorted(set(text[: boundaries[1]]))
    if set(text) != set(alphabet):
        raise ValueError("held-out alphabet is not fully covered by training; no UNK permitted")
    encode = {char: index for index, char in enumerate(alphabet)}
    tokens = np.array([encode[char] for char in text], dtype=np.int64)
    splits, arrays = {}, {}
    for split, start, end in zip(
        ("train", "val", "test"), boundaries, boundaries[1:], strict=False
    ):
        x, y = _windows(tokens[start:end], context)
        arrays[f"{split}_inputs.npy"], arrays[f"{split}_labels.npy"] = x, y
        splits[split] = {
            "raw_range": [start, end],
            "target_range": [start + 1, end],
            "characters": end - start,
            "supervised_targets": end - start - 1,
            "rows": len(x),
            "masked_padding": int((y == -1).sum()),
            "raw_utf8_sha256": hashlib.sha256(text[start:end].encode("utf-8")).hexdigest(),
        }
    output.mkdir(parents=True)
    (output / "input.txt").write_bytes(raw)
    for name, array in arrays.items():
        np.save(output / name, array, allow_pickle=False)
    config = CharacterLMConfig(
        input_seq_len=context,
        vocab_size=len(alphabet),
        n_train=splits["train"]["rows"],
        n_val=splits["val"]["rows"],
        n_test=splits["test"]["rows"],
        alphabet=alphabet,
        source_sha256=expected_sha256,
        source_url=SOURCE_URL,
        source_commit=SOURCE_COMMIT,
        source_characters=len(text),
        source_bytes=len(raw),
        splits=splits,
        sha256={name: sha256(output / name) for name in ["input.txt", *arrays]},
    )
    (output / "meta.json").write_text(
        json.dumps(config.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return config
