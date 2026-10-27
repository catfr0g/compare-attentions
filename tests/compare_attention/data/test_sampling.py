# pylint: disable=redefined-outer-name
"""Генерация примеров kv_retrieval."""

import numpy as np
import pytest

from compare_attention.data.config import IGNORE_INDEX, KVRetrievalConfig
from compare_attention.data.records import KVRetrievalDataset, TokenSplit
from compare_attention.data.sampling import gap_probabilities, generate_dataset, generate_split
from tests.compare_attention.data.factories import config as _config

_SPLITS = ("train", "validation", "test")
_PAIRS = (("train", "validation"), ("train", "test"), ("validation", "test"))
_ARRAYS = ("inputs", "labels")
type _Example = tuple[KVRetrievalConfig, TokenSplit]
type _Datasets = tuple[KVRetrievalDataset, KVRetrievalDataset]


@pytest.fixture
def peaked_gaps() -> np.ndarray:
    return gap_probabilities(64, 4, 0.01)


@pytest.fixture
def validation_example() -> _Example:
    spec = _config()
    return spec, generate_dataset(spec).validation


@pytest.fixture
def same_seed_pair() -> _Datasets:
    spec = _config(seed=11)
    return generate_dataset(spec), generate_dataset(spec)


@pytest.fixture
def train_growth() -> _Datasets:
    small = generate_dataset(_config(n_train=2, n_val=2, n_test=3, seed=7))
    large = generate_dataset(_config(n_train=5, n_val=2, n_test=3, seed=7))
    return small, large


@pytest.fixture
def varied_dataset() -> KVRetrievalDataset:
    return generate_dataset(_config(n_train=6, n_val=5, n_test=4, seed=3))


@pytest.fixture
def replaced_duplicate() -> tuple[np.ndarray, np.ndarray, set[bytes], np.ndarray, np.ndarray]:
    spec = _config()
    first = np.arange(spec.input_seq_len, dtype=np.int64)
    second = first + 1
    labels = np.full(spec.input_seq_len, IGNORE_INDEX, dtype=np.int64)
    calls = {"n": 0}

    def sample(_rng: object, _spec: object, _gap_probs: object) -> tuple[np.ndarray, np.ndarray]:
        calls["n"] += 1
        prompt = first if calls["n"] < 3 else second
        return prompt.copy(), labels.copy()

    taken: set[bytes] = set()
    inputs, out_labels = generate_split(
        np.random.default_rng(0),
        2,
        spec,
        gap_probabilities(spec.input_seq_len, spec.num_kv_pairs, spec.power_a),
        taken,
        sample=sample,
        max_rejects=5,
        split_name="train",
    )
    return inputs, out_labels, taken, first, second


def test_peaked_gaps_cover_the_query_space(peaked_gaps: np.ndarray) -> None:
    assert peaked_gaps.shape == (28,)


def test_peaked_gaps_sum_to_one(peaked_gaps: np.ndarray) -> None:
    assert np.isclose(peaked_gaps.sum(), 1.0)


def test_peaked_gaps_prefer_shorter_distances(peaked_gaps: np.ndarray) -> None:
    assert np.all(peaked_gaps[:-1] > peaked_gaps[1:])


def test_unit_power_is_uniform() -> None:
    uniform = gap_probabilities(64, 4, 1.0)

    assert np.allclose(uniform, 1.0 / len(uniform))


@pytest.mark.parametrize("field", _ARRAYS)
def test_validation_array_has_sequence_length(
    validation_example: _Example,
    field: str,
) -> None:
    spec, split = validation_example

    assert getattr(split, field).shape[1] == spec.input_seq_len


@pytest.mark.parametrize("field", _ARRAYS)
def test_validation_array_is_int64(
    validation_example: _Example,
    field: str,
) -> None:
    _, split = validation_example

    assert getattr(split, field).dtype == np.int64


def test_prompt_tokens_are_non_negative(validation_example: _Example) -> None:
    _, split = validation_example

    assert split.inputs[0].min() >= 0


def test_prompt_tokens_belong_to_the_vocabulary(
    validation_example: _Example,
) -> None:
    spec, split = validation_example

    assert split.inputs[0].max() < spec.vocab_size


@pytest.mark.parametrize("offset", [0, 1])
def test_prefix_tokens_are_unique(
    validation_example: _Example,
    offset: int,
) -> None:
    spec, split = validation_example
    tokens = split.inputs[0, offset : 2 * spec.num_kv_pairs : 2]

    assert len(set(tokens.tolist())) == spec.num_kv_pairs


@pytest.mark.parametrize("offset", [0, 1])
def test_prefix_tokens_come_from_their_half(
    validation_example: _Example,
    offset: int,
) -> None:
    spec, split = validation_example
    half = spec.vocab_size // 2
    tokens = set(split.inputs[0, offset : 2 * spec.num_kv_pairs : 2].tolist())
    allowed = set(range(1, half)) if offset == 0 else set(range(half, spec.vocab_size))

    assert tokens <= allowed


def test_each_pair_is_queried_once(validation_example: _Example) -> None:
    spec, split = validation_example

    assert len(np.flatnonzero(split.labels[0] != IGNORE_INDEX)) == spec.num_kv_pairs


def test_queries_use_even_slots_after_the_pairs(
    validation_example: _Example,
) -> None:
    spec, split = validation_example
    context = 2 * spec.num_kv_pairs
    space = (spec.input_seq_len - context) // 2
    allowed = {context + 2 * gap for gap in range(space)}
    used = set(np.flatnonzero(split.labels[0] != IGNORE_INDEX).tolist())

    assert used <= allowed


def test_query_label_is_the_paired_value(validation_example: _Example) -> None:
    spec, split = validation_example
    context = 2 * spec.num_kv_pairs
    prompt = split.inputs[0]
    labels = split.labels[0]
    expected = dict(zip(prompt[:context:2].tolist(), prompt[1:context:2].tolist(), strict=True))
    paired = {int(prompt[pos]): int(labels[pos]) for pos in np.flatnonzero(labels != IGNORE_INDEX)}

    assert paired == expected


@pytest.mark.parametrize(("split", "field"), [("validation", "inputs"), ("test", "labels")])
def test_larger_train_keeps_earlier_split(train_growth: _Datasets, split: str, field: str) -> None:
    small, large = train_growth
    kept = getattr(getattr(small, split), field)
    same = getattr(getattr(large, split), field)

    assert np.array_equal(kept, same)


def test_larger_train_extends_the_same_prefix(train_growth: _Datasets) -> None:
    small, large = train_growth

    assert np.array_equal(small.train.inputs, large.train.inputs[: small.train.inputs.shape[0]])


@pytest.mark.parametrize(("split", "field"), [("train", "inputs"), ("validation", "labels")])
def test_same_seed_repeats(same_seed_pair: _Datasets, split: str, field: str) -> None:
    first, second = same_seed_pair
    left = getattr(getattr(first, split), field)
    right = getattr(getattr(second, split), field)

    assert np.array_equal(left, right)


def test_another_seed_changes_validation_inputs(same_seed_pair: _Datasets) -> None:
    first, _ = same_seed_pair

    other = generate_dataset(_config(seed=12))

    assert not np.array_equal(first.validation.inputs, other.validation.inputs)


@pytest.mark.parametrize("split", _SPLITS)
def test_split_prompts_are_unique(varied_dataset: KVRetrievalDataset, split: str) -> None:
    rows = getattr(varied_dataset, split).inputs

    assert len({row.tobytes() for row in rows}) == len(rows)


@pytest.mark.parametrize(("left", "right"), _PAIRS)
def test_splits_do_not_share_prompts(
    varied_dataset: KVRetrievalDataset,
    left: str,
    right: str,
) -> None:
    left_keys = {row.tobytes() for row in getattr(varied_dataset, left).inputs}
    right_keys = {row.tobytes() for row in getattr(varied_dataset, right).inputs}

    assert left_keys.isdisjoint(right_keys)


def test_smaller_power_a_pulls_queries_closer() -> None:
    spec = {
        "n_train": 1,
        "n_val": 200,
        "seed": 1,
        "vocab_size": 128,
        "input_seq_len": 64,
        "num_kv_pairs": 4,
    }
    close_set = generate_dataset(_config(**spec, power_a=0.01))
    spread_set = generate_dataset(_config(**spec, power_a=1.0))
    context = 8
    close_at = np.where(close_set.validation.labels != IGNORE_INDEX)[1]
    spread_at = np.where(spread_set.validation.labels != IGNORE_INDEX)[1]

    assert float(((close_at - context) // 2).mean()) < float(((spread_at - context) // 2).mean())


def test_first_accepted_prompt_is_kept(replaced_duplicate: tuple) -> None:
    inputs, _, _, first, _ = replaced_duplicate

    assert np.array_equal(inputs[0], first)


def test_duplicate_prompt_is_replaced_by_the_next_one(replaced_duplicate: tuple) -> None:
    inputs, _, _, _, second = replaced_duplicate

    assert np.array_equal(inputs[1], second)


def test_accepted_prompt_keeps_its_labels(replaced_duplicate: tuple) -> None:
    _, out_labels, _, _, _ = replaced_duplicate

    assert np.all(out_labels == IGNORE_INDEX)


def test_accepted_prompts_are_recorded(replaced_duplicate: tuple) -> None:
    _, _, taken, first, second = replaced_duplicate

    assert taken == {first.tobytes(), second.tobytes()}


def test_repeated_duplicate_prompts_raise() -> None:
    spec = _config()
    prompt = np.zeros(spec.input_seq_len, dtype=np.int64)
    labels = np.full(spec.input_seq_len, IGNORE_INDEX, dtype=np.int64)

    def sample(_rng: object, _given: object, _gap_probs: object) -> tuple[np.ndarray, np.ndarray]:
        return prompt.copy(), labels.copy()

    with pytest.raises(ValueError, match="duplicate"):
        generate_split(
            np.random.default_rng(0),
            2,
            spec,
            gap_probabilities(spec.input_seq_len, spec.num_kv_pairs, spec.power_a),
            set(),
            sample=sample,
            max_rejects=3,
            split_name="train",
        )
