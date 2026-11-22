"""Answer-only adapters for the reviewed variable/state tracking sources."""

# Pylint cannot infer the synthetic importlib package's standard __name__.
# pylint: disable=no-member

import importlib.util


def test_tracking_adapter_available():
    assert importlib.util.find_spec("compare_attention.data.tracking") is not None


def test_fixed_shapes_oracles_and_global_uniqueness(tmp_path):
    import numpy as np

    from compare_attention.data.tracking import prepare_tracking, validate_arrays

    for task, width, vocab, targets in (
        ("ruler_variable_tracking", 118, 116, 5),
        ("state_based_recall", 54, 43, 1),
    ):
        path = tmp_path / task
        config = prepare_tracking(path, task=task, n_train=35, n_val=3, n_test=5)
        assert config.input_seq_len == width
        assert config.vocab_size == vocab
        seen = set()
        for split in ("train", "val", "test"):
            x = np.load(path / f"{split}_inputs.npy", allow_pickle=False)
            y = np.load(path / f"{split}_labels.npy", allow_pickle=False)
            assert x.shape == y.shape == (getattr(config, f"n_{split}"), width)
            assert np.all((y != -1).sum(1) == targets)
            validate_arrays(x, y, config)
            for row in x:
                key = row[: config.prompt_length].tobytes()
                assert key not in seen
                seen.add(key)
        assert len(seen) == 43


def test_source_bytes_rng_collation_and_metadata(tmp_path):
    import hashlib
    import importlib.util
    import json
    import sys
    import types
    from pathlib import Path

    import numpy as np

    from compare_attention.data.tracking import SOURCE_HASHES, oracle, prepare_tracking

    root = Path(__file__).resolve().parents[3]
    source = root / "local/reports/four-task-benchmark/next-dataset-review-v1/tasks"
    package = types.ModuleType("_tracking_reference")
    package.__path__ = []
    sys.modules[package.__name__] = package
    for name, digest in SOURCE_HASHES.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
        vendored = root / "src/compare_attention/data/tracking_source" / name
        assert vendored.read_bytes() == (source / name).read_bytes()
        module_name = package.__name__ + "." + name[:-3]
        spec = importlib.util.spec_from_file_location(module_name, source / name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    for task, module, cls in (
        ("ruler_variable_tracking", "ruler", "RulerTask"),
        ("state_based_recall", "state_based_recall", "StateBasedRecallTask"),
    ):
        path = tmp_path / task
        c = prepare_tracking(path, task=task, n_train=35, n_val=3, n_test=5)
        params = (
            {
                key: getattr(c, key)
                for key in ("n_vars", "v_card", "num_chains", "num_hops", "n_filler")
            }
            if task == "ruler_variable_tracking"
            else {key: getattr(c, key) for key in ("n_bits", "max_bits", "n_swaps")}
        )
        taken = set()
        for split, seed in c.split_seeds.items():
            upstream = getattr(sys.modules[package.__name__ + "." + module], cls)(
                **params, seed=seed
            )
            items = []
            while len(items) < getattr(c, f"n_{split}"):
                item = upstream._sample_one()
                key = item.prompt.tobytes()
                if key in taken:
                    continue
                taken.add(key)
                answer, details = oracle(item.prompt, c)
                np.testing.assert_array_equal(item.answer, answer)
                assert item.metadata["sequence_length"] == c.prompt_length
                if task == "state_based_recall":
                    assert item.metadata["address"] == details["address"]
                    assert item.metadata["moved"] == details["address_moved"]
                items.append(item)
            rx, ry = upstream.collate(items, c.input_seq_len)
            np.testing.assert_array_equal(rx, np.load(path / f"{split}_inputs.npy"))
            np.testing.assert_array_equal(ry, np.load(path / f"{split}_labels.npy"))
        assert json.loads(c.model_dump_json())["source_commit"].startswith("30922c")


def test_rehashed_label_and_prefix_corruption_rejected(tmp_path):
    import numpy as np
    import pytest

    from compare_attention.data.tracking import prepare_tracking, validate_arrays

    path = tmp_path / "ruler"
    c = prepare_tracking(path, task="ruler_variable_tracking", n_train=2, n_val=1, n_test=1)
    x, y = (np.load(path / f"test_{kind}.npy") for kind in ("inputs", "labels"))
    for part in ("answer", "prefix", "dense", "query", "dtype"):
        a, b = x.copy(), y.copy()
        if part == "answer":
            b[0, -1] = (b[0, -1] + 1) % c.vocab_size
        elif part == "prefix":
            a[0, -1] = (a[0, -1] + 1) % c.vocab_size
        elif part == "dense":
            b[0, 0] = 20
        elif part == "query":
            a[0, c.prompt_length - 2] = 0
        else:
            a = a.astype(np.int32)
        with pytest.raises(ValueError):
            validate_arrays(a, b, c)


def test_single_chain_blind_lhs_control(tmp_path):
    import numpy as np

    from compare_attention.data.tracking import oracle, prepare_tracking

    path = tmp_path / "one"
    c = prepare_tracking(
        path,
        task="ruler_variable_tracking",
        parameters={"num_chains": 1},
        n_train=2,
        n_val=1,
        n_test=5,
    )
    for row in np.load(path / "test_inputs.npy"):
        answer, details = oracle(row[: c.prompt_length], c)
        np.testing.assert_array_equal(answer, details["blind_lhs"])


def test_explicit_chunk_padding_preserves_all_five_targets(tmp_path):
    import numpy as np

    from compare_attention.data.tracking import prepare_tracking, validate_arrays

    c = prepare_tracking(
        tmp_path / "padded",
        task="ruler_variable_tracking",
        pad_to_128=True,
        n_train=2,
        n_val=1,
        n_test=1,
    )
    x = np.load(tmp_path / "padded/test_inputs.npy")
    y = np.load(tmp_path / "padded/test_labels.npy")
    assert c.input_seq_len == 128 and c.prompt_length == 114
    assert np.all(x[:, 118:] == 0) and np.all(y[:, 118:] == -1)
    assert np.count_nonzero(y != -1) == 5
    validate_arrays(x, y, c)
