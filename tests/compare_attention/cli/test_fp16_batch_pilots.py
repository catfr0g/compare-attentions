"""Complete raw/parsed fair-batch matrix; CPU-only, without trainer execution."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from compare_attention.cli.train import load_experiment, parse_train_args
from compare_attention.data.store import load_config
from compare_attention.train.config import TrainConfig
from compare_attention.train.flops import estimate_step

ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "experiments/fp16_batch_pilots"
FAMILIES = ("classic", "deltanet")
BUDGET = 10_000_000_000_000_000
COSTS = {"classic": 2_473_901_162_496, "deltanet": 2_399_573_901_312}


@pytest.fixture(name="prepared_meta")
def fixture_prepared_meta(tmp_path):
    """Metadata of the immutable prepared dataset, not newly generated examples."""
    meta = {
        "format_version": 2,
        "name": "kv_retrieval",
        "source": "https://github.com/On-Point-RND/transformer-architecture-generalization",
        "ignore_index": -1,
        "dtype": "int64",
        "vocab_size": 8192,
        "input_seq_len": 512,
        "num_kv_pairs": 64,
        "power_a": 0.01,
        "n_train": 100000,
        "n_val": 1000,
        "n_test": 1000,
        "seed": 1337,
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


def _raw(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _config(path, prepared_meta):
    parsed = load_experiment(path)
    cli = parse_train_args(["--config", str(path)])
    assert cli.model == parsed.model
    assert cli.schedule == parsed.schedule
    assert cli.optim == parsed.optim
    assert cli.data == parsed.data
    return TrainConfig(
        data=prepared_meta,
        model=parsed.model,
        schedule=parsed.schedule,
        optim=parsed.optim,
        dataset=load_config(prepared_meta),
    )


# Keep the whole raw/parsed matrix comparison in one end-to-end contract.
# pylint: disable-next=too-many-locals
def test_complete_matrix_preserves_full_raw_and_parsed_scientific_mapping(prepared_meta):
    expected_paths = {
        f"{family}/batch_{batch}/seed_42.yaml" for family in FAMILIES for batch in (8, 32)
    } | {
        f"{stage}/{family}/batch_32/seed_42.yaml"
        for stage in ("smoke", "burnin")
        for family in FAMILIES
    }
    assert {p.relative_to(MATRIX).as_posix() for p in MATRIX.rglob("*.yaml")} == expected_paths
    configs = {}
    for family in FAMILIES:
        baseline = _raw(ROOT / f"experiments/fp16_long_controls/{family}/seed_42.yaml")
        for batch in (8, 32):
            expected = deepcopy(baseline)
            expected["schedule"].update(
                train_flops=BUDGET,
                batch_size=batch,
                val_every_steps=100 if batch == 8 else 25,
                log_every_steps=100 if batch == 8 else 25,
                checkpoint_every_steps=500 if batch == 8 else 125,
                checkpoint_keep_last=2,
                early_stopping_patience=0,
            )
            path = MATRIX / f"{family}/batch_{batch}/seed_42.yaml"
            assert _raw(path) == expected
            config = _config(path, prepared_meta)
            assert config.schedule.model_dump() == load_experiment(path).schedule.model_dump()
            assert config.dataset.seed == 1337
            assert config.dataset.input_seq_len == config.model.block_size == 512
            assert config.optim.learning_rate == 0.0003
            assert config.schedule.amp_max_scale == 128.0
            assert config.schedule.early_stopping_patience == 0
            cost = estimate_step(config).total
            assert cost == COSTS[family] * (batch // 8)
            configs[family, batch] = config
        for stage, steps in (("smoke", 1), ("burnin", 32)):
            pilot = MATRIX / f"{family}/batch_32/seed_42.yaml"
            expected = _raw(pilot)
            expected["schedule"].update(
                train_flops=estimate_step(configs[family, 32]).total * steps,
                checkpoint_every_steps=steps,
            )
            path = MATRIX / f"{stage}/{family}/batch_32/seed_42.yaml"
            assert _raw(path) == expected
            config = _config(path, prepared_meta)
            cost = estimate_step(config).total
            assert config.schedule.train_flops == cost * steps
            assert (config.schedule.train_flops + cost - 1) // cost == steps
            assert config.schedule.checkpoint_every_steps == steps
            assert config.schedule.early_stopping_patience == 0
    for batch in (8, 32):
        classic = configs["classic", batch].model_dump(mode="json")
        delta = configs["deltanet", batch].model_dump(mode="json")
        assert classic["model"].pop("attention") == "classic"
        assert delta["model"].pop("attention") == "deltanet"
        assert classic == delta
    for family in FAMILIES:
        small, large = (configs[family, batch] for batch in (8, 32))
        for cadence in ("val_every_steps", "log_every_steps", "checkpoint_every_steps"):
            assert getattr(small.schedule, cadence) * estimate_step(small).total == (
                getattr(large.schedule, cadence) * estimate_step(large).total
            )
            assert getattr(small.schedule, cadence) * 8 == getattr(large.schedule, cadence) * 32


def test_recorded_costs_and_integer_ceil_horizons_match_real_estimator(prepared_meta):
    manifest_path = MATRIX / "costs.json"
    assert manifest_path.is_file(), "Missing reproducible CPU cost ledger"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["configs"]
    paths = {p.relative_to(MATRIX).as_posix() for p in MATRIX.rglob("*.yaml")}
    assert len(rows) == len(paths) == 8
    assert {row["path"] for row in rows} == paths
    for row in rows:
        config = _config(MATRIX / row["path"], prepared_meta)
        estimate = estimate_step(config)
        budget = config.schedule.train_flops
        updates = (budget + estimate.total - 1) // estimate.total
        assert row["estimate"] == estimate.to_dict()
        assert row["train_flops"] == budget
        assert row["expected_updates"] == updates
        assert row["expected_completed_flops"] == updates * estimate.total
        assert row["budget_overshoot"] == updates * estimate.total - budget
        assert 0 <= row["budget_overshoot"] < estimate.total
