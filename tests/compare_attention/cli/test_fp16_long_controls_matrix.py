"""Exploratory controls live separately from fixed-budget long baselines."""

from pathlib import Path

import yaml

from compare_attention.cli.train import load_experiment

ROOT = Path(__file__).resolve().parents[3]
CONTROLS = {
    "checkpoint_every_steps": 1000,
    "checkpoint_keep_last": 2,
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 0.001,
    "early_stopping_min_steps": 2000,
    "amp_max_scale": 128.0,
}
COSTS = {"classic": 2473901162496, "deltanet": 2399573901312}


def test_exploratory_matrix_adds_only_approved_training_controls():
    folder = ROOT / "experiments" / "fp16_long_controls"
    expected_paths = {
        f"{family}/seed_{seed}.yaml" for family in COSTS for seed in (42, 6767, 696969)
    } | {f"{stage}/{family}.yaml" for stage in ("smoke", "burnin") for family in COSTS}
    assert {p.relative_to(folder).as_posix() for p in folder.rglob("*.yaml")} == expected_paths
    for family, cost in COSTS.items():
        for seed in (42, 6767, 696969):
            name = f"{family}/seed_{seed}.yaml"
            baseline = ROOT / "experiments/fp16_long" / name
            expected = yaml.safe_load(baseline.read_text())
            expected["schedule"].update(CONTROLS)
            target = folder / name
            assert yaml.safe_load(target.read_text()) == expected
            parsed = load_experiment(target)
            assert parsed.schedule.train_flops == 100_000_000_000_000_000
            assert parsed.schedule.batch_size == 8
        for stage, steps in (("smoke", 1), ("burnin", 32)):
            expected = yaml.safe_load((folder / family / "seed_42.yaml").read_text())
            expected["schedule"]["train_flops"] = cost * steps
            expected["schedule"]["checkpoint_every_steps"] = 16 if steps == 32 else 1
            target = folder / stage / f"{family}.yaml"
            assert yaml.safe_load(target.read_text()) == expected
            assert load_experiment(target).schedule.train_flops == cost * steps
