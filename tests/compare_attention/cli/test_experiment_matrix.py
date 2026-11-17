"""Matched attention experiments preserve the current DeltaNet training budget."""

from copy import deepcopy
from pathlib import Path

import yaml

from compare_attention.cli.train import load_experiment

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (42, 6767, 696969)


def test_attention_matrix_has_matched_configs() -> None:
    baseline = yaml.safe_load((ROOT / "experiments/delta.yaml").read_text(encoding="utf-8"))
    for attention in ("classic", "deltanet"):
        directory = ROOT / "experiments" / attention
        assert directory.is_dir(), f"Missing experiment directory: {directory}"
        assert sorted(path.name for path in directory.iterdir()) == sorted(
            f"seed_{seed}.yaml" for seed in SEEDS
        )
        for seed in SEEDS:
            path = directory / f"seed_{seed}.yaml"
            actual = yaml.safe_load(path.read_text(encoding="utf-8"))
            expected = deepcopy(baseline)
            expected["schedule"]["seed"] = seed
            expected["model"]["attention"] = attention
            assert actual == expected, path
            args = load_experiment(path)
            assert args.schedule.seed == seed
            assert args.model.attention == attention
            assert args.schedule.train_flops == baseline["schedule"]["train_flops"]
