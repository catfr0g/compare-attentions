"""Long-budget experiments preserve the approved FP16 scientific configuration."""

from pathlib import Path

import yaml

from compare_attention.cli.train import load_experiment

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (42, 6767, 696969)
FAMILIES = ("classic", "deltanet")
LONG_BUDGET = 100_000_000_000_000_000


def test_long_matrix_changes_only_budget_from_approved_fp16_baseline() -> None:
    directory = ROOT / "experiments" / "fp16_long"
    expected_paths = {f"{family}/seed_{seed}.yaml" for family in FAMILIES for seed in SEEDS}
    actual_paths = {p.relative_to(directory).as_posix() for p in directory.rglob("*.yaml")}
    assert actual_paths == expected_paths
    for family in FAMILIES:
        for seed in SEEDS:
            baseline_path = ROOT / "experiments" / "fp16" / family / f"seed_{seed}.yaml"
            expected = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
            assert expected["schedule"]["train_flops"] == 1_000_000_000_000_000
            expected["schedule"]["train_flops"] = LONG_BUDGET
            path = directory / family / baseline_path.name
            assert yaml.safe_load(path.read_text(encoding="utf-8")) == expected
            baseline = load_experiment(baseline_path).model_dump()
            baseline["schedule"]["train_flops"] = LONG_BUDGET
            assert load_experiment(path).model_dump() == baseline
