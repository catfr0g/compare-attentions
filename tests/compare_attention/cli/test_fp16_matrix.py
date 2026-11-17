"""FP16 matrix changes only precision, with exact family-specific smoke budgets."""

from copy import deepcopy
from pathlib import Path

import yaml

from compare_attention.cli.train import load_experiment

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (42, 6767, 696969)
COSTS = {"classic": 2473901162496, "deltanet": 2399573901312}


def test_complete_fp16_matrix_matches_full_legacy_mappings() -> None:
    root = ROOT / "experiments"
    fp16 = root / "fp16"
    expected_paths = {f"{family}/seed_{seed}.yaml" for family in COSTS for seed in SEEDS} | {
        f"smoke/{family}.yaml" for family in COSTS
    }
    assert {path.relative_to(fp16).as_posix() for path in fp16.rglob("*.yaml")} == expected_paths
    for family, cost in COSTS.items():
        for seed in SEEDS:
            baseline = root / family / f"seed_{seed}.yaml"
            expected = deepcopy(yaml.safe_load(baseline.read_text(encoding="utf-8")))
            assert expected["schedule"]["dtype"] == "bfloat16"
            expected["schedule"]["dtype"] = "float16"
            path = fp16 / family / baseline.name
            assert yaml.safe_load(path.read_text(encoding="utf-8")) == expected
            parsed = load_experiment(path)
            assert parsed.schedule.dtype == "float16"
            assert parsed.schedule.batch_size == 8
            assert parsed.schedule.compile is True
            assert parsed.schedule.train_flops == 1_000_000_000_000_000
        expected = yaml.safe_load((fp16 / family / "seed_42.yaml").read_text())
        expected["schedule"]["train_flops"] = cost
        path = fp16 / "smoke" / f"{family}.yaml"
        assert yaml.safe_load(path.read_text()) == expected
        assert load_experiment(path).schedule.train_flops == cost


def test_burnin_configs_price_exactly_32_steps_without_other_changes() -> None:
    directory = ROOT / "experiments" / "fp16_burnin"
    assert directory.is_dir(), "Missing FP16 numerical burn-in configs"
    assert {p.name for p in directory.glob("*.yaml")} == {f"{a}.yaml" for a in COSTS}
    for family, cost in COSTS.items():
        baseline = ROOT / "experiments" / "fp16" / family / "seed_42.yaml"
        expected = yaml.safe_load(baseline.read_text(encoding="utf-8"))
        expected["schedule"]["train_flops"] = cost * 32
        path = directory / f"{family}.yaml"
        assert yaml.safe_load(path.read_text(encoding="utf-8")) == expected
        assert load_experiment(path).schedule.train_flops == cost * 32
