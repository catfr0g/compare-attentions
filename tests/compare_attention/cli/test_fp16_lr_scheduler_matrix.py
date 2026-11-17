"""Scheduler study changes only the explicit optimizer schedule versus LR controls."""

from pathlib import Path

import yaml

from compare_attention.cli.train import load_experiment

ROOT = Path(__file__).resolve().parents[3]


def test_scheduler_matrix_preserves_lr_controls_and_declares_schedule():
    directory = ROOT / "experiments/fp16_lr_scheduler"
    expected = {
        f"1e-3/{stage}/{family}/seed_42.yaml"
        for stage in ("full", "exact1", "exact32")
        for family in ("classic", "deltanet")
    }
    assert {p.relative_to(directory).as_posix() for p in directory.rglob("*.yaml")} == expected
    for name in sorted(expected):
        baseline_path = ROOT / "experiments/fp16_lr_pilots" / name
        baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
        baseline["optim"].update(
            lr_schedule="warmup_cosine", warmup_fraction=0.05, min_lr_ratio=0.1
        )
        path = directory / name
        assert yaml.safe_load(path.read_text(encoding="utf-8")) == baseline
        parsed = load_experiment(path)
        assert parsed.optim.learning_rate == 0.001
        assert parsed.optim.lr_schedule == "warmup_cosine"
        assert parsed.optim.warmup_fraction == 0.05
        assert parsed.optim.min_lr_ratio == 0.1
        assert parsed.schedule.batch_size == 32
        assert parsed.schedule.early_stopping_patience == 0
