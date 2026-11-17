"""CLI opt-in controls and explicit resume path remain separate from scientific config."""

from pathlib import Path

from compare_attention.cli.train import load_experiment, parse_train_args


def test_resume_and_training_control_flags_are_parsed(tmp_path):
    checkpoint = tmp_path / "step_00000002.pt"
    args = parse_train_args(
        [
            "--data",
            str(tmp_path / "data"),
            "--train-flops",
            "1000000",
            "--tracker",
            "disabled",
            "--checkpoint-every-steps",
            "1000",
            "--early-stopping-patience",
            "20",
            "--early-stopping-min-delta",
            "0.001",
            "--early-stopping-min-steps",
            "2000",
            "--amp-max-scale",
            "128",
            "--resume-from",
            str(checkpoint),
        ]
    )
    assert args.resume_from == checkpoint
    assert args.schedule.checkpoint_every_steps == 1000
    assert args.schedule.early_stopping_patience == 20
    assert args.schedule.amp_max_scale == 128


def test_existing_yaml_defaults_disable_early_stopping_and_resume():
    root = Path(__file__).resolve().parents[3]
    args = load_experiment(root / "experiments/fp16/classic/seed_42.yaml")
    assert args.schedule.early_stopping_patience == 0
    assert args.schedule.checkpoint_every_steps == 0
    assert args.schedule.amp_max_scale is None
    assert args.resume_from is None
