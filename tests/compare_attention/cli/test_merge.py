"""Мержинг YAML в argv: precedence флагов."""

from pathlib import Path

from compare_attention.cli.train import _merge_yaml_into_argv, load_experiment


def test_yaml_tokens_fill_every_required_section(tmp_path: Path) -> None:
    yaml_path = tmp_path / "x.yaml"
    yaml_path.write_text(
        "data: data/kv_retrieval\n"
        "schedule:\n"
        "  train_flops: 12\n"
        "  compile: false\n"
        "model:\n"
        "  vocab_size: 256\n"
        "kv:\n"
        "  input_seq_len: 32\n"
        "tracker: disabled\n",
        encoding="utf-8",
    )
    defaults = load_experiment(yaml_path)

    argv = _merge_yaml_into_argv(["--config=" + str(yaml_path)], defaults)

    assert any("kv_retrieval" in token for token in argv)
    assert "--train-flops=12" in argv
    assert "--no-compile" in argv
    assert "--vocab-size=256" in argv
    assert "--input-seq-len=32" in argv
    assert "--tracker=disabled" in argv


def test_explicit_flags_win_over_the_yaml_tokens(tmp_path: Path) -> None:
    yaml_path = tmp_path / "x.yaml"
    yaml_path.write_text(
        "data: data/kv_retrieval\nschedule:\n  train_flops: 12\n",
        encoding="utf-8",
    )
    defaults = load_experiment(yaml_path)

    argv = _merge_yaml_into_argv(["--config=" + str(yaml_path), "--train-flops=9"], defaults)

    assert "--train-flops=9" in argv
    assert "--train-flops=12" not in argv


def test_optional_sections_are_not_forged(tmp_path: Path) -> None:
    yaml_path = tmp_path / "x.yaml"
    yaml_path.write_text("data: data/kv_retrieval\nschedule:\n  train_flops: 5\n", encoding="utf-8")
    defaults = load_experiment(yaml_path)

    argv = _merge_yaml_into_argv([], defaults)

    assert not any("--hf-repo" in token for token in argv)
    assert not any("--result-dir" in token for token in argv)


def test_the_enum_flag_is_rendered_by_name(tmp_path: Path) -> None:
    yaml_path = tmp_path / "x.yaml"
    yaml_path.write_text("data: data/kv_retrieval\nschedule:\n  train_flops: 7\n", encoding="utf-8")
    defaults = load_experiment(yaml_path)

    argv = _merge_yaml_into_argv([], defaults)

    assert "--axis=TRAIN_FLOPS" in argv
