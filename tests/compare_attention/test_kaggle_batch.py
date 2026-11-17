"""Последовательные Kaggle-запуски без сети и настоящих секретов."""

import json
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from compare_attention import kaggle_run


class FakeRunner:  # pylint: disable=too-few-public-methods
    """Выполнить локальные этапы и записать команды вместо Kaggle CLI."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.pushes: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        action = command[:3]
        text = "ok"
        if action == ["uv", "build", "--wheel"]:
            dist = Path(command[-1])
            dist.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dist / "compare_attention-0.1.0-py3-none-any.whl", "w") as wheel:
                wheel.writestr("compare_attention/__init__.py", "")
        elif command[:2] == ["uv", "export"]:
            Path(command[-1]).write_text("pyyaml==6.0.3\n", encoding="utf-8")
        elif action == ["kaggle", "datasets", "version"]:
            folder = Path(command[command.index("-p") + 1])
            self.uploads.append(
                {
                    path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
                    for path in folder.iterdir()
                    if path.suffix in {".yaml", ".yml"}
                }
            )
        elif action == ["kaggle", "datasets", "status"]:
            text = "ready"
        elif action == ["kaggle", "kernels", "push"]:
            folder = Path(command[command.index("-p") + 1])
            self.pushes.append(
                {
                    "path": folder,
                    "metadata": json.loads((folder / "kernel-metadata.json").read_text("utf-8")),
                    "code": (folder / "kernel.py").read_text("utf-8"),
                }
            )
        elif action == ["kaggle", "kernels", "status"]:
            text = "complete"
        elif action == ["kaggle", "kernels", "output"]:
            folder = Path(command[command.index("-p") + 1])
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "result.txt").write_text(command[3], encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, text, "")


@pytest.fixture(autouse=True)
def offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Все файлы временные; dotenv и подготовка учётных данных отключены."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(kaggle_run, "load_dotenv", lambda *_args: None)
    monkeypatch.setattr(kaggle_run, "find_dotenv", lambda **_kwargs: "")
    monkeypatch.setattr(kaggle_run, "_prepare_kaggle_cli", lambda: None)
    monkeypatch.setattr(kaggle_run, "prepare_env_file", lambda *_args: None)


def write_config(path: Path, budget: int = 12) -> Path:
    """Минимальный валидный эксперимент."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"data: local/data\nschedule:\n  train_flops: {budget}\ntracker: disabled\n",
        encoding="utf-8",
    )
    return path


def run_config(path: Path, runner: kaggle_run.Runner, slug: str = "batch-run") -> int:
    """Вызвать публичный CLI с fake runner."""
    return kaggle_run.main(
        ["--owner", "user", "--config", str(path), "--slug", slug, "--poll-seconds", "0"],
        run=runner,
    )


def test_directory_runs_yaml_files_in_name_order_to_separate_outputs(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    write_config(configs / "beta.yml", budget=24)
    write_config(configs / "alpha.yaml", budget=12)
    (configs / "notes.txt").write_text("not yaml: [", encoding="utf-8")
    write_config(configs / "nested" / "ignored.yaml")
    runner = FakeRunner()

    assert run_config(configs, runner) == 0

    assert [item["metadata"]["id"] for item in runner.pushes] == [
        "user/batch-run-alpha",
        "user/batch-run-beta",
    ]
    assert [list(upload) for upload in runner.uploads] == [["alpha.yaml"], ["beta.yml"]]
    for name, suffix, budget, upload in zip(
        ["alpha", "beta"], ["yaml", "yml"], [12, 24], runner.uploads, strict=True
    ):
        staging = tmp_path / "kaggle" / f"batch-run-{name}"
        payload = upload[f"{name}.{suffix}"]
        assert payload["data"] == "/kaggle/working/kv_retrieval"
        assert payload["schedule"]["train_flops"] == budget
        kernel = (staging / "kernel.py").read_text("utf-8")
        mount = f"/kaggle/input/datasets/user/compare-attention-wheel/{name}.{suffix}"
        assert f"--config={mount}" in kernel
        assert f"--result-dir=/kaggle/working/batch-run-{name}" in kernel
        assert (staging / "output" / "result.txt").read_text("utf-8") == f"user/batch-run-{name}"
    actions = [command[1:3] for command in runner.commands if command[0] == "kaggle"]
    cycle = [
        ["datasets", "version"],
        ["datasets", "status"],
        ["kernels", "push"],
        ["kernels", "status"],
        ["kernels", "output"],
    ]
    assert actions == cycle * 2


@pytest.mark.parametrize("payload", ["data: [", "", "- not-a-mapping", "data: here\nschedule: {}"])
@pytest.mark.parametrize("batch", [False, True])
def test_invalid_config_is_reported_before_any_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: str, batch: bool
) -> None:
    configs = tmp_path / "configs"
    write_config(configs / "alpha.yaml")
    invalid = configs / "z-invalid.yml"
    invalid.write_text(payload, encoding="utf-8")
    runner = FakeRunner()
    credential_reads = []
    monkeypatch.setattr(kaggle_run, "load_dotenv", lambda *_args: credential_reads.append("dotenv"))
    monkeypatch.setattr(kaggle_run, "_prepare_kaggle_cli", lambda: credential_reads.append("cli"))

    with pytest.raises(SystemExit, match=r"некорректный YAML.*z-invalid.yml"):
        run_config(configs if batch else invalid, runner)

    assert not runner.commands
    assert not credential_reads
    assert not (tmp_path / "kaggle").exists()


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("missing", "путь не существует"),
        ("empty", "нет файлов .yaml/.yml"),
        ("nested", "нет файлов .yaml/.yml"),
    ],
)
def test_missing_configs_are_reported_before_any_action(
    tmp_path: Path, kind: str, message: str
) -> None:
    configs = tmp_path / "configs"
    if kind != "missing":
        configs.mkdir()
        (configs / "notes.txt").write_text("not an experiment", encoding="utf-8")
    if kind == "nested":
        write_config(configs / "nested" / "alpha.yaml")
    runner = FakeRunner()

    with pytest.raises(SystemExit, match=message) as caught:
        run_config(configs, runner)

    assert str(configs) in str(caught.value)
    assert not runner.commands
    assert not (tmp_path / "kaggle").exists()


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Mixed Name_v1.yaml", "batch-run-mixed-name-v1"),
        ("---alpha...beta!.yml", "batch-run-alpha-beta"),
        ("x" * 70 + ".yaml", "batch-run-" + "x" * 40),
    ],
)
def test_batch_slug_is_normalized_and_length_limited(
    tmp_path: Path, filename: str, expected: str
) -> None:
    config = write_config(tmp_path / "configs" / filename)
    runner = FakeRunner()

    assert run_config(config.parent, runner) == 0

    slug = runner.pushes[0]["metadata"]["id"].split("/")[1]
    assert slug == expected
    assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug)
    assert 5 <= len(slug) <= 50


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("same.yaml", "same.yml"),
        ("same_name.yaml", "same-name.yml"),
        ("x" * 60 + "a.yaml", "x" * 60 + "b.yaml"),
    ],
)
def test_slug_collisions_are_reported_before_any_action(
    tmp_path: Path, first: str, second: str
) -> None:
    configs = tmp_path / "configs"
    write_config(configs / first)
    write_config(configs / second)
    runner = FakeRunner()

    with pytest.raises(SystemExit, match="коллизия slug") as caught:
        run_config(configs, runner)

    assert first in str(caught.value)
    assert second in str(caught.value)
    assert not runner.commands
    assert not (tmp_path / "kaggle").exists()


@pytest.mark.parametrize(
    ("base_slug", "filename"),
    [
        ("UPPER-case", "alpha.yaml"),
        ("../escape", "alpha.yaml"),
        ("bad_slug", "alpha.yaml"),
        ("bad--slug", "alpha.yaml"),
        ("x" * 49, "alpha.yaml"),
        ("x", "a.yaml"),
        ("batch-run", "!!!.yaml"),
        ("batch-run", "опыт.yaml"),
    ],
)
def test_invalid_batch_slug_is_reported_before_any_action(
    tmp_path: Path, base_slug: str, filename: str
) -> None:
    config = write_config(tmp_path / "configs" / filename)
    runner = FakeRunner()

    with pytest.raises(SystemExit, match="недопустимый slug"):
        run_config(config.parent, runner, slug=base_slug)

    assert not runner.commands
    assert not (tmp_path / "kaggle").exists()


@pytest.mark.parametrize(
    "failures",
    [
        [("uv", "build", "--wheel")],
        [("uv", "export", "--format")],
        [("kaggle", "datasets", "version"), ("kaggle", "datasets", "create")],
        [("kaggle", "datasets", "status")],
        [("kaggle", "kernels", "push")],
        [("kaggle", "kernels", "status")],
        [("kaggle", "kernels", "output")],
    ],
)
def test_failed_command_stops_before_next_experiment(
    tmp_path: Path, failures: list[tuple[str, ...]]
) -> None:
    configs = tmp_path / "configs"
    write_config(configs / "alpha.yaml")
    write_config(configs / "beta.yml")
    runner = FakeRunner()

    def failing_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        completed = runner(command)
        if tuple(command[:3]) in failures:
            return subprocess.CompletedProcess(command, 1, completed.stdout, "synthetic failure")
        return completed

    code = run_config(configs, failing_run)

    assert code != 0
    assert not (tmp_path / "kaggle" / "batch-run-beta").exists()
    assert all(item["metadata"]["id"] == "user/batch-run-alpha" for item in runner.pushes)


@pytest.mark.parametrize(
    ("raw_status", "status"), [("error", "error"), ("cancelacknowledgement", "cancelled")]
)
def test_unsuccessful_kernel_reports_status_and_stops(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], raw_status: str, status: str
) -> None:
    configs = tmp_path / "configs"
    write_config(configs / "alpha.yaml")
    write_config(configs / "beta.yml")
    runner = FakeRunner()

    def terminal_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        completed = runner(command)
        if command[1:3] == ["kernels", "status"]:
            return subprocess.CompletedProcess(command, 0, raw_status, "")
        return completed

    code = run_config(configs, terminal_run)

    assert code != 0
    error = capsys.readouterr().err
    assert "batch-run-alpha" in error
    assert status in error
    assert not any(command[1:3] == ["kernels", "output"] for command in runner.commands)
    assert not (tmp_path / "kaggle" / "batch-run-beta").exists()


def test_help_describes_directory_mode_without_reading_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("--help не должен читать учётные данные")

    monkeypatch.setattr(kaggle_run, "load_dotenv", forbidden)
    monkeypatch.setattr(kaggle_run, "_prepare_kaggle_cli", forbidden)
    runner = FakeRunner()

    with pytest.raises(SystemExit) as caught:
        kaggle_run.main(["--help"], run=runner)

    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "--config" in help_text
    assert "папка" in help_text
    assert ".yaml/.yml" in help_text
    assert not runner.commands


def test_single_file_keeps_the_requested_slug(tmp_path: Path) -> None:
    """Регрессия: одиночный YAML не получает суффикс имени."""
    config = write_config(tmp_path / "alpha.yaml")
    runner = FakeRunner()

    assert run_config(config, runner, slug="single-run") == 0

    assert [item["metadata"]["id"] for item in runner.pushes] == ["user/single-run"]
    assert (tmp_path / "kaggle" / "single-run" / "output" / "result.txt").is_file()
    assert list(runner.uploads[0]) == ["alpha.yaml"]


def test_flags_mode_keeps_the_requested_slug_and_train_flags() -> None:
    """Регрессия: флаги работают без --config."""
    runner = FakeRunner()

    code = kaggle_run.main(
        ["--owner", "user", "--slug", "flags-run", "--train-flops", "12", "--tracker", "disabled"],
        run=runner,
    )

    assert code == 0
    assert runner.pushes[0]["metadata"]["id"] == "user/flags-run"
    kernel = runner.pushes[0]["code"]
    assert "--train-flops=12" in kernel
    assert "--tracker=disabled" in kernel
    assert "--config=" not in kernel
    assert runner.uploads == [{}]


def test_directory_waits_through_running_statuses_before_next_push(tmp_path: Path) -> None:
    """Регрессия: очередь и running не разрешают следующий эксперимент."""
    configs = tmp_path / "configs"
    write_config(configs / "alpha.yaml")
    write_config(configs / "beta.yml")
    runner = FakeRunner()
    statuses = {
        "user/batch-run-alpha": ["queued", "running", "complete"],
        "user/batch-run-beta": ["running", "complete"],
    }
    downloaded = []

    def waiting_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[1:3] == ["kernels", "push"] and runner.pushes:
            assert downloaded == ["user/batch-run-alpha"]
            assert Path("kaggle/batch-run-alpha/output/result.txt").is_file()
        completed = runner(command)
        if command[1:3] == ["kernels", "status"]:
            return subprocess.CompletedProcess(command, 0, statuses[command[3]].pop(0), "")
        if command[1:3] == ["kernels", "output"]:
            assert not statuses[command[3]]
            downloaded.append(command[3])
        return completed

    assert run_config(configs, waiting_run) == 0

    assert downloaded == ["user/batch-run-alpha", "user/batch-run-beta"]
