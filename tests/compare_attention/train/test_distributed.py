"""Real two-process gloo tracer bullets against the public production train API."""

import json
import os
import subprocess
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.train.backup import BackupTracker
from compare_attention.train.checkpoint import load_checkpoint
from compare_attention.train.distributed import distributed_run
from compare_attention.train.flops import estimate_step
from compare_attention.train.loop import train
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.models.factories import gpt2_config
from tests.compare_attention.train.factories import train_config

ROOT = Path(__file__).resolve().parents[3]
WORKER = Path(__file__).with_name("ddp_worker.py")


def launch(root, *args, module=None):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    allowed = {
        "SYSTEMROOT",
        "WINDIR",
        "PATH",
        "PATHEXT",
        "COMSPEC",
        "OS",
        "USERNAME",
        "USER",
        "LOGNAME",
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "TMPDIR",
        "TMP",
        "TEMP",
        "UV_CACHE_DIR",
        "TORCH_HOME",
        "CLEARML_CONFIG_FILE",
        "CLEARML_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "XDG_CACHE_HOME",
        "RUFF_CACHE_DIR",
        "PYLINTHOME",
        "PYTEST_ADDOPTS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    }
    environment = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    environment.update(
        CUDA_VISIBLE_DEVICES="-1",
        USE_LIBUV="0",
        PYTHONPATH=str(ROOT),
        TORCH_DISABLE_SHARE_RDZV_TCP_STORE="1",
        PYTHON_DOTENV_DISABLED="1",
        CLEARML_OFFLINE_MODE="1",
        HF_HUB_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_REPO="",
        HF_TOKEN="",
    )
    if os.name == "nt":
        environment["GLOO_SOCKET_IFNAME"] = "Loopback Pseudo-Interface 1"
    command = [
        "uv",
        "run",
        "--offline",
        "--no-sync",
        "python",
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--rdzv-backend=c10d",
        "--rdzv-conf=store_type=file",
        f"--rdzv-endpoint={root / uuid.uuid4().hex}",
        "--local-addr=127.0.0.1",
        "--no-python",
        "uv",
        "run",
        "--offline",
        "--no-sync",
        "python",
        *(["-m", module, *args] if module else [str(WORKER), str(root), *args]),
    ]
    with (root / "torchrun.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command, env=environment, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
        ) as process:
            try:
                code = process.wait(timeout=100)
            except subprocess.TimeoutExpired:
                import psutil

                for child in psutil.Process(process.pid).children(recursive=True):
                    child.kill()
                process.kill()
                raise
    assert code == 0, (root / "torchrun.log").read_text(encoding="utf-8", errors="replace")


def fixture(root, attention="classic", dropout=0.0, batch=4):
    data = root / "data"
    prepare_kv_retrieval(
        data_config(vocab_size=32, input_seq_len=16, num_kv_pairs=2, n_train=batch + 3, n_val=3),
        data,
    )
    # Rank zero gets no valid labels on the first update. Later updates wrap the pool.
    labels_path = data / "train_labels.npy"
    labels = np.load(labels_path)
    labels[: batch // 2] = -1
    labels[batch // 2, :-1] = -1
    np.save(labels_path, labels)
    model = gpt2_config(
        vocab_size=32,
        block_size=16,
        n_layer=1,
        n_head=1 if attention == "deltanet" else 2,
        n_embd=16,
        dropout=dropout,
        attention=attention,
    )
    config = train_config(data, model=model, batch_size=batch, val_every_steps=2)
    config = config.model_copy(
        update={
            "schedule": config.schedule.model_copy(
                update={"train_flops": estimate_step(config).total * 4}
            )
        }
    )
    (root / "config.json").write_text(config.model_dump_json(), encoding="utf-8")
    return config


@pytest.mark.parametrize("attention,batch", [("classic", 4), ("classic", 32), ("deltanet", 4)])
def test_torchrun_matches_single_global_updates_with_zero_label_rank_and_wrap(
    tmp_path, attention, batch
):
    if attention == "deltanet":
        pytest.skip(
            "Installed FLA causal_conv1d/recurrent kernels require Triton; CPU-only scope, "
            "no DeltaNet fallback permitted (see extension-delta.txt)"
        )
    config = fixture(tmp_path, attention=attention, batch=batch)
    reference = train(config, BackupTracker(tmp_path / "single"))
    launch(tmp_path)
    actual = json.loads((tmp_path / "rank-0.json").read_text())
    other = json.loads((tmp_path / "rank-1.json").read_text())
    assert actual["steps"] == other["steps"] == reference.steps == 4
    assert actual["train_flops"] == other["train_flops"] == reference.train_flops
    assert (
        actual["provenance"]["initial_parameters_sha256"]
        == reference.provenance["initial_parameters_sha256"]
    )
    traces = [json.loads((tmp_path / f"trace-{rank}.json").read_text()) for rank in range(2)]
    for kind in ("inputs", "labels"):
        source = np.load(tmp_path / f"data/train_{kind}.npy")
        for step in range(4):
            actual_rows = np.array(traces[0][step][kind] + traces[1][step][kind])
            expected_rows = source[
                [(step * batch + offset) % len(source) for offset in range(batch)]
            ]
            np.testing.assert_array_equal(actual_rows, expected_rows)

    def step_losses(directory):
        return [
            json.loads(line)["loss"]
            for line in (directory / "metrics.jsonl").read_text().splitlines()
            if json.loads(line)["type"] == "step"
        ]

    np.testing.assert_allclose(
        step_losses(tmp_path / "distributed"), step_losses(tmp_path / "single"), rtol=0, atol=2e-6
    )
    expected = torch.load(tmp_path / "single/model.pt", weights_only=True)
    observed = torch.load(tmp_path / "distributed/model.pt", weights_only=True)
    assert expected.keys() == observed.keys()
    for name in expected:
        # MHA key bias has analytically zero gradient. FP32 reduction roundoff
        # is amplified by Adam's epsilon; bound only this null-direction slice.
        if name.endswith("attn.layer.in_proj_bias"):
            torch.testing.assert_close(
                observed[name][16:32], expected[name][16:32], rtol=0, atol=5e-5
            )
            for section in (slice(0, 16), slice(32, 48)):
                torch.testing.assert_close(
                    observed[name][section], expected[name][section], rtol=2e-5, atol=2e-6
                )
        else:
            torch.testing.assert_close(observed[name], expected[name], rtol=2e-5, atol=2e-6)
    for field in ("train_loss", "val_loss", "val_accuracy"):
        assert abs(actual["quality"][field] - reference.quality.model_dump()[field]) < 2e-6


def assert_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_equal(a, b)
    else:
        assert left == right


def test_two_rank_dropout_checkpoint_resume_is_bitwise_after_rng_consuming_replay(tmp_path):
    config = fixture(tmp_path, dropout=0.2)
    config = config.model_copy(
        update={
            "schedule": config.schedule.model_copy(
                update={
                    "checkpoint_every_steps": 2,
                    "checkpoint_keep_last": 2,
                    "train_flops": estimate_step(config).total * 7,
                }
            )
        }
    )
    (tmp_path / "config.json").write_text(config.model_dump_json())
    launch(tmp_path, "resume")
    full = load_checkpoint(tmp_path / "full/checkpoints/latest.json")
    resumed = load_checkpoint(tmp_path / "resumed/checkpoints/latest.json")
    assert full["schema_version"] == resumed["schema_version"] == 2
    assert full["distributed"]["topology"]["world_size"] == 2
    assert [row["rank"] for row in full["distributed"]["ranks"]] == [0, 1]
    for field in (
        "model",
        "optimizer",
        "precision",
        "rng",
        "cursor",
        "losses",
        "validation_history",
        "early_stopping",
        "step",
        "flops",
        "distributed",
    ):
        assert_equal(full[field], resumed[field])
    for rank in range(2):
        assert_equal(
            torch.load(tmp_path / f"full-next-{rank}.pt", weights_only=True),
            torch.load(tmp_path / f"resumed-next-{rank}.pt", weights_only=True),
        )
    checkpoints = tmp_path / "full/checkpoints"
    assert sorted(p.name for p in checkpoints.glob("step_*.pt")) == [
        "step_00000006.pt",
        "step_00000007.pt",
    ]
    owned = json.loads((checkpoints / ".owned.json").read_text())
    assert [row["step"] for row in owned] == [4, 6, 7]  # real pre-prune journal


def test_torchrun_public_cli_has_one_canonical_writer_and_explicit_global_cost(tmp_path):
    config = fixture(tmp_path)
    payload = config.model_dump(mode="json", exclude={"dataset"})
    payload.update(tracker="disabled", result_dir=str(tmp_path / "cli"))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(payload))
    launch(tmp_path, "cli")
    assert [(tmp_path / f"cli-exit-{rank}.txt").read_text() for rank in range(2)] == ["0", "0"]
    rows = [json.loads(line) for line in (tmp_path / "cli/metrics.jsonl").read_text().splitlines()]
    steps = [row for row in rows if row["type"] == "step"]
    assert len(steps) == 4
    assert len([row for row in rows if row["type"] == "result"]) == 1
    provenance = rows[-1]["provenance"]
    spec = provenance["distributed"]
    assert spec["global_batch_size"] == 4 and spec["local_batch_size"] == 2
    assert spec["global_step_flops"] == estimate_step(config).total
    assert sum(row["step_flops"] for row in spec["ranks"]) == spec["global_step_flops"]
    assert len({row["initial_parameters_sha256"] for row in spec["ranks"]}) == 1


def test_ddp_uses_active_scaler_and_counts_only_real_updates_on_each_rank(tmp_path):
    fixture(tmp_path)
    launch(tmp_path, "scaler")


@pytest.mark.parametrize("mode", ["nan", "gradient", "zero", "writer", "load"])
def test_failure_is_collective_bounded_and_cleans_up_group(tmp_path, mode):
    fixture(tmp_path)
    launch(tmp_path, f"failure:{mode}")
    assert [(tmp_path / f"{mode}-failed-{rank}.txt").is_file() for rank in range(2)] == [True, True]
    if mode != "writer":
        assert not (tmp_path / mode / "metrics.jsonl").exists()


def test_resume_guard_is_collective_and_precedes_every_tracker_and_write(tmp_path):
    config = fixture(tmp_path)
    config = config.model_copy(
        update={"schedule": config.schedule.model_copy(update={"checkpoint_every_steps": 1})}
    )
    (tmp_path / "config.json").write_text(config.model_dump_json())
    launch(tmp_path, "guards")
    assert [
        json.loads((tmp_path / f"guards-{rank}.json").read_text())["cases"] for rank in range(2)
    ] == [5, 5]


def test_collective_phase_order_mismatch_fails_before_wrong_callbacks_continue(tmp_path):
    fixture(tmp_path)
    launch(tmp_path, "phase")


@pytest.mark.parametrize("mode,budget_steps", [("early", 10), ("budget", 3)])
def test_stopping_and_budget_precedence_agree_on_all_ranks(tmp_path, mode, budget_steps):
    config = fixture(tmp_path)
    config = config.model_copy(
        update={
            "schedule": config.schedule.model_copy(
                update={
                    "checkpoint_every_steps": 1,
                    "train_flops": estimate_step(config).total * budget_steps,
                    "val_every_steps": 1,
                    "early_stopping_patience": 2,
                    "early_stopping_min_steps": 0,
                }
            )
        }
    )
    (tmp_path / "config.json").write_text(config.model_dump_json())
    launch(tmp_path, f"stopping:{mode}")


def test_opt_in_without_torchrun_fails_with_an_actionable_message(monkeypatch):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="torchrun"):
        with distributed_run(True):
            pytest.fail("DDP must not silently become single-device training")
