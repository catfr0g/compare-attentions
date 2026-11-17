"""CUDA AMP keeps FP32 masters and records real, not attempted, updates."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from compare_attention.data.config import KVRetrievalConfig
from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.models.config import GPT2Config
from compare_attention.train.backup import BackupTracker
from compare_attention.train.config import Schedule, TrainConfig
from compare_attention.train.flops import estimate_step
from compare_attention.train.loop import train
from compare_attention.train.precision import Precision
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.models.factories import gpt2_config
from tests.compare_attention.train.factories import train_config


def test_precision_module_exists() -> None:
    assert importlib.util.find_spec("compare_attention.train.precision") is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA AMP")
def test_amp_step_keeps_fp32_masters_and_counts_real_update() -> None:

    model = torch.nn.Linear(4, 2).cuda()
    initial = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    precision = Precision(torch.device("cuda"))
    with precision.autocast():
        output = model(torch.ones(3, 4, device="cuda"))
        assert output.dtype == torch.float16
        loss = output.float().square().mean()
    precision.backward_step(loss, optimizer)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.parameters())
    assert not torch.equal(initial, model.weight)
    details = precision.metadata()
    assert set(details) == {
        "mode",
        "parameter_dtype",
        "autocast_dtype",
        "grad_scaler_enabled",
        "optimizer_steps",
        "skipped_optimizer_steps",
        "initial_scale",
        "final_scale",
    }
    assert details["mode"] == "amp_float16"
    assert details["parameter_dtype"] == "float32"
    assert details["autocast_dtype"] == "float16"
    assert details["grad_scaler_enabled"] is True
    assert details["optimizer_steps"] == 1
    assert details["skipped_optimizer_steps"] == 0
    assert details["initial_scale"] == details["final_scale"] == 128.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA AMP")
@pytest.mark.parametrize("attention", ["classic", "deltanet"])
def test_training_persists_fp32_master_amp_provenance(tmp_path, attention) -> None:

    prepare_kv_retrieval(
        data_config(
            vocab_size=256, input_seq_len=128, num_kv_pairs=2, n_train=4, n_val=3, n_test=1
        ),
        tmp_path / "data",
    )
    config = train_config(
        tmp_path / "data",
        dtype="float16",
        model=gpt2_config(
            vocab_size=256, block_size=128, n_layer=1, n_embd=32, n_head=2, attention=attention
        ),
    )
    result = train(config, BackupTracker(tmp_path / "result"))
    precision = result.provenance["precision"]
    assert precision["optimizer_steps"] == result.steps == 1
    assert precision["skipped_optimizer_steps"] == 0
    assert precision["parameter_dtype"] == "float32"
    assert precision["initial_scale"] == precision["final_scale"] == 128.0
    identity = json.loads((tmp_path / "result" / "run_identity.json").read_text())
    assert identity == result.provenance
    assert identity["initial_parameters_sha256"] != identity["final_parameters_sha256"]
    weights = torch.load(tmp_path / "result" / "model.pt", weights_only=True)
    assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in weights.values())


def test_schedule_accepts_float16_as_compute_dtype() -> None:
    schedule = Schedule(train_flops=1, dtype="float16")
    assert schedule.torch_dtype() == torch.float16


@pytest.mark.parametrize("attention", ["classic", "deltanet"])
def test_fp16_price_matches_bfloat16_without_rng_mutation(attention: str) -> None:

    config = TrainConfig.model_construct(
        data=Path("unused"),
        model=GPT2Config(attention=attention),
        dataset=KVRetrievalConfig(),
        schedule=Schedule(train_flops=1, dtype="float16"),
    )
    before = torch.random.get_rng_state().clone()
    cost = estimate_step(config)
    other = config.model_copy(
        update={"schedule": config.schedule.model_copy(update={"dtype": "bfloat16"})}
    )
    assert cost == estimate_step(other)
    assert cost.total == {"classic": 2473901162496, "deltanet": 2399573901312}[attention]
    assert torch.equal(before, torch.random.get_rng_state())
