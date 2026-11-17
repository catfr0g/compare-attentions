"""CUDA subprocess exercises real CLI, budgets and persisted compiler evidence."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
import yaml

from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.train.flops import estimate_step
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.models.factories import gpt2_config
from tests.compare_attention.train.factories import train_config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires local CUDA")
@pytest.mark.parametrize("attention", ["classic", "deltanet"])
@pytest.mark.parametrize("steps", [1, 3])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_cuda_cli_budget_and_compiler_artifacts(
    tmp_path: Path, attention: str, steps: int, dtype: str
) -> None:
    # pylint: disable=too-many-locals
    data = tmp_path / "data"
    prepare_kv_retrieval(
        data_config(
            vocab_size=256, input_seq_len=128, num_kv_pairs=2, n_train=4, n_val=3, n_test=1
        ),
        data,
    )
    config = train_config(
        data,
        model=gpt2_config(
            vocab_size=256, block_size=128, n_layer=1, n_embd=32, n_head=2, attention=attention
        ),
        dtype=dtype,
        compile=True,
        seed=42,
        val_every_steps=1,
    )
    cost = estimate_step(config)
    budget = cost.total if steps == 1 else cost.total * 2 + 1
    payload = config.model_dump(mode="json", exclude={"dataset"})
    payload["schedule"]["train_flops"] = budget
    payload.update(tracker="disabled", result_dir=str(tmp_path / "result"))
    experiment = tmp_path / "tiny.yaml"
    experiment.write_text(yaml.safe_dump(payload), encoding="utf-8")
    environment = {
        **os.environ,
        "HF_REPO": "",
        "HF_TOKEN": "",
        "HF_HUB_OFFLINE": "1",
        "PYTHON_DOTENV_DISABLED": "1",
        "CLEARML_OFFLINE_MODE": "1",
    }
    process = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "--no-sync",
            "python",
            "-B",
            "-m",
            "compare_attention.cli.train",
            "--config",
            str(experiment),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    target = tmp_path / "result"
    identity = json.loads((target / "run_identity.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (target / "metrics.jsonl").read_text().splitlines()]
    result = events[-1]
    assert result["type"] == "result"
    assert result["steps"] == steps
    assert [event["type"] for event in events[:-1]] == ["step", "validation"] * steps
    assert [event["flops"] for event in events if event["type"] == "step"] == [
        cost.total * index for index in range(1, steps + 1)
    ]
    assert [event["flops"] for event in events if event["type"] == "validation"] == [
        cost.total * index for index in range(1, steps + 1)
    ]
    assert result["train_flops"] == cost.total * steps
    assert 0 <= result["train_flops"] - budget < cost.total
    assert identity["status"] == "completed"
    assert identity["effective_seed"] == identity["requested_seed"] == 42
    assert identity["accounting"] == cost.to_dict()
    assert identity == result["provenance"]
    assert identity["initial_parameters_sha256"] != identity["final_parameters_sha256"]
    if dtype == "float16":
        assert identity["precision"] == {
            "mode": "amp_float16",
            "parameter_dtype": "float32",
            "autocast_dtype": "float16",
            "grad_scaler_enabled": True,
            "optimizer_steps": steps,
            "skipped_optimizer_steps": 0,
            "initial_scale": 128.0,
            "final_scale": 128.0,
        }
        weights = torch.load(target / "model.pt", weights_only=True)
        assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in weights.values())
    compiler = identity["compiler"]
    assert compiler["requested"] is True
    assert compiler["backend"] == "inductor"
    assert compiler["training_backend_invocations"] > 0
    assert compiler["validation_backend_invocations"] > 0
    assert compiler["backend_invocations"] == compiler["successful_backend_invocations"]
    assert compiler["hidden_warmup_steps"] == 0
    print(
        json.dumps(
            {
                "attention": attention,
                "steps": steps,
                "budget": budget,
                "train_flops": result["train_flops"],
                "compiler": compiler,
                "artifact_dir": str(target),
            }
        )
    )
