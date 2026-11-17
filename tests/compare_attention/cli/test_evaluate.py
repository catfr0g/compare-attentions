"""Exercise the real public evaluator, not a mocked evaluation service."""

# Local imports isolate feature discovery and side-effect sentinels in TDD slices.
# pylint: disable=import-outside-toplevel

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from compare_attention.data.config import KVRetrievalConfig
from compare_attention.data.store import _meta_dict
from compare_attention.models.config import GPT2Config
from compare_attention.models.gpt2 import GPT2
from compare_attention.train.identity import parameter_fingerprint


@pytest.fixture(name="artifacts")
def saved_artifacts(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    config = GPT2Config(vocab_size=16, block_size=4, n_layer=1, n_head=1, n_embd=8)
    model = GPT2(config).eval()
    inputs = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 1, 3, 4]], dtype=np.int64)
    with torch.inference_mode():
        predictions = model(torch.from_numpy(inputs)).argmax(-1).numpy()
    labels = np.full_like(inputs, -1)
    labels[0, 1] = predictions[0, 1]
    labels[1, :] = predictions[1, :]
    labels[2, 3] = predictions[2, 3]
    np.save(data / "test_inputs.npy", inputs)
    np.save(data / "test_labels.npy", labels)
    np.save(data / "val_inputs.npy", inputs)
    np.save(data / "val_labels.npy", (predictions + 1) % 16)
    meta = _meta_dict(
        KVRetrievalConfig(
            vocab_size=16, input_seq_len=4, num_kv_pairs=1, n_train=3, n_val=3, n_test=3
        )
    )
    (data / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    weights = tmp_path / "model.pt"
    torch.save(model.state_dict(), weights)
    identity = {
        "config": {"model": config.model_dump(), "schedule": {"dtype": "float16"}},
        "attention": "classic",
        "final_parameters_sha256": parameter_fingerprint(model),
    }
    (tmp_path / "run_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    return weights, data, tmp_path / "report.json"


def command(artifacts, *extra):
    weights, data, _ = artifacts
    return [
        "uv",
        "run",
        "--offline",
        "--no-sync",
        "python",
        "-B",
        "-m",
        "compare_attention.cli.evaluate",
        "--checkpoint",
        str(weights),
        "--data",
        str(data),
        "--device",
        "cpu",
        "--batch-size",
        "2",
        *extra,
    ]


def test_real_module_defaults_test_and_read_only(artifacts):
    weights, data, output = artifacts
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in [weights, *data.iterdir()]
    }
    result = subprocess.run(
        command(artifacts, "--output", str(output)), capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report == json.loads(result.stdout)
    assert report["split"] == "test"
    entrypoint = Path(sys.executable).parent / ("eval.exe" if os.name == "nt" else "eval")
    installed = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "--no-sync",
            str(entrypoint),
            *command(artifacts)[8:],
            "--split",
            "val",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert installed.returncode == 0, installed.stderr
    assert json.loads(installed.stdout)["accuracy"] == 0
    assert report["accuracy"] == 1
    assert report["supervised_count"] == 6
    assert report["rows"] == 3
    assert report["inference_dtype"] == "float32"
    assert report["device"] == "cpu"
    assert report["model_config"]["attention"] == "classic"
    assert report["checkpoint"]["sha256"] == before[weights]
    assert report["runtime"]["torch"] == str(torch.__version__)
    assert set(report["data"]["sha256"]) == {"meta.json", "test_inputs.npy", "test_labels.npy"}
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}


@pytest.mark.parametrize(
    "fault",
    [
        "key",
        "shape",
        "dtype",
        "alias",
        "nonfinite",
        "fingerprint",
        "incomplete_config",
        "attention",
    ],
)
def test_loader_rejects_corruption_before_inference(artifacts, fault):
    from compare_attention.evaluation import evaluate_checkpoint

    weights, data, _ = artifacts
    state = torch.load(weights, weights_only=True, map_location="cpu")
    identity_path = weights.parent / "run_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if fault == "key":
        state.pop("wpe.weight")
    elif fault == "shape":
        state["wpe.weight"] = state["wpe.weight"][:1]
    elif fault == "dtype":
        state = {key: value.double() for key, value in state.items()}
    elif fault == "alias":
        state["lm_head.weight"] = state["lm_head.weight"].clone() + 1
    elif fault == "nonfinite":
        state["wpe.weight"][0, 0] = float("nan")
    elif fault == "fingerprint":
        identity["final_parameters_sha256"] = "0" * 64
    elif fault == "incomplete_config":
        identity["config"]["model"].pop("dropout")
    else:
        identity["attention"] = "deltanet"
    torch.save(state, weights)
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_checkpoint(weights, data, device="cpu")


def test_standalone_yaml_and_conflicting_explicit_model(artifacts):
    from compare_attention.evaluation import evaluate_checkpoint

    weights, data, _ = artifacts
    identity_path = weights.parent / "run_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    config = weights.parent / "training.yaml"
    config.write_text(yaml.safe_dump(identity["config"]), encoding="utf-8")
    assert evaluate_checkpoint(weights, data, config=config, device="cpu")["accuracy"] == 1
    identity_path.unlink()
    assert evaluate_checkpoint(weights, data, config=config, device="cpu")["accuracy"] == 1
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    identity["config"]["model"]["dropout"] = 0.1
    config.write_text(yaml.safe_dump(identity["config"]), encoding="utf-8")
    with pytest.raises(ValueError, match="config"):
        evaluate_checkpoint(weights, data, config=config, device="cpu")


@pytest.mark.parametrize("fault", ["count", "sequence", "vocabulary"])
def test_selected_split_matches_metadata(artifacts, fault):
    from compare_attention.evaluation import evaluate_checkpoint

    weights, data, _ = artifacts
    path = data / "meta.json"
    meta = json.loads(path.read_text())
    key, value = {
        "count": ("n_test", 4),
        "sequence": ("input_seq_len", 6),
        "vocabulary": ("vocab_size", 32),
    }[fault]
    meta[key] = value
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset|split|metadata"):
        evaluate_checkpoint(weights, data, device="cpu")


def test_existing_output_rejected_before_evaluation(artifacts, monkeypatch):
    import compare_attention.cli.evaluate as cli

    weights, data, output = artifacts
    output.write_text("keep me", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval", "--checkpoint", str(weights), "--data", str(data), "--output", str(output)],
    )
    monkeypatch.setattr(cli, "evaluate_checkpoint", lambda **_kw: pytest.fail("evaluation began"))
    with pytest.raises(FileExistsError):
        cli.main()
    assert output.read_text() == "keep me"


def test_no_training_generation_or_dotenv_effects(artifacts, monkeypatch):
    import dotenv

    from compare_attention.data import store
    from compare_attention.evaluation import evaluate_checkpoint

    def forbidden(*_args, **_kwargs):
        pytest.fail("training, generation or credentials side effect")

    monkeypatch.setattr(store, "prepare_kv_retrieval", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    weights, data, _ = artifacts
    assert evaluate_checkpoint(weights, data, device="cpu")["correct"] == 6


# A real serialized resumable checkpoint is deliberately constructed end-to-end.
# pylint: disable-next=too-many-locals
def test_full_checkpoint_and_latest_do_not_restore_training_state(artifacts, monkeypatch):
    import compare_attention.train.checkpoint as checkpoint_module
    from compare_attention.evaluation import evaluate_checkpoint
    from compare_attention.train.checkpoint import capture_state
    from compare_attention.train.config import Schedule, TrainConfig

    weights, data, _ = artifacts
    raw_identity = json.loads((weights.parent / "run_identity.json").read_text())
    model = GPT2(GPT2Config.model_validate(raw_identity["config"]["model"]))
    model.load_state_dict(torch.load(weights, weights_only=True))
    config = TrainConfig(
        data=data,
        model=model.config,
        dataset=KVRetrievalConfig.model_validate(json.loads((data / "meta.json").read_text())),
        schedule=Schedule(train_flops=100, batch_size=2),
    )
    identity = dict(raw_identity, config=config.model_dump(mode="json"))
    optimizer = torch.optim.AdamW(model.parameters())
    payload = capture_state(
        model,
        optimizer,
        config,
        step=0,
        cursor=0,
        flops=0,
        losses=[],
        elapsed=0.0,
        early_stopping={},
        identity=identity,
        precision=None,
    )
    full = weights.parent / "step_00000000.pt"
    torch.save(payload, full)
    pointer = full.parent / "latest.json"
    pointer.write_text(
        json.dumps(
            {"step": 0, "path": full.name, "sha256": hashlib.sha256(full.read_bytes()).hexdigest()}
        )
    )
    rng_before = torch.get_rng_state().clone()
    monkeypatch.setattr(torch.optim.AdamW, "__init__", lambda *_a, **_kw: pytest.fail("optimizer"))
    monkeypatch.setattr(
        checkpoint_module, "restore_state", lambda *_a, **_kw: pytest.fail("resume")
    )
    monkeypatch.setattr(
        checkpoint_module, "restore_rng", lambda *_a, **_kw: pytest.fail("RNG restored")
    )
    for source in (full, pointer):
        result = evaluate_checkpoint(source, data, device="cpu")
        assert result["accuracy"] == 1
        assert result["checkpoint"]["format"] == "training_checkpoint"
        assert result["checkpoint"]["step"] == 0
    assert torch.equal(torch.get_rng_state(), rng_before)
    with full.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        evaluate_checkpoint(pointer, data, device="cpu")
