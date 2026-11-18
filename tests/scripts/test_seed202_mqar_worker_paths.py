"""Fail-closed deployment data binding and unchanged native equality negatives."""

import copy
import os
from pathlib import Path

import pytest
from test_seed202_mqar_worker_mount import DATA, ROOT, populate


@pytest.mark.parametrize("corruption", ["foreign", "ambiguous", "hash", "source"])
def test_worker_mount_rejects_unbound_inputs(tmp_path, monkeypatch, corruption):
    from seed202_pilot import cloud_contract, mqar_worker_paths
    from seed202_pilot.contract import MQAR_DATA

    mount = tmp_path / "input"
    slug = MQAR_DATA["source_ref"].split("/")[1]
    physical = populate(mount, slug, "flat")
    original = ROOT / "genuine-resume/original"
    candidate = Path(os.environ["MQAR_TEST_CANDIDATE"])
    if corruption == "foreign":
        physical = tmp_path / "foreign"
    elif corruption == "ambiguous":
        populate(mount, slug, "owner")
    elif corruption == "source":
        local = tmp_path / "resume"
        local.mkdir()
        (local / "original-kernel.py").write_bytes(b"# foreign source\n")
        original = local / "original"
    else:
        # Actual verifier on incorrect physical bytes, not a patched success.
        assert callable(cloud_contract._data)
        (physical / "data-manifest.json").write_bytes(b"corrupt\n")
    with pytest.raises(ValueError):
        mqar_worker_paths.bind_case(
            candidate / "payload/configs/mqar-k64-classic-seed202.yaml",
            physical,
            original,
            mount_root=mount,
        )


@pytest.mark.parametrize(
    "component,field,value",
    [
        ("schedule", "seed", 303),
        ("schedule", "train_flops", 6000000000000000),
        ("schedule", "batch_size", 32),
        ("schedule", "dtype", "float32"),
        ("schedule", "compile", False),
        ("schedule", "amp_max_scale", 256.0),
        ("model", "attention", "deltanet"),
        ("model", "n_layer", 3),
        ("model", "vocab_size", 4096),
        ("model", "block_size", 1024),
        ("optim", "learning_rate", 0.002),
        ("optim", "lr_schedule", "warmup_cosine"),
    ],
)
def test_worker_mount_rejects_every_foreign_recipe(tmp_path, monkeypatch, component, field, value):
    import json

    from seed202_pilot import cloud_contract, mqar_worker_paths
    from seed202_pilot.contract import MQAR_DATA

    mount = tmp_path / "input"
    physical = populate(mount, MQAR_DATA["source_ref"].split("/")[1], "flat")
    candidate = Path(os.environ["MQAR_TEST_CANDIDATE"])
    source = candidate / "payload/configs/mqar-k64-classic-seed202.yaml"
    raw = json.loads(source.read_bytes())
    raw[component][field] = value
    config = tmp_path / source.name
    config.write_text(json.dumps(raw))
    real_data = cloud_contract._data
    monkeypatch.setattr(cloud_contract, "_data", lambda spec, _root: real_data(spec, DATA))
    with pytest.raises(ValueError, match="UNAPPROVED_SEED202_CONFIG_BYTES"):
        mqar_worker_paths.bind_case(
            config, physical, ROOT / "genuine-resume/original", mount_root=mount
        )


@pytest.mark.parametrize("field", ["data", "seed", "K", "length", "vocab", "budget", "precision"])
def test_unchanged_native_preflight_rejects_config_drift(monkeypatch, field):
    from types import SimpleNamespace

    from seed202_pilot.mqar_worker_paths import FULL_SHA, SOURCE_SHA, training_config

    from compare_attention.train import checkpoint, distributed, distributed_checkpoint

    path = ROOT / "genuine-resume/original/out/checkpoints/step_00000980.pt"
    payload = checkpoint.load_checkpoint(path)
    case = {
        "config": payload["config"],
        "worker_data_binding": {
            "field": "config.data",
            "identity": payload["config"]["data"],
            "physical": str(DATA),
            "original_source_sha256": SOURCE_SHA,
            "original_full_sha256": FULL_SHA,
        },
    }
    config = training_config(case)
    if field == "data":
        config = config.model_copy(update={"data": Path("/kaggle/input/foreign")})
    elif field in {"K", "length", "vocab"}:
        key, value = {
            "K": ("num_kv_pairs", 63),
            "length": ("input_seq_len", 256),
            "vocab": ("vocab_size", 4096),
        }[field]
        config = config.model_copy(
            update={"dataset": config.dataset.model_copy(update={key: value})}
        )
    else:
        key, value = {
            "seed": ("seed", 303),
            "budget": ("train_flops", 6000000000000000),
            "precision": ("dtype", "float32"),
        }[field]
        config = config.model_copy(
            update={"schedule": config.schedule.model_copy(update={key: value})}
        )
    run = SimpleNamespace(
        world_size=2, device=SimpleNamespace(type="cuda"), batch=lambda _batch: 32
    )
    token = distributed._ACTIVE.set(run)
    try:
        with pytest.raises(ValueError, match="DDP checkpoint config mismatch"):
            distributed_checkpoint.preflight(path, config)
    finally:
        distributed._ACTIVE.reset(token)


def test_worker_binding_path_protocols_and_private_native_code(tmp_path, monkeypatch):
    from seed202_pilot import cloud_contract, mqar_worker_paths
    from seed202_pilot.contract import MQAR_DATA

    from compare_attention.train import loop

    mount = tmp_path / "input"
    physical = populate(mount, MQAR_DATA["source_ref"].split("/")[1], "flat")
    real_data = cloud_contract._data
    monkeypatch.setattr(cloud_contract, "_data", lambda spec, _root: real_data(spec, DATA))
    candidate = Path(os.environ["MQAR_TEST_CANDIDATE"])
    case = mqar_worker_paths.bind_case(
        candidate / "payload/configs/mqar-k64-classic-seed202.yaml",
        physical,
        ROOT / "genuine-resume/original",
        mount_root=mount,
    )
    config = mqar_worker_paths.training_config(case)
    canonical = "/kaggle/input/datasets/zimike/ca-kv-1c660aa5f54d88ea9e0c1ce4"
    assert str(config.data) == os.fspath(config.data) == canonical
    assert Path(config.data) == Path(canonical)
    assert config.model_dump(mode="json") == case["config"]
    calls = []
    monkeypatch.setattr(loop, "load_split", lambda path, split: calls.append((path, split)))
    before = dict(vars(loop))
    train = mqar_worker_paths.bound_train(case)
    scope = train.__globals__
    for name in ("train", "_train", "_load"):
        assert scope[name].__code__ is vars(loop)[name].__code__
    for path in (config.data, str(config.data), os.fspath(config.data), Path(config.data)):
        scope["load_split"](path, "train")
    assert calls == [(physical, "train")] * 4
    with pytest.raises(ValueError, match="WORKER_SPLIT_IDENTITY"):
        scope["load_split"](physical, "train")
    assert vars(loop) == before
    altered = copy.deepcopy(case)
    altered["config"]["dataset"]["num_kv_pairs"] = 63
    with pytest.raises(ValueError, match="WORKER_DATA_CONFIG_DRIFT"):
        mqar_worker_paths.training_config(altered)
