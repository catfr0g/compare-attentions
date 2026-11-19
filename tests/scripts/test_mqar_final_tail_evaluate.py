"""Fresh tiny CPU Classic model: loader/evaluator engineering, never science."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_public_parser_loader_records_single_forward_and_val_tail(tmp_path, monkeypatch):
    spec = importlib.util.find_spec("mqar_final_tail_evaluate")
    assert spec is not None, "MQAR public-evaluator record adapter is missing"
    from mqar_final_tail_evaluate import evaluate_public

    from compare_attention.data.config import KVRetrievalConfig
    from compare_attention.models.config import GPT2Config
    from compare_attention.models.gpt2 import GPT2

    # Fresh engineering state only; deliberately distinct VAL and TEST targets.
    config = GPT2Config(vocab_size=16, block_size=8, n_layer=1, n_head=1, n_embd=8)
    model = GPT2(config)
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    torch.save(model.state_dict(), standalone / "model.pt")
    (standalone / "model.json").write_text(json.dumps({"model": config.model_dump(mode="json")}))
    data = tmp_path / "data"
    data.mkdir()
    dataset = KVRetrievalConfig(
        vocab_size=16, input_seq_len=8, num_kv_pairs=1, n_train=3, n_val=3, n_test=3
    )
    from compare_attention.data.store import _meta_dict

    (data / "meta.json").write_text(json.dumps(_meta_dict(dataset)))
    x = np.array([[1, 8, 2, 3, 4, 1, 5, 6]] * 3, dtype=np.int64)
    y = np.array([[-1, -1, -1, -1, -1, 8, -1, -1]] * 3, dtype=np.int64)
    for split in ("val", "test"):
        np.save(data / f"{split}_inputs.npy", x)
        np.save(data / f"{split}_labels.npy", y if split == "val" else np.where(y == 8, 9, y))
    output = tmp_path / "report.json"
    args = [
        "--checkpoint",
        str(standalone / "model.pt"),
        "--config",
        str(standalone / "model.json"),
        "--data",
        str(data),
        "--split",
        "val",
        "--batch-size",
        "2",
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--output",
        str(output),
    ]
    report, records = evaluate_public(
        args, data=data, split="val", records=tmp_path / "records.json"
    )
    assert report["rows"] == len(records) == 3 and report["supervised_count"] == 3
    assert report["checkpoint"]["provenance"] is None
    assert [r["row"] for r in records] == [0, 1, 2]
    assert all(r["targets"] == [8] and r["positions"] == [5] for r in records)
    assert not any(k.startswith("generated") for k in report)
    with pytest.raises(FileExistsError):
        evaluate_public(args, data=data, split="val", records=tmp_path / "records.json")
