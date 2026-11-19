"""Synthetic HTTP / quality metadata only; no real publication authority."""

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_inference_scientific_contract_counts_own_metrics_not_sorting():
    assert importlib.util.find_spec("mqar_final_tail_tracking") is not None, (
        "MQAR service adapter missing"
    )
    from mqar_final_tail_report import quality_metrics
    from mqar_final_tail_tracking import binding, verify_unsampled_events

    metrics = quality_metrics(
        {"loss": 0.3, "accuracy": 0.5, "correct": 32000, "supervised_count": 64000, "rows": 1000}
    )
    rows = [{"metric": "quality", "variant": k, "iter": 0, "value": v} for k, v in metrics.items()]
    got = verify_unsampled_events({"events": rows, "total": len(rows)}, metrics)
    assert got["total"] == len(metrics) and got["unsampled_complete"] is True
    for extra in (
        {"metric": "quality", "variant": "unknown", "iter": 0, "value": 0},
        {"metric": ":monitor:gpu", "variant": "gpu_0_temperature", "iter": 0, "value": 0},
    ):
        with pytest.raises(ValueError):
            verify_unsampled_events({"events": rows + [extra], "total": len(rows) + 1}, metrics)
    bound = binding("1234567890abcdef1234567890abcdef")
    assert bound["original_provider_errors"] == "145778849:ERROR,145850896:ERROR"
    assert bound["optimizer_updates_permitted"] == 0 and bound["query_targets"] == 64000


def test_genuine_old_scientific_resource_separation_and_full_val_integrity():
    import json

    from mqar_final_tail_completion import validate_unsampled_readback
    from mqar_final_tail_data import audit_val_bank
    from mqar_final_tail_test_layout import DATA, INPUT

    raw = json.loads((INPUT / "original/original-clearml-all-events.json").read_bytes())
    final = json.loads((INPUT / "original/out/metrics.jsonl").read_text().splitlines()[-1])
    result = validate_unsampled_readback(
        raw,
        task_id="17a3c249cedc4f37a358ca99f653ac52",
        updates=10780,
        validations=11,
        config=final["config"],
    )
    assert result["event_count"] == 11252
    assert result["scientific_event_count"] == 10812 and result["resource_event_count"] == 440
    assert audit_val_bank(DATA) == {
        "rows": 1000,
        "queries": 64000,
        "model_forwards": 0,
        "scope": "LABEL_INTEGRITY_ONLY_NOT_MODEL_PREDICTIONS",
    }
