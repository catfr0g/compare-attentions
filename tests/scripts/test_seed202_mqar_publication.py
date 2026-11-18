"""Genuine first callback through native saver and real HF SDK HTTP encoding."""

import copy
import importlib
import sys

import pytest
from test_seed202_mqar_builder import HISTORY
from test_seed202_mqar_checkpoint import DOWNLOAD, WHEEL, WORK, native_case


@pytest.mark.parametrize("corrupt,later", [(False, False), (True, False), (False, True)])
def test_genuine_full_first_callback_sdk_http_readback(tmp_path, monkeypatch, corrupt, later):
    from build_seed202_mqar_continuation import stage_science_helpers, stage_source
    from seed202_mqar_http_fixture import HFHTTPFixture

    helpers = stage_science_helpers(WORK / "parent", tmp_path / "science")
    source = stage_source(WORK / "parent", HISTORY, tmp_path / "source")
    old = {k: v for k, v in sys.modules.items() if k.startswith("seed202_pilot")}
    for name in old:
        del sys.modules[name]
    paths = [str(WHEEL), str(source), str(helpers / "operator")]
    sys.path[:0] = paths
    try:
        import torch
        from quality_recovery import RecoveryPublisher
        from quality_services import HFTransport
        from quality_tracker import QualityTracker

        native = importlib.import_module("seed202_pilot.mqar_host_native")

        (full,) = DOWNLOAD.rglob("step_00000980.pt")
        payload = torch.load(full, weights_only=True, map_location="cpu")
        calls = []

        def metadata_only_validate(saved, _case, *, step):
            native.validate_full(saved, native_case(saved), step=step)
            calls.append("AUTHENTIC_FULL_METADATA_ONLY_NOT_TARGET_RESTORE")

        fixture = HFHTTPFixture(monkeypatch, corrupt=corrupt)
        publisher = RecoveryPublisher(
            tmp_path / "recovery.jsonl",
            "engineering/mqar/recovery",
            HFTransport(
                {"HF_TOKEN": "hf_SYNTHETIC_ENGINEERING_ONLY", "HF_REPO": "engineering/mqar"}
            ),
        )
        tracker = QualityTracker(
            tmp_path / "out",
            {"checkpoint_steps": list(range(980, 10781, 980))},
            publisher,
            validate=metadata_only_validate,
        )
        tracker.start(None)
        if corrupt:
            with pytest.raises(RuntimeError, match="RECOVERY_PUBLICATION_UNKNOWN_STOP"):
                tracker.save_checkpoint(payload, 980, 2)
            assert publisher.records()[-1]["state"] == "unknown"
        else:
            tracker.save_checkpoint(payload, 980, 2)
            if later:
                from compare_attention.train.checkpoint import _digest

                # ENGINEERING ONLY: no optimizer was constructed or executed.
                for step in (1960, 2940):
                    synthetic = copy.deepcopy(payload)
                    synthetic.update(
                        step=step,
                        flops=step * 463856467968,
                        cursor=step * 64 % 100000,
                        losses=payload["losses"] * (step // 980),
                    )
                    synthetic["validation_history"] = [
                        {**payload["validation_history"][0], "flops": point * 463856467968}
                        for point in range(980, step + 1, 980)
                    ]
                    from compare_attention.train.stopping import EarlyStopping

                    schedule = payload["config"]["schedule"]
                    stopper = EarlyStopping(
                        schedule["early_stopping_patience"],
                        schedule["early_stopping_min_delta"],
                        schedule["early_stopping_min_steps"],
                    )
                    for event in synthetic["validation_history"]:
                        stopper.observe(event["loss"], event["flops"] // 463856467968)
                    synthetic["early_stopping"] = stopper.state_dict()
                    for state in synthetic["optimizer"]["state"].values():
                        state["step"].fill_(step)
                    synthetic["precision"]["optimizer_steps"] = step
                    synthetic["precision"]["scaler"]["_growth_tracker"] = step % 2000
                    synthetic["integrity_sha256"] = _digest(synthetic)
                    tracker.save_checkpoint(synthetic, step, 2)
                assert not (tmp_path / "out/checkpoints/step_00000980.pt").exists()
            expected_steps = [980, 1960, 2940] if later else [980]
            assert publisher.complete(expected_steps)[0]["state"] == "verified"
        assert calls and set(calls) == {"AUTHENTIC_FULL_METADATA_ONLY_NOT_TARGET_RESTORE"}
        assert fixture.commits == (3 if later else 1)
        assert fixture.encoded_files > 0
    finally:
        for path in paths:
            sys.path.remove(path)
        for name in list(sys.modules):
            if name.startswith("seed202_pilot"):
                del sys.modules[name]
        sys.modules.update(old)
