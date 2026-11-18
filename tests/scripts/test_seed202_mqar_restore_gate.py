"""Synthetic read-only equality gate; no genuine model or optimizer restoration."""

import pytest


@pytest.mark.parametrize("rank", [0, 1])
def test_profile_return_calls_native_local_call_before_any_update(tmp_path, rank, monkeypatch):
    """Engineering control has no model, optimizer, RNG or training operation."""
    import torch
    from seed202_mqar_restore_gate import observe_return

    from compare_attention.train.distributed import _ACTIVE, DistributedRun, dist, local_call

    events = []

    class Control:
        def resume(self):
            events.append("history-replayed")
            return None

    def step():
        raise AssertionError("NO_OPTIMIZER_IN_METADATA_TRACE")

    def capture(control):
        assert isinstance(control, Control)
        events.append("capture-after-return")
        return {"step": 980}

    def equality(expected, actual):
        assert expected == actual == {"step": 980}
        events.append("equal")

    def all_gather(headers, header):
        assert header.tolist() == [0, 0]
        for peer in headers:
            peer.copy_(header)
        events.append(("native-local-call-header", rank))

    def all_gather_object(rows, value):
        assert value == ("mqar_restore_equality_before_981", None)
        rows[:] = [value, value]
        events.append(("native-local-call-gather", rank))

    monkeypatch.setattr(dist, "all_gather", all_gather)
    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)

    receipts = []
    token = _ACTIVE.set(DistributedRun(rank, 2, rank, torch.device("cpu")))
    try:
        with observe_return(
            {"step": 980},
            tmp_path,
            rank,
            resume=Control.resume,
            restore=Control.resume,
            step=step,
            capture=capture,
            collective=local_call,
            compare=equality,
            write=lambda path, value: receipts.append((path, value)),
        ):
            Control().resume()
    finally:
        _ACTIVE.reset(token)
    assert events == [
        "history-replayed",
        "capture-after-return",
        "equal",
        ("native-local-call-header", rank),
        ("native-local-call-gather", rank),
    ]
    assert receipts[0][0] == tmp_path / f"restore-equality-rank-{rank}.json"
    assert receipts[0][1]["rank"] == rank


def test_exception_unwind_is_not_mistaken_for_successful_native_resume_return():
    import dis
    from types import SimpleNamespace

    from seed202_mqar_restore_gate import normal_return

    def successful():
        return None

    def failing():
        raise ValueError("ENGINEERING_ONLY")

    for function in (successful, failing):
        for instruction in dis.get_instructions(function):
            if instruction.opname.startswith("RETURN") or instruction.opname == "RAISE_VARARGS":
                frame = SimpleNamespace(f_code=function.__code__, f_lasti=instruction.offset)
                assert normal_return(frame) is instruction.opname.startswith("RETURN")
    import sys

    previous = sys.getprofile()
    observed = []

    def observer(frame, event, _argument):
        if frame.f_code in {successful.__code__, failing.__code__} and event == "return":
            observed.append((frame.f_code.co_name, normal_return(frame)))

    sys.setprofile(observer)
    try:
        successful()
        try:
            failing()
        except ValueError:
            pass
    finally:
        sys.setprofile(previous)
    assert observed == [("successful", True), ("failing", False)]


def test_read_only_gate_rejects_changed_optimizer_and_rng():
    from seed202_mqar_restore_gate import compare_snapshot

    expected = {
        "model": {"weight": [1.0]},
        "optimizer": {"step": 980},
        "rng": {"torch": [1, 2]},
        "step": 980,
        "cursor": 62720,
        "flops": 454579338608640,
        "losses": [0.2],
        "elapsed": 1.0,
        "early_stopping": {},
        "validation_history": [],
        "config": {},
        "model_structure": {},
        "optimizer_parameters": [],
        "optimizer_class": "engineering",
        "precision": {"scale": 128},
    }
    compare_snapshot(expected, dict(expected), digest=lambda value: repr(value))
    for key in ("optimizer", "rng", "step", "cursor", "precision"):
        with pytest.raises(ValueError, match="RESTORE_EQUALITY_" + key.upper()):
            compare_snapshot(expected, {**expected, key: None}, digest=lambda value: repr(value))
