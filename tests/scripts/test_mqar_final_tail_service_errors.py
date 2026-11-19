"""Original service exception survives diagnostic I/O failure; no secret text retained."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_diagnostic_failure_never_masks_original_service_exception(tmp_path, monkeypatch):
    import mqar_final_tail_service_errors as errors

    def fail(*_args):
        raise OSError("SYNTHETIC_IO_FAILURE")

    monkeypatch.setattr(errors, "persist", fail)
    service = errors.instrument(SimpleNamespace(invoke=lambda _phase, call: call()), tmp_path)
    with pytest.raises(KeyError, match="SYNTHETIC_PRIMARY"):
        service.invoke(
            "recovery_start", lambda: (_ for _ in ()).throw(KeyError("SYNTHETIC_PRIMARY"))
        )
