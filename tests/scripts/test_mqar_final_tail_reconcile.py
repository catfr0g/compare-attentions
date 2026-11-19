"""Public final-HF unknown ACK reconciliation must never publish a fresh action."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_public_hf_reconcile_cannot_create_a_new_journal(tmp_path):
    import mqar_final_tail_service_cli

    assert hasattr(mqar_final_tail_service_cli, "reconcile_hf"), "public HF GET-only bridge missing"
    with pytest.raises(ValueError, match="EXISTING_FINAL_JOURNAL_REQUIRED"):
        mqar_final_tail_service_cli.reconcile_hf(None, SimpleNamespace(work=tmp_path))
