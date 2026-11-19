"""Retain original Gloo numeric assertions with the new secret-safe error contract."""

import pytest

from tests.compare_attention.train import test_distributed_validation as original

WORKER = "tests.compare_attention.train.normal_validation_worker"


def test_normal_safe_contract_preserves_original_bitwise_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(original, "WORKER", WORKER)
    original.test_real_gloo_v5_observation_checkpoint_resume_is_bitwise(tmp_path)


@pytest.mark.parametrize("mode", ["validation_failure", "writer_failure"])
def test_normal_safe_contract_preserves_collective_validation_failure(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(original, "WORKER", WORKER)
    original.test_validation_or_writer_failure_remains_collective_before_checkpoint(tmp_path, mode)
