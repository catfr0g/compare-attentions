"""Real compiled CUDA train/validation/checkpoint/early-stop integration."""

import pytest
import torch

from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.exploration_gate import verify_exploration_run
from compare_attention.train.backup import BackupTracker
from compare_attention.train.checkpoint import load_checkpoint
from compare_attention.train.flops import estimate_step
from compare_attention.train.loop import train
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.models.factories import gpt2_config
from tests.compare_attention.train.factories import train_config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration requires a GPU")
@pytest.mark.parametrize("family", ["classic", "deltanet"])
def test_real_compiled_fp16_early_stop_checkpoint_and_gate(tmp_path, family):
    data = tmp_path / "data"
    prepare_kv_retrieval(data_config(vocab_size=256, input_seq_len=128, n_train=8), data)
    model = gpt2_config(
        vocab_size=256, block_size=128, n_layer=1, n_head=2, n_embd=128, attention=family
    )
    config = train_config(
        data,
        model=model,
        dtype="float16",
        compile=True,
        val_every_steps=1,
        checkpoint_every_steps=1,
        checkpoint_keep_last=2,
        early_stopping_patience=1,
        early_stopping_min_delta=100.0,
        early_stopping_min_steps=0,
        amp_max_scale=128.0,
    )
    config = config.model_copy(
        update={
            "schedule": config.schedule.model_copy(
                update={"train_flops": estimate_step(config).total * 6}
            )
        }
    )
    out = tmp_path / "output"
    result = train(config, BackupTracker(out))
    assert result.steps == 2
    assert result.stop_reason == "early_stopping"
    assert result.provenance["precision"]["optimizer_steps"] == 2
    assert result.provenance["precision"]["skipped_optimizer_steps"] == 0
    assert result.provenance["precision"]["final_scale"] == 128.0
    assert result.provenance["compiler"]["successful_training_backend_invocations"] > 0
    checkpoint = load_checkpoint(out / "checkpoints/latest.json")
    assert checkpoint["step"] == 2
    assert checkpoint["precision"]["optimizer_steps"] == 2
    verified = verify_exploration_run(
        out,
        expected_config=config.model_dump(mode="json"),
        expected_gpu=torch.cuda.get_device_name(),
    )
    assert verified["passed"] and verified["budget_exhausted"] is False
