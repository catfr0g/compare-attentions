"""Free-running causal tests: no answer leakage, unrestricted argmax, full tails."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from compare_attention.data.tracking import prepare_tracking


class FeedbackModel(torch.nn.Module):
    """Wrong first prediction must be fed back instead of the ground-truth token."""

    def __init__(self, vocab):
        super().__init__()
        self.config = SimpleNamespace(attention="classic", vocab_size=vocab, block_size=512)
        self.calls = []

    def forward(self, x):
        self.calls.append(x.clone())
        logits = torch.zeros((*x.shape, self.config.vocab_size), device=x.device)
        logits.scatter_(-1, ((x + 1) % self.config.vocab_size).unsqueeze(-1), 1)
        return logits


def test_greedy_prompt_only_feedback_and_label_independence(tmp_path):
    from compare_attention.tracking_evaluation import greedy_generate

    path = tmp_path / "ruler"
    c = prepare_tracking(path, task="ruler_variable_tracking", n_train=2, n_val=1, n_test=3)
    x = np.load(path / "test_inputs.npy")
    model = FeedbackModel(c.vocab_size)
    out = greedy_generate(model, x[:, : c.prompt_length], steps=c.answer_length, batch_size=2)
    assert out.shape == (3, 5)
    assert [call.shape[1] for call in model.calls] == [114, 115, 116, 117, 118] * 2
    assert torch.equal(model.calls[1][:, -1], torch.as_tensor(out[:2, 0]))
    # Change ALL teacher-forced tokens; generation receives only the true prompt.
    changed = x.copy()
    changed[:, c.prompt_length :] = 0
    again = greedy_generate(
        model, changed[:, : c.prompt_length], steps=c.num_hops + 1, batch_size=2
    )
    np.testing.assert_array_equal(out, again)
    assert not model.training
    assert not out.flags.writeable or out.dtype == np.int64


@pytest.mark.parametrize("task", ["ruler_variable_tracking", "state_based_recall"])
def test_generated_metric_denominators_and_empty_guard(tmp_path, task):
    from compare_attention.evaluation import evaluate_model

    path = tmp_path / "data"
    c = prepare_tracking(path, task=task, n_train=2, n_val=1, n_test=3)
    x, y = (np.load(path / f"test_{kind}.npy") for kind in ("inputs", "labels"))
    metrics = evaluate_model(FeedbackModel(c.vocab_size), x, y, batch_size=2, task_config=c)
    assert metrics["generated_rows"] == 3
    assert metrics["generated_positions"] == 3 * c.answer_length
    assert "teacher_forced_ce_nats_per_token" in metrics
    assert "count_nmae" not in metrics and "bits_per_character" not in metrics
    if task == "ruler_variable_tracking":
        assert metrics["primary_metrics"] == ["generated_chain_exact_match"]
        assert metrics["generated_links_total"] == 3 * 4
    else:
        assert metrics["primary_metrics"] == ["bit_accuracy"]
        assert (
            sum(metrics["slices"][name]["count"] for name in ("address_moved", "address_unchanged"))
            == 3
        )
        for item in metrics["slices"].values():
            if item["count"] == 0:
                assert item["accuracy"] is None
    with pytest.raises(ValueError, match="nonempty"):
        evaluate_model(FeedbackModel(c.vocab_size), x[:0], y[:0], task_config=c)


def test_generation_boundary_ignores_corrupted_answers_and_labels(tmp_path):
    from compare_attention.tracking_evaluation import generated_metrics

    path = tmp_path / "data"
    c = prepare_tracking(path, task="ruler_variable_tracking", n_train=2, n_val=1, n_test=3)
    x, y = (np.load(path / f"test_{kind}.npy") for kind in ("inputs", "labels"))
    first, second = FeedbackModel(c.vocab_size), FeedbackModel(c.vocab_size)
    before = generated_metrics(first, x, y, c, batch_size=2)
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[:, c.prompt_length :] = 0
    changed_y[:, c.prompt_length - 1 :] = 0
    after = generated_metrics(second, changed_x, changed_y, c, batch_size=2)
    assert before["generated_positions"] == after["generated_positions"] == 15
    assert len(first.calls) == len(second.calls) == 10
    for actual, changed in zip(first.calls, second.calls, strict=True):
        assert torch.equal(actual, changed)
