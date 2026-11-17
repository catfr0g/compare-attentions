"""Predicted EOS, mixed completion, feedback and independent generated closure."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from compare_attention.tracking_evaluation import greedy_generate


class EOSModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(attention="classic", vocab_size=8, block_size=512)
        self.calls = []

    def forward(self, x):
        self.calls.append(x.clone())
        logits = torch.zeros((*x.shape, 8))
        next_token = torch.where(x[:, 0] == 4, 3, (x[:, -1] + 1) % 8)
        if x.shape[1] >= 42:
            next_token[:] = 3
        logits[:, -1].scatter_(1, next_token[:, None], 1)
        return logits


def test_eos_mixed_rows_feedback_and_tail():
    prompts = np.full((3, 40), 1, dtype=np.int64)
    prompts[0, 0] = 4
    model = EOSModel()
    result = greedy_generate(model, prompts, steps=40, batch_size=2, eos_token_id=3, pad_token_id=0)
    assert result.shape == (3, 40)
    assert result[0].tolist() == [3] + [0] * 39
    assert result[1].tolist() == [2, 3] + [0] * 38
    assert len(model.calls) == 4
    assert model.calls[1][:, -1].tolist() == [3, 2]
    assert model.calls[0].shape == (2, 40) and model.calls[2].shape == (1, 40)


def test_sorting_metrics_reject_invalid_sequences(tmp_path):
    from compare_attention.data.sorting import prepare_sorting
    from compare_attention.sorting_evaluation import score_predictions

    path = tmp_path / "data"
    c = prepare_sorting(path, n_train=2, n_val=1, n_test=3)
    x, y = (np.load(path / f"test_{k}.npy") for k in ("inputs", "labels"))
    good = y[:, 39:79].copy()
    metrics = score_predictions(x[:, :40], good, c)
    assert metrics["generated_answer_correct"] == 3
    assert metrics["syntax_valid"]["correct"] == 3
    for token in (0, 1, 2, 3, 4):
        bad = good.copy()
        bad[:, 0] = token
        m = score_predictions(x[:, :40], bad, c)
        assert m["generated_answer_correct"] == int(np.all(bad == good, axis=1).sum())
    bad = good.copy()
    bad[:, -1] = 4
    assert score_predictions(x[:, :40], bad, c)["generated_answer_correct"] == 0
    with pytest.raises(ValueError):
        score_predictions(x[:, :40], good[:, :-1], c)


def test_generation_gt_suffix_independence_and_single_capture(tmp_path):
    from compare_attention.data.sorting import prepare_sorting
    from compare_attention.sorting_evaluation import generated_metrics, score_predictions

    path = tmp_path / "data"
    c = prepare_sorting(path, n_train=2, n_val=1, n_test=3)
    x, y = (np.load(path / f"test_{k}.npy") for k in ("inputs", "labels"))
    first, second = EOSModel(), EOSModel()
    before = generated_metrics(first, x, y, c, batch_size=2)
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[:, 40:] = 7
    changed_y[:] = 6
    after = generated_metrics(second, changed_x, changed_y, c, batch_size=2)
    assert before == after
    for a, b in zip(first.calls, second.calls, strict=True):
        assert torch.equal(a, b)
    records = before["generated_records"]
    closed = score_predictions(x[:, :40], np.array([r["prediction"] for r in records]), c)
    assert closed == before and len(records) == 3
    assert before["generated_answer_correct"] == 0
    assert all(r["generated_length"] < 40 for r in records)


def test_missing_eos_uses_declared_cap_not_ground_truth():
    class NeverEOS(EOSModel):
        def forward(self, x):
            self.calls.append(x.clone())
            logits = torch.zeros((*x.shape, 8))
            logits[..., 4] = 1
            return logits

    model = NeverEOS()
    result = greedy_generate(
        model, np.ones((1, 40), dtype=np.int64), steps=40, eos_token_id=3, pad_token_id=0
    )
    assert result.tolist() == [[4] * 40]
    assert len(model.calls) == 40 and model.calls[-1].shape[1] == 79
    assert model.calls[1][0, -1] == 4


def test_order_and_multiset_are_separate_metrics(tmp_path):
    from compare_attention.data.sorting import encode_items, prepare_sorting
    from compare_attention.sorting_evaluation import score_predictions

    c = prepare_sorting(tmp_path / "data", n_train=1, n_val=1, n_test=1)
    original = [(4, 4, 4, 4)] * 4 + [(5, 5, 5, 5)] * 4
    prompt = encode_items(original, terminal=1)[None]
    reversed_items = encode_items(list(reversed(original)))[None]
    m = score_predictions(prompt, reversed_items, c)
    assert m["syntax_valid"]["correct"] == m["multiset_preserved"]["correct"] == 1
    assert m["ascending_order"]["correct"] == m["generated_answer_correct"] == 0
    wrong_multi = encode_items([(4, 4, 4, 4)] * 8)[None]
    m = score_predictions(prompt, wrong_multi, c)
    assert m["ascending_order"]["correct"] == 1
    assert m["multiset_preserved"]["correct"] == m["generated_answer_correct"] == 0
