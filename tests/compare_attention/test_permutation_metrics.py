"""Whole-sequence permutation metrics: masks, micro counts, tails, unrestricted decoder."""

# pylint: disable=import-outside-toplevel
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from compare_attention.evaluation import evaluate_model


def test_masked_metrics_not_batch_means_and_first_copy():
    from compare_attention.permutation_metrics import permutation_metrics

    y = np.array([[1, 34, -1, -1], [34, 65, 91, 97], [65, -1, -1, -1]])
    predicted = np.array([[1, 0, 120, 120], [34, 65, 91, 97], [65, 0, 0, 0]])
    m = permutation_metrics(predicted, y, allowed_token_ids=[1, 34, 65, 91, 97])
    assert m["acc"] == 2 / 3
    assert m["token_acc"] == 6 / 7
    assert m["sequence_correct"] == 2
    assert m["exclude_first_token_acc"] == 3 / 4
    assert m["exclude_first_acc"] == 1 / 2  # one-label row is excluded, not perfect credit
    assert m["last_accuracy"] == 2 / 3
    assert m["outside_group_predictions"] == 1
    assert m["position_supervised_count"] == [3, 2, 1, 1]


def test_full_evaluator_unrestricted_argmax_and_tail(tmp_path):
    from compare_attention.data.permutation import prepare_permutation
    from compare_attention.data.store import load_split

    config = prepare_permutation(tmp_path / "c5", group="c5", n_train=32, n_val=2, n_test=3)
    data = load_split(tmp_path / "c5", "test")

    class Copy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(attention="classic", block_size=512, vocab_size=121)

        def forward(self, inputs):
            logits = torch.zeros((*inputs.shape, 121))
            logits.scatter_(-1, inputs.unsqueeze(-1), 2)
            logits[:, -1, 120] = 10  # outside C5 wins over a valid-class candidate
            return logits

    m = evaluate_model(Copy(), data.inputs, data.labels, task_config=config, batch_size=2)
    assert m["rows"] == 3
    assert m["supervised_count"] == 96
    assert m["acc"] == 0
    assert m["outside_group_predictions"] == 3
    assert m["position_accuracy"][0] == 1
    assert m["primary_metrics"] == ["acc", "token_acc"]
    assert "bits_per_character" not in m
    assert "count_nmae" not in m
    expected = np.count_nonzero(data.inputs[:, :-1] == data.labels[:, :-1])
    assert m["correct"] == expected
    assert m["token_acc"] == expected / 96
    with pytest.raises(ValueError, match="DeltaNet requires CUDA"):
        model = Copy()
        model.config.attention = "deltanet"
        evaluate_model(model, data.inputs, data.labels, task_config=config)
