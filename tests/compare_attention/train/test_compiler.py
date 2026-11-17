"""Real Inductor backend evidence, not a torch.compile request spy."""

import pytest
import torch
from torch._dynamo.exc import SkipFrame

from compare_attention.models.gpt2 import GPT2
from compare_attention.train import compiler, loop
from tests.compare_attention.models.factories import gpt2_config


@pytest.mark.parametrize("phase", ["training", "validation", "unknown"])
@pytest.mark.parametrize("outcome", ["success", "failed", "skipped"])
def test_backend_success_is_recorded_only_after_return_in_its_phase(
    monkeypatch: pytest.MonkeyPatch, phase: str, outcome: str
) -> None:
    evidence = {}
    model = torch.nn.Linear(2, 2)
    failure = (
        SkipFrame("unsupported BF16") if outcome == "skipped" else RuntimeError("backend failed")
    )

    def backend(_graph, _inputs):
        assert evidence["backend_invocations"] == 1
        assert evidence[f"{phase}_backend_invocations"] == 1
        assert evidence["successful_backend_invocations"] == 0
        assert evidence["successful_training_backend_invocations"] == 0
        assert evidence["successful_validation_backend_invocations"] == 0
        assert evidence["backend_invocation_verified"] is False
        if outcome != "success":
            raise failure
        return model

    def compile_backend(_model, *, backend, fullgraph):
        assert fullgraph is False
        return backend

    monkeypatch.setattr(compiler, "lookup_backend", lambda _name: backend)
    monkeypatch.setattr(torch, "compile", compile_backend)
    compiled = compiler.compile_model(model, True, evidence)
    with compiler.compilation_phase(phase):
        if outcome == "success":
            assert compiled(None, []) is model
        else:
            with pytest.raises(type(failure), match=str(failure)):
                compiled(None, [])
    succeeded = outcome == "success"
    assert evidence["backend_invocations"] == 1
    assert evidence["successful_backend_invocations"] == int(succeeded)
    assert evidence["backend_invocation_verified"] is succeeded
    for name in ("training", "validation"):
        assert evidence[f"successful_{name}_backend_invocations"] == int(
            succeeded and phase == name
        )
    assert evidence["train_backend_invocations"] == int(phase == "training")
    assert evidence["eval_backend_invocations"] == int(phase == "validation")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires local CUDA")
@pytest.mark.parametrize("attention", ["classic", "deltanet"])
def test_real_inductor_train_validation_train(attention: str) -> None:
    torch._dynamo.reset()  # pylint: disable=protected-access
    model = (
        GPT2(
            gpt2_config(
                vocab_size=128, block_size=80, n_layer=1, n_embd=32, n_head=2, attention=attention
            )
        )
        .cuda()
        .to(torch.bfloat16)
    )
    evidence = {}
    compiled = loop._compile(model, True, evidence)  # pylint: disable=protected-access
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    inputs = torch.arange(160, device="cuda").reshape(2, 80) % 128
    labels = inputs.clone()
    with torch._dynamo.config.patch(suppress_errors=False):  # pylint: disable=protected-access,no-member
        first = loop._step(compiled, optimizer, inputs, labels)  # pylint: disable=protected-access
        validation = loop._validate(compiled, inputs, labels, 2)  # pylint: disable=protected-access
        second = loop._step(compiled, optimizer, inputs, labels)  # pylint: disable=protected-access
    torch.cuda.synchronize()
    assert all(torch.isfinite(torch.tensor(value)) for value in (first, *validation, second))
    assert evidence["backend_invocations"] > 0
    assert evidence["successful_backend_invocations"] == evidence["backend_invocations"]
    assert evidence["training_backend_invocations"] > 0
    assert evidence["successful_training_backend_invocations"] > 0
    assert evidence["successful_validation_backend_invocations"] > 0
    assert evidence["phase_source"] == "explicit_training_loop_context"
    assert evidence["train_backend_invocations"] > 0
    assert evidence["eval_backend_invocations"] > 0
    assert evidence["backend_invocation_verified"] is True
    assert evidence["scope"] == "dynamo_to_inductor_graphs_not_aot_backward_or_fullgraph_coverage"
    assert all(int(state["step"]) == 2 for state in optimizer.state.values())
