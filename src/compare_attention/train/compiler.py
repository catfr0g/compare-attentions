"""Evidence from actual Dynamo calls to the real Inductor backend."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast

import torch
from torch._dynamo.backends.registry import lookup_backend

_PHASE = ContextVar("compiler_phase", default="unknown")


@contextmanager
def compilation_phase(phase: str):
    """Attribute lazy backend calls to the actual training-loop phase."""
    token = _PHASE.set(phase)
    try:
        yield
    finally:
        _PHASE.reset(token)


def compiler_metadata(requested: bool) -> dict[str, Any]:
    """Counters describe graph compilation, not full-graph or backward coverage."""
    return {
        "requested": requested,
        "backend": "inductor" if requested else None,
        "backend_invocations": 0,
        "successful_backend_invocations": 0,
        "successful_training_backend_invocations": 0,
        "successful_validation_backend_invocations": 0,
        "train_backend_invocations": 0,
        "training_backend_invocations": 0,
        "validation_backend_invocations": 0,
        "unknown_backend_invocations": 0,
        "phase_source": "explicit_training_loop_context",
        "eval_backend_invocations": 0,
        "backend_invocation_verified": False,
        "scope": "dynamo_to_inductor_graphs_not_aot_backward_or_fullgraph_coverage",
        "fullgraph": False,
        "hidden_warmup_steps": 0,
    }


def compile_model(
    model: torch.nn.Module, enabled: bool, evidence: dict[str, Any]
) -> torch.nn.Module:
    """Compile lazily on the real first forward; do not run calibration/probe steps."""
    evidence.update(compiler_metadata(enabled))
    if not enabled:
        return model
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    inductor = lookup_backend("inductor")

    def counting_inductor(graph, example_inputs):
        phase = _PHASE.get()
        for key in ("backend_invocations", f"{phase}_backend_invocations"):
            evidence[key] = int(evidence[key]) + 1
        if phase in ("training", "validation"):
            alias = "train" if phase == "training" else "eval"
            evidence[f"{alias}_backend_invocations"] += 1
        compiled = inductor(graph, example_inputs)
        evidence["successful_backend_invocations"] = (
            int(evidence["successful_backend_invocations"]) + 1
        )
        if phase in ("training", "validation"):
            evidence[f"successful_{phase}_backend_invocations"] += 1
        evidence["backend_invocation_verified"] = True
        return compiled

    return cast(torch.nn.Module, torch.compile(model, backend=counting_inductor, fullgraph=False))
