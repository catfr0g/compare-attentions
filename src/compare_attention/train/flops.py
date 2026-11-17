"""Versioned algorithmic training cost; no model construction or execution."""

from dataclasses import asdict, dataclass

from compare_attention.data.config import KVRetrievalConfig
from compare_attention.data.permutation import PermutationConfig
from compare_attention.data.selective_count import SelectiveCountConfig
from compare_attention.data.sorting import SortingConfig
from compare_attention.data.tiny_shakespeare import CharacterLMConfig
from compare_attention.data.tracking import RulerConfig, StateRecallConfig
from compare_attention.models.config import GPT2Config
from compare_attention.train.config import (
    DELTA_CHUNKED_ABOVE,
    ComparisonAxis,
    Schedule,
    TrainConfig,
)
from compare_attention.train.flops_delta import delta_attention_cost

POLICY = "algorithmic_matmul_conv_v2"


@dataclass(frozen=True)
class StepFlops:
    """Count of included operations in one complete training batch."""

    policy: str
    forward: int
    backward: int
    total: int
    breakdown: dict[str, dict[str, int]]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-compatible schema, including policy identity."""
        return asdict(self)


def _validate(config: TrainConfig) -> None:
    """Revalidate even model_copy/model_construct inputs without reading data."""
    # Exact types reject architecture subclasses whose added operations are not modeled.
    # pylint: disable=unidiomatic-typecheck
    if type(config) is not TrainConfig:
        raise NotImplementedError("only the repository TrainConfig GPT2 architecture is supported")
    for value, kind in (
        (config.model, GPT2Config),
        (config.schedule, Schedule),
    ):
        if type(value) is not kind:
            raise NotImplementedError(f"unsupported configuration type: {type(value).__name__}")
        kind.model_validate(value.model_dump(), strict=True)
    if type(config.dataset) not in (
        KVRetrievalConfig,
        CharacterLMConfig,
        SelectiveCountConfig,
        PermutationConfig,
        RulerConfig,
        StateRecallConfig,
        SortingConfig,
    ):
        raise NotImplementedError(f"unsupported dataset type: {type(config.dataset).__name__}")
    type(config.dataset).model_validate(config.dataset.model_dump(), strict=True)
    model, schedule, dataset = config.model, config.schedule, config.dataset
    if schedule.axis != ComparisonAxis.TRAIN_FLOPS:
        raise NotImplementedError("only train_flops is supported")
    if model.vocab_size != dataset.vocab_size:
        raise ValueError("vocab_size must match the dataset")
    if model.block_size < dataset.input_seq_len:
        raise ValueError("block_size must cover the dataset sequence")
    if schedule.batch_size > dataset.n_train:
        raise ValueError("batch_size must not exceed the training split")
    if (
        model.attention == "deltanet"
        and dataset.input_seq_len > DELTA_CHUNKED_ABOVE
        and schedule.dtype not in ("bfloat16", "float16")
    ):
        raise ValueError("deltanet sequences longer than 64 require bfloat16 or float16")


def estimate_step(config: TrainConfig) -> StepFlops:
    """Count each dense operator and its input/weight gradients separately."""
    _validate(config)
    model = config.model
    batch, time, width = (config.schedule.batch_size, config.dataset.input_seq_len, model.n_embd)
    tokens = batch * time
    breakdown = {}

    def linear(name: str, inputs: int, outputs: int, layers: int = 1) -> None:
        forward = 2 * tokens * inputs * outputs * layers
        # Both floating-point input and weight require gradients at these sites.
        breakdown[name] = {"forward": forward, "backward": 2 * forward}

    if model.attention == "classic":
        linear("attention.qkv", width, 3 * width, model.n_layer)
        linear("attention.output", width, width, model.n_layer)
        for name in ("attention.qk", "attention.av"):
            forward = 2 * batch * time * time * width * model.n_layer
            breakdown[name] = {"forward": forward, "backward": 2 * forward}
    else:
        breakdown.update(
            {
                f"attention.{name}": {key: value * model.n_layer for key, value in term.items()}
                for name, term in delta_attention_cost(batch, time, width, model.n_head).items()
            }
        )
    linear("mlp.expand", width, 4 * width, model.n_layer)
    linear("mlp.contract", 4 * width, width, model.n_layer)
    linear("lm_head", width, model.vocab_size)
    forward = sum(term["forward"] for term in breakdown.values())
    backward = sum(term["backward"] for term in breakdown.values())
    metadata = {
        "schema_version": 1,
        "shape": {
            "batch_size": batch,
            "sequence_length": time,
            "hidden_size": width,
            "num_heads": model.n_head,
            "num_layers": model.n_layer,
            "vocab_size": model.vocab_size,
        },
        "attention": model.attention,
        "architecture": "repository_gpt2_v1",
        "included": (
            ["dense_matmul"]
            if model.attention == "classic"
            else ["dense_matmul", "convolution", "delta_core_contractions"]
        ),
        "mac_flops": 2,
        "causal_convention": (
            "dense_square" if model.attention == "classic" else "fla_causal_recurrence_or_chunk64"
        ),
        "padding_convention": "logical_tokens_no_kernel_padding",
        "recompute": ("none" if model.attention == "classic" else "fla_default_backward_recompute"),
        "excluded": [
            "optimizer",
            "non_contraction_elementwise",
            "non_contraction_reductions",
            "embedding_gather_scatter",
            "loss",
            "validation",
            "kernel_padding",
            "hardware_instructions",
        ],
    }
    return StepFlops(POLICY, forward, backward, forward + backward, breakdown, metadata)
