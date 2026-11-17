"""FLA 0.6.0 default DeltaNet, algorithmic contraction FLOPs (MAC=2).

See docs/flops-delta.md. This is not a hardware instruction counter. Callers
must validate the default adapter, no masks/caches, and fully trainable inputs
and parameters. Normalization and scalar elementwise operations are excluded.
"""

import os
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec

POLICY = "algorithmic_matmul_conv_v2"
SUPPORTED_FLA_VERSION = "0.6.0"


def _validate_runtime() -> None:
    try:
        supported = all(
            version(name) == SUPPORTED_FLA_VERSION
            for name in ("flash-linear-attention", "fla-core")
        )
    except PackageNotFoundError as error:
        raise ValueError("DeltaNet cost requires FLA 0.6.0 and fla-core 0.6.0") from error
    if not supported:
        raise ValueError("DeltaNet cost supports only FLA 0.6.0 and fla-core 0.6.0")
    if os.environ.get("FLA_CONV_BACKEND", "triton") != "triton":
        raise ValueError("DeltaNet cost supports only the default Triton convolution backend")
    if (
        any(
            value != "0" and name.startswith(("FLA_TILELANG", "FLA_GLUON"))
            for name, value in os.environ.items()
        )
        or find_spec("tilelang") is not None
    ):
        raise ValueError("DeltaNet cost does not support alternate kernel backend overrides")


def delta_attention_cost(
    batch_size: int, sequence_length: int, hidden_size: int, num_heads: int
) -> dict[str, dict[str, int]]:
    """Return one default attention layer's forward and backward components."""
    # Named algebra terms retain the source-level derivation for auditability.
    # pylint: disable=too-many-locals,unidiomatic-typecheck
    _validate_runtime()
    shape = (batch_size, sequence_length, hidden_size, num_heads)
    if any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("unsupported DeltaNet shape: require positive static Python integers")
    if hidden_size % num_heads or hidden_size // num_heads not in (16, 32, 64, 128):
        raise ValueError("unsupported DeltaNet head width: require 16, 32, 64 or 128")
    if sequence_length > 64 and sequence_length % 64:
        raise ValueError("unsupported DeltaNet chunk shape: sequence length must divide into 64")
    tokens = batch_size * sequence_length
    projection = 2 * tokens * hidden_size * hidden_size
    costs = {
        name: {"forward": projection, "backward": 2 * projection}
        for name in ("q_projection", "k_projection", "v_projection", "output_projection")
    }
    beta = 2 * tokens * hidden_size * num_heads
    conv = 3 * 2 * tokens * hidden_size * 4
    costs.update(
        beta_projection={"forward": beta, "backward": 2 * beta},
        short_convolution={"forward": conv, "backward": 3 * conv},
        qk_l2norm={"forward": 0, "backward": 0},
        output_rmsnorm={"forward": 0, "backward": 0},
    )
    chunk = 64
    dim = hidden_size // num_heads
    if sequence_length <= 64:
        # Scalar beta-gradient reductions are excluded, but all state matvecs
        # and outer-product multiply-accumulates are included (even in Triton
        # when expressed using elementwise multiply + reduction).
        state = 2 * tokens * num_heads * dim**2
        costs.update(
            recurrent_state={"forward": 3 * state, "backward": 0},
            recurrent_reverse_state={"forward": 0, "backward": 4 * state},
            recurrent_recompute_state={"forward": 0, "backward": 3 * state},
        )
        return costs
    chunks = batch_size * num_heads * (sequence_length // chunk)
    # Each value is per chunk and per head; inverse uses the actual 4x16
    # row-solve / block-merge algorithm, not a generic cubic solve estimate.
    inverse = 4 * 14 * (2 * 16 * 16) + 16 * (2 * 16**3)
    forward = {
        "chunk_gram": 2 * chunk**2 * dim,
        "chunk_inverse": inverse,
        "chunk_wy": 4 * chunk**2 * dim,
        "chunk_state": 4 * chunk * dim**2,
        "chunk_output": 2 * chunk * dim**2 + 4 * chunk**2 * dim,
    }
    backward = {
        "chunk_recompute_wy": 4 * chunk**2 * dim,
        "chunk_recompute_state": 4 * chunk * dim**2,
        "chunk_local_dv": 4 * chunk**2 * dim,
        "chunk_reverse_state": 6 * chunk * dim**2,
        "chunk_qkw_grad": 6 * chunk * dim**2 + 6 * chunk**2 * dim,
        "chunk_wy_grad": 12 * chunk**2 * dim + 4 * chunk**3,
    }
    costs.update(
        {name: {"forward": chunks * value, "backward": 0} for name, value in forward.items()}
    )
    costs.update(
        {name: {"forward": 0, "backward": chunks * value} for name, value in backward.items()}
    )
    return costs
