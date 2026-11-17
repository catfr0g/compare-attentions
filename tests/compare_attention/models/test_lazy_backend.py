"""Classic-only CPU paths must not import optional DeltaNet backend."""

import os
import subprocess


def test_classic_lazy_backend_import():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    code = """
import sys
from compare_attention.models.config import GPT2Config
from compare_attention.models.gpt2 import GPT2
model = GPT2(GPT2Config(vocab_size=43, block_size=54, n_layer=1, n_head=1, n_embd=8))
assert 'compare_attention.models.deltanet' not in sys.modules
assert 'fla' not in sys.modules
"""
    result = subprocess.run(
        ["uv", "run", "--offline", "--no-sync", "python", "-B", "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
