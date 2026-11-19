"""Final10780 scientific identity and read-only structural primitives."""

import hashlib
import json
from pathlib import Path

FINAL = "a21f1b58e50488359e022edb73d6c526814f35b9e8d3d7284d96d778eb897178"
FULL_SHA = "f4cbbbbfa6f1bbae377acf5baa9a07019e00986f26ede4c8c1c76ab22beca3d0"
MODEL_SHA = "7e6931f2a0b36083d7f72bb5ceb1fdf83f8b6bab2a773ba2f181cdf9efbf7c7f"
WHEEL_SHA = "512cb5b073e47deca36d2bfb41f2d183e0711b8817b9a0aa18dbd2e07591b2d3"
REMOTE_DATA = "/kaggle/input/datasets/zimike/ca-kv-1c660aa5f54d88ea9e0c1ce4"
CONFIG_SHA = "24d0189b1c9c65757a4ac93a648e41aeacd3611f2f76a7fed3875d9f82c33811"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def audit(root, historical, data, config):
    from mqar_final_tail_structural import audit as structural

    return structural(Path(root), historical, Path(data), Path(config))
