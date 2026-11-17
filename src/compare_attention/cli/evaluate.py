"""Evaluate existing checkpoint weights without training or data preparation."""

import argparse
import json
from pathlib import Path

from compare_attention.evaluation import evaluate_checkpoint


def main(argv: list[str] | None = None) -> int:
    """Public inference-only entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument("--output")
    parser.add_argument("--config")
    args = vars(parser.parse_args(argv))
    output = args.pop("output")
    if output is not None and (Path(output).exists() or Path(output).is_symlink()):
        raise FileExistsError("evaluation output already exists: " + output)
    result = evaluate_checkpoint(**args)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if output is not None:
        with open(output, "x", encoding="utf-8") as stream:
            stream.write(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    main()
