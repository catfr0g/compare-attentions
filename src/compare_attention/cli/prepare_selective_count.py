"""Prepare source-faithful Selective Count examples in a new immutable directory."""

import argparse
import json
from pathlib import Path

from compare_attention.data.selective_count import prepare_selective_count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--n-symbols", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    args = vars(parser.parse_args(argv))
    config = prepare_selective_count(**args)
    print(json.dumps(config.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
