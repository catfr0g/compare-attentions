"""Prepare a fresh pinned C5 or S5 running-product pool (actual moves, no answer step)."""

import argparse
import json
from pathlib import Path

from compare_attention.data.permutation import prepare_permutation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", choices=("c5", "s5"), required=True)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    config = prepare_permutation(**vars(parser.parse_args(argv)))
    print(json.dumps(config.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
