"""Prepare a NEW fixed-pool RULER or state-based recall dataset; never overwrite."""

import argparse
import json
from pathlib import Path

from compare_attention.data.tracking import TASKS, prepare_tracking


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument(
        "--pad-to-128",
        action="store_true",
        help="Explicit matched RULER padding for the existing Delta chunk64 cost gate",
    )
    config = prepare_tracking(**vars(parser.parse_args(argv)))
    print(json.dumps(config.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
