"""Prepare a NEW pinned sorting fixed pool; no overwrite and no network."""

import argparse

from compare_attention.data.sorting import prepare_sorting


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    config = prepare_sorting(**vars(parser.parse_args(argv)))
    print(config.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
