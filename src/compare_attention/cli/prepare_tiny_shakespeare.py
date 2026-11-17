"""Prepare immutable upstream Tiny Shakespeare with the public array format."""

import argparse
import json
import urllib.request
from pathlib import Path

from compare_attention.data.tiny_shakespeare import SOURCE_URL, prepare_tiny_shakespeare


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, help="Existing exact pinned upstream input.txt bytes"
    )
    parser.add_argument("--context", type=int, default=512)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.source is None:
        # Download to a new sibling, never overwrite source or normalize text on Windows.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        source = args.output.with_suffix(".source.txt")
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            raw = response.read()
        with source.open("xb") as stream:
            stream.write(raw)
    else:
        source = args.source
    config = prepare_tiny_shakespeare(source, args.output, context=args.context)
    print(json.dumps(config.model_dump(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
