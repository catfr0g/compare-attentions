"""CLI двухфазной Kaggle-серии: локальный stage, затем отдельные publish/run."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from compare_attention.kaggle_run import Runner, _prepare_kaggle_cli
from compare_attention.kaggle_series import publish_series, run_selected, stage_series


def authenticate() -> None:
    """Только команды с сетью используют существующую локальную авторизацию."""
    load_dotenv(find_dotenv(usecwd=True))
    _prepare_kaggle_cli()


def main(argv: list[str] | None = None, run: Runner | None = None) -> int:
    """Все конфиги stage передаются явно; run выбирает только разрешённую группу."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage", help="local immutable release; no credentials/network")
    stage.add_argument("--root", type=Path, default=Path.cwd())
    stage.add_argument("--owner", required=True)
    stage.add_argument(
        "--data", type=Path, required=True, help="already prepared matching KV directory"
    )
    stage.add_argument("--cache", type=Path, required=True)
    stage.add_argument("--configs", type=Path, nargs="+", required=True)
    publish = commands.add_parser(
        "publish", help="create two PRIVATE immutable input datasets once"
    )
    publish.add_argument("--manifest", type=Path, required=True)
    execute = commands.add_parser(
        "run", help="sequential fail-fast selected runs; persistent resume"
    )
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument(
        "--select", nargs="+", required=True, help="exact staged repo-relative config names"
    )
    execute.add_argument(
        "--credentials-dataset", help="existing PRIVATE dataset with kaggle-env.env"
    )
    execute.add_argument("--accelerator", default="NvidiaTeslaT4")
    execute.add_argument("--timeout", type=int, default=9 * 3600)
    execute.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args(argv)
    if args.command == "stage":
        manifest = stage_series(args.root, args.configs, args.data, args.cache, args.owner, run)
        print(manifest)
    else:
        authenticate()
        if args.command == "publish":
            publish_series(args.manifest, run)
        else:
            run_selected(
                args.manifest,
                args.select,
                credentials_dataset=args.credentials_dataset,
                accelerator=args.accelerator,
                timeout=args.timeout,
                poll_seconds=args.poll_seconds,
                run=run,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
