"""Staged Kaggle series; --help does not read credentials or contact Kaggle."""

from compare_attention.kaggle_series_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
