"""Запуск обучения на Kaggle.

Использование: uv run scripts/train_kaggle.py --train-flops 1000000000 [прочие флаги train ...]
см. compare_attention.kaggle_run.KaggleArgs.
"""

import sys

from compare_attention.kaggle_run import main

if __name__ == "__main__":
    sys.exit(main())
