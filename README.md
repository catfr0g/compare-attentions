# Compare attention

Базовый Python-проект для экспериментов с механизмами внимания.

Python 3.14, `uv`, PyTorch. Код проекта: `src/compare_attention/`.

## Установка

PyTorch ставится со сборкой CUDA 12.8:

```sh
uv sync --locked
```

## Проверка стиля

```sh
uv sync --group dev --locked
uv run --locked ruff check .
uv run --locked ruff format .
uv run --locked pylint src
uv run --locked pylint tests --max-module-lines=500
```

pylint ограничивает модуль 300 строками в `src` и 500 в `tests`.

Подготовить датасет `kv_retrieval` и проверить стиль:

```sh
uv run --locked prepare-kv-retrieval
uv run --locked prepare-kv-retrieval --n-train 1000 --n-val 100 --n-test 100
uv run --locked clean-kv-retrieval
uv run --locked lint
```

Повторный `prepare-kv-retrieval` с теми же параметрами ничего не перезаписывает. Другие параметры в том же каталоге требуют `--force`.
