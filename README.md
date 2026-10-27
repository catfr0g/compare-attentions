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
uv run --locked pre-commit install
```

Перед коммитом pre-commit запускает `ruff check`, `ruff format`, `pylint` и `pytest`. pylint ограничивает модуль 500 строками.

## Датасет

```sh
uv run --locked prepare-kv-retrieval
uv run --locked prepare-kv-retrieval --n-train 1000 --n-val 100 --n-test 100
uv run --locked clean-kv-retrieval
```

Повторный `prepare-kv-retrieval` с теми же параметрами ничего не перезаписывает. Другие параметры в том же каталоге требуют `--force`.
