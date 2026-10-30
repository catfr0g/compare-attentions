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

Перед коммитом pre-commit запускает `ruff check`, `ruff format`, `pylint` и `pytest` (unit и лёгкие интеграционные; e2e — вручную). pylint ограничивает модуль 500 строками.

## Датасет

```sh
uv run --locked prepare-kv-retrieval
uv run --locked prepare-kv-retrieval --n-train 1000 --n-val 100 --n-test 100
uv run --locked clean-kv-retrieval
```

Повторный `prepare-kv-retrieval` с теми же параметрами ничего не перезаписывает. Другие параметры в том же каталоге требуют `--force`.

## Обучение

Пока уравнивается суммарный бюджет FLOPs на обучение (`train_flops`): цикл суммирует FLOPs реально выполненных операций. Оси `layer_flops` и `parameters` пока не отбирают сетку.

```sh
uv run --locked train --data data/kv_retrieval --train-flops 1000000000000
uv run --locked train --data data/kv_retrieval --train-flops 1000000000000 --attention deltanet --dtype bfloat16
```

Для DeltaNet длина больше 64 считается чанковым ядром и требует `bfloat16`.

## Устройство запуска и трекинг

Результаты одного запуска раскладываются так:

* **Метрики и логи** — [ClearML](https://clearml.io), облачный проект `compare-attention`: конфиг запуска, скаляры потерь по шагам, итог. ClearML сам захватывает stdout/stderr, поэтому консольный вывод с любой машины виден в UI у run'а.
* **Каталог результата** — `--result-dir` (по умолчанию `models/<штамп времени>`): `train.log`, `model.pt` (веса обучения) и `metrics.jsonl` — строка `{type: step, loss, flops}` на каждый шаг обучения и `{type: result, ...}` с итогом запуска.
* **Hugging Face** — с флагом `--hf-repo user/model` или переменной `HF_REPO` из `.env` каталог результата целиком сливается в приватный репозиторий (путь `runs/<имя каталога>`); нужен токен в переменной `HF_TOKEN`. Приоритет: `--hf-repo` → `HF_REPO` (окружение) → `HF_REPO` (файл `.env`); если ничего не задано, выгрузки нет.

Разовый вход в ClearML (ключи с app.clear.ml):

```sh
uv run --locked clearml-init
```

На тренировочных серверах вместо `clearml-init` удобен `.env.example`: скопируйте его в `.env`, заполните ключи — команда `train` сама читает `.env` из текущего каталога (переменные окружения имеют приоритет над файлом). SDK принимает и переменные окружения, и `~/clearml.conf`.

Трекер выбирается флагом: `--tracker clearml` (по умолчанию) или `--tracker disabled` — для запусков без сети.

## YAML-эксперименты

Один YAML-файл = один эксперимент (шаблон — `experiments/smoke.yaml`):

```yaml
data: data/kv_retrieval
schedule:
  train_flops: 1000000000
  batch_size: 8
model:
  vocab_size: 8192
  block_size: 512
  n_layer: 12
  attention: classic
hf_repo: user/compare-attention
```

```sh
uv run --locked train --config experiments/smoke.yaml      # секции YAML — дефолты
uv run --locked train --config experiments/smoke.yaml --train-flops 100    # флаги CLI перекрывают
```

Поведение: результат собирается в `models/<имя эксперимента>/<штамп времени>`; опциональная секция `kv` позволяет подготовить датасет прямо перед запуском, если `data` ещё пустой; вся конфигурация уезжает в ClearML (`task.connect`) и `metrics.jsonl`. Флаг CLI имеет приоритет над значением YAML; `--hf-repo` по-прежнему перекрывается и `.env`, стандартный приоритет: флаг → YAML → окружение.

## Запуск обучения на Kaggle

```sh
uv run --locked scripts/train_kaggle.py --config experiments/smoke.yaml
# эквивалентно зеркальным флагам:
uv run --locked scripts/train_kaggle.py --train-flops 1000000000 --n-layer 2
```

С `--config` YAML-файл эксперимента копируется в тот же приватный dataset (внутри копии `data` переписывается на `/kaggle/working/kv_retrieval`, а секция `kv` подгоняет подготовку данных в ядре), ядро запускает `train --config /kaggle/input/.../<имя>.yaml`. Локальные флаги `--slug`/`--accelerator` и пр. по-прежнему работают; секции YAML перекрываются флагами локально.

Схема: wheel проекта собирается локально (`uv build --wheel`), заливается в私有ный Kaggle dataset, генерируется kernel-скрипт и запускается на выбранном GPU (`--accelerator`, по умолчанию T4). Скрипт поллит статус и после завершения скачивает артефакты в `kaggle/<slug>/output`.

Креды Kaggle: `KAGGLE_USERNAME`/`KAGGLE_KEY` из `.env` (скрипт сам читает `.env`); `kaggle.json`/`kaggle auth login` тоже работают.

Креды ClearML/HF внутри kernel: опираются на Kaggle Secrets — **но Kaggle передаёт их только в запусках, начатых из UI**, поэтому основной механизм другой. Локальный `.env` сокращается до корзины ключей (только `CLEARML_API_*`, `HF_TOKEN`, `HF_REPO`; `KAGGLE_*` никогда не попадают) и кладётся файлом `kaggle-env.env` в тот же приватный dataset, откуда kernel восстанавливает их в окружение. Датасет держать приватным; для страховки — отзываемые ключи: отдельная пара API-ключей ClearML и fine-grained HF-токен только на этот репозиторий.

## Тесты

```sh
uv run --locked pytest            # unit + e2e + лёгкие интеграционные
uv run --locked pytest -m docker  # e2e: сервер ClearML в docker compose (heavy)
```

Слои: `tests/compare_attention/**` — unit (зеркалят `src/`); `tests/integration/` — ClearML SDK в официальном offline-режиме (без сети); `tests/e2e/train/` — полные прогоны CLI; `tests/e2e/clearml/` — настоящий сервер через docker compose, помечены `docker` и по умолчанию выключены.
