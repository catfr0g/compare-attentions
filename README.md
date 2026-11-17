# Compare attention

Базовый Python-проект для экспериментов с механизмами внимания.

Python 3.13, `uv`, PyTorch. Код проекта: `src/compare_attention/`.

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

Бюджет `train_flops` задан в единицах версионированной политики `algorithmic_matmul_conv_v2`. Стоимость полного шага рассчитывается один раз: dense matmul, convolution и DeltaNet state-contractions, включая предусмотренный FLA backward recompute; MAC = 2 FLOPs. Цикл прибавляет эту стоимость после каждого реального шага оптимизатора. Это не полные аппаратные FLOPs: нормы, scalar non-contraction операции, loss, optimizer и validation исключены. Старые результаты `FlopCounterMode` относятся к другой метрике и не смешиваются с v2. Формулы и ограничения: `docs/flops-policy.md`, `docs/flops-delta.md`. Оси `layer_flops` и `parameters` пока не отбирают сетку.

```sh
uv run --locked train --data data/kv_retrieval --train-flops 1000000000000
uv run --locked train --data data/kv_retrieval --train-flops 1000000000000 --attention deltanet --dtype bfloat16
```

Для DeltaNet длина больше 64 считается чанковым ядром и требует `bfloat16`.

## Устройство запуска и трекинг

Результаты одного запуска раскладываются так:

* **Метрики и логи** — [ClearML](https://clearml.io), проект `compare-attention`: конфиг запуска, train loss по шагам, validation loss и accuracy, итог. Ось X для train и validation — накопленные train FLOPs. ClearML захватывает stdout/stderr, поэтому обычные строки прогресса видны в консоли задачи даже без TTY.
* **Каталог результата** — `--result-dir` (по умолчанию `models/<штамп времени>`): `train.log`, `model.pt`, `metrics.jsonl` и `run_identity.json`. Identity схемы 2 содержит requested/effective seed, SHA256 начальных/конечных параметров, стоимость шага с версией политики и фактические вызовы Inductor отдельно для обучения и validation. Тот же объект сохраняется в `provenance` итоговой записи метрик. Журнал содержит `{type: step, loss, flops}` на каждый шаг, `{type: validation, loss, accuracy, flops}` на каждую проверку и `{type: result, ...}` с итогом запуска.
* **Hugging Face** — с флагом `--hf-repo user/model` или переменной `HF_REPO` из `.env` каталог результата целиком сливается в приватный репозиторий (путь `runs/<имя каталога>`); нужен токен в переменной `HF_TOKEN`. Приоритет: `--hf-repo` → `HF_REPO` (окружение) → `HF_REPO` (файл `.env`); если ничего не задано, выгрузки нет.

Разовый вход в ClearML (ключи с app.clear.ml):

```sh
uv run --locked clearml-init
```

На тренировочных серверах вместо `clearml-init` удобен `.env.example`: скопируйте его в `.env`, заполните ключи — команда `train` сама читает `.env` из текущего каталога (переменные окружения имеют приоритет над файлом). SDK принимает и переменные окружения, и `~/clearml.conf`.

Трекер выбирается флагом: `--tracker clearml` (по умолчанию) или `--tracker disabled` — для запусков без сети.

Прогресс дублируется обычными строками в stdout и `train.log`: шаг, FLOPs и доля бюджета, loss, learning rate, токены в секунду и оценка оставшегося времени. В терминале дополнительно остаётся `tqdm` с текущим loss.

Частоты задаются в секции `schedule` или флагами CLI:

| YAML | CLI | По умолчанию |
| --- | --- | --- |
| `log_every_steps` | `--log-every-steps` | 10 шагов; первый и последний шаг пишутся всегда |
| `val_every_steps` | `--val-every-steps` | 100 шагов; в конце validation выполняется всегда |

Validation проверяет весь split `val`, без градиентов. Loss и accuracy учитывают только позиции с целевыми метками, без `IGNORE_INDEX`. Операции validation не входят в бюджет train FLOPs; её время не входит в скорость обучения и оценку оставшегося времени. Если последний шаг совпал с интервалом validation, повторной проверки нет.

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

Для новых серий рекомендуется `scripts/train_kaggle_series.py`: один зафиксированный release всех конфигов, общий проверенный read-only KV input, отдельная доставка credentials и журнал возобновления. Порядок подготовки, публикации, выборочного запуска и проверок описан в `docs/kaggle-series.md`. Ниже сохранён совместимый прежний интерфейс, который собирает пакет отдельно для каждого запуска.

```sh
uv run --locked scripts/train_kaggle.py --config experiments/smoke.yaml
# все YAML-файлы папки, последовательно:
uv run --locked scripts/train_kaggle.py --config experiments --slug compare-attention-train
# эквивалентно зеркальным флагам:
uv run --locked scripts/train_kaggle.py --train-flops 1000000000 --n-layer 2
```

С `--config` YAML-файл эксперимента копируется в тот же приватный dataset (внутри копии `data` переписывается на `/kaggle/working/kv_retrieval`, а секция `kv` задаёт подготовку данных в ядре). Ядро запускает `train --config /kaggle/input/.../<имя>.yaml`. Параметры обучения, включая частоты логов и validation, берутся из YAML. Параметры запуска Kaggle, например `--slug` и `--accelerator`, задаются локальными флагами CLI.

Схема: wheel проекта собирается локально (`uv build --wheel`), загружается в приватный Kaggle dataset, генерируется kernel-скрипт и запускается на выбранном GPU (`--accelerator`, по умолчанию T4). Скрипт проверяет статус и после завершения скачивает артефакты в `kaggle/<slug>/output`.

Если `--config` указывает папку:

* Скрипт берёт только файлы `.yaml` и `.yml` непосредственно в этой папке. Подпапки и другие файлы пропускаются. Порядок запуска — сортировка по имени файла.
* Каждый эксперимент получает slug `<базовый-slug>-<имя-файла-без-расширения>`. Например, `classic.yaml` с базой `compare-attention-train` сохраняет результаты в `kaggle/compare-attention-train-classic/output`.
* Следующий эксперимент начинается только после завершения предыдущего и успешного скачивания его артефактов. При первой ошибке очередь останавливается с ненулевым кодом.
* Все YAML-файлы и коллизии slug проверяются до сборки и загрузки. Пустая папка, отсутствующий путь или некорректный конфиг останавливают запуск до обращения к Kaggle.
* Wheel собирается и загружается отдельно для каждого эксперимента. Это последовательные запуски, а не параллельные задания в Kaggle.

В режиме папки базовый slug должен содержать 1–48 символов: строчные ASCII-буквы, цифры и одиночные внутренние дефисы. Имя файла переводится в нижний регистр, разделители заменяются дефисами; итоговый slug ограничен 50 символами. Имена без ASCII-букв и цифр не поддерживаются. Совпадения после преобразования или усечения считаются ошибкой. Для одного файла slug остаётся без суффикса.

Креды Kaggle: `KAGGLE_USERNAME`/`KAGGLE_KEY` из `.env` (скрипт сам читает `.env`); `kaggle.json`/`kaggle auth login` тоже работают.

Креды ClearML/HF внутри kernel: опираются на Kaggle Secrets — **но Kaggle передаёт их только в запусках, начатых из UI**, поэтому основной механизм другой. Локальный `.env` сокращается до корзины ключей (только `CLEARML_API_*`, `HF_TOKEN`, `HF_REPO`; `KAGGLE_*` никогда не попадают) и кладётся файлом `kaggle-env.env` в тот же приватный dataset, откуда kernel восстанавливает их в окружение. Датасет держать приватным; для страховки — отзываемые ключи: отдельная пара API-ключей ClearML и fine-grained HF-токен только на этот репозиторий.

## Тесты

```sh
uv run --locked pytest            # unit + e2e + лёгкие интеграционные
uv run --locked pytest -m docker  # e2e: сервер ClearML в docker compose (heavy)
```

Слои: `tests/compare_attention/**` — unit (зеркалят `src/`); `tests/integration/` — ClearML SDK в официальном offline-режиме (без сети); `tests/e2e/train/` — полные прогоны CLI; `tests/e2e/clearml/` — настоящий сервер через docker compose, помечены `docker` и по умолчанию выключены.
