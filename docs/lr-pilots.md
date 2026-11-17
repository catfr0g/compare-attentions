# Пилоты постоянного learning rate (LR)

## Статус и границы

Подготовлены 12 YAML: LR `1e-3` и `1e-4`, Classic и DeltaNet,
этапы `exact1`, `exact32`, `full`. Выполнены только CPU-проверки.
**Release НЕ staged, НЕ опубликован, kernel НЕ отправлен.**
Последнее указание родителя запрещает staging во время параллельной разработки scheduler.
Исходная привязка этого кандидата: `12a32eaef77d1d2f13a0d8f67497be3d86c411932067e39c631d4b16fc51813f`.
Она не является разрешением на новую scheduler-реализацию.
Изменение production source другим владельцем ожидаемо; этот оператор его не изменяет.

Запрос scheduler выполняет отдельный владелец. Здесь сохранены только constant-LR
контроли. В raw YAML новых полей scheduler нет. При сравнении старого B32 Optim
нормализуются только отсутствующие значения по реальным defaults `Optim`;
любая effective policy, отличная от `constant`, запрещена.
После интеграции scheduler нужны новый freeze, полная регрессия, независимый review,
новые pins/evidence в НОВОМ LR-операторе и отдельное разрешение родителя.
Старые B32 baseline, оператор, helper и approval нельзя переписывать.

## Матрица и расчет

Полные пилоты отличаются от соответствующего B32 seed42 baseline **только**
`optim.learning_rate`. B32, FP16, compile, seed42, ES off, amp max128, model и данные
не изменены. Бюджет полного пилота — целое `10000000000000000` FLOP.
Gate-конфиги дополнительно меняют только `schedule.train_flops` и
`checkpoint_every_steps` на exact horizon 1 или 32.
Периоды validation/log сохраняются равными 25; checkpoint полного пилота — 125.
`costs.json` содержит breakdown из реальных `TrainConfig`/`estimate_step`,
целочисленный ceil, выполненный бюджет и overshoot для каждого YAML.

| Семейство | FLOP / update | Полных updates | Exact32 FLOP |
|---|---:|---:|---:|
| classic | 9895604649984 | 1011 | 316659348799488 |
| deltanet | 9598295605248 | 1042 | 307145459367936 |

## Исполняемый оператор

`local/experiments/fp16-lr-pilots/run_series.py` не выполняет cloud calls на import.
По умолчанию и с `--check` выполняется только fail-closed offline preflight.
Без `candidate.json` он сообщает `LR release not staged`.
`--launch` читает dotenv **только после** успешной проверки нового LR approval.
Ни этот документ, ни `review-ready.json` не являются approval.

Разрешаемая область будущего запуска — строго шесть singleton jobs LR1e-3:
Classic exact1 → полный scientific gate + сравнение с B32 → DeltaNet exact1 → gate →
Classic exact32 → gate → DeltaNet exact32 → gate → Classic full → gate → DeltaNet full → gate → STOP.
LR1e-4 остается только подготовленной матрицей; B8 и повтор LR3e-4 не запускаются.
При ошибке сохраняется `lr-results.json` с предыдущими принятыми результатами и причиной
отказа. Следующий kernel не отправляется. Resume повторяет проверки скачанных результатов,
но не дает разрешения на автоматический retry failed/ambiguous journal.

Используются неизмененные `verify_stage`/`verify_pilot` и full-snapshot проверки старого
controls helper с SHA256 `fa184277a3d0f82ec5b75d8ab0d965f04a0aa42de74f52ada7d3eed262dbd657`.
Его globals не изменяются и main не вызывается. Чистые bundle/source-evidence проверки
переиспользуются из hash-bound B32 reference operator; его launch approval не используется.

На КАЖДОМ скачанном артефакте дополнительно сравниваются вся runtime mapping и actual
initial-parameter fingerprint с full B32 baseline ТОГО ЖЕ семейства. Полные frozen configs,
data identity и run config сравниваются без исключений, кроме intended LR и gate budget/cadence.
Runtime mismatch означает отказ, а не matched scientific result. GPU bootstrap сохраняет
существующий T4-first hardware gate до dependencies/credentials/training.
Совпадение всех software versions можно подтвердить только после скачивания результата;
оператор не обещает version pin/install.

Preflight связывает полный manifest/config/data/source, archive, unpacked bundle/wheel,
оба helper hashes, operator и собственные тесты/XML, исходные 745-test/source-review evidence,
B32 baseline manifest/identity checksums, и НОВЫЕ `lr-review.json`, `lr-cloud-approval.json`.
Сейчас старые source pins намеренно блокируют новый source. При интеграции нельзя просто
сменить SOURCE_SHA256: source-evidence чтение из reference также должно быть заменено
свежей полной регрессией/review нового source в новом LR evidence, без изменения reference.

## Команды

Запускать из корня репозитория. Проверка не требует `.env` и не публикует данные:

```bash
HF_HUB_OFFLINE=1 HF_TOKEN= HF_REPO= uv run --locked python local/experiments/fp16-lr-pilots/run_series.py --check
HF_HUB_OFFLINE=1 HF_TOKEN= HF_REPO= uv run --locked pytest tests/compare_attention/cli/test_fp16_lr_pilots.py -q --basetemp C:/Users/Michael/AppData/Local/hermes/cache/scratch/lr-prep-review --junitxml local/experiments/fp16-lr-pilots/offline-tests.xml
```

После стабильного source, fresh full regression/review и обновления нового оператора
родитель может выполнить ОДИН offline `stage_series` для всех 12 имен с prepared data и
owner из B32 baseline. Cache: `E:/HermesArtifacts/compare-attention/fp16-lr-pilots`.
Затем родитель записывает точный manifest в `candidate.json`, получает `bindings(manifest)`
и только после независимого review создает `lr-review.json` / `lr-cloud-approval.json`.
Только последующий отдельный parent launch может использовать `--launch`.
Здесь staging намеренно не выполнялся; `prepare_matrix.py` сохраняет старый source pin
и откажется регенерировать матрицу на измененном source.

## Полные YAML (четыре пилота)

### `experiments/fp16_lr_pilots/1e-3/full/classic/seed_42.yaml`

```yaml
data: data/kv_retrieval
schedule:
  train_flops: 10000000000000000
  batch_size: 32
  seed: 42
  dtype: float16
  compile: true
  checkpoint_every_steps: 125
  checkpoint_keep_last: 2
  early_stopping_patience: 0
  early_stopping_min_delta: 0.001
  early_stopping_min_steps: 2000
  amp_max_scale: 128.0
  val_every_steps: 25
  log_every_steps: 25
optim:
  learning_rate: 0.001
  weight_decay: 0.1
model:
  vocab_size: 8192
  block_size: 512
  n_layer: 12
  n_head: 12
  n_embd: 768
  dropout: 0.0
  bias: true
  attention: classic
```

### `experiments/fp16_lr_pilots/1e-3/full/deltanet/seed_42.yaml`

```yaml
data: data/kv_retrieval
schedule:
  train_flops: 10000000000000000
  batch_size: 32
  seed: 42
  dtype: float16
  compile: true
  checkpoint_every_steps: 125
  checkpoint_keep_last: 2
  early_stopping_patience: 0
  early_stopping_min_delta: 0.001
  early_stopping_min_steps: 2000
  amp_max_scale: 128.0
  val_every_steps: 25
  log_every_steps: 25
optim:
  learning_rate: 0.001
  weight_decay: 0.1
model:
  vocab_size: 8192
  block_size: 512
  n_layer: 12
  n_head: 12
  n_embd: 768
  dropout: 0.0
  bias: true
  attention: deltanet
```

### `experiments/fp16_lr_pilots/1e-4/full/classic/seed_42.yaml`

```yaml
data: data/kv_retrieval
schedule:
  train_flops: 10000000000000000
  batch_size: 32
  seed: 42
  dtype: float16
  compile: true
  checkpoint_every_steps: 125
  checkpoint_keep_last: 2
  early_stopping_patience: 0
  early_stopping_min_delta: 0.001
  early_stopping_min_steps: 2000
  amp_max_scale: 128.0
  val_every_steps: 25
  log_every_steps: 25
optim:
  learning_rate: 0.0001
  weight_decay: 0.1
model:
  vocab_size: 8192
  block_size: 512
  n_layer: 12
  n_head: 12
  n_embd: 768
  dropout: 0.0
  bias: true
  attention: classic
```

### `experiments/fp16_lr_pilots/1e-4/full/deltanet/seed_42.yaml`

```yaml
data: data/kv_retrieval
schedule:
  train_flops: 10000000000000000
  batch_size: 32
  seed: 42
  dtype: float16
  compile: true
  checkpoint_every_steps: 125
  checkpoint_keep_last: 2
  early_stopping_patience: 0
  early_stopping_min_delta: 0.001
  early_stopping_min_steps: 2000
  amp_max_scale: 128.0
  val_every_steps: 25
  log_every_steps: 25
optim:
  learning_rate: 0.0001
  weight_decay: 0.1
model:
  vocab_size: 8192
  block_size: 512
  n_layer: 12
  n_head: 12
  n_embd: 768
  dropout: 0.0
  bias: true
  attention: deltanet
```

