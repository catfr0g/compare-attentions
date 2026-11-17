# Парное сравнение общего batch size: FP16, seed 42

Статус: подготовлены конфиги, но ничего не staged, не опубликовано и не запущено.
GPU, обучение, cloud и подбор LR не выполнялись. Вместимость B32 в память и скорость не проверены.
Активная immutable-серия и production-код не изменены.

## Контракт сравнения

Четыре полных пилота: classic и DeltaNet при B8 и B32. Бюджет каждого —
10 000 000 000 000 000 training FLOPs. LR=0.0003, AdamW weight_decay=0.1;
T512, D768, H12, L12, V8192, dropout=0, bias=true, FP16 с FP32 master weights,
compile=true, amp_max_scale=128, train seed=42 и подготовленный dataset seed=1337.
Все YAML содержат полные mappings, а не overlays. Модель и data идентичны исходным
`fp16_long_controls/{family}/seed_42.yaml`; разрешённые изменения только в schedule:
бюджет, batch, val/log/checkpoint cadence, retention=2 и early_stopping_patience=0.
Неактивные min_delta=0.001 и min_steps=2000 сохранены, других control-изменений нет.

| Batch | Validation | Logging | Checkpoint | Keep | Early stopping |
|---|---:|---:|---:|---:|---|
| 8 | 100 | 100 | 500 | 2 | OFF |
| 32 | 25 | 25 | 125 | 2 | OFF |

При B32 цена update ровно в четыре раза выше в каждой семье. Поэтому интервалы
оценки, логирования и сохранения уравнены по training FLOPs и числу примеров:
validation/logging после каждых 800 примеров, checkpoint после каждых 4000.
Первый/последний log и финальная validation — обязательные события trainer, не
дополнительные периодические интервалы. I/O и validation не входят в training FLOPs.
Финальная оценка выполняется и при бюджете короче интервала (smoke).

Batch увеличивает число независимых последовательностей в одном update, а не
доступный контекст: каждая последовательность по-прежнему имеет длину 512.
При фиксированном compute B32 даёт примерно в четыре раза меньше optimizer updates.
Одинаковый интервал validation в updates дал бы B32 вчетверо более редкие наблюдения
по compute/примерам. Early stopping отключён, чтобы patience и warm-up не меняли
достигнутый бюджет или критерии остановки между batch sizes. Одинаковый LR здесь
контролируемая исходная точка, не доказанный оптимум для B32.

## CPU-оценка стоимости и горизонтов

Цена вычислена настоящим `estimate_step` из настоящего `TrainConfig`, построенного
по неизменённой metadata prepared dataset. Перед оценкой проверены реальные mmap
inputs/labels всех трёх splits: int64 и объявленные формы. Ни модель, ни GPU не запускались.
Полные breakdown, policy, shape, SHA256 конфигов, overshoot и metadata сохранены в
`experiments/fp16_batch_pilots/costs.json`. Policy — `algorithmic_matmul_conv_v2`.
Количество updates вычислено без float: `(budget + cost - 1) // cost`.
Финальный budget overshoot допустим только меньше цены одного update.

| Конфиг | FLOPs/update | Ожидаемые updates |
|---|---:|---:|
| `classic/batch_8/seed_42.yaml` | 2473901162496 | 4043 |
| `classic/batch_32/seed_42.yaml` | 9895604649984 | 1011 |
| `smoke/classic/batch_32/seed_42.yaml` | 9895604649984 | 1 |
| `burnin/classic/batch_32/seed_42.yaml` | 9895604649984 | 32 |
| `deltanet/batch_8/seed_42.yaml` | 2399573901312 | 4168 |
| `deltanet/batch_32/seed_42.yaml` | 9598295605248 | 1042 |
| `smoke/deltanet/batch_32/seed_42.yaml` | 9598295605248 | 1 |
| `burnin/deltanet/batch_32/seed_42.yaml` | 9598295605248 | 32 |

## Порядок допуска, без запуска

Для каждой семьи подготовлены свежие B32 smoke (ровно 1 update, checkpoint_every_steps=1)
и burn-in (ровно 32 updates, checkpoint_every_steps=32). Все прочие поля совпадают
с соответствующим B32 full pilot, включая early stopping OFF и val/log cadence=25.
Это численные gates, не научные реплики. Сначала нужны отдельное разрешение,
проверка неизменности source/data/configs, затем fresh gates обеих семей; только после
их acceptance можно отдельно staged первую пару полных B32-пилотов. Этот документ
и создание YAML не дают разрешения на запуск и не утверждают, что gates пройдены.

B8-префикс существующей серии — лишь кандидат baseline. Не объявлять его автоматически
эквивалентным: старые early-stopping и checkpoint/log controls отличаются. Даже если
warm-up 2000 исключал early stop на рассматриваемом префиксе, нужны проверка полной
runtime/provenance identity и одинаковый attained compute, data cursor и события оценки.
Если это нельзя доказать, использовать новые чистые B8-конфиги. Следующие seeds и DDP
этой подготовкой не разрешены.

## Воспроизведение и проверки

CPU-only helper: `local/diagnostics/fp16-batch-pilots-prep/prepare.py` (ignored).
Запуск из корня checkout: `.venv/Scripts/python.exe local/diagnostics/fp16-batch-pilots-prep/prepare.py`.
Он проверяет prepared arrays, записывает только новую матрицу и cost ledger; training/cloud не вызывает.
Точный prepared-data путь и SHA metadata находятся в ledger.

`tests/compare_attention/cli/test_fp16_batch_pilots.py` проверяет точный набор восьми YAML,
полные raw mappings с нормализацией только разрешённых полей, реальные YAML/CLI parser
и TrainConfig, common settings двух семей при каждом batch, цены, integer-ceil horizons
и одинаковые периодические cadence по compute/примерам внутри семьи. Переносимая fixture
содержит только точную metadata prepared dataset; реальная проверка массивов выполнена helper.

RED: отсутствие всей матрицы вызвало ожидаемый assertion failure. GREEN: матрица прошла.
Отдельный ledger-контракт также проверен RED на отсутствующем ledger и GREEN после его создания.
Targeted regression: 6 passed, включая прежние FP16/long/controls matrix tests.
Наблюдались существующие Torch JIT и Pydantic deprecation warnings; production ради них не менялся.
Ruff check/format прошли; Pylint нового теста — 10.00/10. Pre/post SHA256 всех 178
существовавших файлов source/tests/experiments и prepared data совпали, изменений — 0.
Снимки hashes, RED/GREEN JUnit и CPU helpers сохранены в ignored
`local/diagnostics/fp16-batch-pilots-prep/`. Все восемь YAML-блоков ниже отдельно
сверены с файлами, SHA256 восьми конфигов проверены по ledger.

## Полные YAML

### `classic/batch_8/seed_42.yaml`

```yaml
data: data/kv_retrieval
schedule:
  train_flops: 10000000000000000
  batch_size: 8
  seed: 42
  dtype: float16
  compile: true
  checkpoint_every_steps: 500
  checkpoint_keep_last: 2
  early_stopping_patience: 0
  early_stopping_min_delta: 0.001
  early_stopping_min_steps: 2000
  amp_max_scale: 128.0
  val_every_steps: 100
  log_every_steps: 100
optim:
  learning_rate: 0.0003
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

### `classic/batch_32/seed_42.yaml`

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
  learning_rate: 0.0003
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

### `smoke/classic/batch_32/seed_42.yaml`

```yaml
data: data/kv_retrieval
schedule:
  train_flops: 9895604649984
  batch_size: 32
  seed: 42
  dtype: float16
  compile: true
  checkpoint_every_steps: 1
  checkpoint_keep_last: 2
  early_stopping_patience: 0
  early_stopping_min_delta: 0.001
  early_stopping_min_steps: 2000
  amp_max_scale: 128.0
  val_every_steps: 25
  log_every_steps: 25
optim:
  learning_rate: 0.0003
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

### `burnin/classic/batch_32/seed_42.yaml`

```yaml
data: data/kv_retrieval
schedule:
  train_flops: 316659348799488
  batch_size: 32
  seed: 42
  dtype: float16
  compile: true
  checkpoint_every_steps: 32
  checkpoint_keep_last: 2
  early_stopping_patience: 0
  early_stopping_min_delta: 0.001
  early_stopping_min_steps: 2000
  amp_max_scale: 128.0
  val_every_steps: 25
  log_every_steps: 25
optim:
  learning_rate: 0.0003
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

### `deltanet/batch_8/seed_42.yaml`

```yaml
data: data/kv_retrieval
schedule:
  train_flops: 10000000000000000
  batch_size: 8
  seed: 42
  dtype: float16
  compile: true
  checkpoint_every_steps: 500
  checkpoint_keep_last: 2
  early_stopping_patience: 0
  early_stopping_min_delta: 0.001
  early_stopping_min_steps: 2000
  amp_max_scale: 128.0
  val_every_steps: 100
  log_every_steps: 100
optim:
  learning_rate: 0.0003
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

### `deltanet/batch_32/seed_42.yaml`

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
  learning_rate: 0.0003
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

### `smoke/deltanet/batch_32/seed_42.yaml`

```yaml
data: data/kv_retrieval
schedule:
  train_flops: 9598295605248
  batch_size: 32
  seed: 42
  dtype: float16
  compile: true
  checkpoint_every_steps: 1
  checkpoint_keep_last: 2
  early_stopping_patience: 0
  early_stopping_min_delta: 0.001
  early_stopping_min_steps: 2000
  amp_max_scale: 128.0
  val_every_steps: 25
  log_every_steps: 25
optim:
  learning_rate: 0.0003
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

### `burnin/deltanet/batch_32/seed_42.yaml`

```yaml
data: data/kv_retrieval
schedule:
  train_flops: 307145459367936
  batch_size: 32
  seed: 42
  dtype: float16
  compile: true
  checkpoint_every_steps: 32
  checkpoint_keep_last: 2
  early_stopping_patience: 0
  early_stopping_min_delta: 0.001
  early_stopping_min_steps: 2000
  amp_max_scale: 128.0
  val_every_steps: 25
  log_every_steps: 25
optim:
  learning_rate: 0.0003
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

