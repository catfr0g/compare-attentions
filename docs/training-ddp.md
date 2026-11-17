# Обучение с двумя процессами DDP

## Назначение и границы

DDP включается явно. По умолчанию остаётся однопроцессное обучение. Используется существующая цепочка `train → _run → _fit → _tracked_step → _step`, а не отдельный тренер. Поддерживается один узел, `world_size=2`, `LOCAL_RANK=RANK`.

Параметры модели, DeltaNet, AdamW, LR, dtype, compile и бюджет не изменяются автоматически. `schedule.batch_size` означает **глобальный** батч. При B32 каждый процесс обрабатывает 16 строк. Глобальный батч должен делиться на два без остатка. Нельзя запускать старый режим под torchrun с двумя процессами: без явного разрешения DDP тренер завершится с ошибкой до создания трекеров.

Локальные инженерные проверки используют CPU/Gloo, float32 и `compile=false`. Они не подтверждают CUDA, NCCL, целевой FP16/Inductor или ускорение на T4. Отдельный существующий CPU-Inductor тест завершился `Compiler: cl is not found`; компилятор не устанавливался и backend/gates не ослаблялись. Установленная FLA 0.6.0 требует Triton для causal convolution; реальный CPU-forward DeltaNet завершился ошибкой отсутствия активного драйвера. Подмена DeltaNet или разрешение GPU для обхода этой ошибки не применялись.

## Python API

```python
from compare_attention.train.loop import train

result = train(config, tracker=tracker, distributed=True)
# Продолжение: только тот же конфиг, топология и НОВЫЙ persistent result sink.
result = train(config, tracker=new_tracker, distributed=True, resume_from=checkpoint)
```

Вызов выполняется внутри двух процессов, запущенных torchrun. `distributed` и `resume_from` — операционные аргументы; они не добавлены в `TrainConfig` или `Schedule`. В CLI доступны `--distributed`, `--no-distributed`, `--resume-from`. YAML может содержать верхнеуровневый `distributed: true`. Явный CLI-флаг имеет приоритет. Значение false исключено из сериализации `TrainArgs`, чтобы сохранить старый формат по умолчанию.

## Запуск

Шаблон Linux для последующей проверки в разрешённой среде с двумя GPU:

```bash
uv run --offline --no-sync python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 \
  --no-python uv run --offline --no-sync python -m compare_attention.cli.train \
  --config EXPERIMENT.yaml --distributed --result-dir NEW_RESULT_DIR
```

CUDA-процесс выбирает `cuda:LOCAL_RANK`; backend — NCCL. Этот пример не означает, что облачный запуск или целевой FP16 уже проверены.

Проверенный вариант маршрута Windows/CPU использует новый FileStore rendezvous. Перед каждым локальным запуском необходимо установить CPU-mask и изолированные временные каталоги **до импорта Torch**. Файл среды для данной инженерной проверки находится в `local/reports/four-task-benchmark/ddp-implementation-v1/environment.sh`.

```bash
source local/reports/four-task-benchmark/ddp-implementation-v1/environment.sh
export USE_LIBUV=0 TORCH_DISABLE_SHARE_RDZV_TCP_STORE=1
uv run --offline --no-sync python -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 --rdzv-backend=c10d \
  --rdzv-conf=store_type=file --rdzv-endpoint=FRESH_ABSOLUTE_STORE_PATH \
  --local-addr=127.0.0.1 \
  --no-python uv run --offline --no-sync python -m compare_attention.cli.train \
  --config CPU_FIXTURE.yaml --distributed --dtype=float32 --no-compile \
  --tracker=disabled --result-dir NEW_RESULT_DIR
```

Путь rendezvous должен быть новым для каждого запуска. На установленном Windows Torch стандартный dual-stack Gloo зависал при создании группы. CPU-адаптер `gloo_ipv4` использует настоящий `ProcessGroupGloo` с явно заданным IPv4 loopback. Он не изменяет Linux/NCCL-путь. Стандартный Windows TCPStore rendezvous также требует обхода отсутствующего libuv; изменения установленных пакетов не нужны.

Реальный CLI `__main__` и функция `main()` проверяются под torchrun:

```bash
source local/reports/four-task-benchmark/ddp-implementation-v1/environment.sh
uv run --offline --no-sync pytest -q \
  tests/compare_attention/cli/test_distributed.py \
  tests/compare_attention/train/test_distributed.py \
  tests/compare_attention/train/test_distributed_checkpoint.py
```

## Строки, loss и FLOPs

Каждый шаг сначала использует прежний `_rows`: последовательный циклический глобальный батч, включая переход через конец пула. Он делится на два непересекающихся непрерывных участка. Глобальный cursor продвигается на B32, не на 16. Новых строк, padding-сэмплов и скрытых optimizer updates нет.

Каждый процесс вычисляет сумму CE и число supervised tokens. Знаменатель суммируется по процессам. Для backward используется `local_loss_sum * world_size / global_supervised_tokens`, поскольку DDP усредняет градиенты. Ранг без supervised labels участвует в backward с нулевым вкладом. Полностью пустой глобальный батч запрещён. Логируемый loss — глобальная сумма CE, делённая на глобальное число supervised tokens.

FLOPs начисляются только за успешный глобальный optimizer update. Каждый локальный фактический shape независимо оценивается тем же `estimate_step`. Сумма локальных стоимостей должна совпасть с глобальной стоимостью. Исходный глобальный B32 не умножается повторно на два. Индивидуальные оценки записаны в `provenance.distributed.ranks[*].accounting`.

## Синхронизация и трекеры

Трекеры стартуют только на rank0. Запрошенный seed применяется после их старта на обоих рангах, до создания модели. Сравниваются действительные initial parameter fingerprints.

Только rank0 пишет канонические loss/validation/LR events, checkpoints, model.pt, run_identity.json, train.log и выполняет финальный HF upload. Существующий пользовательский `Tracker.log_loss(loss, global_flops)` сохраняет назначение. Callback вызывается после подтверждения завершения optimizer step **всех** рангов и получает глобальный loss. Отдельный timing event в обычный журнал не добавлен. Это позволяет внешнему benchmark-трекеру измерять post-update границы через `monotonic_ns`.

Проверки finite loss и градиентов согласуются до optimizer step. Для активного GradScaler согласование выполняется после unscale и до scaler.step. Сохраняются FP32 masters, прежний cap и строгий запрет пропущенных обновлений. Финальные compiler/precision сведения собираются по рангам; компиляция не считается подтверждённой по одному флагу.

Группа имеет тайм-аут 60 секунд. Ошибки локальных фаз и primary callbacks передаются всем рангам; несовпадающий порядок фаз запрещён. Группа уничтожается в `finally`. При аппаратной/нативной ошибке время выхода также зависит от torchrun; локальные integration tests дополнительно ограничивают процессное дерево тайм-аутом.

## Checkpoint и возобновление

Однопроцессный checkpoint остаётся schema 1. DDP checkpoint — schema 2 с дополнительным ключом:

```text
distributed:
  topology: {world_size, global_batch_size, local_batch_size, sharding, device_type}
  ranks: [{rank, rng, identity}, ...]
```

Общая часть содержит прежние model/optimizer/precision/config/cursor/budget/loss/validation/stopping поля. Перед публикацией сравниваются хэши общей модели, optimizer/scaler и прогресса обоих процессов. Если они различаются, snapshot не публикуется. RNG сохраняется отдельно для каждого ранга: Python, NumPy, Torch CPU и только выбранное CUDA-устройство. В общей части RNG/identity rank0 остаются каноническими и должны точно соответствовать записи rank0.

Интерфейсы:

- `checkpoint.load_checkpoint(path_or_latest_json)` — безопасное чтение и проверка целостности; распознаёт обе схемы.
- `distributed_checkpoint.validate(payload)` — отдельная проверка DDP envelope, топологии, rank records, RNG-структуры, compiler и initialization identity. CPU-аудит CUDA RNG проверяет структуру, но не подтверждает реальное восстановление CUDA generator.
- `distributed_checkpoint.preflight(path, config)` — согласованный same-topology/config resume preflight до старта трекеров.
- При фактическом restore прежний строгий валидатор проверяет параметры, dtype, tied aliases, optimizer/scaler и реальный RNG выбранного устройства до загрузки. RNG повторно восстанавливается после replay истории в новый sink.

Нельзя продолжать exhausted или terminal early-stopped snapshot. Отсутствующие/повторные ранги, несовместимая топология, повреждённый RNG, несовместимый compiler и использованный result sink отвергаются. Проверка nested sinks выполняется на всех рангах до старта любого трекера и записи истории.

Публикует прежний `BackupTracker`: временный файл, flush/fsync, atomic replace, checksum pointer, затем pruning только managed файлов. `.owned.json` — **журнал до pruning**, а не точный retained inventory. Реальная последовательность save2/4/6/7 с keep2 даёт retained6/7 и journal4/6/7; это нормальная семантика производителя.

DDP result/provenance имеет **identity schema 3**. Старые single-device FLOPs, exploration и seed gates её отвергают. DDP envelope также отвергается прежним schema-1 restore/validator. DDP envelope validator не является разрешением использовать старые научные single-device gates или запускать основную серию.

## Численная проверка

На CPU проверены Classic GPT, sequence16, один слой, width16, float32, compile=false, **dropout=0** и четыре updates для сопоставления world1/world2 при B4 и B32. Эта проверка не утверждает эквивалентность случайных dropout masks между разными world sizes. Проверены нулевые labels одного ранга, неравные denominators и wrap. Все фактические rank-local строки сшиваются в исходный глобальный порядок. Initial tensors совпадают с одиночным запуском, а callback проверяет completion markers обоих optimizer steps. Loss сравнивается на каждом шаге с abs tolerance 2e-6. Параметры: rtol 2e-5, atol 2e-6; только аналитически нулевое направление MHA key bias допускает atol 5e-5 из-за усиления FP32 reduction roundoff через Adam epsilon. Это инженерный допуск, не изменение научных gates.

Uninterrupted и interrupted/resumed DDP с dropout сравниваются битово по весам, полному optimizer/scaler, cursor, loss/validation/stopping history, каждому rank RNG и следующим random draws после RNG-потребляющего tracker replay. Время исполнения не входит в битовую эквивалентность.
