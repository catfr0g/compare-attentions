# FP16: конфигурации и облачные проверки

## Область изменения

Пользователь согласовал отдельную проверку FP16 для обеих архитектур. Старые
BF16-файлы `experiments/{classic,deltanet}/seed_{42,6767,696969}.yaml` не изменены.
Новая полная матрица: `experiments/fp16/{classic,deltanet}/seed_{42,6767,696969}.yaml`.
Каждый файл отличается от соответствующего BF16-файла только
`schedule.dtype: float16`. Общие параметры: batch size 8, compile true,
бюджет 1 000 000 000 000 000 FLOPs. Полные отображения проверяются тестом,
включая загрузку через настоящий parser.

Точные одношаговые конфигурации (seed 42):

| Файл | Бюджет FLOPs |
| --- | ---: |
| `experiments/fp16/smoke/classic.yaml` | 2473901162496 |
| `experiments/fp16/smoke/deltanet.yaml` | 2399573901312 |

## Контракт численной точности

FP16 означает AMP: параметры и master weights FP32, autocast FP16,
GradScaler включён. Завершённый `run_identity.json` содержит объект `precision`:

```json
{
  "mode": "amp_float16",
  "parameter_dtype": "float32",
  "autocast_dtype": "float16",
  "grad_scaler_enabled": true,
  "optimizer_steps": 1,
  "skipped_optimizer_steps": 0,
  "initial_scale": 128.0,
  "final_scale": 128.0
}
```

Для обеих архитектур явно задан общий начальный scale `128.0`. Конечный scale
может меняться; gate требует конечных положительных initial/final scale.
Счётчики имеют целочисленный тип
(не bool). Для smoke нужен один реальный optimizer update и ноль пропусков.
Начальный и конечный fingerprints параметров должны быть корректными,
различными SHA256 (64 шестнадцатеричных символа). При FP32 master weights
равенство хешей не объясняется преобразованием dtype и блокирует продолжение.

API `verify_flops_smoke(..., expected_dtype="float16", expected_gpu="Tesla T4")`
сохраняет старые вызовы: `expected_dtype` по умолчанию `None`. FP16 всегда
проверяется по новому контракту, даже без явно переданного expected_dtype.
Identity.config должен полностью совпадать с result.config; result.provenance
должен полностью совпадать с завершённой identity. Сохраняются проверки бюджета,
единственной финальной validation, конечных метрик и успешного compiler backend
в training phase (`successful_training_backend_invocations > 0` и
`backend_invocation_verified is true`), а не только попытки компиляции.

Полные FP16-пилоты проходят тот же контракт precision: optimizer_steps равен
result.steps, пропусков нет. Конечные scale разных seeds **не обязаны совпадать**.
Режим и dtype каждого запуска проверяются независимо; конфигурации различаются
только seed, runtime и accounting должны совпадать. Для schema v2 compiler gate
также требует успешного training backend. Legacy schema v1 без FP16 сохраняется.

## Ранняя проверка runtime

Сгенерированный immutable-series kernel запускает runtime preflight **до** чтения
credentials, установки зависимостей и импорта FLA. Dtype и compile берутся из
`state.identity.configs[name].schedule`, а не из имени GPU или неявных defaults.
`enable_gpu: true` остаётся в metadata.

`/kaggle/working/runtime-preflight.json` содержит только разрешённые поля:
`torch`, `cuda`, `cuda_available`, `device_count`, `gpu` (name/index/compute_capability),
`triton`, `dtype`, `compile`, `native_bf16_supported`, `passed`, `failure`.
Нет выгрузки environment, credentials, URL или произвольного текста исключения.
Отсутствие CUDA/GPU или установленного Triton блокирует установку зависимостей.
Compiled BF16 дополнительно требует native BF16 без emulation. Автоматической
слепой установки Triton или подмены dtype нет.

Общий целевой фактический GPU для обеих smoke — `Tesla T4`. Запрошенный accelerator
не является доказательством выданного оборудования; точное имя проверяется по
завершённой identity. Preflight собирает фактическую CUDA/Triton metadata, но
не доказывает работоспособность всей FP16-модели. Это проверяют обе одношаговые
smoke с успешной training-компиляцией и реальным изменением параметров.

После успешных обеих одношаговых smoke требуется 32 успешных обновления каждой
архитектуры при том же общем scale, на том же T4 и из того же immutable release.
Только затем — отдельные classic-пилоты 42 и 6767, seed gate и остальные четыре
запуска. Создание конфигураций и локальные тесты сами по себе не запускают
облачные задания.

Начальный scale 128 — проверяемый кандидат, а не доказанное исправление T4.
Предыдущий T4-пилот со scale 1024 прошёл восемь обновлений и остановился на
следующем из-за переполнения. На RTX 5070 исходное переполнение не воспроизвелось;
обе production-модели при scale 128 прошли 32 локальных обновления без пропусков.
Это не доказывает устойчивость на T4 или на полном бюджете. Снижение scale также
меняет поведение малых FP16-градиентов, поэтому близкие loss не означают точного
равенства градиентов. Полные запуски остаются закрыты до облачной проверки.

## Прямой SDPA для classic

Classic сохраняет `nn.MultiheadAttention` как контейнер исходных параметров и
инициализации, но выполняет packed QKV → native SDPA → исходный out projection.
Новых Linear/RNG draws, изменения ключей checkpoint или tied weights нет.
Backend остаётся auto; другой CUDA kernel и полное устранение причины NaN этим
не гарантированы. DeltaNet, batch8, FP16/FP32 masters, scale128 и optimizer loop
не меняются.

Диагностический kernel
`zimike/ca-t4-direct-sdpa-diagnostic-20261102/1` прошёл32 actual updates на Tesla
T4,0 skips, finite loss/quality/actual FP32 checkpoint, успешные training и
validation backend returns. Начальный fingerprint совпал с неудачным MHA-путём.
Evidence: `local/diagnostics/t4-direct-sdpa/verified-candidate-gate.json`.
Это проверка отдельного кандидата, не допуск полной серии: после переноса
в production source нужен новый immutable release и повтор обеих одношаговых
и обеих32-шаговых проверок. Два seed-пилота перед остальными четырьмя сохранены.

## Политика FLOPs

`algorithmic_matmul_conv_v2` и формулы не меняются при переходе BF16 → FP16.
Это алгоритмический счётчик matmul/conv, не полный hardware FLOPs; активации и
скалярные операции optimizer исключены. Действуют `docs/flops-policy.md` и
`docs/flops-delta.md`. FP16 не разрешает переоценивать стоимость или смешивать
старые dispatcher totals с v2 под одним equal-compute label.
