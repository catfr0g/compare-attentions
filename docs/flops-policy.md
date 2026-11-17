# Политика алгоритмических FLOPs

## Назначение и версия

Идентификатор политики: `algorithmic_matmul_conv_v2`.
Версия JSON-схемы: `metadata.schema_version = 1`.
Архитектура: `repository_gpt2_v1`, то есть текущий `models/gpt2.py`:
абсолютные embeddings, pre-norm, MLP ширины `4D`, tied vocabulary head.

Метрика определяет алгоритмическую стоимость включённых операторов одного
полного обучающего batch. Это **не** число аппаратных инструкций и **не** полная
стоимость всех floating-point операций обучения. Старые наблюдаемые значения
`FlopCounterMode`, пропускающие пользовательские FLA/Triton-ядра, нельзя сравнивать
с этой метрикой как с той же единицей бюджета.

Изменение формулы, включённых операций, causal/padding-соглашения или политики
recompute требует нового идентификатора политики. Изменение JSON-контракта
требует новой версии схемы. Интеграция должна включать идентификатор политики,
версию схемы и фактическую форму batch в fingerprint стоимости/запуска. Нельзя
повторно использовать старую стоимость после изменения этих значений. Версия
метрики сама по себе не является ключом кэша компилятора: runtime, исходный код,
полная конфигурация и аппаратная платформа должны оставаться в его идентичности.

## API и контракт результата

```python
estimate_step(config: TrainConfig) -> StepFlops
```

`StepFlops` — frozen dataclass. Верхнеуровневые поля:

| Поле | Тип | Значение |
| --- | --- | --- |
| `policy` | `str` | `algorithmic_matmul_conv_v2` |
| `forward` | `int` | Сумма прямого прохода по компонентам |
| `backward` | `int` | Сумма градиентов и предусмотренного FLA recompute |
| `total` | `int` | Строго `forward + backward` |
| `breakdown` | `dict[str, dict[str, int]]` | Компонент → `{"forward": int, "backward": int}` |
| `metadata` | `dict[str, object]` | Соглашения и фактические размеры |

`to_dict() -> dict` возвращает отделённую JSON-сериализуемую копию всех шести
полей. Изменение вложенного словаря копии не меняет исходный результат. `frozen`
запрещает присваивание полей dataclass; это не обещание глубокой неизменяемости
содержащихся в нём словарей. Вызывающий код не должен менять исходные словари.

Ключи `metadata`:

- `schema_version`: `1`;
- `architecture`: `repository_gpt2_v1`;
- `attention`: `classic` или `deltanet`;
- `shape`: целые `batch_size`, `sequence_length`, `hidden_size`, `num_heads`,
  `num_layers`, `vocab_size`;
- `mac_flops`: `2`;
- `included`: для classic `["dense_matmul"]`; для DeltaNet
  `["dense_matmul", "convolution", "delta_core_contractions"]`;
- `causal_convention`: для classic `dense_square`; для DeltaNet
  `fla_causal_recurrence_or_chunk64`;
- `padding_convention`: `logical_tokens_no_kernel_padding`;
- `recompute`: для classic `none`; для DeltaNet `fla_default_backward_recompute`;
- `excluded`: `optimizer`, `non_contraction_elementwise`,
  `non_contraction_reductions`, `embedding_gather_scatter`, `loss`, `validation`,
  `kernel_padding`, `hardware_instructions`.

Компоненты суммированы по всем слоям; `lm_head` учитывается один раз. В classic
имена компонентов: `attention.qkv`, `attention.output`, `attention.qk`,
`attention.av`, `mlp.expand`, `mlp.contract`, `lm_head`. В DeltaNet все ключи
`delta_attention_cost` получают префикс `attention.` и множитель числа слоёв.
Нулевые компоненты сохраняются, чтобы явно показывать исключённые FLA norms.

## Фактическая форма и границы применимости

Обозначения: `B = schedule.batch_size`, `T = dataset.input_seq_len`,
`D = model.n_embd`, `H = model.n_head`, `L = model.n_layer`, `V = model.vocab_size`.
`model.block_size` — ёмкость таблицы позиций, **не** длина вычисляемой
последовательности. Требуется `block_size >= T`.

Стоимость относится к полному batch из `B` строк. Циклическая выборка строк
обучения должна всегда давать ровно `B` строк; неполный последний batch имеет
другую стоимость и не покрывается постоянным значением для полного batch.
Функция не читает содержимое датасета и не проверяет реализацию цикла выборки.
Перед созданием модели рабочий цикл отдельно сверяет фактические train/val inputs
и labels с `(n_split, input_seq_len)` и `int64`. Несогласованные массивы отклоняются
до оценки стоимости и создания optimizer; наличие корректной metadata не заменяет
проверку загруженных тензоров.

Поддерживаются только точные типы текущих `TrainConfig`, `GPT2Config`, `Schedule`
и `KVRetrievalConfig`, полностью обучаемый GPT-2 и ось `train_flops`.
Поля повторно валидируются, в том числе после обходящих validation
`model_copy`/`model_construct`. Проверяются совместимость vocab, вместимость
block, размер train split и dtype длинного DeltaNet. Неизвестное внимание,
невалидные размеры, неизвестные типы/подклассы и неподдерживаемые FLA режимы
вызывают `ValueError`/`NotImplementedError`, а не частичный положительный счёт.

Нет произвольного freezing параметров, KV cache, prefix/state input,
variable-length packing, дополнительного activation checkpointing или
кастомных масок. Эти режимы требуют нового поддерживаемого контракта.
`compile`, dtype, bias и dropout не меняют сумму включённых операций при
одинаковых допустимых формах. Это не утверждение об одинаковом времени их работы.

Оценка не строит модель, не делает forward/backward/optimizer step, не читает
тензоры параметров и не изменяет RNG. Для DeltaNet проверяется версия и backend
установленного пакета без импорта/запуска CUDA-ядра.

## Включённые и исключённые операции

Один multiply-accumulate стоит 2 FLOPs: умножение и сложение. Плотное произведение
`[m,k] @ [k,n]` стоит `2mkn`; крайняя сумма также считается по MAC-соглашению,
а не как `2k-1` на выходной элемент.

Включены все dense linear/matmul GPT: Q/K/V, выходная проекция внимания, обе
MLP-проекции и vocabulary head, а также соответствующие матричные градиенты.
Для DeltaNet включены свёртки и математические contractions состояния/чанков,
даже если Triton выражает contraction через scalar multiply/reduce или outer
product. Это не основание учитывать произвольные прочие elementwise/reduction
операции. Точные FLA формулы и recompute определены в `docs/flops-delta.md`.

Исключены bias additions и bias-gradient sums, residual/embedding additions,
GELU/SiLU/sigmoid, softmax, LayerNorm/RMSNorm/L2Norm, scaling/gating, dropout,
маски и их построение, прочие не-contraction scalar/reduction операции,
cross-entropy, clipping, optimizer/AdamW/weight decay, zero_grad,
gradient accumulation additions, gather/scatter embeddings, перемещения
памяти, инициализация, компиляция и validation. Эти исключения одинаковы для
обеих архитектур. Embedding lookup не является dense GEMM; градиента по целым
индексам нет. Scatter-add в embedding weights не считается отдельным matmul.

## Classic: формулы прямого и обратного проходов

| Компонент | Forward | Backward |
| --- | --- | --- |
| Q/K/V projections | `6 B T D² L` | `12 B T D² L` |
| Attention output projection | `2 B T D² L` | `4 B T D² L` |
| QKᵀ | `2 B T² D L` | `4 B T² D L` |
| AV | `2 B T² D L` | `4 B T² D L` |
| MLP D→4D | `8 B T D² L` | `16 B T D² L` |
| MLP 4D→D | `8 B T D² L` | `16 B T D² L` |
| Vocabulary head | `2 B T D V` | `4 B T D V` |

Итого `forward = L (24 B T D² + 4 B T² D) + 2 B T D V`.
Для каждого включённого classic matmul существуют две матричные производные:
по floating-point input и по весам/второму операнду. Они дают `backward = 2 ×
forward` **для этих операторов**, а не универсальное правило стоимости всей
модели. Tied head не добавляет вторую head-проекцию: вычисление градиента
общего веса через vocabulary head уже включено. Градиенты embeddings остаются
исключёнными scatter-операциями.

Для DeltaNet нельзя умножать общую сумму forward на 3. Backward берётся из
отдельных компонентов `delta_attention_cost`, включая предусмотренный
recompute свёрток, WY и состояния.

## Causal и padding

Classic использует канонические **плотные квадратные** QKᵀ/AV на всех `T × T`
позициях. Причинная маска не даёт множителя 1/2: треугольная оптимизация Flash
Attention является свойством выполнения, а не данной метрики. Плотный PyTorch
SDPA math backend служит проверяемым эталоном этих matmul.

Все логические `T` позиции и `V` logits включаются независимо от `IGNORE_INDEX`
и числа supervised tokens. Стоимость loss исключена; нулевой математический
градиент на masked labels не удаляет строки из канонического dense GEMM.
Не учитываются аппаратные tile padding и внутренние округления размеров.
Для causal convolution считаются полные kernel taps, включая умножения на
нулевое левое дополнение; число выходов равно `T`. DeltaNet учитывает только
поддержанные полные чанки либо short-sequence recurrence. Неполные длинные
чанки отклоняются; нельзя молча округлять T вниз.

## Проверки

`tests/compare_attention/train/test_flops.py` проверяет явный маленький breakdown,
JSON-контракт, независимое перечисление scalar MAC, исключения/формы, чистоту
оценки и fail-closed режимы. Полный маленький GPT-2 с tied weights и двумя
слоями сверяется по forward и backward с `FlopCounterMode` на CPU, SDPA MATH,
с bias и без bias. Это тест модели, не runtime-проба оценщика; optimizer не
создаётся. RNG теста изолирован через `fork_rng`.

Свёртка сверяется с зарегистрированным счётчиком на одном независимом канале,
после чего стоимость суммируется по каналам и трём q/k/v потокам. Backward
включает отдельный повторный forward свёртки. В установленном PyTorch
`2.11.0+cu128` зарегистрированная формула grouped convolution weight-gradient
игнорирует аргумент `_groups`: агрегированный depthwise backward не является
полным эталоном. Сверка использует эквивалентный ungrouped single-channel
случай, где покрытие формулы полное, а не подгоняет политику под этот дефект.

Для production classic `B=8,T=512,D=768,H=12,L=12,V=8192`:
`forward=824633720832`, `backward=1649267441664`,
`total=2473901162496` FLOPs на полный batch. Эти значения получены из формул;
это не измерение GPU и не прогноз скорости. DeltaNet имеет отдельные
reference-тесты и ограничения, описанные в `docs/flops-delta.md`.
