# Последовательный benchmark: протокол v1

Статус: **Tiny Shakespeare подготовлен локально; облачное обучение не начато**. Это четыре семейства задач, а не усреднённый leaderboard.

## Порядок и границы

1. MQAR / KV retrieval: существующий результат принят. Старые веса и frozen-архивы не изменять; старый обученный DeltaNet больше не исследовать и не дообучать.
2. Tiny Shakespeare: новая пара Classic / текущий DeltaNet, оба с нуля, seed42. После проверки родительским агентом — последовательный запуск и научная приёмка каждого результата.
3. selective_count: только после приёмки Tiny. Сейчас только протокол; реализация и обучение отсутствуют.
4. permutation: после selective_count; сначала C5, затем S5 как варианты одного семейства. Сейчас только протокол.

Порядок уже согласован пользователем. Повторное согласование этого же порядка не требуется. Научный review конкретного бюджета, frozen-кандидата и операторских инструментов всё равно обязателен. Не запускать облако из этой подготовки.

## Tiny: данные и метрики

Команда подготовки: `uv run --no-sync python -m compare_attention.cli.prepare_tiny_shakespeare --output local/datasets/tiny-shakespeare-v1`. Повторная запись существующего каталога запрещена. Для воспроизведения без сети передать `--source` с точными upstream bytes. Исходник закреплён commit `6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e` репозитория karpathy/char-rnn, путь `data/tinyshakespeare/input.txt`, SHA256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`. UTF-8 decode строгий; CR/LF и Unicode не нормализуются. Никакого pickle, синтетического или скопированного вручную корпуса.

Разбиение **90/5/5 contiguous raw characters до окон**: границы `[0,1003854)`, `[1003854,1059624)`, `[1059624,1115394)`. Словарь — sorted alphabet только train; все heldout символы проверены на покрытие, 65 символов, UNK отсутствует. Это намеренно отличается от nanoGPT 90/10 и полного алфавита перед split. Рецепт nanoGPT закреплён commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`, `data/shakespeare_char/prepare.py`, SHA256 `d5fbea0686d146748cf36017d6c34136a2fe9c41f065f1429529ad55a992c692`. Его случайные перекрывающиеся train crops здесь не используются: сохраняется текущий детерминированный циклический row-stream trainer.

При context512 строки идут с шагом512. Логит в позиции t предсказывает следующий символ: `x[t]=chars[t]`, `y[t]=chars[t+1]`; дополнительного сдвига в loss нет. Соседние окна могут использовать один граничный символ как предыдущую target и следующую input, но каждая target внутри split встречается ровно один раз за проход. Окна не пересекают split. Первый символ каждого split — только input; межсплитовые переходы не оцениваются. Контекст сбрасывается на каждом окне; это оценка оконным контекстом до512, не sliding-prefix с полными512 предшественниками для каждого символа.

Хвост дополняется input id0 (это обычный символ newline), labels=-1; причинность исключает влияние будущего padding на валидные позиции. Ничего не отбрасывается кроме первых символов split как targets. Train/val/test: 1961/109/109 строк; 1003853/55769/55769 targets; 179/39/39 masked positions. Arrays int64. Все пути, raw-range, SHA256 исходника, сплитов и шести NPY — в `meta.json`. Загрузчик проверяет hashes всех файлов перед TrainConfig/pricing/eval.

Trainer сохраняет ordered rows, cyclic modulo, no shuffle. Полный batch32 продолжается через границу прохода; последний batch не отбрасывается. Существующий mean CE за update усредняет только немаскированные targets текущего batch; batch с tail имеет чуть меньше targets. Eval суммирует NLL по всем targets и делит на их точное число, включая неполный последний batch. **Primary**: test CE (nats/char), BPC=CE/ln2. **Secondary**: test character accuracy. Val CE/BPC — для кривых; test не используется для выбора бюджета/гиперпараметров. Не сравнивать character accuracy с MQAR accuracy.

## Tiny: фиксированная пара

`experiments/tiny_shakespeare/v1/{classic,deltanet}/seed_42.yaml`: 4 блока, width128, heads4, context512, batch32, vocab65, dropout0, bias=true; отличается только attention. Новый CLI selector `dataset_task` сериализуется для Tiny, но отсутствует в legacy KV dumps при значении по умолчанию; точный frozen K64 parser/security contract сохранён без изменения старых тестов/архивов. AdamW LR0.001 constant, WD0.1; FP16 с FP32 master weights, AMP cap128; compile=true, native backend gates; early stopping OFF. Snapshot каждые125 updates плюс final, keep_last2; val/log каждые25 updates плюс final. Никаких sweep, новых seed, изменения архитектуры, LR, cap или математического пути.

Число параметров отличается штатным attention: Classic867200, DeltaNet873472. FLOPs считаются существующим `algorithmic_matmul_conv_v2`: MAC=2, declared dense/conv/custom contractions; optimizer, loss, elementwise, validation и setup исключены. Полные padded logical shapes оплачиваются, несмотря на masked tail.

Исходные YAML сохраняют предложенные 10^16 FLOPs, **не разрешённые к автоматическому запуску**: это примерно1258 и1493 повторений крошечного корпуса. Рекомендуемый для review первый общий бюджет — **10^14 FLOPs на модель** (около12.6/14.9 проходов). Это предложение, не применённое изменение YAML и не гарантия сходимости. Точные integer ceilings, exposures и альтернативы — `local/reports/four-task-benchmark/budget.json`. Время/стоимость T4 не измерены; из локальной CPU скорости облачное время не выводить. Высокий бюджет несёт явный риск переобучения и лишних HF checkpoints.

## Обязательный операторский gate

Это новая scientific release, а не старый K64 export. Зависимости не добавлялись; локальные проверки выполнялись через `uv run --no-sync`, wheel через `uv build`, requirements через `uv export --locked`. Не делать `uv sync` с удалением существующего environment-only matplotlib и не считать его cloud dependency. Новый cloud installer также должен использовать uv: старый pip-based installation plan — только исторический reference, не готовый Tiny installer. Старые `scripts/export_colab.py`, `colab_package.py`, `colab_verify.py` закрепляют K64 config/data/source; их нельзя запускать на Tiny под старой подписью. Использовать их проверенные HF/checkpoint/service принципы и текущий native trainer, но выпустить отдельный reviewed operator binding для Tiny. Не подменять тренировочный wrapper inference-only wrapper.

Обязательны online ClearML + private HF, предварительная HF write/readback proof, periodic FULL resumable checkpoints (model/optimizer/scaler/RNG/cursor/history), hash-bound final evaluation, native compiler/precision/full snapshot checks; никакого offline fallback. Локальный CPU smoke с disabled trackers — только разрешённый инженерный тест, не production policy. Не объявлять сохранение локального snapshot доказательством удалённого HF backup.

Отдельный private env вне архива/source/logs: ровно7 ключей `CLEARML_API_ACCESS_KEY`, `CLEARML_API_SECRET_KEY`, `CLEARML_API_HOST`, `CLEARML_WEB_HOST`, `CLEARML_FILES_HOST`, `HF_TOKEN`, `HF_REPO`. Секреты на этапе подготовки не читать. Kaggle primary, Colab fallback только при quota/failure и выборе пользователя. Нужны проверенные уникальные mount resolver для двух наблюдавшихся layouts, fixed NVIDIA child environment с canonical root/library checks; не переносить видимость GPU или RNG topology rules из inference. Runtime reference: Python3.13.16, Torch2.11.0+cu130, Triton3.6.0, FLA0.6, T4; два видимых T4 — train на существующем cuda:0, без DDP и visibility masking. Прямой полный запуск с startup gates; отдельных 1/32-step GPU jobs по умолчанию нет.

Теги нового оператора: `series:four-task-v1`, `dataset:tiny-shakespeare-char-v1`, `stage:tiny`, `family:classic|deltanet`, `seed:42`, `schedule:constant`, `termination:budget_exhausted|early_stopping|failed` по факту. Старые теги сохранять. Сейчас автоматическое назначение этих тегов не реализовано generic CLI; это блокер production operator.

После каждого singleton: exact version download → complete scientific/runtime/full-snapshot gate → durable acceptance → следующий singleton. Только после приёмки обоих Tiny разрешён selective_count. Для кривых одна плоскость/одинаковые X/Y limits; отдельные CE/BPC против actual training FLOPs и updates, baseline явно подписан. Общую среднюю метрику четырёх задач не вычислять. Генерация текста (одинаковый prompt/temperature/seed) необязательна и не является primary score. Matplotlib локально может быть установлен через uv pip, но не объявлен dependency и не предполагается в cloud.

## Будущие задачи: закреплённые семантики, не реализация

Первичный источник On-Point-RND/transformer-architecture-generalization, commit `444bf8466d26b39ea5df90ae0016c62e53392b1f`. Локальные точные копии и SHA в `local/reports/four-task-benchmark/upstream/pins.json`.

- `tasks/selective_count.py` SHA256 `159b9fb63b83203990f9e60311b35f00c4e4d8aa4cdd506c8c06ac594565ba23`: последний символ — query; считать его во **всей последовательности, включая последний query**. Единственная supervised target на последней input позиции. По умолчанию m32, n=10m; это исходные defaults, не утверждённый будущий launch config. Класс `CLASS_LOW+count`; exact accuracy и NMAE=mean(abs(clipped(predicted-CLASS_LOW,0,max_count)-true_count))/(n/m+1). Нельзя заменить подсчётом первого символа или обычным LM сдвигом.
- `tasks/permutation.py` SHA256 `27d20f29e4ad2741897b8b17fa6d3b50e1b66e3c8514b81d30c9cf3433edc464`: C5 затем S5, одна family. Каждый move имеет running-composition label в той же позиции, включая тривиальную первую copy target. Композиция справа: `(a o b)[j]=a[b[j]]`, `state <- state o move`; S5 120 перестановок, C5 пять степеней цикла, общий vocab121 включая PAD. Primary exact-sequence и micro token accuracy; дополнительную accuracy без первой copy показывать отдельно. Reference default lengths [4,33), не новый согласованный benchmark length. Политику fixed pools/dedup/test/exposure и стоимость выбрать отдельно перед реализацией.
- `tasks/base.py` SHA256 `c9088836a4af2407f30b307d4f6d4c149b04bda7e4fabc39e159b02b5bc67e1a`: collate с `labels` не сдвигает targets; prompt+answer masked collate ставит single answer на последнюю prompt позицию. Это разные supervised semantics, не универсальная next-token задача. Reference generator uses live sampling/dedup against validation or fixed pool with replacement; конечные prepared pools здесь не следует объявлять полностью идентичным upstream stream.

## Текущее состояние

Prepared/local-only. Существующие MQAR результаты приняты; никакие старые веса не загружались. Tiny Classic выполнен реально на CPU, DeltaNet CPU capability attempt ограничен установленным FLA backend; без подмены ядра. Облако, remote publication, secrets, GPU train и commits отсутствуют. Доказательства, блокеры и frozen-candidate hashes — в локальном отчёте/handoff. Новый production operator, parent scientific review и облачный runtime gate ещё требуются.
