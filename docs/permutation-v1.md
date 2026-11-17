# Permutation v1 — локальный научный кандидат

## Границы

Облачный запуск не разрешён. Оператор, credentials, installs и commits в этом этапе не используются. Все публичные Permutation prepare/train/eval выполнены с CUDA mask=-1 на CPU. **Нарушение границ:** первый запуск affected regression был ошибочно выполнен без CUDA mask; существующие legacy CUDA/auto-device тесты использовали локальный RTX5070. Эти доказательства сохранены отдельно, затем выполнен новый CPU-masked regression. Утверждать, что весь этап обошёлся без GPU, нельзя; требуется review parent. Подробности: `scope-incident.json`. MQAR → Tiny → Count приняты ранее; их результаты, данные и веса остаются неизменными. Перенос весов между задачами или группами запрещён. Будущий порядок: C5 Classic → C5 DeltaNet → принятие пары C5 → S5 Classic → S5 DeltaNet. Все четыре модели стартуют с нуля; текущий DeltaNet не заменяется.

## Задача и источник

Источник: `On-Point-RND/transformer-architecture-generalization`, commit `444bf8466d26b39ea5df90ae0016c62e53392b1f`; исходник permutation SHA256 `27d20f29e4ad2741897b8b17fa6d3b50e1b66e3c8514b81d30c9cf3433edc464`. Пять элементов, 120 перестановок в лексикографическом порядке. В обеих группах полный vocabulary121: PAD0 и перестановки1..120. Циклическая C5 использует source indices `[0,33,64,90,96]`, то есть token IDs `[1,34,65,91,97]`, а не непрерывные классы1..5.

Каждый входной токен — очередной множитель. На той же позиции требуется текущий правый продукт: `(a o b)[j] = a[b[j]]`. Никакого LM shift, добавленного answer-step или единственного последнего Count target нет. Первая метка всегда равна первому входу. Существующий same-position CE и IGNORE_INDEX=-1 подходят без изменения loss/training loop.

## Решение о длине требует review

Основной кандидат использует **32 реальных хода**, а не512. Model block_size512 остаётся верхней границей. Нативный GPT2 уже поддерживает `time <= block_size`; estimator уже считает `dataset.input_seq_len`. Поэтому padding512, изменение архитектуры, уменьшение max-context или ослабление validation не требуются. Публичный frozen-wheel CPU путь проверяет именно actual32/max512.

Это явная адаптация default distribution источника, а не заявление полного совпадения: source default `(4,33)` выбирает4..32 хода. В C5 при длине4 существует лишь625 последовательностей; 100000 глобально уникальных примеров с равномерными короткими длинами потребовали бы rejection bias. Fixed32 сохраняет source-supported integer parameter, даёт достаточный пул и размечает все префиксы1..32. Профиль actual512 в budget.json — только расчёт отдельного будущего протокола, не padding, не выполненное обучение, не разрешение на расширение. Сначала parent выбирает длину и бюджет.

## Данные

Для каждой группы train100000/val1000/test1000, dataset seed1337. Split seeds получены `SeedSequence(1337).generate_state(3)` в порядке train,val,test. Вызов генератора каждой строки точно повторяет source fixed-length `_sample_one`; целые длины не расходуют RNG. Глобальная уникальность prompt внутри каждой группы проверяется между всеми сплитами; rejected draws считаются. Разные группы — отдельные задачи, а не смешанный поток.

Все204000 строк сопоставлены с настоящими source `_sample_one` и collator. Дополнительно все префиксы проверены независимым point-action oracle, который применяет множители в обратном временном порядке к каждой точке. Noncommuting S5 countercase отличает ошибочный порядок; одной коммутативной C5 для этого недостаточно. Подмена факторов/PAD, corrupt labels, сдвиг и маска отвергаются. При подготовке нет PAD и все32 позиции размечены.

Static split-seeded глобально уникальный пул и ordered cyclic loader отличаются от source val-first общего RNG и online/with-replacement pool. Эти различия зафиксированы, source stream API отдельно упражнён; равенство training-stream exposure не заявляется.

## Метрики

`acc` — whole-sequence exact match: все размеченные позиции строки правильны. `token_acc` — micro accuracy по всем меткам, включая первую copy. Argmax производится по полному121-way выходу. PAD и классы вне группы считаются ошибками; restricted decoding не применяется. Отдельно доступны exclude-first EM/token, last-position accuracy, actual position counts/accuracy и outside-group/PAD counts. Микроагрегация учитывает tail batch и неодинаковые маски; exclude-first EM не даёт пустой строке ложный полный балл.

Старый generic trainer сохраняет прежний смысл `val_accuracy`: это **token micro accuracy**, не sequence EM. Для sequence-primary и positional метрик требуется новый публичный task-specific eval; будущий оператор должен связать его с acceptance. Нельзя переименовать исторический token scalar в sequence accuracy. Tiny BPC, Count clipped NMAE и MQAR semantics не изменены. BPC/NMAE для Permutation не выдаются.

Baselines: copy текущего фактора, constant product class, выбранный только по TRAIN, и oracle. Аналитическая group-uniform token chance1/5 или1/120 не ограничивает модельный decoder. Sequence chance `q^-32` относится к независимым случайным guesses; это не предположение о независимости ошибок обученной модели. Copy первого токена даёт долю1/32 тривиальных правильных ответов; secondary exclude-first раскрывает этот вклад. CSV/PNG гистограмм показывают только DATA, не performance.

## Цена и инженерная проверка

4 layers ×128, 4heads, batch32, model seed42, FP16 с FP32 masters/cap128, compile, LR0.001 constant, WD0.1, early stopping off. Обе группы имеют одинаковую форму/vocabulary и одинаковую стоимость внутри архитектуры. `budget.json` считает10^14/10^15/10^16 existing estimator policy `algorithmic_matmul_conv_v2` для actual32 и512: успешные updates, examples, labels, coverage/repeats, validation/FULL checkpoint events и нижнюю оценку объёма snapshot transfers. Validation, setup, optimizer и I/O не включены в training FLOPs; T4 walltime не измерен и не предсказывается.

Предварительная рекомендация: общий первый10^14 на архитектуру/группу, не автоматическое наследование Count10^15. Для actual32 это около6 проходов пула, а10^15 — около60. Provisional validation1000 updates, FULL checkpoint5000, log250 отражают экспозицию коротких последовательностей; parent выбирает cadence до оператора. В YAML стоит provisional10^14, но launch_allowed=false.

CPU engineering — **ровно два optimizer updates**, а не двухтокенная задача и не32-step GPU smoke. Длина входа32, все production dimensions/max-context сохранены; только CPU FP32/eager явно переопределены. Полный test1000 строк имеет32000 targets на группу. FULL snapshots проверяются нативным checkpoint validator. DeltaNet на CPU не поддерживается; evaluator явно отказывает. Это не CUDA/FP16/compiler parity и не научное сравнение качества.

## Публичные команды

```bash
uv run --no-sync python -m compare_attention.cli.prepare_permutation --group c5 --output NEW_C5_DIR
uv run --no-sync python -m compare_attention.cli.prepare_permutation --group s5 --output NEW_S5_DIR
uv run --no-sync python -m compare_attention.cli.train --config experiments/permutation/v1/c5/classic/seed_42.yaml --data C5_DIR --result-dir NEW_RESULT
uv run --no-sync python -m compare_attention.cli.evaluate --checkpoint NEW_RESULT/checkpoints/latest.json --data C5_DIR --split test --batch-size 32 --device cuda --dtype float16 --output NEW_EVAL.json
```

Последние команды показывают интерфейс, не cloud approval. Для production нужен отдельный review/оператор с проверенными Count/Tiny generic helpers и неизменёнными compiler/precision/FULL-snapshot gates. В этом кандидате оператор намеренно не реализован.

## Доказательства

`local/reports/four-task-benchmark/permutation-prep/`: protocol.json, budget.json, data-audit.json, CSV/PNG, actual CLI logs, JUnit, source/wheel/архив manifests и verification.json. Различаются свежие scoped проверки и byte-identical inherited core; broad исторический suite/MSVC43skips не повторяется. Wheel получен честным stdlib repack принятой dist-info metadata с actual source и проверенным RECORD, без установки build dependencies. Accepted archives/data/models не перезаписываются.
