# Явное восстановление export07: до запуска нужен отдельный proof

## Область действия

Дополнительный host-only модуль `scripts/colab_api_recovery.py` не входит в frozen
пакет. Он поддерживает только исходный `deltanet-k64-seed42-colab07b`, revision
`f0d0efa4c8eb784677c79fed09b1ee6a27e04e2a`, checkpoint `step_00046500.pt`, SHA256
`ce288e4db2d47925c3876f93b9a874d344851010c9bc6bbd4acb7149602355bf`,
27 788 535 байт. Исходный бюджет — `10000000000000000` FLOPs; целевой счетчик —
47265, продолжение — 765 обновлений. Параметров для замены этих значений нет.

Это не разрешение облачного запуска. Нужны независимое review конечных файлов,
новый actual T4, успешный setup **точного export07**, затем проверенный native
restore/readback proof на этом T4. CPU-тест не доказывает совместимость T4.
Исходные wheel, lock, config, data, tools и gates не изменяются.

## Команды и порядок

Без `--execute` каждая команда возвращает план без I/O. Значения переменных ниже
задает оператор; секретные значения в командную строку не передаются. `RUN` и
`RUNTIME` должны быть новыми, `EXPORT` указывает на существующий export07.

```bash
python scripts/colab_api_run.py setup --recovery --execute \
  --runtime "$RUNTIME" --run-name "$RUN" --export-dir "$EXPORT" \
  --env-file "$PRIVATE_ENV" --client-file "$CLIENT" --token-file "$TOKEN" \
  --journal "$SETUP_JOURNAL"

# Только после setup completed/returncode=0.
python scripts/colab_api_run.py restore --execute \
  --runtime "$RUNTIME" --run-name "$RUN" --setup-journal "$SETUP_JOURNAL" \
  --client-file "$CLIENT" --token-file "$TOKEN" --journal "$RESTORE_JOURNAL"

python scripts/colab_api_run.py status --execute \
  --runtime "$RUNTIME" --run-name "$RUN" --setup-journal "$SETUP_JOURNAL" \
  --client-file "$CLIENT" --token-file "$TOKEN" --journal "$STATUS_JOURNAL"
```

`--recovery` допустим только у setup. Обычная команда train для recovery-bound
job запрещена. Restore не запускает тренировку, Task, forward или optimizer.step.
Исходный restore CLI выполняет read-only fetch с точным revision, затем отдельный
credential-free/offline процесс frozen venv применяет настоящие native validators.
Нужны точные config/data path/runtime/initialization/accounting/model/optimizer/
scaler/RNG/cursor/history, нетерпимое к несовместимости восстановление и readback.
Ошибки останавливают фазу без изменения checkpoint или ослабления validator.

Вне result создаются:

- `/content/compare-attention-jobs/<run>/restore/recovery.py` — отдельный helper;
- `restore/semantic.json` — измеренные identity/runtime, счетчики, история,
  состояние precision/controller и fingerprint восстановленных параметров;
- `restore/proof.json` — durable hash-bound binding, immutable spec, пути,
  SHA snapshot/backup/latest/semantic/helper, включая origin task и новый run.
- `recovery-input/` — отдельно восстановленные snapshot и ownership metadata.

До resume оператор должен read-only сохранить и проверить полные proof/semantic
файлы и их SHA, а не только положительный status. Status возвращает публичные
`recovery.proof_sha256` и `helper_sha256`; это наблюдение, не approval. Для review
сохраняют также соответствующие bytes helper. Нет автоматического выбора proof.

```bash
# Только после независимого review actual T4 proof. Новые hash-переменные —
# буквальные SHA256 просмотренных файлов, не строки-заглушки и не секреты.
python scripts/colab_api_run.py resume --execute \
  --runtime "$RUNTIME" --run-name "$RUN" --setup-journal "$SETUP_JOURNAL" \
  --client-file "$CLIENT" --token-file "$TOKEN" --journal "$RESUME_JOURNAL" \
  --recovery-helper-sha256 "$REVIEWED_HELPER_SHA256" \
  --restore-proof-sha256 "$REVIEWED_PROOF_SHA256"
```

Resume требует successful setup и restore. Guard повторно проверяет package,
archive, metadata, snapshot и approved proof/helper; повторяется в supervisor
перед вызовом неизмененного runner с единственным literal `--resume-from`.
Бюджет не заменяется оставшимся бюджетом. Result остается совершенно новым;
никакие metrics или identity туда заранее не помещаются. Proof рядом с job
сохраняет origin lineage и должен входить в отдельный checksum-bound acceptance
bundle вместе с новой итоговой identity и новым ClearML task ID. Исходные task,
HF prefix и отброшенная ветвь после 46500 не смешиваются с новой ветвью.

## Доказательства и остановка

Fresh train и resume используют **один** exclusive `train/` stage. Restore имеет
свой exclusive stage. Pending identity flush/fsync завершается до Popen. Любой
существующий stage/journal, включая неизвестный spawn или неуспех, запрещает
повторный запуск. Новый host journal не снимает remote barrier. Никаких retry,
автоматического resume, редактирования snapshot или повторной публикации.

Status различает `inherited_updates=46500` и `new_updates`, вычисленные только из
нативного FLOPs-счетчика. Replay не считается новым compute. Popen и returncode
supervisor не заменяют scientific acceptance. После тренировки неизмененный
runner выполняет исходные scientific/full-snapshot gates, test evaluation,
provenance и полный HF upload/readback. До их фактического успеха completion
заявлять нельзя.

Локальные тесты используют synthetic transport и настоящий tiny native replay
из actual export07 wheel: оригинальные metrics/identity удалены, у resumed job
проверены реальные четыре новых CPU обновления, точные replay rows и равенство
model/optimizer/RNG/cursor/history непрерывному шестишаговому запуску. Никаких
optimizer updates approved K64 run эти тесты не выполняют. Тесты не доказывают
actual T4 restore, доступность HF/ClearML или прохождение full K64 gates.
