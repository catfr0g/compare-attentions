# HOST-мост для принятых qualified Sorting recovery

## Назначение и граница

Мост предназначен только для перехода от двух уже принятых Sorting seed202
recovery к MQAR K64 Classic, затем к MQAR K64 currentDeltaNet. Он не повторяет
обучение и не превращает исходные ERROR в COMPLETE. Он не является разрешением
на публикацию, выделение GPU или повтор неудачного запуска.

Поддерживаемые исходные composite закреплены полным SHA256 в
`scripts/seed202_qualified_bridge.py`. Их `later_pilots_authorized=false`,
qualification, исходные ошибки и разные telemetry qualifications сохраняются.
У Delta исходные scientific telemetry gaps равны нулю; у Classic сохраняются
исторические пропуски. Причина исходных ошибок остаётся UNKNOWN.

Новые MQAR принимаются только неизменённым `state.replay_acceptance` и полным
`acceptance.complete`. Qualified-политика не применяется к новым MQAR, к их
ошибкам или к произвольному объекту с похожими полями.

## Поддерживаемые файлы

- `scripts/build_seed202_qualified_bridge.py`: строит новый кандидат из точного
  SHA-проверенного V7 ZIP. Меняет только конечный список HOST-интерфейсов.
- `scripts/seed202_qualified_bridge.py`: tagged predecessor witness, проверка
  исходных файлов, замкнутых import roots и replay.
- `scripts/seed202_qualified_imports.py`: связывает `--science-dir` с reviewed
  input descriptor и держит существующие Win32 source leases до чтения credentials,
  во время SDK/observer работы и до выхода из public command.
- `tests/scripts/test_seed202_qualified_launch.py`: genuine predecessors и
  public HOST CLI; SDK-публикация и последующее завершение — transport fixtures.
- `tests/scripts/test_seed202_qualified_security.py`: отрицательные границы.
- `tests/scripts/seed202_qualified_http_fixture.py`: настоящий SDK HTTP encoder,
  Bearer attachment и decoder с перехватом только сетевого I/O.

Исходный V7 и оба recovery-пакета не изменяются. Scientific wheel, worker,
конфигурации, данные, зависимости, обучение и native MQAR gate не перестраиваются.
Новый архив содержит `HOST-DIFF.json` рядом с архивом; исходный V7 manifest
сохраняется отдельно внутри payload. Полный diff надо проверить независимо.

## Явный входной контракт сборщика

JSON `--inputs` содержит:

- `science`: существующий распакованный scientific package;
- `science_archive`: его исходный неизменённый ZIP;
- `transition_review`: настоящий независимо выданный Delta transition review;
- `cases`: ровно `sorting-classic-seed202` и `sorting-deltanet-seed202`.

Каждый case содержит `archive`, `root` и `spec`. В `spec` ровно десять путей:
`release`, `result`, `native`, `services`, `provider`, `review`, `plan`, `data`,
`input`, `output`. `output` должен быть уже существующим принятым composite.
`release` связан с `root/payload`, а `data` — с scientific package. Входы не
содержат `.env`, токенов, приватных URL или service headers.

```sh
export TMPDIR=<existing-host-scratch-directory>
export TEMP="$TMPDIR" TMP="$TMPDIR"
export UV_CACHE_DIR="$TMPDIR/qualified-bridge-uv"
export PYLINTHOME="$TMPDIR/qualified-bridge-pylint"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/qualified-bridge-inductor"
export CUDA_VISIBLE_DEVICES=-1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_TELEMETRY=1
uv run --offline --no-sync python -B scripts/build_seed202_qualified_bridge.py \
  --v7-archive <original-v7.zip> --inputs <artifact-paths.json> \
  --destination <fresh-candidate-directory>
```

Сборщик создаёт каталог только один раз. Для повторной сборки из frozen builder
надо передать `--test-root <payload/source/tests/qualified_launch>`.
Кандидат наследует исходный path allowlist; переносимость аргументов не снимает
ограничения производственных preparation leases.

## Replay и freshness

Для Delta выполняется исходный authenticated native/FULL и whole-composite
replay. Для Classic повторно проверяется полный composite, source/input archive
и исходные raw bindings; уже независимо принятый native proof используется
без повторного запуска старого валидатора. Старый Classic валидатор создаёт
модель внутри `fork_rng`, поэтому его повторный запуск не допускается здесь.
Это наследование конкретного принятого доказательства, не новый native audit.

Replay работает в отдельном CPU-only UV child без сети, forward, backward,
optimizer construction и RNG restore. Мемоизация хранит только process-local
результат чистого replay. Перед каждым использованием снова проверяются полные
input bytes и замкнутые inventories; cache не является разрешением.

Все внешние replay/import inputs включаются в существующую подготовку и
Win32 write-denial leases. Existing LIVE 120 seconds, два writer lock,
DPAPI/HMAC receipt, exact plan/review/action/nonce/target, private/source checks,
quota, UNKNOWN GET-only reconciliation и SDK capabilities не заменяются.
403 остаётся неоднозначным 403, а не 404.

## Проверка и независимый review

```sh
export SEED202_BRIDGE_TEST_CANDIDATE=<candidate-directory>
export SEED202_BRIDGE_TEST_DELTA=<genuine-delta-qualified-acceptance.json>
uv run --offline --no-sync pytest -q \
  tests/scripts/test_seed202_qualified_launch.py \
  tests/scripts/test_seed202_qualified_security.py
```

Без явно переданных accepted artifacts эти тесты пропускаются, а не создают
фиктивные научные входы. Transport fixtures используют только synthetic
credentials и явно маркируют MQAR completion как `NO_GPU_NO_NATIVE_MQAR_RESULT`.
Такой объект не является результатом реального MQAR эксперимента. Native science
leaf заменяется только внутри этих tests; production implementation неизменён.

Нужны отдельные окончательные lint/regression evidence, source/test manifests,
проверка final ZIP и независимый review финальных bytes. Не использовать
промежуточный development archive как готовый или одобренный launch candidate.
После независимой технической проверки — локальный fix commit с явными путями
и штатными check-only UV hooks. **NO PUSH** до нового разрешения пользователя.
