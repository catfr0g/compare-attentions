# MQAR Classic seed202: продолжение с update981

## Исправление только identity/physical mount (после rejected frozen02)

Worker `mqar_worker_paths.bind_case` выводит неизменную owner-qualified
`config.data` из SHA-проверенного original saved main (`19ff980d…`) и отдельно
проверяет SHA исходного FULL (`e2289ced…`). Обе допустимые схемы provider mount
проходят один реальный public parser и проверку всех неизменных MQAR bytes.
Только ожидаемое поле `config.data` отображается в исходную identity; исходные
FULL/config/history/checksums не редактируются. Чужой или неоднозначный mount
останавливает worker. `training_config` повторно читает physical metadata настоящим
TrainConfig и требует полного равенства всех полей, кроме явно отображленного data.
Path identity не перенаправляет filesystem протокол: physical split access привязан
отдельно в частном namespace к точным исходным code objects `train/_train/_load`.
Глобальные scientific modules и wheel остаются неизменными, native resume preflight
по-прежнему сравнивает весь config строго; model math/optimizer/RNG policy не меняются.

Новый CPU engineering proof вызывает public worker entry, реальный complete cloud
FULL validator с неизменным code object и model-free schema constructor transport,
затем исходный public train/distributed strict preflight и оба физических split loader.
Он останавливается ДО seeding/model/optimizer/RNG execution. Это НЕ CUDA restore:
настоящая target restoration остаётся обязательной внутри единственной будущей
разрешённой allocation. Старые saved-main parser proofs отдельно унаследованы,
не выдаются за новый full-preflight proof. Новый candidate требует focused независимого
review. Ни этот документ, ни fixture `ENGINEERING_ONLY` не дают launch permission.


Этот пакет предназначен для отдельного независимого review. Он не выдаёт разрешение
на публикацию или запуск. Команды обращения к сервисам ниже выполняет владелец
исполнения после независимого PASS. Автор реализации их не выполняет.

Исходный job `145778849/v1` остаётся `ERROR`. Продолжение использует сохранённый
FULL после 980 успешных updates. Следующий update — 981. Исходный предел — 10780
updates и запрошенные `5e15` FLOPs. Осталось 9800 updates. Повтор первых 980 updates
запрещён. При новой ошибке прекратите всю серию. Повторное выделение GPU запрещено.

## Локальная среда

Выполняйте команды из корня репозитория в PowerShell. Используйте только `uv`.
Задайте переменные до запуска Python. Не выполняйте локальное создание настоящей
модели, создание optimizer, forward/backward или восстановление RNG.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TMPDIR = 'C:/Users/Michael/AppData/Local/hermes/cache/scratch'
$artifact = 'E:/HermesArtifacts/compare-attention/sorting-mqar-seed202-pilot-v1'
$parent = "$env:TMPDIR/mqar202-repair-work-v1/parent"
$baseline = "$artifact/remaining-three-after-tail-v1/qualified-launch-bridge-v1/development-07/sorting-mqar-seed202-pilot-v1.zip"
$history = "$artifact/periodic-resume-repair-v1/candidate-05-final03/sorting-mqar-seed202-pilot-v1.zip"
uv run --offline --no-sync python -B scripts/seed202_mqar_package.py `
  --parent $parent --baseline $baseline --history $history `
  --destination "$artifact/mqar-classic-resume-repair-v1/NEW-UNIQUE-STAGING"
```

Эта команда создаёт staging с состоянием `STAGED_FOR_ENGINEERING_TESTS_NOT_READY`.
Staging не является завершённым кандидатом. Для review используйте только отдельно
замороженный кандидат с итоговыми lint, JUnit, inventory и SHA256. Не меняйте старые
архивы, исходную выгрузку, отзывы или journals.

Для локальной проверки и заморозки задайте три новых уникальных каталога.
Проверка сначала выполняет Ruff check/format и Pylint E/F, затем affected pytest.
Число тестов рассчитывается по уникальным JUnit testcase IDs.

```powershell
$staged = "$artifact/mqar-classic-resume-repair-v1/NEW-UNIQUE-STAGING"
$quality = "$artifact/mqar-classic-resume-repair-v1/NEW-UNIQUE-QUALITY"
$frozen = "$artifact/mqar-classic-resume-repair-v1/NEW-UNIQUE-FROZEN"
uv run --offline --no-sync python -B scripts/seed202_mqar_quality.py `
  --staged $staged --destination $quality --parent $parent
if ($LASTEXITCODE -ne 0) { throw 'OWNER_QUALITY_STOP' }
uv run --offline --no-sync python -B scripts/seed202_mqar_package.py `
  --staged $staged --verification-dir $quality --destination $frozen
if ($LASTEXITCODE -ne 0) { throw 'FREEZE_STOP' }
```

Заморозка проверяет соответствие исходников отчёту и обязательные тесты маршрутов.
Она не создаёт независимый PASS или разрешение исполнения.

Для отдельного безопасного повтора saved-main proof используйте новый каталог.
Команда ниже выполняет только инженерный fixture: обе схемы mounts, настоящий
hardware predicate на заданной metadata, настоящий UV parser и CPU sentinel.
Она не создаёт модель или optimizer, не восстанавливает RNG и не обращается к облаку.

```powershell
$env:MQAR_TEST_CANDIDATE = $frozen
$env:SEED202_EXTRACTED_WHEEL = "$quality/authenticated-wheel"
$env:PYTHONPATH = "$frozen/payload/source/scripts;$frozen/payload/source/tests/scripts;$frozen/payload/source/src"
$smoke = "$artifact/mqar-classic-resume-repair-v1/NEW-UNIQUE-CPU-SMOKE"
$smokeTemp = "$env:TMPDIR/NEW-UNIQUE-MQAR-CPU-SMOKE"
if ((Test-Path $smoke) -or (Test-Path $smokeTemp)) { throw 'CREATE_ONLY_SMOKE_REQUIRED' }
uv run --offline --no-sync python -B -m pytest `
  "$frozen/payload/source/tests/scripts/test_seed202_mqar_saved_main.py" `
  -q -k frozen_saved_main -p no:cacheprovider --tb=short `
  "--basetemp=$smokeTemp" "--junitxml=$smoke/saved-main.xml"
if ($LASTEXITCODE -ne 0) { throw 'SAVED_MAIN_CPU_SENTINEL_STOP' }
```

Итоговый `artifact-index.json` указывает archive, manifest, source classification,
JUnit и оба `saved-main-*-command.json`. Последние содержат точную фактически
выполненную UV command line, SHA256 kernel, маску CPU и exit code 73.
Это не target restore proof. Его producer остаётся CUDA observer перед update981.

На Windows вызывайте Git Bash по полному пути
`C:/Program Files/Git/bin/bash.exe`, если он нужен. Имя `bash` может указывать на
другую программу.

## Входы будущего исполнения

Назначьте `$candidate` на проверенный замороженный кандидат. Назначьте `$science`
на полное аутентифицированное извлечение scientific archive. Назначьте `$mqar`
на исходный K64 cache без копирования массивов. Файлы `$auth` и `$private` задаёт
владелец исполнения. Не выводите их содержимое, хеши или значения в отчёты.

```powershell
$frozen = "$artifact/mqar-classic-resume-repair-v1/continuation-frozen-02"
$candidate = "$artifact/MQAR-POSTREVIEW-NEW-UNIQUE"
$evidenceReview = 'C:/PATH/TO/INDEPENDENT-EVIDENCE-REVIEW.json'
$executionDecision = 'C:/PATH/TO/EXTERNAL-EXECUTION-DECISION.json'
$env:PYTHONPATH = "$frozen/payload/source/scripts"
uv run --offline --no-sync python -B "$frozen/payload/source/scripts/seed202_mqar_postreview.py" `
  --candidate $frozen --review $evidenceReview --decision $executionDecision `
  --destination $candidate
if ($LASTEXITCODE -ne 0) { throw 'POSTREVIEW_BINDING_STOP' }
$science = "$artifact/full-budget-tail-recovery-v1/science"
$mqar = 'E:/HermesArtifacts/compare-attention/fp16-k64-small/272c2da7a9ef619711e4e92a1a98400ac0ed64b08f481d13752093d450820975/data'
$auth = 'C:/PATH/TO/OWNER-AUTH.env'
$private = 'C:/PATH/TO/OWNER-PRIVATE.env'
$execution = "$artifact/mqar-classic-resume-repair-v1/NEW-UNIQUE-EXECUTION"
$journalRoot = "$execution/journals"
$plan = "$execution/plan.json"
$review = "$execution/independent-review.json"
$lifetime = "$execution/fresh-official-lifetime.json"
$case = 'mqar-k64-classic-seed202'
$env:PYTHONPATH = "$candidate/payload/source/src"
$bridge = Get-Content -Raw "$candidate/payload/qualified-launch/bridge.json" | ConvertFrom-Json
$delta = $bridge.cases.'sorting-deltanet-seed202'.spec.output
$common = @('--candidate-dir', $candidate, '--case', $case,
  '--parent-archive', "$candidate/payload/inherited/parent-candidate03.zip.bin",
  '--auth-env', $auth, '--private', $private, '--science-dir', $science,
  '--mqar-data', $mqar, '--lifetime', $lifetime, '--plan', $plan, '--review', $review)
uv run --offline --no-sync python -B -m seed202_pilot.cli check --candidate-dir $candidate
uv run --offline --no-sync python -B -m seed202_pilot.cli make-plan `
  --candidate-dir $candidate --case $case --journal $journalRoot `
  --acceptance $delta --output $plan
```

Reviewer должен выдать `$review` для точного SHA256 этого singleton plan. Не
создавайте собственный PASS. Сохраните отдельно решение владельца исполнения.
Получите свежие lifetime и quota evidence. Старая terminal observation не является
свежей проверкой возможности запуска.

До создания singleton plan отдельный reviewer проверяет
`payload/deploy/mqar-k64-classic-seed202/evidence-required.json`.
Его scope — `MQAR_EVIDENCE_ONLY_V1`: точные worker/resume/config/science/metadata bytes,
пять versioned inputs и SHA256 kernel после обнуления только двух полей
`BINDING.evidence_sha256` и `BINDING.authority_sha256`. SHA256 итогового архива
не входит в этот scope. Поэтому циклического archive selfhash нет.

Внешний `$evidenceReview` содержит ровно `scope=INDEPENDENT_EVIDENCE_REVIEW_V1`,
`evidence_sha256`, `decision=PASS`, непустой `reviewer`. Внешний `$executionDecision`
содержит ровно `scope=EXECUTION_DECISION_V1`, тот же `evidence_sha256`,
`review_sha256` точных байтов review, `decision=ALLOW` и nonce из evidence.
Эта инструкция задаёт формат; она не выдаёт эти решения. Frozen02 оставляет
review и execution decision обязательными и не содержит настоящего PASS.

Команда postreview создаёт новый каталог и новый архив.
`candidate.json` и `artifact-index.json` этого каталога указывают на новый ZIP,
его фактический unique member count и checksum того же файла. Это binding внешних
evidence bytes; action seals остаются обязательными.
Она монтирует в пятом
dataset ровно пять файлов: `resume.zip.bin`, `evidence.json`,
`independent-review.json`, `execution-decision.json`, `authority-inventory.json`.
Publisher загружает каждый файл; readback сравнивает полный listing, каждый байт
и numeric ID версии 1. Saved main проверяет этот inventory до и после warm imports.
Worker и host acceptance повторяют проверку. Удаление provenance или rank receipt
не превращает продолжение в fresh run: обязательность задаёт frozen requested mode.

## Раздельные публикации и единственная отправка job

Выполняйте действия строго в порядке `publish`, `publish-resume`, `submit`.
Каждое действие использует отдельные preparation, proof, seal и journal.
Сначала завершите полные сравнения байтов в PREP. Затем получите LIVE proof под
двумя writer locks. Срок LIVE namespace proof — менее 120 секунд. Команды не
повторяют неизвестную мутацию.

```powershell
foreach ($action in @('publish', 'publish-resume', 'submit')) {
  $j = "$journalRoot/$case/$action"
  $prepared = "$execution/$action-preparation.json"
  $proof = "$execution/$action-proof.json"
  $approval = "$execution/$action-approval.json"
  uv run --offline --no-sync python -B -m seed202_pilot.cli prepare @common `
    --action $action --journal $j --output $prepared
  if ($LASTEXITCODE -ne 0) { throw 'PREP_STOP' }
  uv run --offline --no-sync python -B -m seed202_pilot.cli probe @common `
    --action $action --journal $j --preparation $prepared --output $proof
  if ($LASTEXITCODE -ne 0) { throw 'LIVE_STOP' }
  uv run --offline --no-sync python -B -m seed202_pilot.cli seal `
    --plan $plan --review $review --proof $proof --output $approval
  if ($LASTEXITCODE -ne 0) { throw 'SEAL_STOP' }
  uv run --offline --no-sync python -B -m seed202_pilot.cli $action @common `
    --approval $approval --journal $j --preparation $prepared
  if ($LASTEXITCODE -ne 0) { throw 'STOP_GET_ONLY_RECONCILIATION_REQUIRED' }
}
```

При потерянном ACK остановите цикл. Не повторяйте publish или submit. Используйте
только GET-only reconcile с исходными action, approval и journal. В примере ниже
`$action`, `$approval` и `$j` должны оставаться от остановленного действия.

```powershell
uv run --offline --no-sync python -B -m seed202_pilot.cli reconcile @common `
  --action $action --approval $approval --journal $j
```

После успешного reconcile продолжите с ещё не выполненного следующего действия.
Не запускайте весь цикл заново. Все четыре исходных versioned inputs остаются
неизменными. Новый worker заменяет только первый input. Отдельный resume input
становится пятым. HTTP403 остаётся `FORBIDDEN_AMBIGUOUS`; не считайте его HTTP404.

## Наблюдение и полная приёмка

Оставьте `$approval` от `submit`. Используйте отдельный collection journal.
При `ERROR` остановитесь. Не вызывайте обычный COMPLETE-only collector.

```powershell
$approval = "$execution/submit-approval.json"
$collectJournal = "$journalRoot/$case/collect"
$download = "$execution/download"
$index = "$execution/download-index.json"
$services = "$execution/services.json"
$snapshot = "$execution/service-snapshot"
$accepted = "$execution/acceptance.json"
$observed = uv run --offline --no-sync python -B -m seed202_pilot.cli status @common `
  --approval $approval --journal "$journalRoot/$case/submit" | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $observed.typed_status -ne 'COMPLETE') {
  throw 'TERMINAL_COMPLETE_REQUIRED_NO_RETRY'
}
uv run --offline --no-sync python -B -m seed202_pilot.cli collect @common `
  --approval $approval --journal $collectJournal --destination $download --index $index
if ($LASTEXITCODE -ne 0) { throw 'COLLECT_STOP' }
uv run --offline --no-sync python -B -m seed202_pilot.cli services @common `
  --approval $approval --journal $collectJournal --destination $download `
  --service-snapshot $snapshot --output $services
if ($LASTEXITCODE -ne 0) { throw 'SERVICES_STOP' }
uv run --offline --no-sync python -B -m seed202_pilot.cli accept @common `
  --approval $approval --journal $collectJournal --destination $download --index $index `
  --services-evidence $services --task-data $mqar --task-labels "$mqar/val_labels.npy" `
  --wheel "$candidate/payload/inherited/training-wheel.whl" `
  --config-path "$candidate/payload/configs/$case.yaml" `
  --periodic-destination "$execution/periodic-replay" --output $accepted
if ($LASTEXITCODE -ne 0) { throw 'WHOLE_ACCEPTANCE_STOP' }
uv run --offline --no-sync python -B -m seed202_pilot.cli next `
  --candidate-dir $candidate `
  --qualified-predecessor "$candidate/payload/predecessor/qualified-acceptance.json" `
  --acceptance $delta --acceptance $accepted
```

Приёмка должна сохранить исходный ERROR, точный исторический prefix, новые updates,
оба target restore equality receipts, 11 публикаций и полную историю ClearML.
Публикация сохранённой точки980 является новой `SALVAGED` публикацией. У исходного
job первой HF backup не было. MQAR primary metric — per-query accuracy; whole-row
exact match учитывается отдельно. `next` проверяет барьер; он не отправляет Delta.

Инженерные HTTP/metadata/no-optimizer fixtures не доказывают CUDA restore или
научный результат. Actual restore equality должна пройти внутри единственного
будущего продолжения до update981. Научный wheel должен сохранить SHA256
`512cb5b073e47deca36d2bfb41f2d183e0711b8817b9a0aa18dbd2e07591b2d3`.
