# Полный K64 в Google Colab T4 — дополнительный backend

Этот launcher переносит **существующие frozen wheel, два YAML и семь файлов данных**. Он не собирает wheel из рабочего дерева, не готовит новый датасет, не вызывает Kaggle API и не меняет журнал предыдущих запусков. По умолчанию запускается только свежий полный DeltaNet, seed 42. Classic доступен лишь при явном выборе `--family classic`.

Kaggle остаётся поддерживаемым backend: его integration, frozen source и journal не меняются. В Colab **ClearML и HF включены по умолчанию**. **Обязательное recovery-хранилище — приватный HF**, с полными native snapshots каждые125 updates и в конце. Drive не требуется; явный дополнительный Drive output (`--storage drive`) допустим только при уже смонтированном Drive; HF-защита при этом остаётся обязательной. Новый экспорт использует schema-v3 (`storage="hf"`, `periodic_full_checkpoints=true`). Старые v1/v2 экспорты, включая Drive-bound `colab-k64-export-03`, остаются историческими: не переписывайте их notebook, manifest или SHA256.

## 1. Экспорт на Windows

Из корня репозитория, обычным Python 3.13+; установка зависимостей не нужна:

```bash
python scripts/export_colab.py \
  --series E:/HermesArtifacts/compare-attention/fp16-k64-small/272c2da7a9ef619711e4e92a1a98400ac0ed64b08f481d13752093d450820975/series.json \
  --output-dir E:/HermesArtifacts/compare-attention/colab-k64-export-04
```

Каталог назначения должен отсутствовать. Результат:

- `colab-package.zip`: разрешённые release/data/tools/support и manifest с размерами/SHA256 каждого файла;
- `launch_colab.ipynb`: notebook с **реальным SHA256 именно этого ZIP**;
- `export-report.json`: SHA256/размер ZIP, исходная source identity, число файлов.
- `colab.env.example`: семь **пустых** полей, отдельно от ZIP. Это не файл с рабочими ключами.

Сохраняйте notebook и ZIP одной пары. Не редактируйте SHA256, чтобы обойти ошибку. Экспорт проверяет immutable source/release/helper pins, оба полных конфига, data/series fingerprints, размеры и hashes исходных и уже упакованных bytes. Он не копирует `.env`, credentials, токены, рабочее дерево, старые результаты, checkpoints или Kaggle metadata. Включённый старый verifier содержит исторические пути как константы; вызывается только его read-only `verify_snapshot`, не rollout.

## 2. Офлайн-проверка без NVIDIA, Torch и установки

Подставьте SHA256 из `export-report.json`. Каталог распаковки должен отсутствовать:

```bash
python scripts/colab_package.py \
  --archive E:/HermesArtifacts/compare-attention/colab-k64-export-04/colab-package.zip \
  --sha256 SHA256_ИЗ_EXPORT_REPORT \
  --output-dir E:/HermesArtifacts/compare-attention/colab-k64-inspect-04

python scripts/colab_runner.py \
  --package-dir E:/HermesArtifacts/compare-attention/colab-k64-inspect-04 \
  --inspect
```

`--verify-only` — синоним `--inspect`. Без action-флага runner также выполняет только проверку/dry-run. Проверяются все package bytes и отсутствие лишних файлов. ZIP traversal, абсолютные пути, symlinks и дубликаты отвергаются до извлечения кода. Извлечённые файлы помечаются read-only. Package manifest не является цифровой подписью: доверенная точка — переданный отдельно SHA256 ZIP/notebook, а не manifest из неизвестного архива.

## 3. Запуск пользователем в Colab

1. Для ручного fallback откройте соответствующий `launch_colab.ipynb` в Colab. Через панель Files загрузите ZIP в `/content/colab-package.zip`. Приватный env загрузите **отдельно** в `/content/private/kaggle-env.env`; при необходимости создайте каталог через Files. Не загружайте ключи внутрь ZIP или package. Автоматический API-controller использует те же фиксированные paths; этот notebook не требует Drive или Google Drive OAuth.
2. Выберите GPU **T4** в настройках среды. Notebook запрашивает T4 в metadata, но это не резервирование оборудования.
3. Выполните первую ячейку. Она проверяет существующий Python **3.13.x**, ровно один T4 и compute capability **7.5** через `nvidia-smi`, **до credentials, packages и обучения**. CPU, 5070, другая GPU или старый Python не допускаются. Не устанавливайте драйвер и не заменяйте kernel в обход отказа.
4. Во второй ячейке задайте `ARCHIVE_PATH`, локальные новые `WORK_DIR`/`VENV`, **новый** `RESULT_DIR` (по умолчанию `/content/compare-attention-results/deltanet-full-seed42-run01`) и `ENV_FILE` (по умолчанию `/content/private/kaggle-env.env`). Используйте существующий приватный Kaggle `.env`, **вне ZIP, package, venv и RESULT_DIR**. Не вставляйте значения в notebook/чат. Drive mount/OAuth не выполняется. Эта ячейка проверяет ZIP SHA256 до импорта извлечённых scripts, безопасно распаковывает и проверяет package. Colab Secrets не используются.
5. Выполните ячейку setup. Она создаёт изолированный stdlib venv из имеющегося Python. Не меняет packages глобального Colab kernel. Не тренирует.
6. Только последняя training-ячейка явно запускает полный foreground run. Значение `FAMILY='deltanet'` и `RESUME_FROM=None` оставляют свежий pending DeltaNet. Не используйте SMALL weights и не запускайте Classic повторно без намеренного выбора.

Не используйте Run all до проверки путей. Новая GPU-среда может иметь собственные лимиты и закончить сессию раньше полного бюджета. Этот launcher не обходит квоты, не делает keep-alive/reconnect и не запускает параллельные аккаунты. Время и доступность T4 не обещаются.

### Точный installation plan

Сначала существующий Python 3.13 создаёт **новый** изолированный stdlib venv командой `<BASE_PYTHON> -m venv <VENV> --without-pip`: `ensurepip` не вызывается. Затем каждый install выполняется через уже имеющийся base pip: `<BASE_PYTHON> -m pip --python <VENV>/bin/python install …`, без `--upgrade`. Это документированный pip способ работы с venv без установленного pip (официальный раздел «Managing a different Python interpreter»). Код pip берётся из base environment, но точный `--python` направляет установку в новый venv; глобальные packages и kernel не меняются. Новый pip, `get-pip` и системные пакеты не устанавливаются:

1. `torch==2.11.0+cu130` из официального `https://download.pytorch.org/whl/cu130`;
2. `triton==3.6.0 --no-deps`;
3. **весь** неизменённый frozen `requirements.txt --no-deps`, включая CUDA toolkit/cu12 libraries;
4. frozen wheel `--no-deps`.

Файл requirements не фильтруется и не переэкспортируется. Затем probe запускается **через `<VENV>/bin/python`**, проверяет изоляцию, Torch/CUDA/Triton, T4/SM75, реальное CUDA FP16 матричное умножение и действенность `PYTHON_DOTENV_DISABLED`. Ошибка CUDA/драйвера приводит к отказу; нет fallback на cu128, latest Torch, CPU, BF16, другой model или отключение compile. Нужен существующий base pip с поддержкой `--python`; его отсутствие или ошибка создания venv останавливают setup без bootstrap-загрузок, глобальных установок или замены kernel.

### CLI внутри Colab

Notebook выполняет эквивалент следующих команд отдельными действиями:

```bash
python /content/compare-attention-package/tools/colab_runner.py \
  --package-dir /content/compare-attention-package \
  --venv /content/compare-attention-venv --setup

python /content/compare-attention-package/tools/colab_runner.py \
  --package-dir /content/compare-attention-package \
  --venv /content/compare-attention-venv \
  --result-dir /content/compare-attention-results/deltanet-full-seed42-run01 \
  --env-file /content/private/kaggle-env.env \
  --run
```

## 4. Научные и storage-ограничения

Frozen настройки: 4 слоя, embedding 128, 4 головы, V8192, L512, K64, batch32, seed42, LR0.001 constant, WD0.1, FP16 AMP с FP32 master weights и scale cap128, compile, FLOPs budget `1e16`, early stopping OFF. Validation/log каждые25, checkpoint каждые125, keep2. DeltaNet ожидает47265 успешных updates; native verifier пересчитывает бюджет по исходному estimator. Нет отдельных exact1/exact32 jobs и скрытых warm-up optimizer updates. CUDA probe не создаёт model/optimizer.

Runner переопределяет только local data path, result directory, явный `--tracker clearml` и resume при запросе. Нет CLI для произвольной замены batch/LR/budget/precision. Данные read-only под `/content`; venv локален. Trainer атомарно публикует полные checkpoints локально; scoped hook после исходного `BackupTracker.save_checkpoint` синхронно вызывает отдельный HF-only subprocess. Исходные wheel/trainer/model/locks не меняются. После CUDA FP16 probe runner вызывает `check`, затем **`colab_checkpoint_backup.py probe --env-file … --result-dir … --package-dir …` до `train`**. Эта фаза проверяет приватный HF HEAD, записывает tiny proof в приватный HF и читает его по pinned commit; startup receipt сохраняется рядом с ещё отсутствующим result directory. Ошибка останавливает запуск до model updates: local-only fallback отсутствует. Не меняйте `WORK_DIR` между сессиями для resume: путь входит в строгую идентичность конфига.

Каждый native checkpoint (125 updates и финальный) публикуется целиком в `runs/<run>/checkpoints/step_XXXXXXXX.pt`; snapshot, `latest.json` и secret-free `backup.json` входят в один commit с ожидаемым parent HEAD. Проверяются pinned bytes, неизменность локального snapshot и binding run/package/source/data. Локально и в HF сохраняются последние два принадлежащих этому run snapshots; посторонние файлы не удаляются. `<run>.hf-checkpoint-receipt.json` лежит рядом с result dir. Потеря сессии теряет локальные данные после последнего подтверждённого checkpoint, не весь прогресс. ClearML scalars и финальный HF upload сами по себе такую защиту не обеспечивают.

После успешного train последовательно выполняются:

1. frozen public `verify_exploration_run` с полным ожидаемым конфигом и фактическим именем T4;
2. pinned native `verify_snapshot`: model, optimizer, scaler, history, RNG, binding;
3. public `python -m compare_attention.cli.evaluate`, **test** split, batch32, CUDA, float16.
4. provenance с SHA256 полного результата, без credentials;
5. существующий `compare_attention.train.hf.upload_results` отправляет **всю** папку в `runs/<имя RESULT_DIR>` приватного model repo, включая checkpoints, metrics, native gate, новый test evaluation и provenance;
6. `HfApi.model_info(revision='main', token=...)` проверяет private/HEAD; `list_repo_files(revision=<HEAD SHA>, repo_type='model', token=...)` проверяет точный набор ожидаемых путей. Затем `hf_hub_download(..., revision=<HEAD SHA>, force_download=True, token=...)` читает каждый загруженный файл заново. Его SHA256 и неизменность локальных bytes проверяются до verified receipt. Это дополнительный download-трафик и место в HF cache.

Ошибочный gate блокирует eval. `colab-native-gate.json`, `test-evaluation.json`, `colab-provenance.json` сохраняются в локальном result dir и финальном HF upload. Исходные dataset/weights не изменяются. `--inspect` не заменяет эти научные gates: это лишь офлайн-проверка package.

Colab — **новый provider/runtime**, не доказанно matched Kaggle. Фактический Python patch, Torch, CUDA, cuDNN, Triton/GPU записываются; reference Kaggle Python3.13.16 хранится отдельно. Python3.13.15 разрешён, но отличие не скрывается. `matched_runtime=false`, `bitwise_initialization_claimed=false`; новая инициализация остаётся seed42 без обещания побитового совпадения с исторической средой.

### Явное продолжение после обрыва

Для явного resume скачайте полный snapshot с `latest.json` из **конкретного pinned HF revision**, проверьте hashes и run/package/source/data binding, затем добавьте `--resume-from /content/recovered-run/latest.json` или задайте `RESUME_FROM` в notebook. Простая копия только model weights не является resume. Каждый resume требует **нового RESULT_DIR**, тех же package/data absolute paths и совместимости исходного strict trainer. Он вправе отказать при несовпадении runtime/RNG/versions/конфига. Автопродолжения и автоматической повторной отправки нет. Не подставляйте Kaggle checkpoint как Colab resume по умолчанию.

### Credentials

Требуются ровно эти семь имён, как в Kaggle (значения здесь намеренно отсутствуют):

```dotenv
CLEARML_API_ACCESS_KEY=
CLEARML_API_SECRET_KEY=
CLEARML_API_HOST=
CLEARML_WEB_HOST=
CLEARML_FILES_HOST=
HF_TOKEN=
HF_REPO=
```

`--env-file` обязателен для run и upload-only. Hardware preflight выполняется до приватного файла; все семь значений проверяются после CUDA probe, до remote recovery/model/training; пустые поля, некорректные HTTP(S) endpoints и нелитеральный `owner/repo` отвергаются без offline fallback. `dotenv_values(interpolate=False)` читает только указанный файл; автоматический dotenv отключён. Остальные поля файла не передаются детям. **Не коммитьте и не расшаривайте рабочий env-файл.** Setup/inspect/gates/eval не получают credentials. Унаследованные HF/ClearML/Kaggle/PIP overrides, Python path/home/optimization/interactive overrides удаляются.

`colab_services.py train --package-dir … --hf-checkpoints -- <исходные trainer flags>` повторяет hardware guard, инжектирует только пять ClearML полей и лишь затем импортирует исходный train CLI/SDK. HF_TOKEN/HF_REPO до CLI не доходят: иначе исходный CLI сделал бы upload раньше gates/test. Это не отключение HF: periodic checkpoint subprocess получает только HF keys, как и отдельная финальная upload-фаза. Она отдельно получает только HF ключи и включает сеть **до импорта HF SDK**, у которого offline flags кэшируются при импорте. SDK stdout/stderr в upload подавляются, ошибки санитизированы; training streams редактируют значения приватных полей, в том числе раздробленные записи.

После setup можно отдельно проверить наличие/валидность ключей без API calls, model и обучения: те же `--package-dir`, `--venv`, `--env-file` плюс `--check-credentials`. Успех выводит только имена полей, не значения. Это не проверка реальных remote permissions.

### Восстановление только HF upload

При сетевой ошибке upload/readback run завершается ошибкой, локальные результаты сохраняются, пока жива сессия; подтверждённые periodic snapshots уже находятся в HF. Локальный `/content` исчезнет при завершении среды. **Не запускайте обучение повторно**: выполните те же CLI пути с `--upload-only` вместо `--run`. Новая GPU/пересоздание result dir для этого не нужны; нужен установленный соответствующий venv. Проверяется совпадение package identity и hashes завершённых gated results. Повторно вызывается только upload и pinned-revision readback. Успешный `<RESULT_DIR.name>.hf-upload-receipt.json` сохраняется **рядом** с result directory, не внутри загруженной папки, чтобы не создавать циклический receipt hash. В нём только revision/files/hashes/verified, без repo/token. При несовпадении readback verified receipt не создаётся. Уже существующий receipt не перезаписывается.

## 5. Локальная проверка реализации

```bash
.venv/Scripts/python.exe -m pytest tests/scripts/test_colab_package.py tests/scripts/test_colab_runner.py tests/scripts/test_colab_services.py tests/scripts/test_colab_streams.py -q
```

`export_colab.bootstrap_code(sha, *, archive_path, work_dir, venv, result_dir, env_file)` возвращает только Python source: точный PREFLIGHT первым, затем path bindings, SHA256, безопасная распаковка и verify. Его использует вторая notebook-ячейка и API-controller; ни OAuth, ни ключей внутри source нет. Allowlist включает только launcher/helper scripts, не host API-controller.

CPU-тесты используют tiny byte fixtures вместе с настоящими frozen wheel/configs и fake subprocess для hardware/setup/train. Они выполняют реальный export/extract/inspect CLI и проверяют JSON/AST notebook. **Это не запуск обучения на Colab и не проверка совместимости реального Colab driver с cu130.** Полный экспорт04 из канонических данных создаёт оператор только после combined tests/review. Эти CPU проверки не подтверждают реальные HF permissions, ClearML connection или обучение; реальные сервисы и T4 проверяются отдельно перед одобренным запуском.
