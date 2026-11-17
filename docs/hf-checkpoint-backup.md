# Периодические полные checkpoint в Hugging Face

## Контракт

Выбраны **ClearML + приватный Hugging Face**, без обязательного Google Drive.
Это вспомогательный транспорт: frozen wheel, исходный trainer, модели,
optimizer, конфигурация, seed, подготовленные данные и Kaggle не меняются.

Интеграция запуска:

1. После проверки frozen package и до создания tracker/model выполнить:

   ```sh
   python tools/colab_checkpoint_backup.py probe \
     --env-file /content/compare-attention-private/runtime.env \
     --package-dir /content/package \
     --result-dir /content/results/run-unique
   ```

2. В **ClearML training child**, вокруг неизменённого trainer entry point:

   ```python
   from colab_checkpoint_backup import checkpoint_backups

   with checkpoint_backups(env_file, package_dir):
       runpy.run_module("compare_attention.cli.train", run_name="__main__")
   ```

   API принимает **два** аргумента: конкретный result-dir берётся из экземпляра
   исходного `BackupTracker`. Новый транспорт — `colab_checkpoint_backup`,
   не старый `colab_checkpoints`.

3. В экспорт добавить все три stdlib-модуля:
   `colab_checkpoint_backup.py`, `colab_checkpoint_protocol.py`,
   `colab_checkpoint_store.py`. Прежние final gates, test evaluation и финальный
   uploader остаются обязательными и выполняются после training.

## Что сохраняется и проверяется

- Сначала завершается исходный `BackupTracker.save_checkpoint`; его полный
  `.pt` уже содержит model/optimizer/scaler/RNG/config/history. Транспорт не
  десериализует и не преобразует эти байты.
- Существующая cadence (каждые 125 шагов и native final checkpoint) не меняется.
  Синхронный отдельный Python publisher вызывается только после успешного
  native save; возврат из hook означает проверенный remote checkpoint.
- В исходном `train/loop.py::_fit` накопление `control.elapsed` происходит
  **до** `control.checkpoint()`. Final checkpoint также после timed loop.
  Поэтому дополнительный HF I/O увеличивает wall time, но не train-throughput
  denominator и не training FLOPs.
- Native `latest.json` и staged копия сверяются по SHA256; неканоничные пути,
  отрицательные/bool шаги, симлинки и повреждённые байты отклоняются.
- Remote namespace: `runs/<result-name>/checkpoints/`. `.pt`, `latest.json`
  и `backup.json` публикуются **одним** optimistic `create_commit` с
  `parent_commit`; удаление старого owned checkpoint входит в тот же commit.
- После commit заново скачиваются **полные байты** `.pt` по возвращённому OID,
  вычисляется SHA256, проверяются pointer, metadata, listing и актуальный
  private HEAD. LFS metadata/HTTP success сами по себе не считаются readback.
- В HEAD остаются два принадлежащих этому запуску checkpoint. Перед удалением
  старого проверяются ownership proof и его remote байты. Неизвестные файлы,
  другие run prefixes и изменённые чужим writer байты не удаляются.
- `backup.json` также записывается локально в `result/checkpoints` после
  успешной проверки, чтобы final uploader не получил лишних remote файлов
  относительно локального result-prefix.
- `probe` резервирует уникальный run prefix nonce-JSON и читает его по pinned
  OID. Он **не создаёт result-dir** и не конструирует Task/model/optimizer.
  Existing public target запрещён; отсутствующий repo создаётся только
  `private=True`, при подтверждённом 404. Изменение публичности запрещает успех.
- Receipt расположен **рядом** с result-dir:
  `<result-name>.hf-checkpoint-receipt.json`. Содержит только source/data/package
  identity, nonce ownership, шаг, SHA256 и revision; никаких repo/token/hosts,
  credential-file values или hashes private файла.

## Изоляция и ошибки

Parent hook не читает credentials и не меняет RNG/environment. В subprocess
передаётся только путь отдельного env-файла; `clean_environment()` удаляет
наследованные ClearML/HF/Kaggle и Python overrides. Только child вызывает
существующий lazy credential loader и включает HF online **до** SDK import.
ClearML keys не инжектируются в publisher. SDK output/исключения подавляются;
наружу возвращается только постоянное secret-safe сообщение.

Download `local_dir`, HF_HOME и Xet cache ограничены временными каталогами и
удаляются после операции; глобальный кеш всех checkpoint не накапливается.
Нужен свободный диск для stage + readback сверх двух native snapshots.

Любая ошибка останавливает training; локальный полный checkpoint сохраняется.
Перед отправкой commit записывается durable `publication_unknown` / `probe_unknown`
barrier. После ambiguous write выполняется только read-only reconciliation;
даже найденный candidate OID **не** превращает receipt в verified success.
Автоматического upload/GPU retry нет. Повтор уже verified шага допустим только
после новой проверки remote pointer и полных байтов, без повторного commit.

**Retain-2 — не физический quota cap:** HF Git/LFS/Xet history может сохранять
старые объекты. Нет history rewrite, destructive GC, обхода квот или согласия
на paid upgrade. Проблема capacity/permissions должна остановить запуск.

## Восстановление

```sh
python tools/colab_checkpoint_backup.py restore \
  --env-file /content/compare-attention-private/runtime.env \
  --package-dir /content/package --run run-unique \
  --revision <40-hex-verified-commit> --target /content/restored-new
```

Команда читает pinned metadata/pointer/full snapshot, проверяет frozen package,
source/data и SHA256, затем создаёт **новый** restore folder. Уже существующий
folder не перезаписывается. Trainer не запускается; полученный `.pt` передают
исходному `--resume-from` с другим свежим result-dir. Исходный строгий validator
остаётся authority для совместимости config, optimizer, RNG и data paths.
Наличие скачанных байтов само по себе не доказывает target-runtime resume.

## Проверки

Offline tests используют revisioned fake HF store, реальные SDK 2.3.0 operation
objects/signatures и настоящий CPU `BackupTracker.save_checkpoint`; проверяют
atomic commit/readback, retention, privacy, corrupt/ambiguous/race paths,
restore, scoped restoration и неизменность Python/NumPy/Torch CPU RNG.
Отдельный Python `-S` процесс доказывает stdlib-only import без Torch/HF/dotenv.
SDK grounding: установленный `huggingface-hub==2.3.0`, `hf_api.py` и
`file_download.py` (`parent_commit`, returned `oid`, `local_dir`, `revision`).

Evidence: `local/reports/colab-api/hf-checkpoints/`.
Эти проверки **не** являются live account write proof, GPU execution или
битовым доказательством target-runtime training/resume. Реальные credentials,
HF probe, новый export04 и полный T4 run выполняет parent после review.
