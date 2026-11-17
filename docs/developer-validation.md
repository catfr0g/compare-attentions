# Проверка репозитория перед коммитом

## Установка hook

Используйте существующее окружение проекта. Эта операция не устанавливает пакеты,
не меняет зависимости и не задаёт `core.hooksPath`:

```sh
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 sh scripts/install_hooks.sh
```

Hook в `.git/hooks/pre-commit` запускает `pre_commit` через
`uv run --offline --no-sync python -B`. Каждый Python-инструмент также запускается
через UV. Отсутствующая зависимость останавливает проверку; автоматической установки
нет. Hook не следует обходить через `--no-verify`, `SKIP` или замену hooksPath.

Ruff проверяет код и формат, но не исправляет файлы. Pylint блокирует диагностики
категорий E/F для изменённых first-party Python-файлов. Полная проверка стиля Pylint
не заявлена. Эти проверки не заменяют научную квалификацию выпуска.

Subprocess tests evaluator, lazy-backend и CUDA CLI также используют явный
`uv run --offline --no-sync`; module invocations добавляют `python -B`.
Изменён только способ запуска тестового процесса, не assertions и не backend.
Исторический exploratory Colab replay остановлен после обнаружения старых
`sys.executable` child commands. Его частичный log сохранён; он не является
полным PASS и не доказывает соблюдение строгой per-child UV policy. Финальная
проверка hook выполняет исправленный отдельный переносимый набор.

## Неизменяемые upstream snapshots

Каталоги `src/compare_attention/data/sorting_source` и `tracking_source` содержат
буквальные upstream snapshots коммита `30922c4100992a245a23103a790ad737ea8ede4c`.
Они исключены только из Ruff/Pylint. Их SHA256 заданы в `SOURCE_HASHES`
модулей `data/sorting.py` и `data/tracking.py`; переносимый
`test_ruff_exempt_snapshots_keep_all_declared_upstream_pins` проверяет все эти
pins по реальным файлам. Sorting дополнительно вызывает `verify_source()`.
Не применяйте к snapshots форматирование,
модернизацию аннотаций или автоматические исправления.

## Переносимая CPU-проверка

Hook проверяет поддерживаемые core, integration, e2e и developer-policy tests.
Маска `CUDA_VISIBLE_DEVICES=-1` устанавливается до импорта Python. Это локальная
инженерная проверка, не CUDA/T4 qualification. Существующие CUDA skips и docker
prerequisites остаются явными. Не следует называть результат универсальным PASS.

Только тест `test_compiled_model_reaches_a_finite_loss` помечен
`requires_cpu_inductor`. На Windows без доступного CUDA и без выбранного
`CXX` (по умолчанию `cl`) в PATH он сообщает `UNVERIFIED Windows CPU Inductor`.
Проверка не ловит исключения компиляции или численные ошибки. При доступном
компиляторе либо CUDA тест выполняется без изменения backend. Остальные training
checks не пропускаются из-за отсутствия компилятора. Нужна отдельная проверка
на машине с требуемым toolchain; strict CUDA acceptance замороженных пилотов
не меняется.

## Историческая локальная квалификация

Ниже перечислены сохранённые, не включённые в публикацию untracked suites.
Hook исключает именно эти локальные suites, а не целые классы core-проверок.
Они используют отдельные исторические операторы, реальные замороженные archives
или локальные upstream/evidence directories:

| Suite | Отдельная предпосылка |
| --- | --- |
| `tests/compare_attention/test_sorting_candidate.py` | Исторический Sorting candidate и его exact source/test closure; текущий HEAD не обязан совпадать со старым archive |
| `tests/compare_attention/cli/test_fp16_lr_pilots.py` | Локальный LR operator, support и baseline evidence |
| `tests/compare_attention/cli/test_tracking.py` | Локальные tracking preparation reports/configs |
| `tests/compare_attention/data/test_selective_count_reference.py` | Локальный независимый upstream reference checkout |
| `tests/compare_attention/data/test_tracking.py` | Локальные reference task snapshots |
| `tests/compare_attention/train/test_phase_handshake.py` и `phase_handshake_worker.py` | Exact accepted historical wheel для differential replay |

Исходные tests и raw evidence сохранены без изменений. Они не превращены в
переносимые CI-тесты простым удалением assertions или подменой архивов. Повторять
их следует в отдельном квалификационном shadow с аутентифицированными inputs и
исторической относительной структурой, не против нового live HEAD. Для новых
квалификаций используйте Git-ignored `local/qualification/`, а не portable tests.

## Отложенный Colab WIP

Накопленные `scripts/colab*.py`, `scripts/export_colab.py`, соответствующие
`tests/scripts/test_colab*.py` и `docs/colab*.md` оставлены вне этой публикации.
Проверка обнаружила, что legacy `colab_runner.installation_plan()` генерирует pip
bootstrap, а `colab_api_run_remote.supervisor_code()` и `launch_code()` создают
subprocesses без явного очищенного `env`. Это требует отдельной UV/bootstrap и
process-boundary closure. Не следует публиковать их как новую готовую безопасную
реализацию или менять для этого замороженные scientific releases.

Переносимые Colab transport tests можно выполнить отдельно без cloud actions;
они не доказывают отсутствие дефекта на непокрытой границе. Дополнительные
исторические `test_colab_api_recovery_config.py`, `test_colab_entrypoint.py` и
`test_colab_package.py` требуют конкретные локальные archives и также сохранены
в исходном виде. Ни Colab, ни Kaggle, ни HF/ClearML mutations при Git-проверке
не разрешены и не нужны.

Файлы `.env`, private inputs, `local/`, runtime fixtures, datasets, weights,
reports и machine-generated XML не включаются в коммит.
