# Неизменяемые входы серии Kaggle

## Назначение

Новый интерфейс отделяет локальную подготовку от публикации и запуска. Он не заменяет
`scripts/train_kaggle.py`: прежние режимы YAML-файла, папки и флагов сохранены.
Новая серия принимает явный список всех конфигураций. Подкаталоги семейств сохраняются,
поэтому `classic/seed_42.yaml` и `deltanet/seed_42.yaml` не конфликтуют.

## Порядок работы

Команды выполняются из корня репозитория. `PREPARED_KV` — уже подготовленная папка,
соответствующая полному KV spec всех конфигураций. Подготовка не изменяет YAML экспериментов.

```sh
uv run --locked scripts/train_kaggle_series.py stage \
  --owner OWNER --data PREPARED_KV --cache LOCAL_CACHE \
  --configs experiments/classic/seed_42.yaml experiments/classic/seed_6767.yaml \
  experiments/classic/seed_696969.yaml experiments/deltanet/seed_42.yaml \
  experiments/deltanet/seed_6767.yaml experiments/deltanet/seed_696969.yaml
```

Команда выводит путь `series.json`. Сохраните его как `MANIFEST`. `stage` не читает `.env`
и не вызывает Kaggle. Он фиксирует разрешённые исходники рабочего дерева, включая
незакоммиченные изменения, `pyproject.toml`, `uv.lock`, README, полные разобранные конфиги
и manifest подготовленных данных. Wheel собирается из отдельного снимка, а не из меняющегося
рабочего дерева. При повторении неизменённого `stage` wheel и requirements не пересоздаются.

Следующие команды используют существующую локальную авторизацию Kaggle. Они выполняют
облачные записи и требуют отдельного разрешения оператора.

```sh
uv run --locked scripts/train_kaggle_series.py publish --manifest MANIFEST
uv run --locked scripts/train_kaggle_series.py run --manifest MANIFEST \
  --credentials-dataset OWNER/EXISTING_PRIVATE_DATASET \
  --select experiments/classic/seed_42.yaml experiments/classic/seed_6767.yaml
```

Публикуются два отдельных приватных dataset: release и подготовленные KV-данные.
SDK получает явное `public=False`; обновление версии существующего dataset не выполняется.
После создания проверяется статус точного dataset. Wheel, requirements и конфиги упакованы
в ZIP с расширением `.bundle`, чтобы сервис не распаковывал архив при публикации.
Kernel проверяет SHA256 архива, все файлы release, полный manifest KV и все семь файлов KV.
Данные остаются в read-only input, не генерируются повторно и не копируются в output.
Пути монтирования определяются по уникальному идентификатору dataset, без выбора первого wheel
или requirements из всех подключённых inputs.

`--credentials-dataset` подключает **существующий приватный** input с `kaggle-env.env`.
Это совместимо с прежней передачей разрешённых ключей через приватный dataset.
Файл читается только в удалённом kernel. При отсутствии этого флага используется Kaggle Secrets.
Ключи авторизации не входят в immutable release, manifest данных и fingerprints. Локальные
`stage` и тесты не читают настоящий `.env`. Не передавайте credentials в конфигурациях.

После двух запусков оператор обязан вызвать отдельный seed gate:

```python
import json
from pathlib import Path
from compare_attention.kaggle_series import run_selected
from compare_attention.seed_gate import verify_seed_pair

manifest = Path("MANIFEST")
state = json.loads(manifest.read_text(encoding="utf-8"))
def result_dir(name):
    slug = state["runs"][name]["slug"]
    return manifest.parent / "runs" / slug / "output" / slug

verify_seed_pair(result_dir("experiments/classic/seed_42.yaml"),
                 result_dir("experiments/classic/seed_6767.yaml"))
# Только если gate прошёл:
run_selected(manifest, ["experiments/classic/seed_696969.yaml",
                       "experiments/deltanet/seed_42.yaml",
                       "experiments/deltanet/seed_6767.yaml",
                       "experiments/deltanet/seed_696969.yaml"],
             credentials_dataset="OWNER/EXISTING_PRIVATE_DATASET")
```

API `stage_series(root, configs, prepared_data, cache, owner, run=None)` возвращает путь manifest.
`publish_series(manifest, run=None)` публикует inputs один раз.
`run_selected(manifest, names, ..., run=None)` использует тот же release для выбранной группы.
API не загружает локальный `.env`: вызывающий код отвечает за авторизацию. Gate не вызывается
автоматически: вызывающий код обязан проверить первые два результата перед оставшимися четырьмя.

## Возобновление и доказательства завершения

Журнал `series.json` записывается атомарной заменой. Для каждого запуска сохраняются состояния
`staged`, `submitting`, `submitted`, `completed`, `verified`, а также точный номер версии kernel.
Опрос и скачивание используют `owner/slug/version`, не последнюю неизвестную версию.
Повтор команды с теми же выбранными конфигами после ошибки скачивания повторяет download
с `--force`, но не build, export, upload или обучение. Уже проверенные результаты сверяются
с manifest. Следующий push разрешён только после полного скачивания и проверки предыдущего.

Kernel формирует отдельный `result-manifest.json`: идентичность серии/конфига/kernel и checksums
файлов результата. Оркестратор проверяет manifest, checkpoint, единственный финальный result,
конечные метрики и достижение заданного FLOP-бюджета. Проверка seed-независимости остаётся
за отдельным gate; файл обучения `run_identity.json` оркестратор не подменяет.

Локальные фазы build/export/stage/upload/push/wait/download/verify записываются в журнал.
Удалённые verify_inputs/install_dependencies/install_wheel/train записываются в
`phase-timings.json`. Эти таймеры не являются доказательством ускорения GPU или компиляции.

## Ограничения

- Облачное выполнение и фактическое уменьшение времени требуют отдельной проверки на Kaggle.
  Офлайн-тесты используют fake runner; локально выполняются настоящие build/export и проверки данных.
- Один журнал рассчитан на одного последовательного оператора. Параллельные writers не поддержаны.
- Неопределённый результат push (`submitting`) и неуспешное обучение требуют ручной сверки.
  Автоматический повтор обучения запрещён. Прерывание между удалённым create и записью журнала
  также требует ручной сверки; существующий dataset не перезаписывается.
- Для изменённой серии с уже существующим KV dataset create-only публикация останавливается;
  автоматическое присоединение чужого/ранее созданного dataset без сверки не реализовано.
- SHA256 проверяется полностью, поэтому повторный stage и начало kernel читают все bytes KV.
  Это стоимость проверки целостности, не повторная генерация.
- Общий pip wheelhouse, перенос Torch/Inductor/Triton caches и изменение compile flags не включены.
  Совместимость межkernel-кеша не проверена; экономия компиляции не заявляется.
