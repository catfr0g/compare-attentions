# Управление исследовательскими прогонами

## Назначение

Early stopping и checkpoints позволяют ограничивать исследовательские расходы.
Они не меняют размер модели, batch или правила алгоритмического подсчёта FLOPs.
По умолчанию обе возможности отключены: старые YAML продолжают работать до бюджета.

## Конфиг

Пример секции для новой серии `experiments/fp16_long_controls/`:

```yaml
schedule:
  train_flops: 100_000_000_000_000_000
  batch_size: 8
  dtype: float16
  compile: true
  checkpoint_every_steps: 1000
  checkpoint_keep_last: 2
  early_stopping_patience: 20
  early_stopping_min_delta: 0.001
  early_stopping_min_steps: 2000
  amp_max_scale: 128.0
```

Все параметры одинаковы для classic и DeltaNet. Сиды и остальные настройки находятся
в полных отдельных YAML. `experiments/fp16/` и `experiments/fp16_long/` не перезаписаны.

## Early stopping

- Контролируется validation loss, меньше — лучше.
- Улучшение должно быть **строго больше** `early_stopping_min_delta`.
- `early_stopping_patience` — число последовательных validation без такого улучшения,
  а не число optimizer steps. При текущей cadence 100 значение 20 соответствует
  20 оценкам validation.
- До `early_stopping_min_steps` обновляется лучший результат, но счётчик patience
  не накапливается. Это защита от остановки во время начального обучения.
- Для выключения: `early_stopping_patience: 0` или CLI
  `--early-stopping-patience 0`.
- При достижении бюджета его завершение имеет приоритет. Итоговая validation
  записывается один раз, включая совпадение с обычной cadence.
- В result сохраняются `stop_reason` и состояние контроллера. В identity при ранней
  остановке используется `status: early_stopped`, а не `completed`.
- Итоговый `model.pt` содержит веса последнего выполненного обновления, **не автоматически
  восстановленные лучшие веса**. Лучший validation loss и его шаг сохраняются отдельно
  в состоянии контроллера.

Нельзя напрямую выдавать ранне остановленные прогоны за сравнение качества при равных FLOPs.
Для финального fixed-budget сравнения отключите early stopping и сравните одинаковые
бюджеты либо сопоставьте кривые на общей реально достигнутой оси FLOPs.
Строгие старые smoke/seed gates не должны принимать раннюю остановку как полный бюджет.

## Checkpoints и продолжение

`checkpoint_every_steps: 0` отключает периодические checkpoints.
При включении состояние сохраняется после успешных обновлений, каждые N шагов
и при штатном завершении (включая early stopping). Сохраняются последние
`checkpoint_keep_last` snapshots. Указатель `checkpoints/latest.json` публикуется
атомарно после успешной записи; неудачная новая запись не должна уничтожать предыдущую.

Полное состояние включает model, optimizer, GradScaler, RNG Python/NumPy/Torch,
позицию в данных, фактические steps/FLOPs, историю потерь и validation и early stopping.
Сохранение и validation не входят во время тренировочных итераций.

Для продолжения используйте точный `.pt` и **новый** каталог результата:

```sh
uv run --locked python -m compare_attention.cli.train \
  --config experiments/fp16_long_controls/classic/seed_42.yaml \
  --resume-from E:/results/previous/checkpoints/step_00001000.pt \
  --result-dir E:/results/resumed-classic-42
```

Checkpoint должен соответствовать тому же полному конфигу, данным, runtime и исходной
инициализации. Изменение бюджета, early stopping, precision или другой научной настройки
при resume запрещено: это уже отдельный эксперимент. История до checkpoint переносится
в новый журнал; выполненный бюджет не обнуляется. Автоматической повторной отправки
неудачного Kaggle kernel нет. Для облачного resume checkpoint нужно отдельно предоставить
как приватный read-only input; локальный CLI-флаг сам этого не делает.

`model.pt` остаётся финальным weights-only файлом для загрузки модели; это не полный
resumable snapshot. В прежнем неудачном long-run checkpoints не было, поэтому восстановить
его optimizer/RNG или продолжить с шага 26 548 невозможно.

## FP16 scale

`amp_max_scale: null` сохраняет прежнюю динамику GradScaler.
`amp_max_scale: 128.0` ограничивает рост сверху исходным scale 128, одинаково для моделей.
Параметры остаются FP32, forward/evaluation используют FP16 autocast.
Проверка реальных optimizer updates и остановка при skip/nonfinite не отключаются.

Высокий scale предыдущего неудачного прогона — основание проверить рост scaler,
но не доказательство физической причины. Ограничение scale должно пройти отдельные
CPU/CUDA-тесты и свежие облачные gates; его нельзя считать доказанным устранением NaN
по одному локальному тесту. Метаданные precision позволяют проверить применённую политику.

## Публикация и проверка

Новый source фиксируется в новом неизменяемом release. Перед длинными пилотами требуются
регрессия, независимая проверка, одношаговые и многошаговые T4 checks обеих моделей.
Общий ранее подготовленный KV input переиспользуется после проверки manifest.
Сначала запускаются только classic/42 и DeltaNet/42; оставшиеся сиды ждут анализа кривых.
