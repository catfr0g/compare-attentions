# Colab API: phase 2 — управление одним runtime

## Границы

Stdlib CLI `scripts/colab_api.py` сохраняет readonly команды `auth-login`,
`auth-status`, `specs`, `t4-eligible`, `runtimes`, `get runtimes/ID`.
Default по-прежнему `t4-eligible`, без выделения GPU. Добавлены `create`,
`status`/`reconcile`, `stop`/`delete`, `operation operations/ID`.

**Это runtime management, не full-training readiness.** Эти команды не исполняют
Python в runtime, не устанавливают пакеты и не обучают. Фактические Python/GPU/Drive
проверяет отдельный фиксированный preflight. Старые notebook, frozen ZIP, пять
ручных скриптов, core/lockfile и условия обучения не меняются. Kaggle сохранён,
Colab — дополнительный backend. ClearML/HF и отдельный приватный `.env` остаются
конфигурацией обучения; Drive — дополнительная persistence, не их замена.
CLI `.env` не читает, Colab Secrets не требуются.

Реальные OAuth и выделение ОДНОГО T4 — только после независимого review финальных
байтов и явного разрешения. Локальные fake HTTP не доказывают eligibility аккаунта,
свободный GPU, квоту, оплату или готовность к обучению. Прошлый readonly успех не
гарантирует выделение. CLI не создаёт проекты, не включает API, не меняет квоты или
биллинг. Обещания бесплатности нет. Google/организация могут ограничить доступ.

## OAuth: management token отдельно

Владелец вручную настраивает Colab API, consent screen/audience/test users и OAuth
client типа Desktop app. JSON `installed` содержит client_id, client_secret,
project_id, Google auth_uri/token_uri. Старый Google auth_uri принимается, но вход
идёт через /o/oauth2/v2/auth. Web client/service account не заменяют Desktop flow.
JSON/токены храните вне репозитория и архивов; передавайте только путь, не секреты.

Readonly: `https://www.googleapis.com/auth/colaboratory.readonly`.
Management: `https://www.googleapis.com/auth/colaboratory`.
Full scope разрешает readonly локально, обратное неверно. Create/delete требуют
реально выданного full scope. `--management` допустим только для `auth-login`;
по умолчанию вход остаётся readonly. Первичный ответ без запрошенного scope
отклоняется. Refresh без scope сохраняет подтверждённый scope; явное сокращение
полного scope отклоняется без перезаписи token file даже при readonly команде.

**Старый readonly token.json не заменять.** Management login с default readonly
путём запрещён даже с --overwrite. Нужен новый management-token.json. Любой
существующий token file защищён от перезаписи без явного --overwrite.

Windows bash из корня проекта; CLIENT — пример приватного пути:

```bash
PY=.venv/Scripts/python.exe
CLIENT='C:/Users/Michael/private/colab-desktop.json'
MANAGEMENT='C:/Users/Michael/AppData/Local/compare-attention/colab-api/management-token.json'
JOURNAL='local/reports/colab-api/one-t4-runtime.json'

# Старый readonly token остаётся default.
$PY scripts/colab_api.py auth-status
$PY scripts/colab_api.py --client-file "$CLIENT" t4-eligible
$PY scripts/colab_api.py --client-file "$CLIENT" runtimes
$PY scripts/colab_api.py --client-file "$CLIENT" get runtimes/ID

# Только после review и разрешения расширить OAuth.
$PY scripts/colab_api.py --client-file "$CLIENT" --token-file "$MANAGEMENT" --management auth-login
$PY scripts/colab_api.py --token-file "$MANAGEMENT" auth-status
```

Auth-status офлайн: scope_valid означает чтение, management_valid — полный scope;
показываются только реально присутствующие whitelisted scopes, expiry и наличие
refresh token. Это не серверная проверка валидности. Для refresh нужен тот же
--client-file. Default Windows: %LOCALAPPDATA%/compare-attention/colab-api/token.json;
иначе $LOCALAPPDATA или ~/.local/share/compare-attention/colab-api/token.json.
Файлы .env в качестве OAuth не принимаются.

Login: PKCE S256, случайный state, callback 127.0.0.1 со свободным портом, 300 с.
Callback не журналируется. Код не надо копировать в чат/терминал. --no-browser
показывает consent URL, но callback должен прийти на ту же машину: это не OOB-flow.
Token file атомарный, POSIX 0600; Windows ACL наследуется, CLI его не меняет.
Владелец ограничивает права приватного каталога.

## Create и GET reconciliation

```bash
# Только plan: нет OAuth, сети, записи journal/lock/каталогов.
$PY scripts/colab_api.py --journal "$JOURNAL" create

# Только после отдельного разрешения выделить ОДИН T4.
$PY scripts/colab_api.py --client-file "$CLIENT" --token-file "$MANAGEMENT" --journal "$JOURNAL" --execute create
$PY scripts/colab_api.py --client-file "$CLIENT" --token-file "$MANAGEMENT" --journal "$JOURNAL" status
$PY scripts/colab_api.py --client-file "$CLIENT" --token-file "$MANAGEMENT" --journal "$JOURNAL" --wait-timeout 120 --poll-interval 2 reconcile
$PY scripts/colab_api.py --client-file "$CLIENT" --token-file "$MANAGEMENT" operation operations/ID
```

Перед POST заново читается runtimespecs: только eligible is True и точная тройка
VARIANT_GPU / T4 / SHAPE_STANDARD. Число 1, строка true, HIGHMEM, L4 и CPU не подходят;
fallback нет. Старый t4-eligible показывает STANDARD/HIGHMEM для совместимости.
UUID4 requestId и RFC-1034 runtimeId генерируются один раз и записываются ДО POST
в атомарный no-clobber journal с create_unknown. POST содержит только runtimeSpec.
Version не задана: Google выбирает текущий образ, это не image pin и не обещание
bitwise-эквивалентности. Фактическую version можно отдельно прочитать командой get.

Существующий journal всегда запрещает новый POST, даже после ошибки/удаления.
Timeout/обрыв/прерывание — неизвестный результат: только GET reconcile.
**Не повторять create и не выбирать новый journal/runtimeId для обхода запрета.**
GET 404 после неоднозначного POST не доказывает, что выделения не было/не будет.
При потере/ручном изменении journal остановитесь и отдельно проверьте аккаунт.
Защита локальна для данного journal, не account-wide; journal не криптографическое
доказательство владения. OS lock допускает одного writer (JOURNAL_BUSY для второго).
Файл .lock остаётся, но ОС освобождает блокировку после выхода/падения процесса.
Не удаляйте его для обхода активного writer.

Status и reconcile — синонимы: Colab API только GET, journal обновляется локально.
OAuth refresh при необходимости отдельно обращается к token endpoint.
Operation: GET /v1/operations/ID, НЕ /v1beta/operations/ID.

- creating: operation ещё не завершена; отсутствующее done/false — не успех.
- create_failed: завершённая operation с error; только code и безопасный reason.
- create_unknown/absent_unknown: выделение не подтверждено; retry нет.
- ready: только GET точного имени и spec, не Python/Drive/training readiness.
- mismatch: имя/spec отличаются; чужой runtime удалять нельзя.
- delete_unknown: DELETE мог уйти, acknowledgment нет; повтор запрещён.
- delete_acknowledged: DELETE подтверждён, отсутствие ещё не проверено.
- deleted: подтверждённый DELETE, затем GET с HTTP 404.

По умолчанию одно наблюдение. --wait-timeout: 0..300 с, --poll-interval: 1..30 с
(default 2). Deadline возвращает последнее наблюдение, не выдуманный успех.
HTTP socket timeout 20 с; начатый запрос может закончиться после polling deadline.
Ошибка сети не превращается в ready/deleted или разрешение повторного POST.
Journal хранит последнее наблюдение, не вечный live status.

## Stop = delete/unassign, не pause

**Stop и delete — синонимы разрушительного unassign assignment. Ephemeral-файлы
могут быть потеряны. Это не checkpoint, не пауза и не гарантия сохранности результатов.**
Persistence перед удалением проверяет оператор.

```bash
# Только plan/предупреждение: нет сети и изменения journal.
$PY scripts/colab_api.py --journal "$JOURNAL" stop
# Только явно разрешённое удаление runtime ИЗ ЭТОГО journal.
$PY scripts/colab_api.py --client-file "$CLIENT" --token-file "$MANAGEMENT" --journal "$JOURNAL" --execute stop
$PY scripts/colab_api.py --client-file "$CLIENT" --token-file "$MANAGEMENT" --journal "$JOURNAL" reconcile
```

Произвольный positional runtime для stop/delete запрещён. Перед DELETE GET проверяет
точное имя и spec. Delete_unknown записывается до отправки; любой уже начатый DELETE
запрещает повтор. Только пустой успешный ответ считается acknowledgment, затем
требуется GET 404. GET 200, 403/503 или timeout не означают deleted.
После неоднозначного DELETE даже GET 404 оставляет delete_unknown: acknowledgment
не доказан. Автоматического повторного удаления/аллокации нет.

## Защита и локальные доказательства

Фиксированный HTTPS Google host и allowlists GET/POST/DELETE; redirect запрещён
для API и OAuth. JSON до 2 MiB. Literal identifiers проверяются до lookup,
не исправляются. Traversal/query/fragment/foreign endpoints отвергаются.
Весь connectionInfo, proxy tokens, подписанные URL, operation metadata/response и
сырые provider messages/details не выводятся и не сохраняются. Ошибки — безопасные
коды/известные reasons. Credentials не передаются в CLI arguments.

Tests/scripts/test_colab_api*.py: только fake credentials/local HTTP, включая
реальный subprocess create → status → stop → GET 404; OAuth/refresh/PKCE,
single-writer/no-clobber, прерывания, exact spec и redaction. Management discovery
revision 20261105. Логи/XML/lint: local/reports/colab-api/management/.
Реальные OAuth/GPU, установка пакетов и обучение при этой реализации не выполнялись.

## Официальные источники

- https://developers.google.com/colab/api/reference/rest
- https://developers.google.com/colab/api/reference/rest/v1beta/runtimes/create
- https://developers.google.com/colab/api/reference/rest/v1beta/runtimes/get
- https://developers.google.com/colab/api/reference/rest/v1beta/runtimes/delete
- https://colaboratory.googleapis.com/$discovery/rest?version=v1beta
- https://colaboratory.googleapis.com/$discovery/rest?version=v1
- https://developers.google.com/identity/protocols/oauth2/native-app

