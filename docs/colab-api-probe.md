# Colab API: фиксированная проверка оборудования и входных файлов

`scripts/colab_api_probe.py` — отдельный оператор **только readiness**, не запуск обучения.
Kaggle, frozen-export и существующие ручные Colab-операторы не изменяются.

## Границы

Первым исполняется точный `colab_runner.PREFLIGHT`: Python 3.13.x, ровно один
T4 / Tesla T4 с capability 7.5 по `nvidia-smi`. До прохождения guard нет
проверки Drive или чтения архива. Приватные входные файлы runtime не открываются;
локальный OAuth нужен для самого API до запуска remote guard.
После guard выполняется только stdlib-код:

- `os.path.ismount('/content/drive')` и наличие `MyDrive`;
- наличие `/content/drive/MyDrive/compare-attention/colab-package.zip`;
- потоковый SHA-256 этого архива, без распаковки или импорта его содержимого;
- только `is_file()` для отдельного `compare-attention/kaggle-env.env`.

**Содержимое env никогда не открывается.** Нет Drive mount/OAuth, Colab Secrets,
установки pip, загрузок/выгрузок, записи на Drive, optimizer или обучения.
Нет аргумента для произвольного Python-кода или endpoint.

Ожидаемый SHA-256:
`e23fe304d1276bfe9e64e187f1c39556173ebd19927330c2bce50b06599cc3cd`.
Он независимо вычислен из неизменяемого
`E:/HermesArtifacts/compare-attention/colab-k64-export-03/colab-package.zip`;
размер файла — 157427451 байт. В переданном контексте hash и размер были
склеены: это не 68/73-символьный SHA. Никакой архив не пересобирался.

## CLI

Из корня репозитория; для Windows используется существующий `.venv`:

```bash
# План: без credentials, refresh, HTTP и записей в cloud/journal.
.venv/Scripts/python.exe scripts/colab_api_probe.py --runtime runtimes/EXACT_ID

# Только оператор после review, management OAuth и разрешения на один runtime:
.venv/Scripts/python.exe scripts/colab_api_probe.py \
  --runtime runtimes/EXACT_ID --execute \
  --client-file 'C:/PRIVATE/desktop-client.json' \
  --token-file 'C:/PRIVATE/colab-management-token.json' \
  --journal local/reports/colab-api/probe/one-readiness-attempt.jsonl
```

OAuth использует существующий `auth.access_token(..., required_scope=auth.MANAGE)`.
Readonly-token не заменяет отдельное management-разрешение. Сам оператор не
выполняет login, создание runtime или подтверждение согласия пользователя.
Runtime ID проверяется до auth; не следует подставлять URL или нормализовать ID.

Exit 0: план либо `complete`; exit 3: `blocked`; exit 2: ошибка/неоднозначность.
`complete` допустим только после terminal tool response с совпавшими session и
execution UUID, успешным guard и всеми проверками входных файлов.
Отсутствующий Drive/env или несовпавший архив означает `blocked`, а не
успешную готовность и не неудачу обучения. Ephemeral fallback отсутствует.

Это **не** проверка записи/восстановления checkpoints, содержимого env,
ClearML/Hugging Face, Torch/cu130, FP16 или optimizer. Fixture-тесты не доказывают
наличие реального GPU. Headless mount не обещается: Drive может потребовать
интерактивную авторизацию и остановить эту фазу на `blocked`.

## Протокол и однократность

Единственный endpoint — `https://colaboratory.googleapis.com/mcp`, stdlib urllib,
HTTPS и запрет redirect. Реализован **явно pinned legacy MCP 2025-11-25**:
`initialize` → `notifications/initialized` → `tools/call`. Версия ответа и tools
capability проверяются до kernel/session writes. MCP-Session-Id проверяется и
хранится только в памяти; последующие запросы несут его и MCP-Protocol-Version.

Важное ограничение актуальных primary docs: последняя описанная редакция MCP
**2026-07-28 уже не использует initialization handshake**. Здесь намеренно
реализован запрошенный handshake старой редакции, а не смесь двух протоколов.
Google public tool docs не устанавливают поддержку этой версии сервером:
её необходимо подтвердить live отдельно. Modern-only сервер даст безопасный
stop; автоматического fallback, повторной инициализации или resend нет.

Ответы JSON и SSE ограничены 1 MiB. Response parser имеет бюджет 30 секунд
между проверками и socket timeout 30 секунд (ожидание read может довести
response phase примерно до 60 секунд); прогресс не продлевает бюджет.
Notifications и SSE keepalive не считаются terminal result. Неверный JSON-RPC ID,
RPC/tool error, отсутствие structuredContent, усечённый output, executionError,
неверные поля или marker приводят к fail-closed. Необработанные stdout/stderr,
provider errors, result strings, URLs, токены и transport session IDs не выводятся
и не попадают в journal; принимается только whitelist marker JSON.

Journal создаётся `O_EXCL`, mode 0600, fsync: resource session и execution UUID
сохраняются **до первого cloud call**, pending-фаза — перед каждым действием.
Внутри repo разрешён только ignored `local/reports/colab-api/probe/*.jsonl`;
можно выбрать отдельный приватный каталог вне repo. На Windows действуют
унаследованные ACL каталога; файл содержит только IDs, фазы и whitelist readiness.
Существующий journal всегда блокирует повтор, в том числе после успеха.
Автоматического resume/new execution по тому же journal нет.

`create_session` вызывается один раз с новым собственным RFC1034 ID;
`execute_code` — один раз с заранее сохранённым UUID. Disconnect/timeout означает
неизвестное состояние: **не повторять execute_code**. Google предупреждает, что
повтор executionId после final response может выполнить код заново.

После подтверждения точного собственного session name оператор пытается закрыть
только эту kernel session через документированный `delete_session`, затем
проверяет её отсутствие через `list_sessions`. При ambiguous execution это также
останавливает собственный kernel; чужие session не удаляются. Ошибка cleanup
не скрывает основную ошибку; cleanup pending/required остаётся в journal.
Неоднозначный create_session без подтверждения не даёт права удалять неизвестную
session. **Родительский оператор обязан освободить выделенный runtime после
короткой проверки, особенно при blocked**, и независимо проверить удаление.
Удаление kernel session не освобождает сам GPU runtime.

## Primary references, проверены 5 ноября 2026

- Google endpoint/tool overview: <https://developers.google.com/colab/api/reference/mcp>
- create_session schema/create-or-get: <https://developers.google.com/colab/api/reference/mcp/tools_list/create_session>
- execute_code, terminal result и повтор UUID: <https://developers.google.com/colab/api/reference/mcp/tools_list/execute_code>
- own-session cleanup: <https://developers.google.com/colab/api/reference/mcp/tools_list/delete_session>
- readback: <https://developers.google.com/colab/api/reference/mcp/tools_list/list_sessions>
- Legacy transport: <https://modelcontextprotocol.io/specification/2025-11-25/basic/transports>
- Legacy lifecycle: <https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle>
- Latest transport/versioning distinction: <https://modelcontextprotocol.io/specification/2026-07-28/basic/transports>
- Latest lifecycle/backward compatibility: <https://modelcontextprotocol.io/specification/2026-07-28/basic/lifecycle>

## Локальная проверка

TDD: наблюдались RED отсутствующего transport, SSE, handshake, probe, parser/journal,
затем исправления GREEN; отдельно выявлен и закрыт небезопасный unbounded JSON read.
Тесты используют fake auth/ответы и настоящий loopback HTTP server (JSON и SSE),
но не настоящие Google credentials, MCP writes или GPU. Они проверяют порядок
реально сгенерированного guard, отсутствие чтения env, blocked, sanitization,
no-retry/no-clobber и own-session cleanup/readback.

```bash
.venv/Scripts/python.exe -m pytest tests/scripts/test_colab_api_probe*.py -q
.venv/Scripts/python.exe -m ruff check scripts/colab_api_probe.py scripts/colab_api_mcp.py tests/scripts/test_colab_api_probe*.py
.venv/Scripts/python.exe -m ruff format --check scripts/colab_api_probe.py scripts/colab_api_mcp.py tests/scripts/test_colab_api_probe*.py
.venv/Scripts/python.exe -m pylint scripts/colab_api_probe.py scripts/colab_api_mcp.py tests/scripts/test_colab_api_probe*.py
```

XML и transcript находятся в `local/reports/colab-api/probe/`.
Live Google handshake, реальный T4 и Drive readiness этим изменением **не проверены**.
