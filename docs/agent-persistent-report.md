# M6B4 — persistent LangGraph и human-in-the-loop

## 1. Backends checkpoint'ов

`AGENT_CHECKPOINTER=sqlite` — значение по умолчанию для локальной разработки:
оно сохраняет историю в `AGENT_SQLITE_PATH` (по умолчанию
`./var/agent_checkpoints.sqlite`) и не требует отдельного сервиса. В Compose
приложение получает `AGENT_CHECKPOINTER=postgres`; `AsyncPostgresSaver` берёт
тот же `DATABASE_URL`, что и backend, заменяя SQLAlchemy-драйвер `+asyncpg` на
URI psycopg v3. `memory` оставлен для очень коротких тестов/экспериментов.

`agent_lifespan()` открывает saver на всё время жизни FastAPI и ровно один раз
вызывает `await checkpointer.setup()` до сборки графа. На HTTP-запросах setup
не выполняется.

## 2. Postgres в Compose

Новый Postgres не создавался: сервис `app` в `compose.yaml` использует уже
имеющийся `postgres`, его `DATABASE_URL` и добавляет только
`AGENT_CHECKPOINTER=postgres`. После первого старта `setup()` создаёт в этой
же БД таблицы `checkpoints`, `checkpoint_writes`, `checkpoint_blobs` и
`checkpoint_migrations`; их можно увидеть командой:

```powershell
docker compose exec postgres psql -U chat -d chat -c "\dt"
```

Проверка выполнена 2026-09-29 после `AsyncPostgresSaver.setup()`:

```text
 checkpoint_blobs
 checkpoint_migrations
 checkpoint_writes
 checkpoints
```

При однократном запуске проверки на Windows для Psycopg потребовался
`SelectorEventLoop` вместо стандартного Proactor loop. Compose работает на
Linux, где этого ограничения нет; локальным режимом по умолчанию остаётся
SQLite.

Alembic не управляет этими таблицами: `alembic/env.py` исключает их через
`include_name`, поэтому autogenerate не предложит удалить схему LangGraph.

## 3. Опасная операция и идемпотентность

Опасный tool — `send_email`: повторная отправка клиенту является внешним
side-effect и может создать неверную коммуникацию. Граф содержит два разных
узла: `prepare_email -> confirm_and_execute_email`.

**До `interrupt()`:** только детерминированная сборка и валидация черновика
(`request_id`, адрес, тема, тело), без вызова почтового провайдера.
**После `interrupt()`:** только при `Command(resume=True)` вызывается
`send_email(draft)`. При resume LangGraph запускает узел заново, поэтому
разделение не позволяет отправить письмо дважды из-за выполнения кода до
паузы. Сейчас функция — безопасный integration seam без SMTP credentials;
реальный провайдер будет подключён в ней.

## 4. Interrupt и resume

Офлайн-скрипт `uv run python scripts/time_travel_demo.py` печатает такой
фрагмент (сокращённые идентификаторы checkpoint'ов опущены):

```text
interrupt payload: {'type': 'approve_send_email', 'preview': {...}}
demo-approved: __interrupt__ = {'type': 'approve_send_email', 'preview': {...}}
approved sent = True
```

Подтверждение передаётся каноническим API `await graph.ainvoke(Command(resume=True), config)`;
устаревшие runtime-breakpoints не используются.

## 5. Time travel

Скрипт читает `aget_state_history()` и показывает checkpoint перед отправкой:

```text
checkpoint_id                         next
...                                   ('confirm_and_execute_email',)
...                                   ('prepare_email',)
...                                   ('__start__',)
state before send: {'draft': {...}, 'sent': False,
                    'next': ('confirm_and_execute_email',)}
rejected sent = False
approved sent = True
```

Для отказа и одобрения используются два `thread_id` с одинаковым входом.
`resume` записывается в checkpoint-lineage и повторный resume того же
interrupt не является альтернативным решением.

## 6. Streaming

`POST /agent/stream` принимает стабильный `thread_id`, вход письма и роль,
а затем передаёт `graph.astream(..., stream_mode=["updates", "messages"])`
в `text/event-stream`. `updates` показывает завершённые узлы, `messages`
оставлен для будущих LLM-токенов; после потока endpoint читает snapshot и
выдаёт отдельный SSE `interrupt` с `__interrupt__`, если граф приостановлен.
Это компактнее, чем `astream_events(version="v2")`, который полезен для
трассировки внутренних вызовов, но заметно многословнее для клиентского API.

Пример:

```powershell
curl -N -X POST http://localhost:8000/agent/stream -H "Content-Type: application/json" -d '{"thread_id":"demo-1","input":{"request_id":"REQ-1","recipient":"user@example.test","subject":"Ответ","body":"Инструкция"}}'
```

Продолжить тот же поток после решения можно тем же endpoint с тем же
`thread_id`, исходным `input` и полем `"resume": true` (либо `false`).

## 7. Permission policy

`read-only` не отправляет письмо, `write-with-approve` всегда останавливается
на подтверждении, а `full` может выполнить отправку без паузы; роль передаётся
через `configurable.user_role` вместе со стабильным `thread_id`.

## 8. Ограничения и дальнейшая работа

SMTP/API-провайдер пока не подключён намеренно, поэтому production-интеграции
нужны credential storage, retry/idempotency key по `request_id` и аудит
доставки. SSE endpoint запускает новую фазу для входа; отдельный resume API и
прокидывание потока в Telegram останутся следующими задачами. Для production
также следует включить strict msgpack policy LangGraph и мониторинг размера
checkpoint-таблиц. На Windows standalone-проверкам Psycopg требуется
Selector event loop; это не относится к Linux-контейнеру production Compose.
