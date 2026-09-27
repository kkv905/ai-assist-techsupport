# Векторное хранилище заявок

Рабочая Qdrant-коллекция — `documents`; вектор модели `BAAI/bge-m3` имеет
1024 компоненты. Коллекция использует COSINE и HNSW `m=16`, `ef_construct=100`.
Это явная фиксация стандартных значений Qdrant: для небольшого текущего корпуса
она даёт хороший баланс памяти и recall, не требуя преждевременной настройки.

Загрузка выполняется командой `uv run python scripts/load_to_qdrant.py`. Она
читает 12 предметных статей из `data/support_articles.json`, разбивает каждую на
10 самостоятельных смысловых фрагментов и загружает 120 точек. UUID5 от
`source:chunk_index` делает повторный запуск upsert-ом тех же точек, а не
дублем. В payload присутствуют `source`, `text`, `created_at`, `tenant_id`,
`category` и `status`; для `source`, `created_at`, `tenant_id`, `status`
созданы payload-индексы.

## Метрика

Проверка запускается на тех же векторах командой
`uv run python scripts/compare_qdrant_metrics.py`. Скрипт создаёт временные
`documents_cosine` и `documents_dot`, печатает таблицу с полными UUID и удаляет
обе учебные коллекции в `finally`; в production остаётся только `documents`.
Ниже приведены фактические результаты прогона 27.09.2026. Векторы bge-m3 нормализуются в
`EmbeddingService`, поэтому COSINE и DOT сохраняют порядок: при единичной норме
скалярное произведение равно cosine similarity.

| Запрос | Top-5 COSINE | Top-5 DOT | Совпало |
| --- | --- | --- | --- |
| access denied в АИС Правоохрана | `b65fb445, 9213e691, f655e696, 50d63204, 9a8d6504` | `b65fb445, 9213e691, f655e696, 50d63204, 9a8d6504` | Да |
| XML не импортируется в ЦРСВЭД | `b8bdce0b, 66e94822, 56a360ef, 44894335, 6888e40c` | `b8bdce0b, 66e94822, 56a360ef, 44894335, 6888e40c` | Да |
| ошибка 500 в Постконтроле | `5445cd57, aee899bb, 9fae8e88, a2389cbf, 88b97793` | `5445cd57, aee899bb, 9fae8e88, a2389cbf, 88b97793` | Да |
| заблокирован пароль сотрудника | `0fdab60a, 0f00c8bd, db5fd998, d8f826e2, 51315123` | `0fdab60a, 0f00c8bd, db5fd998, d8f826e2, 51315123` | Да |
| не подписывается документ ЭП | `0afed31d, c25ebcaf, 209b26e7, c817058c, a12e81c4` | `0afed31d, c25ebcaf, 209b26e7, c817058c, a12e81c4` | Да |

В production оставлен COSINE: это стандартная и явно выраженная метрика для
семантического поиска, безопасная и если в будущем источник векторов перестанет
нормализовать их. DOT возможен без изменения порядка только при сохранении
инварианта нормализации.

## Фильтры metadata

Во всех примерах `store` — `VectorStore`, а `query_vector` получен через
`embed_texts([query])[0]`. В таблице метрик UUID сокращены до первых 8 символов;
ниже у фильтров приведены полные идентификаторы реального прогона.

### Match: источник статьи

```python
from qdrant_client.models import FieldCondition, Filter, MatchValue

only_authorization = Filter(
    must=[FieldCondition(key="source", match=MatchValue(value="faq_authorization_2025.md"))]
)
points = await store.search(query_vector, top_k=3, query_filter=only_authorization)
```

Top-3: `b65fb445-921d-5a2a-b19c-0d77402b74dd`,
`9213e691-9f2a-5044-b57d-bfd1c915f5b5`,
`f655e696-4809-57e3-97a3-4c2b30469d81`.

### Range: только свежие документы

```python
from datetime import UTC, datetime, timedelta
from qdrant_client.models import DatetimeRange, FieldCondition, Filter

fresh = Filter(
    must=[FieldCondition(
        key="created_at", range=DatetimeRange(gte=datetime.now(UTC) - timedelta(days=30))
    )]
)
points = await store.search(query_vector, top_k=3, query_filter=fresh)
```

Для запроса об архивных обращениях без фильтра первым оказался
`b7c7963d-b20c-5b6a-a9c6-d2b7b9006e5b` из `policy_archived_cases.md` (он старше
30 дней); с фильтром он исключён. Top-3 свежей выдачи:
`148d30a2-7597-505d-9beb-c73e8d0f6201`,
`5959f7a0-9c7a-54fd-b9fe-a9f8c42ea0c4`,
`9213e691-9f2a-5044-b57d-bfd1c915f5b5`.

### Композитный must + must_not: tenant без архива

```python
from qdrant_client.models import FieldCondition, Filter, MatchValue

active_west = Filter(
    must=[FieldCondition(key="tenant_id", match=MatchValue(value="customs-west"))],
    must_not=[FieldCondition(key="status", match=MatchValue(value="archived"))],
)
points = await store.search(query_vector, top_k=3, query_filter=active_west)
```

Top-3 для запроса о подписи: `0afed31d-5f2a-51ea-813d-b359d8e925ba`,
`c25ebcaf-14c3-5138-bd66-ca220b98a9e8`,
`209b26e7-18df-50ec-a3bf-e1bbc450b60f`; `policy_archived_cases.md:*` не попадает в выдачу. Это
типичный production-паттерн для изоляции тенанта и исключения устаревших знаний.
