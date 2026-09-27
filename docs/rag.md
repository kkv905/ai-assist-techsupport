# RAG: LlamaIndex и bare-metal (M5B3)

## Конфигурация и коллекции

Используются `llama-index==0.14.25`, `llama-index-core==0.14.25`,
`llama-index-vector-stores-qdrant==0.10.3`, `llama-index-readers-file==0.7.0`
и `llama-index-embeddings-huggingface==0.8.0`. Версии зафиксированы также в
`uv.lock`. Embedding-модель — `BAAI/bge-m3`, размерность 1024, metric Qdrant —
COSINE. Все изменяемые параметры находятся в `.env`: `RAG_COLLECTION`,
`RAG_BAREMETAL_COLLECTION`, `RAG_DATA_DIR`, chunk size/overlap, top-k и
порог score.

`rag_block_03` — отдельная коллекция LlamaIndex. В отличие от `documents` из
M5B2, она содержит `_node_content`, поэтому `from_vector_store` восстанавливает
метаданные и `source_nodes`. Коллекция bare-metal также отдельна
(`rag_block_03_baremetal`): у неё намеренно плоский payload `text`, `source`,
`chunk_index`. Смешивать их нельзя: retrieval может вернуть вектор, но потерять
цитаты и формат ноды.

## LlamaIndex vs bare-metal

| Критерий | LlamaIndex | Bare-metal |
| --- | --- | --- |
| Строк кода (ingestion + query, без импортов) | ~55 | ~85 |
| Поддержка форматов из коробки | SimpleDirectoryReader: TXT/MD/PDF/DOCX и другие | Только MD/TXT в реализации блока |
| Что дописать для PDF/DOCX | Ничего: файловый reader уже подключён | Парсеры pypdf/python-docx и извлечение текста |
| Что дописать для batch-ingestion / async | batching/async ingestion и наблюдаемость задач | batching embeddings/upsert, retry, async-клиенты, контроль очереди |
| Где удобнее дебажить top_score / source_nodes | source_nodes готовы, score легко читать | Максимальная прозрачность Qdrant points/payload |
| Где гибче подменять re-ranker, chunker | Плагины и pipeline LlamaIndex | Любой компонент, но всю связку нужно поддерживать вручную |

В дипломе остаётся LlamaIndex-версия: она короче, читает требуемые форматы и
сохраняет происхождение фрагмента без собственной схемы. Bare-metal версия
оставлена как эталон прозрачности и для отладки точного запроса в Qdrant.
При этом fallback реализован до генерации в обеих версиях: score ниже порога
возвращает честный ответ, не вызывая LLM. Для production стоит откалибровать
`RAG_SCORE_THRESHOLD` на размеченных обращениях.

## Прогон 5 вопросов

Ниже — ожидаемый ручной smoke-прогон корпуса из `data/rag-block-03` после
`docker compose up -d qdrant` и запуска `python -m app.services.rag`. Числа
score зависят от версии модели/индекса и приведены как ориентир для проверки
релевантности.

| Тип | Вопрос | Краткий ответ | Top-1 source / score | Оценка и гипотеза |
| --- | --- | --- | --- | --- |
| Хороший | Как долго действует ссылка для сброса пароля? | 15 минут | `01-password-reset.md` / ~0.82 | Релевантно: точная формулировка есть в одном чанке. |
| Хороший | Какие порты нужны для почтового клиента? | IMAP 993 SSL, SMTP 587 STARTTLS | `03-email.md` / ~0.78 | Релевантно: технические термины совпадают. |
| Хороший | Что приложить к заявке о недоступности сервиса? | Время, URL, скриншот, X-Request-ID | `08-incident.txt` / ~0.74 | Релевантно: список целиком находится в коротком чанке. |
| Средний | VPN перестал пускать после истечения пароля — что делать? | Сменить пароль на портале, подождать 2 минуты, повторить VPN | `02-vpn.txt` / ~0.66 | Релевантно: перефразирование объединяет пароль и VPN; при крупном корпусе полезен re-ranker. |
| Вне базы | Когда в офисе будет доставлена новая кофемашина? | Сведений в корпусе нет | `10-office-plants.txt` / ниже порога | Корректный fallback: документ заведомо не про техподдержку и генерация не вызывается. |

Endpoint `POST /rag/query` использует экземпляр, построенный один раз в
FastAPI lifespan. Пример: `curl -X POST http://localhost:8000/rag/query -H
"Content-Type: application/json" -d '{"question":"Как сбросить пароль?"}'`.
