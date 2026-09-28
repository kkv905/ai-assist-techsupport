# Оценка качества RAG (M5B6)

## Конфигурация

Production RAG использует `gpt-5.2`, `BAAI/bge-m3` (1024 измерения), recursive
chunking 512/64, top-K 5, score guard 0.3 и выключенный BGE re-ranker. Судья
зафиксирован отдельно: `deepseek-chat` через OpenAI-совместимый endpoint
`https://api.deepseek.com`; embeddings судьи — `text-embedding-3-small`
через OpenAI. Это не модель, отвечающая пользователю.

## Golden dataset

`tests/eval/golden_dataset_raw.csv` содержит исходный черновик из 31 вопроса.
Каждая из 31 записей в `tests/eval/golden_dataset.json` вручную проверена:
убраны общие формулировки, добавлены короткие эталонные ответы и исходные
контексты. Корпус включает статьи по VPN, учётным записям, почте, Wi-Fi,
безопасности и workplace.

## Как запустить воспроизводимый прогон

```powershell
uv sync --extra eval --extra tracing
uv run python scripts/run_eval.py --label baseline
uv run python scripts/run_eval.py --label chunk_1024
uv run python scripts/run_eval.py --label top_k_5
```

Каждый запуск создаёт timestamped CSV и JSON с агрегатами в
`tests/eval/results/`. CSV хранит ответ, полные retrieved contexts, latency и
все пять метрик. Ошибка судьи сохраняется в отдельной колонке, а не скрывается
усреднением. До запуска необходимы Qdrant, `OPENAI_API_KEY` и ключ выбранного
судьи (`DEEPSEEK_API_KEY` для текущего значения).

Во время прогона `scripts/run_eval.py` append-only сохраняет каждую готовую
строку в `tests/eval/results/.{label}.checkpoint.jsonl`. Повторная команда с
тем же `--label` читает checkpoint и выполняет только вопросы без готовой
строки; итоговые CSV/JSON создаются после завершения всего набора.

## Baseline

Источник чисел — `tests/eval/results/2026-09-28_1545_baseline.csv`.

| Вариант | faithfulness | answer relevancy | context precision | context recall | has citation | avg latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline: 512/64, top-K 3 | 0.6468 | 0.5284 | 0.6452 | 0.6452 | 0.6774 | 5593 ms |

## Эксперимент A — chunking

Изменён только размер чанка: `512 → 1024`; overlap и top-K остались `64` и
`3`. Артефакт: `tests/eval/results/2026-09-28_1754_chunk_1024.csv`.

| Вариант | faithfulness | answer relevancy | context precision | context recall | has citation | avg latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline: 512/64, top-K 3 | 0.6468 | 0.5284 | 0.6452 | 0.6452 | 0.6774 | 5593 ms |
| A: 1024/64, top-K 3 | 0.6774 | 0.5206 | 0.6452 | 0.6452 | 0.7097 | 7082 ms |

## Эксперимент B — retrieval

Изменён только top-K: `3 → 5`; chunking остался `512/64`.
Артефакт: `tests/eval/results/2026-09-28_2103_top_k_5.csv`.

| Вариант | faithfulness | answer relevancy | context precision | context recall | has citation | avg latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline: 512/64, top-K 3 | 0.6468 | 0.5284 | 0.6452 | 0.6452 | 0.6774 | 5593 ms |
| B: 512/64, top-K 5 | 0.7097 | 0.5257 | 0.6452 | 0.6452 | 0.7097 | 7573 ms |

Беру **B: top-K 5**, потому что у него максимальные faithfulness (0.7097) и
has_citation (0.7097), при одинаковых precision/recall. Default закреплён в
`app/core/config.py` и `.env.example`. Цена выбора — latency выше baseline на
1980 ms, а answer relevancy ниже на 0.0027.

## Phoenix

Phoenix доступен в Compose на `http://localhost:6006`; OTLP endpoint задаётся
`PHOENIX_COLLECTOR_ENDPOINT`. 28 сентября при временном
`PHOENIX_TRACING_ENABLED=true` выполнены 20 разнообразных запросов к
`/rag/query`. Phoenix принял 80 спанов: по 20 `RAG query`, `RAG retrieval`,
`RAG synthesis` и `ChatCompletion`. У каждого запроса один trace: retrieval
и synthesis — дочерние query, LLM — дочерний synthesis; retriever-span хранит
top-K, reranker и число найденных документов. HallucinationEvaluator остаётся
опциональным следующим шагом.

## Failure analysis

В B пять худших строк имеют `faithfulness=0`, `context_recall=0`,
`context_precision=0`, `answer_relevancy=0`, `has_citation=0` и fallback
«По базе не нашёл, могу эскалировать.»:

| user input | retrieved contexts, кратко | диагноз |
| --- | --- | --- |
| Как настроить proxy? | Wi-Fi-инструкция вместо proxy | низкий faithfulness + низкий recall: retrieval |
| Не работает DNS | VPN-инструкция вместо DNS | низкий faithfulness + низкий recall: retrieval |
| Что делать с phishing-письмом? | password и office-plants | низкий faithfulness + низкий recall: retrieval |
| Старая форма после релиза | MFA-инструкция | низкий faithfulness + низкий recall: retrieval |
| Потерял рабочее устройство | MFA-инструкция | низкий faithfulness + низкий recall: retrieval |

Это не generation-проблема: score guard корректно не даёт LLM выдумать ответ.
Eval использовал `RAG_DATA_DIR=data/rag-block-03`, а часть эталонных тем лежит
вне этого поднабора `data/` и физически не индексировалась. До нового A/B надо
либо переиндексировать полный `data/`, либо сузить golden до этого корпуса и
сначала получить новый baseline.

## Ограничения и план

RAGAS — LLM-as-judge, поэтому повторный прогон может отличаться на 5–10%.
Нельзя сравнивать historical baseline после смены judge model.
`answer_relevancy=0.5257` и `has_citation=0.7097` ниже целевых 0.7 и 0.95;
после согласования corpus/golden стоит проверить re-ranker и усилить правило
о citation в ответе.
