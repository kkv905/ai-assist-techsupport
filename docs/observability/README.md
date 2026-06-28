# Observability

Артефакты ручной проверки Phoenix UI:

- `phoenix-overview.png` — обзорная страница Projects, где видно проект `ai-assistant-techsupport`, общее число трейсов и медианную latency.
- `phoenix-project.png` — страница проекта со списком span'ов.

Что видно на `phoenix-project.png`:

- проект `ai-assistant-techsupport` открылся в Phoenix UI;
- во вкладке `Spans` отображается span `ChatCompletion` со статусом `ok`;
- в таблице видны поля `input`, `output`, `start time`;
- справа в блоке `Stats` видны `Total Traces = 1`, `Total Cost < $0.01`, `Latency P50 = 1.5s`, `Latency P99 = 1.5s`.

Этот скриншот подтверждает, что запрос к `/chat` породил trace, который успешно дошел до Phoenix.
