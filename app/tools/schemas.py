from app.prompts.loader import load_text_prompt

SEARCH_KNOWLEDGE_BASE_TOOL_NAME = "search_knowledge_base"

def build_search_knowledge_base_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": SEARCH_KNOWLEDGE_BASE_TOOL_NAME,
            "description": load_text_prompt(f"tools/{SEARCH_KNOWLEDGE_BASE_TOOL_NAME}.md"),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Краткое описание технической проблемы пользователя: "
                            "симптом, текст ошибки, сервис, технология, система "
                            "или ситуация, по которой требуется найти решение "
                            "в базе знаний техподдержки."
                        ),
                        "minLength": 3,
                    },
                    "ips": {
                        "type": "string",
                        "description": (
                            "Подразделение или группа техподдержки, к которой "
                            "относится вопрос пользователя. Примеры значений: "
                            "'', 'АИС Правоохрана', 'АИС ЦРСВЭД', 'АИС Постконтроль'. "
                            "Если подразделение не указано или не определяется из запроса, "
                            "передать пустую строку."
                        ),
                        "enum": [
                            "",
                            "АИС Правоохрана",
                            "АИС ЦРСВЭД",
                            "АИС Постконтроль",
                        ],
                    },
                },
                "required": ["query", "ips"],
                "additionalProperties": False,
            },
        },
    }

TOOLS = [
    build_search_knowledge_base_tool()
]