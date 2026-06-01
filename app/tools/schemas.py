

tools = [
    {
        "type": "function",
        "name": "search_knowledge_base",
        "description": "Поиск",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Описание",
                },
                "product": {
                    "type": "string",
                    "description": "Описание",
                },
            },
            "required": ["query", "product"],
        },
    },
]