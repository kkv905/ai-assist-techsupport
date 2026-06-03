from app.tools.schemas import TOOLS
from app.tools.handlers import search_knowledge_base
from app.prompts.loader import load_prompt
import json


def main():
    # result = search_knowledge_base("Ошибка авторизации access denied", "АИС Правоохрана")
    # result = search_knowledge_base("Не работает приложение, ошибка 500", "")
    # print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=== Вывод проверки load_prompt('system_v1.j2') ===")
    print(load_prompt("system_v1.j2"))
    print("=== Вывод проверки TOOLS ===")
    print(json.dumps(TOOLS, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()