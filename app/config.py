import os
from dotenv import load_dotenv
load_dotenv()

def get_env_required(key: str) -> str:
    """
    Считываем значение ключа из env
    :param key: Название ключа
    :return: Значение ключа
    """
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Не найдена переменная {key}. Проверь файл .env и загрузку через load_dotenv().")
    return value

def get_openai_api_key() -> str:
    """Получаем ключ для OpenAI"""
    return get_env_required("OPENAI_API_KEY")