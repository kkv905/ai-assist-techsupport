from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPTS_DIR = Path(__file__).resolve().parent

def build_prompt_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(PROMPTS_DIR),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )

def load_prompt(file_name: str, **context: object) -> str:
    """ Загружаем промпт если jinja шаблон """
    env = build_prompt_environment()
    template = env.get_template(file_name)
    return template.render(**context).strip()

def load_text_prompt(relative_path: str) -> str:
    """ Загружаем промпт если .md/.txt шаблоны """
    prompt_path = PROMPTS_DIR / relative_path

    if not prompt_path.exists():
        raise FileNotFoundError(f"Файл пропта не найден: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8").strip()

