import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE_PATH = LOG_DIR / "function_calling.jsonl"

def setup_logger() -> logging.Logger:
    """Настраивает JSONL-логгер для полного цикла function calling."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("function_calling")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(
        LOG_FILE_PATH,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    return logger

def write_log_event(
    logger: logging.Logger,
    event: str,
    run_id: str,
    **payload: Any,
) -> None:
    """Записывает одно событие в лог в формате JSONL."""
    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "run_id": run_id,
        **payload,
    }

    logger.info(
        json.dumps(
            log_record,
            ensure_ascii=False,
            default=str,
        )
    )