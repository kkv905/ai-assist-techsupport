import asyncio
from typing import Any
from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


class AsyncLLMClient:
    def __init__(self, llm, cache, ttl: int = 3600):
        self.llm = llm
        self.cache = cache
        self.ttl = ttl
    