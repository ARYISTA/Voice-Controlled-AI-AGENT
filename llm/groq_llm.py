"""
llm/groq_llm.py — Cloud LLM via Groq API (free tier available).

Groq provides very fast inference on open-source models (Llama 3, Mixtral).
Use this as a fallback when Ollama is not available.

Setup:
  1. Create a free account at https://console.groq.com
  2. Generate an API key
  3. Export: GROQ_API_KEY=<your-key>
  4. Set: LLM_BACKEND=groq
"""

import time

from llm.base import BaseLLM, IntentResult
from utils.logger import get_logger
from config import GROQ_API_KEY, GROQ_MODEL

logger = get_logger(__name__)


class GroqLLM(BaseLLM):
    """LLM backend using Groq's cloud API."""

    def __init__(self, api_key: str = GROQ_API_KEY, model: str = GROQ_MODEL):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Set LLM_BACKEND=ollama to use a local model instead."
            )
        self.api_key = api_key
        self.model = model
        logger.info("GroqLLM configured: model=%s", model)

    def classify_intent(self, user_text: str) -> IntentResult:
        system = self._build_intent_system_prompt()
        raw = self._call(user_text, system_prompt=system)
        result = self._parse_intent_json(raw, user_text)
        logger.info("Intent: %s (confidence=%.2f)", result.intent, result.confidence)
        return result

    def chat(self, prompt: str, system_prompt: str = "") -> str:
        return self._call(prompt, system_prompt=system_prompt)

    def _call(self, user_message: str, system_prompt: str = "") -> str:
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("groq not installed. Run: pip install groq")

        client = Groq(api_key=self.api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        elapsed = time.perf_counter() - t0

        content = response.choices[0].message.content
        logger.debug("Groq response in %.2fs: %s…", elapsed, content[:80])
        return content
