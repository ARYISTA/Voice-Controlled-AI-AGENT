"""
llm/ollama_llm.py — Local LLM via Ollama.

Ollama runs open-source models (Llama 3, Mistral, Phi-3, etc.) locally.

Setup:
  1. Install Ollama: https://ollama.com/download
  2. Pull a model: ollama pull llama3
  3. Ensure Ollama is running: ollama serve
"""

import json
import time
from typing import Optional

import requests

from llm.base import BaseLLM, IntentResult, Intent
from utils.logger import get_logger
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = get_logger(__name__)


class OllamaLLM(BaseLLM):
    """
    LLM backend that calls a locally-running Ollama instance.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = 900,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._chat_endpoint = f"{self.base_url}/api/chat"
        logger.info("OllamaLLM configured: model=%s url=%s", model, base_url)

    # ── Public API ────────────────────────────────────────────────────────────

    def classify_intent(self, user_text: str) -> IntentResult:
        """Classify the intent of `user_text` using Ollama."""
        system = self._build_intent_system_prompt()
        raw = self._call(user_text, system_prompt=system)
        result = self._parse_intent_json(raw, user_text)

        if result.error:
            logger.warning("Intent parse warning: %s", result.error)
        else:
            logger.info(
                "Intent: %s (confidence=%.2f)", result.intent, result.confidence
            )
        return result

    def chat(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a free-form response."""
        return self._call(prompt, system_prompt=system_prompt)

    # ── Private ───────────────────────────────────────────────────────────────

    def _call(self, user_message: str, system_prompt: str = "") -> str:
        """Make a blocking call to the Ollama /api/chat endpoint."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2},
        }

        try:
            t0 = time.perf_counter()
            resp = requests.post(
                self._chat_endpoint,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.perf_counter() - t0

            content = data.get("message", {}).get("content", "")
            logger.debug("Ollama response in %.2fs: %s…", elapsed, content[:80])
            return content

        except requests.exceptions.ConnectionError:
            msg = (
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? (ollama serve)"
            )
            logger.error(msg)
            raise ConnectionError(msg)

        except requests.exceptions.Timeout:
            msg = f"Ollama request timed out after {self.timeout}s."
            logger.error(msg)
            raise TimeoutError(msg)

        except Exception as exc:
            logger.exception("Ollama call failed: %s", exc)
            raise
