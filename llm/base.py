"""
llm/base.py — Abstract LLM interface + Intent data model.

All LLM backends must implement `classify_intent` and `chat`.
The IntentResult dataclass is the canonical output of intent classification.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


# ── Intent Registry ───────────────────────────────────────────────────────────

class Intent(str, Enum):
    CREATE_FILE      = "create_file"
    WRITE_CODE       = "write_code"
    SUMMARIZE_TEXT   = "summarize_text"
    GENERAL_CHAT     = "general_chat"

    @classmethod
    def values(cls):
        return [e.value for e in cls]


# ── Intent Result ─────────────────────────────────────────────────────────────

@dataclass
class IntentResult:
    """
    Structured output of the intent classification step.

    Fields
    ------
    intent     : One of the Intent enum values.
    entities   : Key–value pairs extracted from the utterance
                 (e.g. {"filename": "retry.py", "language": "python"}).
    confidence : Float in [0, 1] indicating model confidence.
    raw_text   : The original user utterance (for reference).
    error      : Populated if classification failed.
    """
    intent: str
    entities: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    raw_text: str = ""
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "intent":     self.intent,
            "entities":   self.entities,
            "confidence": round(self.confidence, 3),
        }

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ── Abstract LLM ─────────────────────────────────────────────────────────────

class BaseLLM(ABC):
    """
    Abstract interface for all LLM backends.

    Every backend implements two methods:
      1. classify_intent  — structured intent extraction
      2. chat             — free-form text generation
    """

    @abstractmethod
    def classify_intent(self, user_text: str) -> IntentResult:
        """
        Analyse `user_text` and return an IntentResult.

        The implementation must always return a valid IntentResult,
        even on failure (set the `error` field instead of raising).
        """

    @abstractmethod
    def chat(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate a free-form text response.

        Parameters
        ----------
        prompt        : The user's input / task description.
        system_prompt : Optional system / instruction prompt.

        Returns
        -------
        str — the model's response text.
        """

    # ── Shared Helpers ────────────────────────────────────────────────────────

    def _build_intent_system_prompt(self) -> str:
        return f"""You are an intent classification engine for a voice-controlled AI agent.

Analyse the user's text and respond ONLY with a valid JSON object — no preamble, no markdown.

JSON schema:
{{
  "intent":     "<one of: {', '.join(Intent.values())}>",
  "entities":   {{<key-value pairs relevant to the intent>}},
  "confidence": <float 0.0–1.0>
}}

Intent guide:
- create_file     : user wants a file or folder created/saved
- write_code      : user wants code generated (often implies create_file too)
- summarize_text  : user wants text condensed/explained
- general_chat    : anything else (questions, greetings, conversation)

Entity examples:
- write_code      : {{"language": "python", "task": "retry function", "filename": "retry.py"}}
- create_file     : {{"filename": "notes.txt", "content": "..."}}
- summarize_text  : {{"text": "<the text to summarise>"}}
- general_chat    : {{"topic": "..."}}

If unsure, use general_chat with confidence 0.5.
"""

    def _parse_intent_json(self, raw: str, user_text: str) -> IntentResult:
        """
        Attempt to parse the LLM's raw output as an IntentResult.
        Falls back to general_chat on any parse error.
        """
        import re

        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

        # Find the first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return IntentResult(
                intent=Intent.GENERAL_CHAT,
                confidence=0.5,
                raw_text=user_text,
                error=f"No JSON found in LLM response: {raw[:200]}",
            )

        try:
            data = json.loads(match.group())
            intent_str = data.get("intent", Intent.GENERAL_CHAT).lower()

            # Validate intent
            if intent_str not in Intent.values():
                intent_str = Intent.GENERAL_CHAT

            return IntentResult(
                intent=intent_str,
                entities=data.get("entities", {}),
                confidence=float(data.get("confidence", 0.7)),
                raw_text=user_text,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            return IntentResult(
                intent=Intent.GENERAL_CHAT,
                confidence=0.5,
                raw_text=user_text,
                error=f"JSON parse error: {exc}",
            )
